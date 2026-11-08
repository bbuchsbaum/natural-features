#!/usr/bin/env python3
"""Dependency-light descriptive acoustic/phonological overlap audit.

Run with a manifest; see docs/transcript_phonology.md. No neural inference or
pair-independent significance tests. These RDM helpers deliberately live here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.pronunciation import (
    PronunciationLexicon,
    coverage_report,
    extract_pronunciations,
)
from natural_features.features.speech.phone_sequences import (
    ItemFeatures,
    NgramSchema,
    PhoneEditDistance,
    articulatory_sequence,
    pronunciation_counts,
    sequence_distances,
)
from natural_features.workflows.sentence_comparison import (
    AcousticConfig,
    FeatureScaler,
    combine_scaled_blocks,
    extract_sentence_acoustics,
    feature_distances,
)


@dataclass(frozen=True)
class DistanceMatrix:
    item_ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self):
        v = np.asarray(self.values, dtype=float)
        if v.shape != (len(self.item_ids),) * 2 or len(set(self.item_ids)) != len(
            self.item_ids
        ):
            raise ValueError("RDM dimensions or item IDs invalid")
        if not np.allclose(v, v.T, equal_nan=True) or np.isinf(v).any():
            raise ValueError("RDM must be symmetric with matching missing support")
        if np.any(np.isfinite(np.diag(v)) & (np.diag(v) != 0)):
            raise ValueError("valid self-distances must be zero")
        object.__setattr__(self, "values", v)


def analysis_pairs(
    item_ids: Sequence[str],
    *,
    groups: Mapping[str, str] | None = None,
    within_group: bool = False,
    explicit: Sequence[Sequence[str]] | None = None,
) -> list[tuple[int, int]]:
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("duplicate item IDs")
    if groups is not None and set(groups) != set(item_ids):
        raise ValueError("groups must cover exactly the item IDs")
    if within_group and groups is None:
        raise ValueError("within-group analysis requires group labels")
    if explicit is None:
        pairs = list(itertools.combinations(range(len(item_ids)), 2))
    else:
        index = {item: i for i, item in enumerate(item_ids)}
        pairs = []
        for pair in explicit:
            if (
                len(pair) != 2
                or pair[0] not in index
                or pair[1] not in index
                or pair[0] == pair[1]
            ):
                raise ValueError("invalid explicit analysis pair")
            pairs.append(tuple(sorted((index[pair[0]], index[pair[1]]))))
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate unordered analysis pair")
    return [
        (i, j)
        for i, j in pairs
        if not within_group or groups[item_ids[i]] == groups[item_ids[j]]
    ]


def ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks for exact ties, evaluated only on selected analysis pairs."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2)[inverse]


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    ac, bc = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(ac), np.linalg.norm(bc)
    if na <= 1e-12 * max(1.0, np.linalg.norm(a)) or nb <= 1e-12 * max(
        1.0, np.linalg.norm(b)
    ):
        return None
    return float(np.clip(np.dot(ac / na, bc / nb), -1, 1))


def _design(
    controls: Mapping[str, np.ndarray], *, ranked: bool, groups: Sequence[str] | None
) -> tuple[np.ndarray, list[str]]:
    n = (
        len(next(iter(controls.values())))
        if controls
        else len(groups)
        if groups is not None
        else 0
    )
    columns = [np.ones(n)]
    names = ["intercept"]
    for name, values in controls.items():
        v = ranks(values) if ranked else np.asarray(values, float)
        std = v.std() if len(v) else 0
        columns.append((v - v.mean()) / std if std > 0 else np.zeros(n))
        names.append(name)
    if groups is not None:
        columns.extend(
            np.array([g == level for g in groups], float)
            for level in sorted(set(groups))[1:]
        )
        names.extend(f"group:{level}" for level in sorted(set(groups))[1:])
    return np.column_stack(columns), names


def _diagnostics(design: np.ndarray, names: list[str]) -> dict:
    if not len(design):
        return {
            "rank": 0,
            "columns": names,
            "rank_deficient": True,
            "condition_number": None,
            "residual_df": 0,
        }
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    return {
        "rank": rank,
        "columns": names,
        "rank_deficient": rank < design.shape[1],
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1])
        if rank == design.shape[1]
        else None,
        "residual_df": len(design) - rank,
    }


def audit_rdms(
    rdms: Mapping[str, DistanceMatrix],
    *,
    pairs: Sequence[tuple[int, int]],
    controls: Mapping[str, np.ndarray] | None = None,
    pair_groups: Sequence[str] | None = None,
) -> dict:
    """One common finite mask for every raw and conditional comparison."""
    if not rdms:
        raise ValueError("at least one representation required")
    ids = next(iter(rdms.values())).item_ids
    if any(r.item_ids != ids for r in rdms.values()):
        raise ValueError("RDM item IDs/order differ; align explicitly")
    if any(not 0 <= i < j < len(ids) for i, j in pairs) or len(set(pairs)) != len(
        pairs
    ):
        raise ValueError("pairs must be unique upper-triangle item indices")
    vectors = {
        name: np.array([r.values[i, j] for i, j in pairs]) for name, r in rdms.items()
    }
    controls = {name: np.asarray(v, float) for name, v in (controls or {}).items()}
    if any(v.shape != (len(pairs),) for v in controls.values()):
        raise ValueError("controls must have one value per candidate analysis pair")
    if pair_groups is not None and len(pair_groups) != len(pairs):
        raise ValueError("pair groups must match candidate pairs")
    valid = {name: np.isfinite(v) for name, v in vectors.items()}
    valid.update({f"control:{name}": np.isfinite(v) for name, v in controls.items()})
    common = np.logical_and.reduce(list(valid.values()))
    selected = {name: v[common] for name, v in vectors.items()}
    selected_controls = {name: v[common] for name, v in controls.items()}
    selected_groups = (
        [g for g, keep in zip(pair_groups, common) if keep]
        if pair_groups is not None
        else ["all"] * int(common.sum())
    )
    comparisons = {}
    designs = {}
    for ranked, method in ((False, "pearson"), (True, "spearman")):
        values = {name: ranks(v) if ranked else v for name, v in selected.items()}
        design, names = _design(
            selected_controls, ranked=ranked, groups=selected_groups
        )
        designs[method] = _diagnostics(design, names)
        residuals = {
            name: v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
            for name, v in values.items()
        }
        for a, b in itertools.combinations(values, 2):
            comparisons.setdefault(f"{a} vs {b}", {})[method] = {
                "raw": correlation(values[a], values[b]),
                "conditional": correlation(residuals[a], residuals[b])
                if designs[method]["residual_df"] >= 2
                else None,
            }
    joint, joint_names = _design(
        {**{f"rdm:{k}": v for k, v in selected.items()}, **selected_controls},
        ranked=False,
        groups=selected_groups,
    )
    return {
        "schema": "DescriptiveRDMOverlap/v1",
        "candidate_pairs": len(pairs),
        "matched_pairs": int(common.sum()),
        "pairs": [
            {
                "item_a": ids[i],
                "item_b": ids[j],
                "included": bool(common[k]),
                "excluded_by": [name for name, mask in valid.items() if not mask[k]],
                "distances": {
                    name: float(v[k]) if np.isfinite(v[k]) else None
                    for name, v in vectors.items()
                },
                "controls": {
                    name: float(v[k]) if np.isfinite(v[k]) else None
                    for name, v in controls.items()
                },
            }
            for k, (i, j) in enumerate(pairs)
        ],
        "comparisons": comparisons,
        "control_design": designs,
        "joint_design": _diagnostics(joint, joint_names),
        "conditional_policy": "OLS residual correlation; Spearman ranks distances and numeric controls on common pairs before residualizing; group indicators unranked",
        "inference": "descriptive_only_no_pair_independent_p_values",
        "undefined_policy": "null for insufficient support, constant vectors or exhausted residual degrees of freedom",
    }


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def run_comparison(
    manifest_path: Path,
    out: Path,
    *,
    selection: str,
    reference_ids: Sequence[str] | None = None,
    within_group: bool = False,
    acoustic_blocks: Sequence[str] = ("spectral_shape", "modulation"),
    embeddings: Path | None = None,
    embedding_metadata: Path | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    items = manifest["items"]
    ids = tuple(item["item_id"] for item in items)
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError("manifest requires at least two unique item IDs")
    if not acoustic_blocks or len(set(acoustic_blocks)) != len(acoustic_blocks):
        raise ValueError("choose distinct acoustic blocks")
    if out.exists():
        raise FileExistsError(
            f"output already exists: {out}; use a new output directory"
        )
    supplemental = (
        PronunciationLexicon.from_mapping(
            manifest["supplemental"], name="manifest_supplement"
        )
        if manifest.get("supplemental")
        else None
    )
    overrides = {
        item: {int(i): p for i, p in choices.items()}
        for item, choices in manifest.get("overrides", {}).items()
    }
    compose = manifest.get("compose", False)
    if type(compose) is not bool:
        raise ValueError("manifest 'compose' must be true or false")
    seqs = extract_pronunciations(
        dict(zip(ids, (i["text"] for i in items))),
        selection=selection,
        supplemental=supplemental,
        overrides=overrides,
        compose=compose,
    )
    groups = (
        {i["item_id"]: str(i["group"]) for i in items}
        if all("group" in i for i in items)
        else None
    )
    pairs = analysis_pairs(
        ids, groups=groups, within_group=within_group, explicit=manifest.get("pairs")
    )
    phone_blocks = {
        "unigram": NgramSchema(n=1).transform(seqs),
        "bigram_within": NgramSchema().transform(seqs),
        "bigram_cross": NgramSchema(boundary="cross").transform(seqs),
    }
    counts = pronunciation_counts(seqs)
    audio = {
        item["item_id"]: AudioStimulus.from_wav(manifest_path.parent / item["audio"])
        for item in items
    }
    cfg = AcousticConfig(**manifest.get("acoustic_config", {}))
    acoustics = extract_sentence_acoustics(audio, config=cfg)
    if set(acoustic_blocks) - set(acoustics.blocks) or "controls" in acoustic_blocks:
        raise ValueError(
            "unknown acoustic block or controls selected as representation"
        )
    if reference_ids is None:
        raise ValueError(
            "explicit scaling reference IDs are required; use all IDs only for a descriptive full-set audit"
        )
    scaled, scalers = {}, {}
    for name in acoustic_blocks:
        scaler = FeatureScaler.fit(acoustics.blocks[name], reference_ids=reference_ids)
        scaled[name] = scaler.transform(acoustics.blocks[name])
        if not scaled[name].names:
            raise ValueError(
                f"{name}: no valid varying reference coordinates; explicitly select a supported block set"
            )
        scalers[name] = scaler.to_dict()
    combined = combine_scaled_blocks(scaled, weights={name: 1.0 for name in scaled})
    edit = PhoneEditDistance()
    rdms = {
        name: DistanceMatrix(ids, feature_distances(b))
        for name, b in phone_blocks.items()
    }
    rdms["phone_edit"] = DistanceMatrix(ids, sequence_distances(seqs, edit))
    rdms.update(
        {
            f"acoustic_{name}": DistanceMatrix(
                ids, feature_distances(b, metric="rms_euclidean")
            )
            for name, b in scaled.items()
        }
    )
    rdms["acoustic_combined"] = DistanceMatrix(
        ids, feature_distances(combined, metric="euclidean")
    )
    embedding_provenance = None
    embedding_table = None
    if (embeddings is None) != (embedding_metadata is None):
        raise ValueError("embeddings require a companion provenance JSON")
    if embeddings:
        embedding_provenance = json.loads(embedding_metadata.read_text())
        if not all(
            embedding_provenance.get(k) for k in ("model", "revision", "text_sha256")
        ):
            raise ValueError(
                "embedding provenance requires model, revision and per-item text_sha256"
            )
        if embedding_provenance["text_sha256"] != {
            s.item_id: s.metadata["text_sha256"] for s in seqs
        }:
            raise ValueError("embedding text hashes do not match these transcripts")
        with np.load(embeddings, allow_pickle=False) as data:
            if tuple(data["item_ids"].tolist()) != ids or data["values"].ndim != 2:
                raise ValueError("embedding item IDs/order or dimensions differ")
            values = data["values"].copy()
        embedding_provenance["file_sha256"] = hashlib.sha256(
            embeddings.read_bytes()
        ).hexdigest()
        embedding_table = ItemFeatures(
            ids,
            tuple(f"embedding_{j}" for j in range(values.shape[1])),
            values,
            np.isfinite(values),
            embedding_provenance,
        )
        rdms["text_embedding"] = DistanceMatrix(ids, feature_distances(embedding_table))
    controls = {}
    for j, name in enumerate(counts.names):
        controls[name + "_difference"] = np.array(
            [
                abs(counts.values[i, j] - counts.values[k, j])
                if counts.valid[i, j] and counts.valid[k, j]
                else np.nan
                for i, k in pairs
            ]
        )
    duration = acoustics.blocks["controls"].values[:, 0]
    level = acoustics.blocks["controls"].values[:, 1]
    controls["duration_difference"] = np.array(
        [abs(duration[i] - duration[j]) for i, j in pairs]
    )
    controls["rms_difference"] = np.array([abs(level[i] - level[j]) for i, j in pairs])
    words = [set(w.normalized for w in s.words) for s in seqs]
    controls["lexical_jaccard_distance"] = np.array(
        [
            1 - len(words[i] & words[j]) / len(words[i] | words[j])
            if words[i] | words[j]
            else np.nan
            for i, j in pairs
        ]
    )
    pair_groups = (
        [json.dumps(sorted((groups[ids[i]], groups[ids[j]]))) for i, j in pairs]
        if groups
        else None
    )
    audit = audit_rdms(rdms, pairs=pairs, controls=controls, pair_groups=pair_groups)
    audit["pair_policy"] = {
        "within_group": within_group,
        "groups": groups,
        "explicit_pairs": manifest.get("pairs"),
    }
    audit["provenance"] = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "example_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "embeddings": embedding_provenance,
        "scaling_reference_ids": list(reference_ids),
    }
    out.mkdir(parents=True)
    _write_json(out / "pronunciations.json", [s.to_dict() for s in seqs])
    _write_json(
        out / "articulatory_sequences.json",
        {s.item_id: articulatory_sequence(s) if s.eligible else None for s in seqs},
    )
    _write_json(
        out / "coverage.json",
        coverage_report(seqs, pairs=[(ids[i], ids[j]) for i, j in pairs]),
    )
    _write_json(out / "scalers.json", scalers)
    _write_json(out / "phone_edit.json", edit.to_dict())
    _write_json(out / "overlap.json", audit)
    _write_json(out / "manifest.json", manifest)
    for name, block in {
        **phone_blocks,
        "counts": counts,
        **{f"audio_{k}": v for k, v in acoustics.blocks.items()},
        **{f"scaled_{k}": v for k, v in scaled.items()},
        "acoustic_combined": combined,
    }.items():
        block.save(out / name)
    if embedding_table:
        embedding_table.save(out / "text_embeddings")
    np.savez_compressed(
        out / "rdms.npz",
        item_ids=np.array(ids),
        **{k: r.values for k, r in rdms.items()},
    )
    for index, item_id in enumerate(ids):
        for name, block in acoustics.native[item_id].items():
            # Item IDs are data, never filesystem paths.
            prefix = out / f"native_{index:04d}_{name}"
            np.savez_compressed(
                prefix.with_suffix(".npz"),
                values=block.series.values,
                times_s=block.series.times_s,
                bounds_s=block.series.time_bounds_s,
                valid=block.valid,
                weights_s=block.weights_s,
            )
            _write_json(
                prefix.with_suffix(".json"),
                {
                    "item_id": item_id,
                    "feature_names": block.series.coords["feature"],
                    "timebase": block.series.timebase.to_dict(),
                    "temporal_context": block.series.temporal_context.to_dict(),
                    "coverage_s": block.coverage_s,
                    "metadata": block.series.metadata,
                },
            )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--selection", required=True, choices=("retain", "dictionary_order")
    )
    scaling = parser.add_mutually_exclusive_group(required=True)
    scaling.add_argument("--reference-ids", nargs="+")
    scaling.add_argument(
        "--fit-on-all",
        action="store_true",
        help="descriptive audit only; not held-out validation",
    )
    parser.add_argument("--within-group", action="store_true")
    parser.add_argument(
        "--acoustic-blocks", nargs="+", default=["spectral_shape", "modulation"]
    )
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--embedding-metadata", type=Path)
    args = parser.parse_args()
    refs = (
        [item["item_id"] for item in json.loads(args.manifest.read_text())["items"]]
        if args.fit_on_all
        else args.reference_ids
    )
    audit = run_comparison(
        args.manifest,
        args.out,
        selection=args.selection,
        reference_ids=refs,
        within_group=args.within_group,
        acoustic_blocks=args.acoustic_blocks,
        embeddings=args.embeddings,
        embedding_metadata=args.embedding_metadata,
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "matched_pairs": audit["matched_pairs"],
                "candidate_pairs": audit["candidate_pairs"],
            }
        )
    )


if __name__ == "__main__":
    main()
