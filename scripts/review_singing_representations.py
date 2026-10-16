"""Independent matrix/scalar replay of the frozen singing representation report."""

import argparse
import json
import hashlib
import subprocess
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEAD = "b20a5d30aae14d90c7a475909e8acf59faec95cd"
reportpath = ROOT / "tests/benchmarks/reports/singing_representation_v2.json"
protocolpath = ROOT / "tests/benchmarks/thresholds/singing_representation_v2.json"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, default=reportpath)
parser.add_argument("--out-json", type=Path, required=True)
args = parser.parse_args()
reportpath = args.report


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    return Path(path) if Path(path).is_absolute() else ROOT / path


p = json.load(protocolpath.open())
r = json.load(reportpath.open())
assert (
    sha(protocolpath)
    == r["protocol_sha256"]
    == "fab25fe7c7533022846b5b99cb918f6095ecb49cdab1393f260f32fdadde47fd"
)
for path, h in p["frozen_sources"].items():
    assert sha(ROOT / path) == h
    assert (
        hashlib.sha256(
            subprocess.check_output(["git", "show", HEAD + ":" + path], cwd=ROOT)
        ).hexdigest()
        == h
    )
for spec in p["inputs"].values():
    assert sha(resolve(spec["path"])) == spec["sha256"]
assert (
    r["reservation_sha256"] == p["inputs"]["reservation"]["sha256"]
    and r["qualified"] is False
    and r["fresh_data_consumed"] is False
)
manifest = json.load(resolve(p["inputs"]["manifest"]["path"]).open())
old = json.load(resolve(p["inputs"]["v1_report"]["path"]).open())
inventory = json.load(resolve(p["inputs"]["mapping"]["path"]).open())
assert old["manifest_sha256"] == p["inputs"]["manifest"]["sha256"]
items = {x["id"]: x for x in manifest["items"]}
previous = {x["id"]: x for x in old["items"]}
arms = {x["id"]: x for x in r["arms"]}
assert len(items) == len(arms) == len(r["arms"]) == 32 and set(items) == set(arms)
max_error = 0.0
count = Counter()
nulls = Counter()
changes = set()
details = []


def equal(a, b, tag):
    global max_error
    if a is None or b is None:
        assert a is None and b is None, (tag, "null mismatch")
        return
    if isinstance(a, bool) or isinstance(b, bool):
        assert a == b, (tag, a, b)
        return
    x, y = np.asarray(a, float), np.asarray(b, float)
    assert x.shape == y.shape, (tag, "shape")
    error = float(np.max(abs(x - y))) if x.size else 0.0
    max_error = max(max_error, error)
    assert np.allclose(x, y, atol=1e-11, rtol=0), (tag, "difference", error)


def normalized(v):
    return None if np.sum(v) == 0 else (v / np.sum(v)).tolist()


def edit(ref, hyp):
    # Independent rolling-prefix dynamic program with the declared S/D/I tie order.
    prev = [(0, 0, j) for j in range(len(hyp) + 1)]
    for i, rr in enumerate(ref, 1):
        cur = [(0, i, 0)]
        for j, hh in enumerate(hyp, 1):
            if rr == hh:
                cur.append(prev[j - 1])
                continue
            s, d, ins = prev[j - 1]
            alts = [(s + 1, d, ins)]
            s, d, ins = prev[j]
            alts.append((s, d + 1, ins))
            s, d, ins = cur[-1]
            alts.append((s, d, ins + 1))
            cur.append(min(alts, key=lambda t: (sum(t), *t)))
        prev = cur
    s, d, i = prev[-1]
    return {
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "distance": s + d + i,
        "per": (s + d + i) / len(ref) if ref else None,
    }


for itemid, item in items.items():
    out = arms[itemid]
    c = out["comparison"]
    legacy = previous[itemid]["evaluation"]
    artifact = previous[itemid]["artifacts"]["generated_posterior"]
    assert sha(artifact["path"]) == artifact["sha256"] == out["posterior_sha256"]
    for k in ["model", "condition", "singer", "song"]:
        assert out[k] == item[k]
    with np.load(artifact["path"], allow_pickle=False) as z:
        P = np.asarray(z["values"], float)
        t = z["times_s"]
        labels = z["labels"].tolist()
        support = z["bounds_s"]
        hop = json.loads(z["timebase_json"].item())["hop_s"]
    assert (
        P.shape == (len(t), len(labels))
        and len(set(labels)) == len(labels)
        and np.isfinite(P).all()
        and (P >= 0).all()
    )
    assert np.allclose(P.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7), (
        itemid,
        "input normalization",
    )
    roles = [item["token_roles"][s] for s in labels]
    phonecols = np.array([role == "phone" for role in roles])
    phoneP = P[:, phonecols].sum(axis=1)
    # Each cell is the intersection of its native hop, its explicit support,
    # and its nearest-neighbour Voronoi region. Compute directly per row.
    cells = []
    for i, tt in enumerate(t):
        lo = max(
            tt - hop / 2,
            support[i, 0],
            (t[i - 1] + tt) / 2 if i else min(support[:, 0]),
        )
        hi = min(
            tt + hop / 2,
            support[i, 1],
            (tt + t[i + 1]) / 2 if i + 1 < len(t) else max(support[:, 1]),
        )
        assert lo < hi
        cells.append((lo, hi))
    cells = np.array(cells)
    widths = np.diff(cells, axis=1)[:, 0]
    rawid = P.argmax(axis=1)
    eventmask = np.ones(len(t), bool)
    eventmask[1:] = (rawid[1:] != rawid[:-1]) | (cells[1:, 0] - cells[:-1, 1] > 1e-12)
    eventrows = np.flatnonzero(eventmask & phonecols[rawid])
    eventcols = rawid[eventrows]
    mappings = {}
    relations = {}
    for name in p["mappings"]:
        seqs = []
        rels = []
        for label, role in zip(labels, roles):
            seq = []
            rel = "unsupported"
            if role == "phone" and name == "v1_legacy_exact":
                seq = item["canonical_inventory"].get(label, [])
                seq = [seq] if isinstance(seq, str) else seq
                rel = "sequence" if len(seq) > 1 else "exact" if seq else "unsupported"
            elif role == "phone":
                normalized_label = label.translate(str.maketrans("", "", "ˈˌ"))
                decision = inventory["mappings"].get(normalized_label)
                if decision:
                    seq = decision["phones"]
                    rel = decision["relation"]
                elif normalized_label in inventory.get("ambiguous", {}):
                    rel = "ambiguous"
            seqs.append(seq)
            rels.append(rel)
        mappings[name] = seqs
        relations[name] = rels
    for label, role, s1, s2 in zip(labels, roles, *mappings.values()):
        if s1 != s2:
            changes.add((label, tuple(s1), tuple(s2)))
    refs = [x for x in item["reference_intervals"] if x["role"] == "phone"]
    refseq = [x["phone"] for x in refs]
    expectedphones = list(
        dict.fromkeys(q for seqs in mappings.values() for seq in seqs for q in seq)
    )
    expectedphones = list(dict.fromkeys(expectedphones + refseq))
    assert c["phones"] == expectedphones
    index = {v: i for i, v in enumerate(expectedphones)}
    K = len(index)
    tokenmass = widths @ P
    eventcounts = np.bincount(eventcols, minlength=len(labels))
    assert len(c["token_audit"]) == len(labels)
    for j, a in enumerate(c["token_audit"]):
        assert a["token"] == labels[j] and a["role"] == roles[j]
        equal(a["native_time_posterior_mass"], tokenmass[j], (itemid, "token mass", j))
        assert a["greedy_event_count"] == eventcounts[j]
        for name in mappings:
            d = a["mappings"][name]
            assert (
                d["phones"] == mappings[name][j]
                and d["relation"] == relations[name][j]
                and d["mapped"] == bool(mappings[name][j])
            )
        count["token_audits"] += 1
    for name, seqs in mappings.items():
        predicted = [
            q
            for col in eventcols
            for q in (seqs[col] or ["__unmapped_model_token__:" + labels[col]])
        ]
        metrics = edit(refseq, predicted)
        saved = c["recognition"][name]
        assert saved["reference_phone_count"] == len(refseq) and saved[
            "decoded_phone_count"
        ] == len(predicted)
        for metric, v in metrics.items():
            equal(saved[metric], v, (itemid, name, metric))
        if name == "v1_legacy_exact":
            assert (
                predicted == legacy["recognition"]["decoded_phones"]
                and refseq == legacy["recognition"]["reference_phones"]
            )
            for metric, v in metrics.items():
                equal(
                    legacy["recognition"][metric], v, (itemid, "archived PER", metric)
                )
        count["recognition_records"] += 1
    for section, boundkey, oldprofile, oldref in [
        (
            "composition",
            "window_bounds_s",
            "composition_profiles",
            "reference_composition_profiles",
        ),
        (
            "ordered_bins",
            "bin_bounds_s",
            "ordered_bin_profiles",
            "reference_bin_profiles",
        ),
    ]:
        bounds = np.array(item[boundkey])
        lo = bounds[:-1]
        hi = bounds[1:]
        B = len(lo)
        W = np.maximum(
            0,
            np.minimum(hi[:, None], cells[None, :, 1])
            - np.maximum(lo[:, None], cells[None, :, 0]),
        )
        obs = W.sum(axis=1)
        starts = np.array([x["start_s"] for x in refs])
        ends = np.array([x["end_s"] for x in refs])
        R = np.zeros((len(refs), K))
        for j, q in enumerate(refseq):
            R[j, index[q]] = 1
        D = (
            np.maximum(
                0,
                np.minimum(hi[:, None], ends[None, :])
                - np.maximum(lo[:, None], starts[None, :]),
            )
            @ R
        )
        occurrences = ((lo[:, None] <= starts[None, :]) & (starts[None, :] < hi[:, None])).astype(
            float
        ) @ R
        targets = {
            "duration": list(map(normalized, D)),
            "occurrence": list(map(normalized, occurrences)),
        }
        for target, profiles in targets.items():
            for b, v in enumerate(profiles):
                equal(
                    c[section]["reference_" + target + "_composition"][b],
                    v,
                    (itemid, section, "reference", target, b),
                )
                count["reference_profiles"] += 1
        E = np.zeros((B, len(labels)), int)
        for b in range(B):
            E[b] = np.bincount(
                eventcols[(t[eventrows] >= lo[b]) & (t[eventrows] < hi[b])],
                minlength=len(labels),
            )
        integrated = W @ P
        conditional = W @ np.divide(
            P, phoneP[:, None], out=np.zeros_like(P), where=phoneP[:, None] > 0
        )
        for name, seqs in mappings.items():
            A = np.zeros((len(labels), K))
            Aevent = np.zeros_like(A)
            mapped = np.array([bool(s) for s in seqs])
            for col, seq in enumerate(seqs):
                for q in seq:
                    A[col, index[q]] += 1 / len(seq)
                    Aevent[col, index[q]] += 1
            masses = {
                "raw_posterior_integral": integrated @ A,
                "framewise_phone_conditional": conditional @ A,
                "greedy_ctc_token_events": E @ Aevent,
            }
            branch = c[section]["mappings"][name]
            for b in range(B):
                ph = integrated[b, phonecols].sum()
                mp = integrated[b, mapped].sum()
                ct = W[b, phoneP > 0].sum()
                cm = conditional[b, mapped].sum()
                ev = E[b].sum()
                me = E[b, mapped].sum()
                cov = {
                    "observation_s": obs[b],
                    "phone_mass": ph,
                    "mapped_phone_mass": mp,
                    "unmapped_or_ambiguous_phone_mass": ph - mp,
                    "mapping_coverage": mp / ph if ph else None,
                    "conditional_observed_phone_time_s": ct,
                    "conditional_mapped_mass": cm,
                    "conditional_mapping_coverage": cm / ct if ct else None,
                    "raw_phone_event_count": ev,
                    "mapped_raw_phone_event_count": me,
                    "unmapped_or_ambiguous_raw_phone_event_count": ev - me,
                    "expanded_canonical_event_count": masses["greedy_ctc_token_events"][
                        b
                    ].sum(),
                    "raw_missing": masses["raw_posterior_integral"][b].sum() == 0,
                    "conditional_missing": masses["framewise_phone_conditional"][
                        b
                    ].sum()
                    == 0,
                    "event_missing": masses["greedy_ctc_token_events"][b].sum() == 0,
                }
                assert cov.keys() == branch["coverage"][b].keys()
                for key, v in cov.items():
                    equal(
                        branch["coverage"][b][key],
                        v,
                        (itemid, section, name, "coverage", key, b),
                    )
                count["coverage_records"] += 1
            for pooling, M in masses.items():
                saved = branch["representations"][pooling]
                for b, mass in enumerate(M):
                    profile = normalized(mass)
                    equal(
                        saved["profiles"][b],
                        profile,
                        (itemid, section, name, pooling, "profile", b),
                    )
                    count["predicted_profiles"] += 1
                    if profile is None:
                        nulls[section + "|" + name + "|" + pooling + "|profile"] += 1
                    for target, profiles in targets.items():
                        expected = (
                            None
                            if profile is None or profiles[b] is None
                            else 0.5
                            * sum(abs(x - y) for x, y in zip(profile, profiles[b]))
                        )
                        equal(
                            saved["comparisons"][b][target + "_total_variation"],
                            expected,
                            (itemid, section, name, pooling, target, b),
                        )
                        count["comparison_slots"] += 1
                        if expected is None:
                            nulls[
                                section + "|" + name + "|" + pooling + "|" + target
                            ] += 1
                            count["null_comparisons"] += 1
                        else:
                            count["non_null_comparisons"] += 1
                    if pooling == "greedy_ctc_token_events":
                        equal(
                            saved["event_totals"][b],
                            mass.sum(),
                            (itemid, "event total"),
                        )
                    if (
                        name == "v1_legacy_exact"
                        and pooling == "raw_posterior_integral"
                    ):
                        aligned = (
                            None
                            if profile is None
                            else [profile[index[q]] for q in legacy["phones"]]
                        )
                        equal(
                            legacy[oldprofile][b],
                            aligned,
                            (itemid, section, "archived raw profile"),
                        )
                        refaligned = (
                            None
                            if targets["duration"][b] is None
                            else [
                                targets["duration"][b][index[q]]
                                for q in legacy["phones"]
                            ]
                        )
                        equal(
                            legacy[oldref][b],
                            refaligned,
                            (itemid, section, "archived reference"),
                        )
                        equal(
                            legacy["profile_comparisons"][section][b][
                                "total_variation"
                            ],
                            saved["comparisons"][b]["duration_total_variation"],
                            (itemid, section, "archived TV"),
                        )
                        count["legacy_raw_bins"] += 1
    count["items"] += 1
receipt = {
    "reviewed_commit": HEAD,
    "protocol_sha256": sha(protocolpath),
    "report_sha256": sha(reportpath),
    "status": "PASS",
    "counts": dict(count),
    "maximum_absolute_numeric_difference": max_error,
    "absolute_tolerance": 1e-11,
    "mapping_changes": [
        {"token": x, "v1": list(a), "v2": list(b)} for x, a, b in sorted(changes)
    ],
    "null_counts": dict(sorted(nulls.items())),
    "independence": "Only JSON, NumPy and standard library; no production scoring/mapping/pooling imports. Matrix integration/mapping, independent scalar cell construction and edit-distance recurrence. No models or fresh data.",
}
args.out_json.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "null_counts"}, indent=2))
print("null counts retained in receipt")
