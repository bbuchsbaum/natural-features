"""Versioned, narrow English phone decisions for singing posterior audits."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

_RELATIONS = {"exact", "coarse-equivalence", "sequence", "ambiguous", "unsupported"}


@dataclass(frozen=True)
class PhoneDecision:
    token: str
    normalized_token: str
    phones: tuple[str, ...]
    relation: str
    reason: str
    source: str
    alternatives: tuple[tuple[str, ...], ...] = ()
    canonical_kind: str = "nus"

    @property
    def mapped(self) -> bool:
        return bool(self.phones) and self.relation != "ambiguous"


class PhoneInventory:
    def __init__(self, payload: Mapping[str, Any]):
        if payload.get("schema") != "EnglishSingingPhoneInventory/v2":
            raise ValueError("expected EnglishSingingPhoneInventory/v2")
        self.payload = dict(payload)
        self._decisions = self._read(payload.get("mappings", {}), mapped=True)
        self._ambiguous = self._read(payload.get("ambiguous", {}), mapped=False)

    @staticmethod
    def _read(raw: Any, *, mapped: bool) -> dict[str, PhoneDecision]:
        if not isinstance(raw, Mapping):
            raise ValueError("inventory decisions must be an object")
        result = {}
        for token, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"decision for {token!r} must be an object")
            relation = str(value.get("relation", ""))
            phones = tuple(str(x) for x in value.get("phones", ()))
            alternatives = tuple(tuple(str(x) for x in option) for option in value.get("alternatives", ()))
            if relation not in _RELATIONS or any(not p for p in phones) or any(not option or any(not p for p in option) for option in alternatives):
                raise ValueError(f"invalid decision for {token!r}")
            if mapped and (relation in {"ambiguous", "unsupported"} or not phones):
                raise ValueError(f"mapped decision for {token!r} needs phones and a mapped relation")
            if not mapped and (relation != "ambiguous" or phones or not alternatives):
                raise ValueError(f"ambiguous decision for {token!r} needs alternatives only")
            source = str(value.get("source", "Pinned checkpoint vocabulary; IPA symbol semantics; NUS-48E annotation inventory."))
            result[str(token)] = PhoneDecision(str(token), str(token), phones, relation, str(value.get("reason", "")), source, alternatives, str(value.get("canonical_kind", "nus")))
        return result

    def decide(self, token: str) -> PhoneDecision:
        normalized = str(token).replace("ˈ", "").replace("ˌ", "")
        found = self._decisions.get(normalized) or self._ambiguous.get(normalized)
        if found is None:
            return PhoneDecision(str(token), normalized, (), "unsupported", "No English decision is declared for this token.", "none")
        return PhoneDecision(str(token), normalized, found.phones, found.relation, found.reason, found.source, found.alternatives, found.canonical_kind)


def load_phone_inventory(path: str | Path) -> PhoneInventory:
    return PhoneInventory(json.loads(Path(path).read_text(encoding="utf-8")))


def audit_inventory(labels: Sequence[str], token_roles: Mapping[str, str]) -> dict[str, Any]:
    if set(labels) != set(token_roles) or len(labels) != len(set(labels)):
        raise ValueError("token_roles must name exactly the unique labels")
    roles = Counter(str(token_roles[label]) for label in labels)
    if set(roles) - {"blank", "boundary", "special", "phone"}:
        raise ValueError("invalid token role")
    return {"n_labels": len(labels), "roles": dict(sorted(roles.items()))}


def audit_decisions(labels: Sequence[str], token_roles: Mapping[str, str], inventory: PhoneInventory) -> dict[str, Any]:
    result = audit_inventory(labels, token_roles)
    counts = Counter(inventory.decide(label).relation for label in labels if token_roles[label] == "phone")
    result["phone_decisions"] = {key: int(counts.get(key, 0)) for key in sorted(_RELATIONS)}
    result["mapped_phone_columns"] = int(sum(counts[key] for key in ("exact", "coarse-equivalence", "sequence")))
    result["unresolved_phone_columns"] = int(counts["ambiguous"] + counts["unsupported"])
    return result
