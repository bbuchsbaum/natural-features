"""Load the packaged public feature-catalogue contract."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Any

import yaml

_CONTRACT_PACKAGE = "natural_features.data"
_CONTRACT_RESOURCE = "r_public_feature_contracts.yaml"


@lru_cache(maxsize=1)
def load_r_public_feature_contracts() -> dict[str, Any]:
    """Return the packaged R public feature contract manifest."""
    resource = files(_CONTRACT_PACKAGE).joinpath(_CONTRACT_RESOURCE)
    with resource.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), dict):
        raise RuntimeError(f"Invalid packaged public feature contract: {_CONTRACT_RESOURCE}")
    return payload


@lru_cache(maxsize=1)
def public_feature_ids() -> frozenset[str]:
    """Return feature IDs in the public R-compatible catalogue."""
    features = load_r_public_feature_contracts()["features"]
    return frozenset(str(feature_id) for feature_id in features)
