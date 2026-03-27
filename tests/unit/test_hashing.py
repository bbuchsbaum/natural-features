from __future__ import annotations

import numpy as np

from natural_features.util.hashing import stable_hash


def test_stable_hash_is_order_independent_for_dict_keys() -> None:
    left = {"b": 2, "a": 1, "nested": {"y": 2, "x": 1}}
    right = {"a": 1, "nested": {"x": 1, "y": 2}, "b": 2}
    assert stable_hash(left) == stable_hash(right)


def test_stable_hash_changes_with_ndarray_content() -> None:
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([1, 2, 4], dtype=np.float32)
    assert stable_hash({"arr": a}) != stable_hash({"arr": b})
