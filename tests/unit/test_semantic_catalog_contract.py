from __future__ import annotations

import numpy as np
import pytest

from natural_features.core.registry import Registry
from natural_features.features.language.predictability import (
    _aggregate_subwords_to_words,
    _token_negative_log_probabilities,
)


def test_catalogue_has_no_same_name_fallback_controls() -> None:
    registry = Registry.with_builtin_specs()

    for spec in registry.list():
        assert "fallback" not in spec.tags, spec.name
        assert "strict_dependency" not in spec.params, spec.name
        execution = spec.params.get("execution_mode")
        if execution is not None:
            assert execution["default"] == "strict", spec.name
            assert execution["choices"] == ["strict"], spec.name


def test_catalogue_rejects_unknown_parameter_schema_types() -> None:
    registry = Registry()

    with pytest.raises(ValueError, match="Unsupported catalogue parameter type"):
        registry.register(
            {
                "name": "test.invalid_schema",
                "impl": "natural_features.features.preprocess:text_tokenize",
                "params": {
                    "method": {
                        "type": "stringly_typo",
                        "default": "anything",
                    }
                },
            }
        )


def test_token_surprisal_matches_analytic_softmax_oracle() -> None:
    logits = np.asarray(
        [
            [np.log(3.0), 0.0],
            [0.0, 0.0],
        ]
    )
    observed = _token_negative_log_probabilities(
        logits,
        np.asarray([0, 1]),
    )

    np.testing.assert_allclose(
        observed,
        np.asarray([-np.log(0.75), np.log(2.0)]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_subword_surprisal_sums_within_each_word() -> None:
    observed = _aggregate_subwords_to_words(
        np.asarray([0.2, 0.3, 0.7]),
        np.asarray([[0, 1], [1, 3], [4, 7]]),
        [(0, 3), (4, 7)],
        ["one", "two"],
    )

    np.testing.assert_allclose(observed[:, 0], np.asarray([0.5, 0.7]))
