from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np

from natural_features.features.language.embed import bert_word_embeddings, lm_hidden_states
from natural_features.features.preprocess import text_tokenize


class _Tensor:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


def _install_context_oracle(monkeypatch) -> dict[str, object]:  # noqa: ANN001
    state: dict[str, object] = {"tokenizer_calls": [], "model_calls": 0}

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args: object, **kwargs: object) -> "Tokenizer":
            assert kwargs["use_fast"] is True
            return cls()

        def __call__(self, text: str, **kwargs: object) -> dict[str, _Tensor]:
            assert kwargs == {
                "add_special_tokens": True,
                "return_offsets_mapping": True,
                "return_tensors": "pt",
                "truncation": False,
            }
            state["tokenizer_calls"].append(text)  # type: ignore[union-attr]
            assert text == "bank bank"
            return {
                "input_ids": _Tensor([[101, 7, 7, 102]]),
                "offset_mapping": _Tensor([[[0, 0], [0, 4], [5, 9], [0, 0]]]),
            }

    class Model:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> "Model":
            return cls()

        def __call__(self, **kwargs: object) -> object:
            assert kwargs["output_hidden_states"] is True
            state["model_calls"] = int(state["model_calls"]) + 1
            # The repeated lexical item has a different state at its second
            # position. An implementation that evaluates each word alone
            # cannot reproduce this joint-sequence oracle in one model call.
            layer0 = np.zeros((1, 4, 2), dtype=np.float32)
            layer1 = np.asarray(
                [[[0.0, 0.0], [10.0, 1.0], [20.0, 2.0], [0.0, 0.0]]],
                dtype=np.float32,
            )
            return SimpleNamespace(hidden_states=(_Tensor(layer0), _Tensor(layer1)))

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: _NoGrad()
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModel = Model
    transformers.AutoModelForCausalLM = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return state


def test_bert_words_are_jointly_contextual_and_word_aligned(monkeypatch) -> None:  # noqa: ANN001
    state = _install_context_oracle(monkeypatch)

    result = bert_word_embeddings(
        text_tokenize("bank bank"),
        model="fake-bert",
        layers=[1],
    )

    np.testing.assert_array_equal(
        result.values[:, 0, :],
        np.asarray([[10.0, 1.0], [20.0, 2.0]], dtype=np.float32),
    )
    assert state == {"tokenizer_calls": ["bank bank"], "model_calls": 1}
    assert result.metadata["context_encoding"] == "joint_sequence"
    assert result.metadata["word_alignment"] == "tokenizer_offsets"


def test_causal_hidden_states_use_the_same_joint_context_contract(monkeypatch) -> None:  # noqa: ANN001
    state = _install_context_oracle(monkeypatch)

    result = lm_hidden_states(
        text_tokenize("bank bank"),
        model="fake-causal",
        layers=[1],
        pooling="last_subword",
    )

    np.testing.assert_array_equal(
        result.values[:, 0, :],
        np.asarray([[10.0, 1.0], [20.0, 2.0]], dtype=np.float32),
    )
    assert state == {"tokenizer_calls": ["bank bank"], "model_calls": 1}
    assert result.metadata["backend"] == "transformers_causal_lm"
