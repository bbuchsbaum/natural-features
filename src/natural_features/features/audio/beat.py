"""Native Beat This frame activations with explicit assets and context."""

from __future__ import annotations

import hashlib
import inspect
from importlib.metadata import version
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np

from natural_features.core.backend_errors import (
    BackendDependencyError, BackendInferenceError, BackendLoadError,
)
from natural_features.core.execution import resolve_execution_mode
from natural_features.core.feature_types import FeatureSeries
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.audio._music_contract import (
    asset_digest, computation_context, mono_audio, music_series,
)

_SR, _HOP, _FFT = 22050, 441, 1024
_CHUNK, _BORDER = 1500, 6
_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"


def _dependencies():
    try:
        import torch
        from beat_this.inference import split_predict_aggregate
        from beat_this.model.beat_tracker import BeatThis
        from beat_this.preprocessing import LogMelSpect
        from beat_this.utils import replace_state_dict_key

        backend_version = version("beat-this")
    except (ImportError, OSError) as exc:
        raise BackendDependencyError("Beat This", "install natural-features[beats]") from exc
    if backend_version != "1.1.0":
        raise BackendDependencyError("Beat This", "timing contract requires beat-this==1.1.0")
    return SimpleNamespace(
        torch=torch, model=BeatThis, frontend=LogMelSpect,
        predict=split_predict_aggregate, replace_keys=replace_state_dict_key,
        version=backend_version,
        frontend_version=version("torchaudio"), rotary_version=version("rotary-embedding-torch"),
    )


def _checkpoint_path(checkpoint, *, cache_dir, local_files_only, expected_sha256, torch):
    path = Path(checkpoint).expanduser()
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("checkpoint_sha256 must be a 64-character SHA256 hex digest")
    if not path.is_file():
        # Only named upstream assets can be fetched; arbitrary path typos never
        # become URLs. Offline cache lookup does not call torch's network loader.
        if not re.fullmatch(r"(?:final|small)[0-2]", checkpoint):
            raise BackendLoadError("Beat This", f"checkpoint file does not exist: {checkpoint}")
        cache = Path(cache_dir) if cache_dir is not None else Path(torch.hub.get_dir()) / "checkpoints"
        path = cache / f"beat_this-{checkpoint}.ckpt"
        if not path.is_file():
            if local_files_only:
                raise BackendLoadError("Beat This", f"checkpoint is not cached: {path}")
            if expected_sha256 is None:
                raise ValueError("downloading requires checkpoint_sha256 from a trusted asset manifest")
            cache.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache, delete=False) as output:
                    temporary = Path(output.name)
                    digest = hashlib.sha256()
                    with urlopen(f"{_URL}/{checkpoint}.ckpt", timeout=60) as response:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            digest.update(block)
                            output.write(block)
                if digest.hexdigest() != expected_sha256.lower():
                    raise ValueError("downloaded checkpoint SHA256 mismatch")
                temporary.replace(path)
            except Exception as exc:
                raise BackendLoadError("Beat This", "checkpoint download failed") from exc
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256.lower():
            raise BackendLoadError("Beat This", "checkpoint SHA256 mismatch")
    return path


def _frontend_bounds(n_samples: int) -> np.ndarray:
    """Enclosing real-sample support of centered, reflection-padded STFTs."""
    centers = np.arange(1 + n_samples // _HOP) * _HOP
    left, right = centers - _FFT // 2, centers + _FFT // 2 - 1

    def reflect(index):
        return np.where(index < 0, -index, np.where(index >= n_samples, 2 * n_samples - 2 - index, index))

    lo = np.minimum(reflect(left), reflect(right))
    hi = np.maximum(reflect(left), reflect(right))
    lo = np.where((left <= 0) & (right >= 0), 0, lo)
    hi = np.where((left <= n_samples - 1) & (right >= n_samples - 1), n_samples - 1, hi)
    return np.column_stack([lo, hi + 1]) / _SR


def _chunk_context(frontend_bounds: np.ndarray) -> np.ndarray:
    """Source bounds for upstream keep_first overlapping chunk aggregation."""
    n = len(frontend_bounds)
    starts = np.arange(-_BORDER, n - _BORDER, _CHUNK - 2 * _BORDER)
    if n > _CHUNK - 2 * _BORDER:
        starts[-1] = n - (_CHUNK - _BORDER)
    result = np.full((n, 2), np.nan)
    assigned = np.zeros(n, dtype=bool)
    for start in starts:
        input_lo, input_hi = max(0, start), min(n, start + _CHUNK)
        output_lo, output_hi = max(0, start + _BORDER), min(n, start + _CHUNK - _BORDER)
        rows = np.arange(output_lo, output_hi)
        rows = rows[~assigned[rows]]
        result[rows] = [frontend_bounds[input_lo:input_hi, 0].min(),
                        frontend_bounds[input_lo:input_hi, 1].max()]
        assigned[rows] = True
    if not np.all(assigned):
        raise BackendInferenceError("Beat This", "chunk aggregation left frames without context")
    return result


def music_beat_activations(
    stimulus: AudioStimulus, *, checkpoint: str = "final0", cache_dir: str | None = None,
    checkpoint_sha256: str | None = None, local_files_only: bool = True,
    representation: str = "activation", device: str = "cpu", channel_policy: str = "mean",
    execution_mode: str | None = None,
) -> FeatureSeries:
    """Return beat/downbeat scores on the native 50 Hz grid.

    Inputs must already be 22050 Hz. Activation scores are sigmoid-transformed
    logits, not calibrated confidence. Local checkpoints are content identified;
    downloads require an explicit expected SHA256. Input windows determine the
    bidirectional context, including upstream overlapping 30-second chunks.
    """
    mode, _ = resolve_execution_mode(execution_mode=execution_mode)
    if not isinstance(local_files_only, bool):
        raise ValueError("local_files_only must be a boolean")
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint):
        raise ValueError("checkpoint must name a local file or cached model")
    checkpoint = str(checkpoint)
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if representation not in {"activation", "logit"}:
        raise ValueError("representation must be 'activation' or 'logit'")
    wav = mono_audio(stimulus, channel_policy=channel_policy)
    if stimulus.sr_hz != _SR:
        raise ValueError("Beat This requires 22050 Hz; resample explicitly before extraction")
    if len(wav) <= _FFT // 2:
        raise ValueError("Beat This reflection padding requires more than 512 audio samples")
    deps = _dependencies()
    path = _checkpoint_path(
        checkpoint, cache_dir=cache_dir, local_files_only=local_files_only,
        expected_sha256=checkpoint_sha256, torch=deps.torch,
    )
    try:
        revision = asset_digest(path)
        # Load through an open local stream: no missing-file network fallback.
        with path.open("rb") as stream:
            payload = deps.torch.load(stream, map_location="cpu", weights_only=True)
        params = {k: v for k, v in payload["hyper_parameters"].items()
                  if k in inspect.signature(deps.model).parameters}
        model = deps.model(**params)
        model.load_state_dict(deps.replace_keys(payload["state_dict"], "model.", ""))
        model = model.to(device).eval()
        frontend = deps.frontend(device=device)
    except Exception as exc:
        raise BackendLoadError("Beat This", "failed to load the local checkpoint or frontend") from exc
    try:
        with deps.torch.inference_mode():
            spect = frontend(deps.torch.tensor(wav, dtype=deps.torch.float32, device=device))
            expected = 1 + len(wav) // _HOP
            if tuple(spect.shape) != (expected, 128):
                raise ValueError("unexpected frontend frame count or mel dimension")
            predicted = deps.predict(
                spect=spect, chunk_size=_CHUNK, border_size=_BORDER,
                overlap_mode="keep_first", model=model,
            )
            columns = [predicted[key].detach().cpu().numpy() for key in ("beat", "downbeat")]
        if any(column.shape != (expected,) for column in columns):
            raise ValueError("backend returned an invalid activation shape")
        logits = np.column_stack(columns).astype(np.float64)
        if not np.all(np.isfinite(logits)):
            raise ValueError("backend returned non-finite logits")
    except Exception as exc:
        raise BackendInferenceError("Beat This", "frame inference failed") from exc
    values = logits
    if representation == "activation":
        exp = np.exp(-np.abs(logits))
        values = np.where(logits >= 0, 1 / (1 + exp), exp / (1 + exp))
    offset = float(stimulus.start_offset_s)
    bounds = _frontend_bounds(len(wav)) + offset
    context = computation_context(
        stimulus, policy="bidirectional", bounds_s=_chunk_context(bounds),
        preprocessing={"sample_rate_hz": _SR, "n_fft": _FFT, "hop_samples": _HOP,
                       "center": True, "padding": "reflect", "chunk_frames": _CHUNK,
                       "border_frames": _BORDER, "overlap": "keep_first"},
    )
    return music_series(
        stimulus, values=values, times_s=offset + np.arange(expected) * _HOP / _SR,
        bounds_s=bounds, names=["beat", "downbeat"], extractor="audio.music.beat_activations",
        params={"checkpoint": checkpoint, "asset_digest": revision, "representation": representation,
                "channel_policy": channel_policy, "device": device, "backend_version": deps.version,
                "torchaudio_version": deps.frontend_version, "rotary_version": deps.rotary_version,
                "torch_version": deps.torch.__version__, "local_files_only": local_files_only},
        context=context, hop_s=_HOP / _SR, model_revision=revision,
        extra={"execution_mode": str(mode), "fallback_used": False, "backend": "beat_this",
               "representation": representation, "calibrated_confidence": False},
    )
