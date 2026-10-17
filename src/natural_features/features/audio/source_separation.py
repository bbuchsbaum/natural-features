"""Pinned Demucs loading and waveform separation shared by audio extractors."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.metadata import version
from math import gcd
from pathlib import Path
import tempfile
from urllib.request import urlopen

import numpy as np

from natural_features.core.backend_errors import BackendDependencyError, BackendLoadError

SEPARATOR_SHA256 = "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"
SEPARATOR_FILE = "955717e8-8726e21a.th"
SEPARATOR_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/" + SEPARATOR_FILE


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resample(samples, source_sr, target_sr):
    if source_sr == target_sr:
        return np.asarray(samples, dtype=np.float32)
    from scipy.signal import resample_poly
    factor = gcd(source_sr, target_sr)
    return resample_poly(samples, target_sr // factor, source_sr // factor, axis=0).astype(np.float32)


class SeparatorRuntime:
    def __init__(self, torch, model, device, versions, checkpoint=None):
        self.torch, self.model, self.device, self.versions = torch, model, device, versions
        self.checkpoint = checkpoint

    @property
    def sources(self):
        return self.model.sources

    def _apply(self, stereo, *, legacy_mono_normalization):
        from demucs.apply import apply_model
        x = np.asarray(stereo, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 2 or not np.all(np.isfinite(x)):
            raise ValueError("separator expects finite stereo audio")
        torch = self.torch
        mix = torch.as_tensor(x.T.copy(), device=self.device)
        # F0 historically normalized the channel mean and removed its DC.  The
        # level path instead uses all channel samples so antiphase stereo keeps
        # its physical energy.
        reference = mix.mean(0) if legacy_mono_normalization else mix
        mean = reference.mean() if legacy_mono_normalization else 0.0
        scale = (reference.std(unbiased=False) if legacy_mono_normalization else
                 reference.double().square().mean().sqrt().to(dtype=mix.dtype))
        silent = bool(scale <= 1e-10) if legacy_mono_normalization else bool(scale == 0)
        if silent:
            return np.zeros((len(x), len(self.sources), 2), np.float32)
        with torch.inference_mode():
            stems = apply_model(self.model, ((mix - mean) / scale)[None], shifts=0,
                                split=True, overlap=0.25, device=self.device, num_workers=0)[0]
            restored = stems * scale
            restored = restored - restored.mean(dim=-1, keepdim=True)
        # Demucs shape: source, channel, sample.
        result = restored.permute(2, 0, 1).cpu().numpy()
        if result.shape != (len(x), len(self.sources), 2) or not np.all(np.isfinite(result)):
            raise ValueError("separator returned invalid stems")
        return result

    def separate(self, stereo):
        """Legacy F0-compatible mono stem representation."""
        return self._apply(stereo, legacy_mono_normalization=True).mean(axis=2)

    def separate_stereo(self, stereo):
        return self._apply(stereo, legacy_mono_normalization=False)


@lru_cache(maxsize=2)
def _cached_separator_runtime(checkpoint, device):
    import torch
    from demucs.states import load_model
    try:
        with torch.random.fork_rng(devices=[]):
            model = load_model(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
        if (model.samplerate != 44100 or model.audio_channels != 2 or
                set(model.sources) != {"vocals", "drums", "bass", "other"} or float(model.segment) != 7.8):
            raise ValueError("unexpected separator sample rate, channels or source labels")
        return SeparatorRuntime(torch, model.to(device).eval(), device,
                                {name: version(name) for name in ("demucs", "torch", "scipy")}, checkpoint)
    except Exception as exc:
        raise BackendLoadError("source_separation", "failed to load the pinned separator model") from exc


def load_separator_runtime(separator_checkpoint=None, cache_dir=None, local_files_only=True, device="cpu", *, extractor="source_separation", sha256=_sha256):
    """Load only the fixed HTDemucs checkpoint, downloading only on request."""
    try:
        import torch
        import demucs  # noqa: F401
        import scipy  # noqa: F401
        if version("demucs") != "4.0.1":
            raise ImportError("requires demucs==4.0.1")
    except (ImportError, OSError) as exc:
        raise BackendDependencyError(extractor, "install natural-features[source_separation]") from exc
    cache = Path(cache_dir).expanduser() if cache_dir else Path(torch.hub.get_dir()) / "checkpoints"
    path = Path(separator_checkpoint).expanduser() if separator_checkpoint else cache / SEPARATOR_FILE
    if not path.is_file():
        if separator_checkpoint or local_files_only:
            raise BackendLoadError(extractor, f"pinned Demucs checkpoint is not available: {path}")
        cache.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                temporary = Path(stream.name)
                with urlopen(SEPARATOR_URL, timeout=60) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
            if sha256(temporary) != SEPARATOR_SHA256:
                raise ValueError("downloaded separator SHA256 mismatch")
            temporary.replace(path)
        except Exception as exc:
            raise BackendLoadError(extractor, "pinned separator download failed") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    try:
        if sha256(path) != SEPARATOR_SHA256:
            raise BackendLoadError(extractor, "model SHA256 mismatch; only the pinned asset is supported")
    except OSError as exc:
        raise BackendLoadError(extractor, "model asset could not be inspected") from exc
    try:
        return _cached_separator_runtime(str(path.resolve()), device)
    except BackendLoadError as exc:
        raise BackendLoadError(extractor, "failed to load the pinned separator model") from exc
