# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Speaker similarity (SIM) metric.

SIM is the cosine similarity between WavLM-large speaker embeddings of a
generated audio clip and a reference clip. This is the metric computed by
BytedanceSpeech/seed-tts-eval's ``cal_sim.sh`` path, where the aggregate mean is
reported as ``ASV`` and its variance as ``ASV-var``.

Only the minimal embedding + cosine-similarity logic is reimplemented here (on
top of the vendored :mod:`ecapa_tdnn` model and the ``s3prl`` WavLM upstream);
the full seed-tts-eval repository is intentionally not vendored.
"""

import sys as _sys
import types as _types

import torchaudio as _torchaudio


# --- Compatibility shims for modern torchaudio (>=2.1) -----------------------
# The s3prl WavLM upstream (loaded lazily via torch.hub when the embedding model
# is built) was written against an older torchaudio. These shims must run before
# that upstream is imported, so they live at module import time.
#
# (1) torchaudio.set_audio_backend was removed -> make it a harmless no-op.
if not hasattr(_torchaudio, "set_audio_backend"):
    _torchaudio.set_audio_backend = lambda *args, **kwargs: None
# (2) torchaudio.sox_effects was removed. s3prl imports it but the SIM forward
#     path never calls it, so register a stub module so the imports succeed.
if "torchaudio.sox_effects" not in _sys.modules and not hasattr(_torchaudio, "sox_effects"):
    _sox = _types.ModuleType("torchaudio.sox_effects")

    def _sox_unavailable(*args, **kwargs):
        raise RuntimeError("torchaudio.sox_effects is unavailable in this torchaudio version")

    _sox.apply_effects_file = _sox_unavailable
    _sox.apply_effects_tensor = _sox_unavailable
    _sys.modules["torchaudio.sox_effects"] = _sox
    _torchaudio.sox_effects = _sox
# -----------------------------------------------------------------------------

import librosa  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from pyannote.metrics.base import BaseMetric  # noqa: E402
from pyannote.metrics.types import Details, MetricComponents  # noqa: E402
from torchaudio.transforms import Resample  # noqa: E402

from argmaxtools.utils import get_logger  # noqa: E402

from ...types import PipelineType  # noqa: E402
from ..metric import MetricOptions  # noqa: E402
from ..registry import MetricRegistry  # noqa: E402
from .ecapa_tdnn import ECAPA_TDNN_SMALL  # noqa: E402


logger = get_logger(__name__)


# Mapping from supported model name to the embedding feature dimension expected
# by ECAPA_TDNN_SMALL. Only wavlm_large is needed to reproduce seed-tts-eval SIM,
# but the others are kept for parity with the upstream verification script.
_MODEL_FEAT_DIM = {
    "wavlm_large": 1024,
    "wavlm_base_plus": 768,
    "hubert_large": 1024,
    "wav2vec2_xlsr": 1024,
    "unispeech_sat": 1024,
}

_FEAT_TYPE = {
    "wavlm_large": "wavlm_large",
    "wavlm_base_plus": "wavlm_base_plus",
    "hubert_large": "hubert_large_ll60k",
    "wav2vec2_xlsr": "wav2vec2_xlsr",
    "unispeech_sat": "unispeech_sat",
}


@MetricRegistry.register_metric(PipelineType.SPEAKER_SIMILARITY, MetricOptions.SIM)
class SpeakerSimilarity(BaseMetric):
    """Speaker Similarity (SIM).

    Computes the cosine similarity between speaker embeddings of a generated clip
    and a reference clip, using a WavLM-large based ECAPA-TDNN speaker encoder.

    Each call evaluates one (generated, reference) pair and returns its SIM in
    ``[-1, 1]`` (higher = closer speaker identity). The accumulated/global value
    is the mean SIM across all evaluated pairs (``ASV``); :attr:`variance`
    exposes the population variance across pairs (``ASV-var``).

    This mirrors BytedanceSpeech/seed-tts-eval: ``librosa`` loads each clip at its
    native sample rate, both clips are resampled to 16 kHz, embedded by the same
    model, and compared with cosine similarity. The fine-tuned speaker-encoder
    checkpoint (e.g. ``wavlm_large_finetune.pth``) is loaded with
    ``strict=False`` exactly as upstream does.

    Parameters
    ----------
    model_name : str
        Speaker-encoder upstream to use. Defaults to ``"wavlm_large"``.
    checkpoint : str, optional
        Path to the fine-tuned speaker-encoder checkpoint. Required to reproduce
        seed-tts-eval numbers; if omitted, only the (randomly initialised) head
        is used, which will not match upstream.
    use_gpu : bool
        If True, run the model on ``device``. Defaults to False (CPU), matching
        the validated Phase-1 setup.
    device : str
        Torch device to use when ``use_gpu`` is True. Defaults to ``"cuda:0"``.
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_name: str = "wavlm_large",
        checkpoint: str | None = None,
        use_gpu: bool = False,
        device: str = "cuda:0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_name not in _MODEL_FEAT_DIM:
            raise ValueError(f"Unsupported model_name {model_name!r}; expected one of {sorted(_MODEL_FEAT_DIM)}")
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.use_gpu = use_gpu
        self.device = device
        self._model = None

    @classmethod
    def metric_name(cls) -> str:
        return "speaker similarity"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["sim_sum", "sim_squared_sum", "count"]

    # -- model / embedding helpers --------------------------------------------

    def _resolve_checkpoint(self) -> str | None:
        """Resolve the checkpoint spec to a local path.

        Supports ``hf://{owner}/{repo}/{filename}`` (fetched via
        ``huggingface_hub``, cache-aware — CI-friendly) in addition to a plain
        local filesystem path.
        """
        if self.checkpoint is None:
            return None
        if self.checkpoint.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            parts = self.checkpoint[len("hf://") :].split("/")
            if len(parts) < 3:
                raise ValueError(f"Expected hf://owner/repo/filename, got {self.checkpoint!r}")
            return hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))
        return self.checkpoint

    def _get_model(self):
        """Lazily build and cache the speaker-encoder model."""
        if self._model is not None:
            return self._model

        model = ECAPA_TDNN_SMALL(
            feat_dim=_MODEL_FEAT_DIM[self.model_name],
            feat_type=_FEAT_TYPE[self.model_name],
            config_path=None,
        )
        checkpoint_path = self._resolve_checkpoint()
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict["model"], strict=False)
        else:
            logger.warning(
                "SIM is running WITHOUT the fine-tuned speaker-encoder checkpoint: the ECAPA head "
                "is randomly initialised, so scores are non-discriminative (typically ~0.99 for any "
                "pair) and NOT comparable to seed-tts-eval. Pass it via "
                "`-mc sim.checkpoint=/path/to/wavlm_large_finetune.pth` (or hf://owner/repo/file)."
            )
        if self.use_gpu:
            model = model.cuda(self.device)
        model.eval()
        self._model = model
        return model

    def _embed(self, audio_path: str) -> torch.Tensor:
        """Load an audio file, resample to 16 kHz, and return its embedding."""
        wav, sr = librosa.load(audio_path, sr=None, mono=False)
        if wav.ndim == 2:
            # librosa returns (channels, samples) for multi-channel audio; take a
            # single channel, matching the upstream verification script.
            wav = wav[0, :]

        wav = torch.from_numpy(wav).unsqueeze(0).float()
        wav = Resample(orig_freq=sr, new_freq=self.SAMPLE_RATE)(wav)
        if self.use_gpu:
            wav = wav.cuda(self.device)

        model = self._get_model()
        with torch.no_grad():
            return model(wav)

    def score(self, generated_audio: str, reference_audio: str) -> float:
        """Return the SIM (cosine similarity) for a single pair of audio files."""
        emb_generated = self._embed(generated_audio)
        emb_reference = self._embed(reference_audio)
        return F.cosine_similarity(emb_generated, emb_reference).item()

    # -- BaseMetric interface --------------------------------------------------

    def compute_components(self, generated_audio: str, reference_audio: str, **kwargs) -> Details:
        sim = self.score(generated_audio, reference_audio)
        return {"sim_sum": sim, "sim_squared_sum": sim * sim, "count": 1.0}

    def compute_metric(self, detail: Details) -> float:
        count = detail["count"]
        if count == 0:
            return 0.0
        return detail["sim_sum"] / count

    @property
    def variance(self) -> float:
        """Population variance of SIM across all evaluated pairs (ASV-var)."""
        count = self.accumulated_["count"]
        if count == 0:
            return 0.0
        mean = self.accumulated_["sim_sum"] / count
        return self.accumulated_["sim_squared_sum"] / count - mean * mean


@MetricRegistry.register_metric(PipelineType.SPEECH_GENERATION, MetricOptions.SIM)
class SpeechGenerationSpeakerSimilarity(SpeakerSimilarity):
    """SIM metric for speech-generation pipelines.

    The benchmark runner invokes every metric uniformly as
    ``metric(hypothesis=prediction, reference=sample.reference, ...)``. For a
    speech-generation pipeline the ``hypothesis`` is a ``GeneratedAudio`` (the
    synthesized clip) and ``reference`` is the prompt *transcript* (consumed by
    WER, not SIM). SIM instead needs the target-speaker *audio*, which voice-
    cloning pipelines record on the prediction as ``reference_audio_path``; we
    fall back to a ``reference_audio`` / ``ref_audio`` keyword if a pipeline does
    not set it.

    Accepts the same constructor kwargs as :class:`SpeakerSimilarity`
    (``model_name``, ``checkpoint``, ``use_gpu``, ``device``).
    """

    def compute_components(self, reference, hypothesis, **kwargs) -> Details:
        generated_audio = hypothesis.audio_path
        reference_audio = (
            getattr(hypothesis, "reference_audio_path", None)
            or kwargs.get("reference_audio")
            or kwargs.get("ref_audio")
        )
        if not reference_audio:
            raise ValueError(
                "SIM requires a target-speaker reference clip, but none was found. "
                "Use a voice-cloning pipeline that sets `reference_audio_path` on its "
                "GeneratedAudio prediction, or supply a `ref_audio` in the sample's extra_info."
            )
        return super().compute_components(generated_audio=generated_audio, reference_audio=reference_audio)
