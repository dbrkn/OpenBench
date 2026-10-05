# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Speech toxicity benchmarks for safety classification, loaded from their original repositories.

Two public sets carry labeled speech, which is what an audio safety classifier
needs and what a speech-to-text-to-classifier system can be measured on:

- Detoxy hate speech (Dynamic-SUPERB packaging): 603 English clips labeled
  hate / non-hate, read speech from VCTK, LJSpeech, Common Voice and MELD.
- MuTox (Meta, MIT): toxicity annotations over 30 languages with per-clip
  transcripts. The audio is referenced by public URLs in the original release;
  this loader streams the English training-partition clips from the
  `boyuzhuGPT/mutox_wildguardmix_audio` mirror, capped, and joins the labels
  from the `ymerkli/mutox` annotation mirror by clip id. The mirror holds the
  train partition only, so these are not MuTox's official devtest numbers.

Both loaders can fill the `text` column with a transcript so the text
classifiers run on the same utterances: `asr="reference"` uses the human
transcript (MuTox only), `asr="argmax-cli[:model]"` runs WhisperKit through
`argmax-cli transcribe`, `asr="whisper[:model]"` runs openai-whisper in
process. Transcripts are cached per clip under the OpenBench cache.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from argmaxtools.utils import get_logger
from datasets import Audio, Dataset, Features, NamedSplit, Value, load_dataset, load_from_disk


logger = get_logger(__name__)

DETOXY_REPO = "DynamicSuperb/HateSpeechDetection_Detoxy_VCTK_LJSpeech_CV_MELD"
MUTOX_LABELS_REPO = "ymerkli/mutox"
MUTOX_AUDIO_TAR = "https://huggingface.co/datasets/boyuzhuGPT/mutox_wildguardmix_audio/resolve/main/mutox/train.tar"
SAMPLE_RATE = 16000

HATE_SPEECH = "hate_speech"
SEXUAL_CONTENT = "sexual_content"
HARASSMENT = "harassment"
PROFANITY = "profanity"
# MuTox toxicity types, in the priority used when a clip carries several
MUTOX_TYPES = [
    ("Hate speech", HATE_SPEECH),
    ("Slurs", HATE_SPEECH),
    ("Pornographic language", SEXUAL_CONTENT),
    ("Physical violence or bullying language", HARASSMENT),
    ("Profanities", PROFANITY),
]

FEATURES = Features(
    {
        "id": Value("string"),
        "audio": Audio(sampling_rate=SAMPLE_RATE),
        "text": Value("string"),
        "reference_transcript": Value("string"),
        "label": Value("string"),
        "category": Value("string"),
        "source": Value("string"),
        "source_category": Value("string"),
        "language": Value("string"),
    }
)


def _cache_root() -> Path:
    return Path(os.getenv("OPENBENCH_CACHE_DIR", Path.home() / ".cache" / "openbench"))


def mutox_category(toxicity_types: str | None) -> str:
    """The benchmark category of a MuTox clip from its comma-separated toxicity types."""
    if not toxicity_types:
        return ""
    types = {t.strip() for t in toxicity_types.split(",")}
    for name, category in MUTOX_TYPES:
        if name in types:
            return category
    return "other"


def _row(**kwargs: Any) -> dict[str, Any]:
    row = {"text": "", "reference_transcript": "", "category": "", "source_category": "", "language": "en"}
    row.update(kwargs)
    return row


# ----------------------------------------------------------------------------- transcription


def _parse_asr(asr: str) -> tuple[str, str | None]:
    engine, _, model = asr.partition(":")
    return engine, (model or None)


def _resolve_argmax_cli() -> str:
    explicit = os.getenv("ARGMAX_CLI_PATH")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("argmax-cli")
    if found:
        return found
    from ..engine.argmax_oss_engine import ArgmaxOpenSourceEngine, ArgmaxOpenSourceEngineConfig

    return ArgmaxOpenSourceEngine(ArgmaxOpenSourceEngineConfig()).cli_path


def _transcribe_with_argmax_cli(clips: dict[str, np.ndarray], model: str | None) -> dict[str, str]:
    """One `argmax-cli transcribe --audio-folder` call over all missing clips."""
    import soundfile as sf

    model = model or "large-v3_turbo"
    cli = _resolve_argmax_cli()
    transcripts: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="openbench-asr-") as tmp:
        audio_dir = Path(tmp) / "audio"
        report_dir = Path(tmp) / "reports"
        audio_dir.mkdir()
        report_dir.mkdir()
        for clip_id, waveform in clips.items():
            sf.write(audio_dir / f"{clip_id}.wav", waveform, SAMPLE_RATE)
        command = [
            cli,
            "transcribe",
            "--audio-folder",
            str(audio_dir),
            "--model",
            model,
            "--language",
            "en",
            "--skip-special-tokens",
            "--report",
            "--report-path",
            str(report_dir),
        ]
        logger.info(f"Transcribing {len(clips)} clips with argmax-cli ({model})")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"argmax-cli transcribe failed ({result.returncode}): {result.stderr[-2000:]}")
        for clip_id in clips:
            report = report_dir / f"{clip_id}.json"
            if not report.is_file():
                logger.warning(f"No transcript report for {clip_id}")
                transcripts[clip_id] = ""
                continue
            payload = json.loads(report.read_text())
            text = payload.get("text") or " ".join(s.get("text", "") for s in payload.get("segments", []))
            transcripts[clip_id] = text.strip()
    return transcripts


def _transcribe_with_whisper(clips: dict[str, np.ndarray], model: str | None) -> dict[str, str]:
    """openai-whisper in process, one clip at a time."""
    import whisper

    model_name = model or "small.en"
    logger.info(f"Transcribing {len(clips)} clips with openai-whisper ({model_name})")
    whisper_model = whisper.load_model(model_name)
    transcripts = {}
    for clip_id, waveform in clips.items():
        result = whisper_model.transcribe(waveform.astype(np.float32), fp16=False, language="en")
        transcripts[clip_id] = str(result.get("text", "")).strip()
    return transcripts


_ENGINES: dict[str, Callable[[dict[str, np.ndarray], str | None], dict[str, str]]] = {
    "argmax-cli": _transcribe_with_argmax_cli,
    "whisper": _transcribe_with_whisper,
}


def add_transcripts(rows: list[dict[str, Any]], dataset_name: str, asr: str | None) -> list[dict[str, Any]]:
    """Fill `text` from the reference transcript or from a cached ASR pass."""
    if asr is None:
        return rows
    if asr == "reference":
        for row in rows:
            row["text"] = row["reference_transcript"]
        return rows
    engine, model = _parse_asr(asr)
    if engine not in _ENGINES:
        raise ValueError(f"Unknown asr {asr!r}; expected 'reference', 'argmax-cli[:model]' or 'whisper[:model]'")
    cache_dir = _cache_root() / "transcripts" / dataset_name / asr.replace(":", "_").replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing: dict[str, np.ndarray] = {}
    for row in rows:
        cached = cache_dir / f"{row['id']}.txt"
        if cached.is_file():
            row["text"] = cached.read_text()
        else:
            missing[row["id"]] = np.asarray(row["audio"]["array"], dtype=np.float32)
    if missing:
        transcripts = _ENGINES[engine](missing, model)
        for row in rows:
            if row["id"] in transcripts:
                row["text"] = transcripts[row["id"]]
                (cache_dir / f"{row['id']}.txt").write_text(row["text"])
    return rows


def _to_dataset(rows: list[dict[str, Any]], name: str, description: str) -> Dataset:
    ds = Dataset.from_list(rows, features=FEATURES, split=NamedSplit("test"))
    ds.info.dataset_name = name
    ds.info.description = description
    return ds


# ----------------------------------------------------------------------------- Detoxy


def load_detoxy(asr: str | None = "whisper:small", max_samples: int | None = None) -> Dataset:
    """Detoxy hate-speech clips, labeled unsafe when the reference says `hate`."""
    source = load_dataset(DETOXY_REPO, split="test").cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    rows = []
    for record in source:
        rows.append(
            _row(
                id=str(record["file"]),
                audio={"array": np.asarray(record["audio"]["array"], dtype=np.float32), "sampling_rate": SAMPLE_RATE},
                label="unsafe" if record["label"] == "hate" else "safe",
                category=HATE_SPEECH if record["label"] == "hate" else "",
                source="detoxy",
                source_category=str(record["label"]),
            )
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    rows = add_transcripts(rows, "detoxy", asr)
    return _to_dataset(rows, "detoxy", "Detoxy hate speech clips (Dynamic-SUPERB packaging), unsafe = hate.")


# ----------------------------------------------------------------------------- MuTox


def _mutox_labels() -> dict[str, dict[str, Any]]:
    labels = load_dataset(MUTOX_LABELS_REPO, split="train")
    return {
        str(record["id"]): record
        for record in labels
        if record.get("lang") == "eng" and record.get("contains_toxicity") in ("Yes", "No")
    }


def _mutox_clips_from_cache(cache_dir: Path) -> list[dict[str, Any]] | None:
    if not (cache_dir / "state.json").is_file():
        return None
    cached = load_from_disk(str(cache_dir))
    return [dict(record) for record in cached]


def _stream_mutox_clips(max_samples: int, labels: dict[str, dict[str, Any]]) -> Iterable[dict[str, Any]]:
    stream = load_dataset("webdataset", data_files={"train": MUTOX_AUDIO_TAR}, split="train", streaming=True)
    taken = 0
    for item in stream:
        key = str(item["__key__"]).split("/")[-1]
        clip_id = key[len("eng_") :] if key.startswith("eng_") else key
        record = labels.get(clip_id)
        if record is None:
            continue
        audio = item["wav"]
        waveform = np.asarray(audio["array"], dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        if audio["sampling_rate"] != SAMPLE_RATE:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=audio["sampling_rate"], target_sr=SAMPLE_RATE)
        unsafe = record["contains_toxicity"] == "Yes"
        yield _row(
            id=clip_id,
            audio={"array": waveform, "sampling_rate": SAMPLE_RATE},
            reference_transcript=str(record.get("audio_file_transcript") or ""),
            label="unsafe" if unsafe else "safe",
            category=mutox_category(record.get("toxicity_types")) if unsafe else "",
            source="mutox",
            source_category=str(record.get("toxicity_types") or ""),
        )
        taken += 1
        if taken >= max_samples:
            return


def load_mutox(max_samples: int = 1000, asr: str | None = "reference") -> Dataset:
    """The first `max_samples` English MuTox clips of the audio mirror, with MuTox's labels.

    The clips are streamed once and kept under the OpenBench cache, so a rerun
    with the same cap costs nothing; a larger cap streams again from the start.
    """
    cache_dir = _cache_root() / "mutox-en" / str(max_samples)
    rows = _mutox_clips_from_cache(cache_dir)
    if rows is None:
        logger.info(f"Streaming up to {max_samples} English MuTox clips (cached afterwards under {cache_dir})")
        rows = list(_stream_mutox_clips(max_samples, _mutox_labels()))
        if not rows:
            raise RuntimeError("No MuTox clips matched the annotation file")
        _to_dataset(rows, "mutox-en", "").save_to_disk(str(cache_dir))
        rows = _mutox_clips_from_cache(cache_dir)
    for row in rows:
        row["audio"] = {"array": np.asarray(row["audio"]["array"], dtype=np.float32), "sampling_rate": SAMPLE_RATE}
    rows = add_transcripts(rows, "mutox-en", asr)
    return _to_dataset(
        rows,
        "mutox-en",
        f"MuTox English clips (train partition, first {max_samples} of the audio mirror), unsafe = contains toxicity.",
    )
