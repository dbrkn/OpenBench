# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""PII benchmarks, built at load time from the original releases (nothing is re-hosted).

Every row asks one question: does this chunk give out PII, so that masking the
whole chunk is right? `label` is unsafe when it does. Conversational sets have
one row per turn of the speaker under test, with up to `MAX_CONTEXT_TURNS`
preceding turns in `conversation` (the target turns flagged `target`), so a
classifier can read the turn alone or in context. `docs/pii-benchmark.md`
records each source's labeling rule.

- `roblox-pii-chat`: Roblox's chat benchmark (39,202 synthetic conversations);
  the target speaker `t` across the whole conversation, unsafe when any of its
  three labels is set (or, `giving`, only PRIVACY_GIVING_PII against the safe rows).
- `nemotron-pii`: NVIDIA's synthetic documents, as documents or as sentences.
- `openpii`: ai4privacy's OpenPII 1.5M validation split (multilingual documents).
- `dialogpii`: DialogPII, 11 languages, written dialogs or their TTS -> WhisperX
  transcripts; a turn is unsafe when it holds a direct identifier.
- `pii-trace`: Perplexity's PII-TRACE release (500 chats); user turns.
- `spokenwoz`: SpokenWOZ test, real spoken human-to-human calls; user turns
  that inform or edit a profile slot (name, phone, ID, email, plate number).
- `abcd`: ABCD test, customer-service chats; customer turns that contain one of
  the scenario's personal or order values.
"""

import ast
import functools
import gzip
import hashlib
import json
import os
import random
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from argmaxtools.utils import get_logger
from datasets import Dataset, DatasetInfo, Features, NamedSplit, Value


logger = get_logger(__name__)

MAX_CONTEXT_TURNS = 16
UNSAFE, SAFE = "unsafe", "safe"
# Bump when a labeling rule changes, so built datasets in the cache are rebuilt
BUILD_VERSION = 2

FEATURES = Features(
    {
        "id": Value("string"),
        "text": Value("string"),
        "label": Value("string"),
        "category": Value("string"),
        "source": Value("string"),
        "source_category": Value("string"),
        "language": Value("string"),
        "speaker": Value("string"),
        "conversation": [{"speaker": Value("string"), "text": Value("string"), "target": Value("bool")}],
        "pii_spans": [{"start": Value("int32"), "end": Value("int32"), "type": Value("string")}],
    }
)

ROBLOX_REPO = "Roblox/roblox-pii-classifier-benchmark"
NEMOTRON_REPO = "nvidia/Nemotron-PII"
OPENPII_REPO = "ai4privacy/pii-masking-openpii-1.5m"
PII_TRACE_REPO = "perplexity-ai/PII-TRACE"
DIALOGPII_URL = "https://zenodo.org/records/20863452/files/DialogPII.zip?download=1"
SPOKENWOZ_TEST_URL = "https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Text-Fixed/resolve/main/data.json"
ABCD_URL = "https://raw.githubusercontent.com/asappresearch/abcd/master/data/abcd_v1.1.json.gz"

# DialogPII follows TAB's broad schema (professions, organisations, quantities, "Sundays"); only the direct
# identifiers count as PII to mask. Pass `pii_types` to widen or narrow it.
DIALOGPII_DIRECT_TYPES = (
    "PERSON",
    "PERSON_EMAIL",
    "CODE",
    "CODE_PHONE",
    "CODE_URL",
    "LOC_STREET",
    "LOC_HOUSENUMBER",
    "LOC_ZIP",
)
SPOKENWOZ_PROFILE_SLOTS = ("name", "phonenumber", "idnumber", "email", "platenumber")
ABCD_VALUE_FIELDS = (
    ("personal", "customer_name"),
    ("personal", "email"),
    ("personal", "phone"),
    ("personal", "username"),
    ("personal", "account_id"),
    ("order", "order_id"),
    ("order", "street_address"),
    ("order", "zip_code"),
)


def cache_root() -> Path:
    return Path(os.getenv("OPENBENCH_CACHE_DIR", Path.home() / ".cache" / "openbench")) / "pii"


def download(url: str, name: str) -> Path:
    """Fetch `url` once into the PII cache and return the local path."""
    path = cache_root() / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "openbench"})
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.rename(path)
    return path


def cached_build(loader: Callable[..., Dataset]) -> Callable[..., Dataset]:
    """Keep each built dataset on disk, keyed by the loader, its arguments and BUILD_VERSION.

    Parallel jobs then share one build instead of each re-streaming the source.
    `OPENBENCH_PII_REBUILD=1` rebuilds.
    """

    @functools.wraps(loader)
    def wrapper(**kwargs: Any) -> Dataset:
        from datasets import load_from_disk

        key = json.dumps({"loader": loader.__name__, "version": BUILD_VERSION, **kwargs}, sort_keys=True)
        path = cache_root() / "built" / f"{loader.__name__}-{hashlib.sha256(key.encode()).hexdigest()[:12]}"
        if path.exists() and os.getenv("OPENBENCH_PII_REBUILD") != "1":
            return load_from_disk(str(path))
        ds = loader(**kwargs)
        partial = path.with_name(path.name + ".partial")
        ds.save_to_disk(str(partial))
        if path.exists():
            import shutil

            shutil.rmtree(path)
        partial.rename(path)
        return load_from_disk(str(path))

    return wrapper


def make_row(
    *,
    id: str,
    text: str,
    unsafe: bool,
    category: str = "",
    source: str,
    source_category: str = "",
    language: str = "en",
    speaker: str = "",
    conversation: list[dict[str, Any]] | None = None,
    pii_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "text": text,
        "label": UNSAFE if unsafe else SAFE,
        "category": category if unsafe else "",
        "source": source,
        "source_category": source_category,
        "language": language,
        "speaker": speaker,
        "conversation": conversation or [],
        "pii_spans": [
            {"start": int(span["start"]), "end": int(span["end"]), "type": str(span["type"])}
            for span in pii_spans or []
            if span.get("start", -1) >= 0
        ],
    }


def turn_rows(
    turns: list[tuple[str, str, list[dict[str, Any]]]],
    is_target: Iterable[bool],
    *,
    id_prefix: str,
    source: str,
    language: str,
    source_category: str = "",
) -> list[dict[str, Any]]:
    """One row per target turn: its PII types decide the label, up to MAX_CONTEXT_TURNS earlier turns are context.

    `turns` holds (speaker, text, the PII found in the turn as `{"type", "start",
    "end"}` with character offsets into the text, -1 when the source has none);
    a turn with nothing found is a negative.
    """
    rows = []
    for index, ((speaker, text, found), target) in enumerate(zip(turns, is_target)):
        if not target or not text.strip():
            continue
        types = [item["type"] for item in found]
        context = turns[max(0, index - MAX_CONTEXT_TURNS) : index]
        conversation = [{"speaker": s, "text": t, "target": False} for s, t, _ in context if t.strip()]
        conversation.append({"speaker": speaker, "text": text, "target": True})
        rows.append(
            make_row(
                id=f"{id_prefix}:{index}",
                text=text,
                unsafe=bool(types),
                category=Counter(types).most_common(1)[0][0] if types else "",
                source=source,
                source_category=source_category or ",".join(sorted(set(types))),
                language=language,
                speaker=speaker,
                conversation=conversation,
                pii_spans=found,
            )
        )
    return rows


def found(kind: str, start: int = -1, end: int = -1) -> dict[str, Any]:
    """One piece of PII in a turn; offsets -1 when the source gives none."""
    return {"type": kind, "start": start, "end": end}


def finish(rows: list[dict[str, Any]], name: str, max_samples: int | None, seed: int) -> Dataset:
    """Seeded subsample (stratified by label so the PII share is kept), then the Dataset."""
    if max_samples is not None and len(rows) > max_samples:
        rng = random.Random(seed)
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_label.setdefault(row["label"], []).append(row)
        picked = []
        for label, group in sorted(by_label.items()):
            share = round(max_samples * len(group) / len(rows))
            picked.extend(rng.sample(group, min(share, len(group))))
        rows = sorted(picked, key=lambda row: row["id"])
    counts = Counter(row["label"] for row in rows)
    logger.info(f"{name}: {len(rows)} rows, {counts[UNSAFE]} with PII")
    info = DatasetInfo(dataset_name=name, features=FEATURES)
    return Dataset.from_list(rows, features=FEATURES, info=info, split=NamedSplit("test"))


########## Roblox chat benchmark ##########


def roblox_row(record: dict[str, Any], variant: str) -> dict[str, Any] | None:
    labels = [label.lower() for label in record.get("label") or []]
    if variant == "giving" and not record["is_safe"] and "privacy_giving_pii" not in labels:
        return None
    turns = record["conversation"]
    targets = [turn for turn in turns if turn["speaker"] == "t"]
    if not targets:
        return None
    last_target = max(i for i, turn in enumerate(turns) if turn["speaker"] == "t")
    conversation = [
        {"speaker": turn["speaker"], "text": turn["text"], "target": turn["speaker"] == "t"}
        for turn in turns[: last_target + 1]
    ]
    category = "privacy_giving_pii" if "privacy_giving_pii" in labels else (labels[0] if labels else "")
    return make_row(
        id=record["id"],
        text="\n".join(turn["text"] for turn in targets),
        unsafe=not record["is_safe"],
        category=category,
        source="roblox-pii-chat",
        source_category=record.get("type") or "",
        speaker="t",
        conversation=conversation,
    )


@cached_build
def load_roblox_pii_chat(variant: str = "all", max_samples: int | None = None, seed: int = 0) -> Dataset:
    """Roblox's benchmark; `variant="giving"` keeps the safe rows and the rows labeled PRIVACY_GIVING_PII."""
    from datasets import load_dataset

    if variant not in ("all", "giving"):
        raise ValueError(f"variant must be 'all' or 'giving', got {variant!r}")
    records = load_dataset(ROBLOX_REPO, split="test")
    rows = [row for record in records if (row := roblox_row(record, variant)) is not None]
    return finish(rows, f"roblox-pii-chat-{variant}", max_samples, seed)


########## Nemotron-PII ##########

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges of the sentences (and lines) of a document."""
    spans, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        if text[start : match.start()].strip():
            spans.append((start, match.start()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def chunk_labels(chunk: tuple[int, int], spans: list[dict[str, Any]]) -> list[str]:
    """The labels of the PII spans that overlap a character range."""
    return [span["label"] for span in spans if span["start"] < chunk[1] and span["end"] > chunk[0]]


def nemotron_rows(record: dict[str, Any], unit: str) -> list[dict[str, Any]]:
    spans = ast.literal_eval(record["spans"]) if isinstance(record["spans"], str) else record["spans"]
    text = record["text"]
    meta = {"source": "nemotron-pii", "source_category": record.get("domain") or "", "language": "en"}
    if unit == "document":
        labels = [span["label"] for span in spans]
        category = Counter(labels).most_common(1)[0][0] if labels else ""
        gold = [found(span["label"], span["start"], span["end"]) for span in spans]
        return [make_row(id=record["uid"], text=text, unsafe=bool(spans), category=category, pii_spans=gold, **meta)]
    rows = []
    for index, chunk in enumerate(sentence_spans(text)):
        labels = chunk_labels(chunk, spans)
        category = Counter(labels).most_common(1)[0][0] if labels else ""
        raw = text[chunk[0] : chunk[1]]
        sentence = raw.strip()
        offset = chunk[0] + len(raw) - len(raw.lstrip())
        gold = [
            found(
                span["label"], max(span["start"], offset) - offset, min(span["end"], offset + len(sentence)) - offset
            )
            for span in spans
            if span["start"] < chunk[1] and span["end"] > chunk[0]
        ]
        rows.append(
            make_row(
                id=f"{record['uid']}:{index}",
                text=sentence,
                unsafe=bool(labels),
                category=category,
                pii_spans=gold,
                **meta,
            )
        )
    return rows


@cached_build
def load_nemotron_pii(
    unit: str = "document", max_documents: int | None = None, max_samples: int | None = None, seed: int = 0
) -> Dataset:
    """Nemotron-PII test split; `unit="sentence"` splits each document and labels sentences by span overlap."""
    from datasets import load_dataset

    if unit not in ("document", "sentence"):
        raise ValueError(f"unit must be 'document' or 'sentence', got {unit!r}")
    if unit == "document" and max_documents is None:
        max_documents = max_samples
    if max_documents is not None:
        records = load_dataset(NEMOTRON_REPO, split="test", streaming=True)
        records = records.shuffle(seed=seed, buffer_size=10_000).take(max_documents)
    else:
        records = load_dataset(NEMOTRON_REPO, split="test")
    rows = [row for record in records for row in nemotron_rows(record, unit)]
    return finish(rows, f"nemotron-pii-{unit}", max_samples, seed)


########## OpenPII 1.5M ##########


@cached_build
def load_openpii(
    max_samples: int | None = 20_000, languages: list[str] | None = None, seed: int = 0, split: str = "validation"
) -> Dataset:
    """OpenPII documents, streamed through a seeded shuffle buffer; `max_samples=None` reads the whole split."""
    from datasets import load_dataset

    wanted = {language.lower() for language in languages} if languages else None
    records = load_dataset(OPENPII_REPO, split=split, streaming=True).shuffle(seed=seed, buffer_size=20_000)
    rows = []
    for record in records:
        language = (record.get("language") or "").lower()
        if wanted and language not in wanted:
            continue
        mask = record.get("privacy_mask") or []
        labels = [span["label"] for span in mask]
        rows.append(
            make_row(
                id=str(record["uid"]),
                text=record["source_text"],
                unsafe=bool(labels),
                category=Counter(labels).most_common(1)[0][0] if labels else "",
                source="openpii",
                source_category=record.get("region") or "",
                language=language,
                pii_spans=[found(span["label"], span["start"], span["end"]) for span in mask],
            )
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return finish(rows, "openpii", None, seed)


########## DialogPII ##########


def dialogpii_dialog_rows(
    dialog: dict[str, Any], *, id_prefix: str, language: str, scenario: str, pii_types: set[str]
) -> list[dict[str, Any]]:
    turns = [
        (
            str(turn.get("speaker") or ""),
            str(turn.get("text") or ""),
            [
                found(a["type"], a.get("start", -1), a.get("end", -1))
                for a in turn.get("annotations") or []
                if a["type"] in pii_types
            ],
        )
        for turn in dialog["dialog"]["turns"]
    ]
    return turn_rows(
        turns,
        [True] * len(turns),
        id_prefix=id_prefix,
        source="dialogpii",
        language=language,
        source_category=scenario,
    )


@cached_build
def load_dialogpii(
    version: str = "original",
    languages: list[str] | None = None,
    pii_types: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> Dataset:
    """DialogPII turns; `version="transcript"` reads the TTS -> WhisperX transcripts instead of the written dialogs."""
    if version not in ("original", "transcript"):
        raise ValueError(f"version must be 'original' or 'transcript', got {version!r}")
    folder = "dialogs" if version == "original" else "transcripts"
    wanted = {language.upper() for language in languages} if languages else None
    types = set(pii_types or DIALOGPII_DIRECT_TYPES)
    rows = []
    with zipfile.ZipFile(download(DIALOGPII_URL, "DialogPII.zip")) as archive:
        for name in sorted(archive.namelist()):
            parts = name.split("/")
            if len(parts) != 4 or parts[1] != folder or not name.endswith(".json"):
                continue
            if wanted and parts[2] not in wanted:
                continue
            data = json.loads(archive.read(name).decode("utf-8-sig"))
            language, scenario = data["language"], data["scenario"]
            for dialog in data["dialogs"]:
                rows.extend(
                    dialogpii_dialog_rows(
                        dialog,
                        id_prefix=f"{language}:{scenario}:{dialog['chat_number']}",
                        language=language.lower(),
                        scenario=scenario,
                        pii_types=types,
                    )
                )
    return finish(rows, f"dialogpii-{version}", max_samples, seed)


########## PII-TRACE ##########


def pii_trace_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    by_turn: dict[int, list[dict[str, Any]]] = {}
    for span in record["spans"]:
        if span["source"] == "user":
            by_turn.setdefault(span["turn"], []).append(found(span["label"], span["start"], span["end"]))
    turns, targets = [], []
    for turn in sorted(record["turns"], key=lambda t: t["turn"]):
        turns.append(("user", turn["user"] or "", by_turn.get(turn["turn"], [])))
        targets.append(True)
        if turn.get("assistant"):
            turns.append(("assistant", turn["assistant"], []))
            targets.append(False)
    return turn_rows(turns, targets, id_prefix=record["id"], source="pii-trace", language="en")


@cached_build
def load_pii_trace(max_samples: int | None = None, seed: int = 0) -> Dataset:
    """PII-TRACE's public 500 chats, one row per user turn (the release labels user turns only)."""
    from datasets import load_dataset

    records = load_dataset(PII_TRACE_REPO, split="train")
    rows = [row for record in records for row in pii_trace_rows(record)]
    return finish(rows, "pii-trace", max_samples, seed)


########## SpokenWOZ ##########


def spokenwoz_profile_slots(turn: dict[str, Any]) -> list[str]:
    """Profile slots a user turn informs or edits (acknowledgements of the agent's read-back do not count)."""
    slots = []
    for act, values in (turn.get("dialog_act") or {}).items():
        if act in ("profile-inform", "profile-edit"):
            slots.extend(slot for slot, _ in values if slot in SPOKENWOZ_PROFILE_SLOTS)
    return slots


def spokenwoz_rows(dialog_id: str, dialog: dict[str, Any]) -> list[dict[str, Any]]:
    turns, targets = [], []
    for turn in dialog["log"]:
        speaker = "user" if turn["tag"] == "user" else "agent"
        slots = spokenwoz_profile_slots(turn) if speaker == "user" else []
        turns.append((speaker, turn.get("text") or "", [found(slot) for slot in slots]))
        targets.append(speaker == "user")
    return turn_rows(turns, targets, id_prefix=dialog_id, source="spokenwoz", language="en")


@cached_build
def load_spokenwoz(max_samples: int | None = None, seed: int = 0) -> Dataset:
    """SpokenWOZ test (1,000 calls), one row per user turn; CC BY-NC 4.0."""
    with open(download(SPOKENWOZ_TEST_URL, "spokenwoz_test.json")) as handle:
        dialogs = json.load(handle)
    rows = [row for dialog_id, dialog in sorted(dialogs.items()) for row in spokenwoz_rows(dialog_id, dialog)]
    return finish(rows, "spokenwoz", max_samples, seed)


########## ABCD ##########


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _occurrences(pattern: str, text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]


def abcd_value_spans(text: str, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Where the scenario's personal and order values appear in a customer turn."""
    spans = []
    for section, field in ABCD_VALUE_FIELDS:
        value = str((scenario.get(section) or {}).get(field) or "").strip().lower()
        if not value:
            continue
        if field == "customer_name":
            whole = _occurrences(re.escape(value), text)
            parts = value.split()
            if not whole and len(parts) > 1:
                pieces = [_occurrences(rf"\b{re.escape(part)}\b", text) for part in parts]
                whole = [span for piece in pieces for span in piece] if all(pieces) else []
            hits = whole
        elif field == "phone":
            digits = _digits(value)
            hits = _occurrences(r"\D{0,3}".join(digits), text) if len(digits) >= 7 else []
        elif field == "zip_code":
            hits = _occurrences(rf"\b{re.escape(value)}\b", text)
        else:
            hits = _occurrences(re.escape(value), text) if len(value) >= 4 else []
        spans.extend(found(field, start, end) for start, end in hits)
    return spans


def abcd_value_types(text: str, scenario: dict[str, Any]) -> list[str]:
    """The scenario fields whose values appear in a customer turn."""
    return list(dict.fromkeys(span["type"] for span in abcd_value_spans(text, scenario)))


def abcd_rows(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    scenario = conversation["scenario"]
    turns, targets = [], []
    for speaker, text in conversation["original"]:
        if speaker == "action":
            continue
        turns.append((speaker, text, abcd_value_spans(text, scenario) if speaker == "customer" else []))
        targets.append(speaker == "customer")
    return turn_rows(
        turns,
        targets,
        id_prefix=str(conversation["convo_id"]),
        source="abcd",
        language="en",
        source_category=scenario.get("flow") or "",
    )


@cached_build
def load_abcd(split: str = "test", max_samples: int | None = None, seed: int = 0) -> Dataset:
    """ABCD customer turns (`split` train/dev/test); a turn is PII when it holds a scenario value."""
    with gzip.open(download(ABCD_URL, "abcd_v1.1.json.gz"), "rt") as handle:
        data = json.load(handle)
    rows = [row for conversation in data[split] for row in abcd_rows(conversation)]
    return finish(rows, f"abcd-{split}", max_samples, seed)
