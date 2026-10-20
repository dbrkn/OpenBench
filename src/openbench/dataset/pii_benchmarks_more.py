# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""More PII benchmarks for coverage: real text, more languages, more domains, other models' training data.

Same row format and rules as `pii_benchmarks` (one segment per row, gold spans
as character offsets in the segment). Long documents are cut into segments of
at most `MAX_SEGMENT_CHARS` at paragraph, then sentence, boundaries, never
through a gold span. `docs/pii-benchmark.md` records each source's rule.

- `tab`: Text Anonymization Benchmark test (real ECHR judgments, human-annotated); DIRECT identifiers, or
  DIRECT + QUASI, unioned over the document's annotators.
- `pii-bench-ru`: hivetrace PII-Bench (Russian), the domain split; the dialogue rows become one row per message.
- `gretel-finance`: Gretel synthetic financial documents test, 7 languages.
- `gretel-en`: Gretel PII masking en v1 test (offsets recovered from the unique values).
- `ai4privacy-400k` / `ai4privacy-500k`: ai4privacy validation splits, capped per language; rows whose text has
  leaked mask placeholders are dropped.
- `spy`: SPY legal questions and medical consultations, placeholders filled with seeded Faker values.
- `privy`: Privy protocol traces (JSON, XML, HTML, SQL), small test split.
- `e3jsi-multi`: E3-JSI synthetic multilingual PII NER (7 languages; open-vocabulary types).
- `urchade-mistral`: the synthetic set urchade's GLiNER PII models were trained on (a contamination check).
- `spokenwoz-dev`: the 500 SpokenWOZ dev calls, same rule as `spokenwoz`.
"""

import ast
import json
import random
import re
import zipfile
from collections import Counter
from typing import Any

from argmaxtools.utils import get_logger
from datasets import Dataset

from .pii_benchmarks import cached_build, finish, found, make_row, spokenwoz_rows, turn_rows


logger = get_logger(__name__)

MAX_SEGMENT_CHARS = 1000
_PLACEHOLDER = re.compile(r"\[[A-Z_]+(?:_\d+)?\]")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+")


def hub_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def chunk_ranges(text: str, spans: list[tuple[int, int]], max_chars: int = MAX_SEGMENT_CHARS) -> list[tuple[int, int]]:
    """Segment boundaries for a document: paragraphs, then sentences, merged so that no span is cut."""
    cuts = {0, len(text)}
    for match in re.finditer(r"\n\s*\n", text):
        cuts.add(match.end())
    pieces = sorted(cuts)
    for a, b in zip(pieces, pieces[1:]):
        if b - a > max_chars:
            cuts.update(m.end() for m in _SENTENCE_BREAK.finditer(text, a, b))
    inside = [(start, end) for start, end in spans]
    boundaries = sorted(c for c in cuts if not any(s < c < e for s, e in inside))
    ranges, start = [], 0
    for cut in boundaries[1:]:
        if cut - start >= max_chars // 4 or cut == len(text):
            ranges.append((start, cut))
            start = cut
    if start < len(text):
        ranges.append((start, len(text)))
    return [(a, b) for a, b in ranges if text[a:b].strip()]


def document_rows(
    doc_id: str,
    text: str,
    spans: list[tuple[int, int, str]],
    *,
    source: str,
    language: str = "en",
    source_category: str = "",
    max_chars: int = MAX_SEGMENT_CHARS,
) -> list[dict[str, Any]]:
    """One row per segment of a document, gold spans shifted into the segment."""
    rows = []
    for index, (a, b) in enumerate(chunk_ranges(text, [(s, e) for s, e, _ in spans], max_chars)):
        raw = text[a:b]
        segment = raw.strip()
        offset = a + len(raw) - len(raw.lstrip())
        inside = [found(kind, s - offset, e - offset) for s, e, kind in spans if s >= a and e <= b and e > s]
        rows.append(
            make_row(
                id=f"{doc_id}:{index}",
                text=segment,
                unsafe=bool(inside),
                category=Counter(i["type"] for i in inside).most_common(1)[0][0] if inside else "",
                source=source,
                source_category=source_category,
                language=language,
                pii_spans=inside,
            )
        )
    return rows


def segment_row(row_id: str, text: str, spans: list[tuple[int, int, str]], **meta: Any) -> dict[str, Any]:
    gold = [found(kind, s, e) for s, e, kind in spans if e > s]
    return make_row(
        id=row_id,
        text=text,
        unsafe=bool(gold),
        category=Counter(g["type"] for g in gold).most_common(1)[0][0] if gold else "",
        pii_spans=gold,
        **meta,
    )


def cap_per_language(rows: list[dict[str, Any]], per_language: int | None, seed: int) -> list[dict[str, Any]]:
    if per_language is None:
        return rows
    rng = random.Random(seed)
    by_language: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_language.setdefault(row["language"], []).append(row)
    kept = []
    for language, group in sorted(by_language.items()):
        kept.extend(group if len(group) <= per_language else rng.sample(group, per_language))
    return kept


########## TAB ##########


def tab_document_spans(annotations: list[dict[str, Any]], identifier_types: set[str]) -> list[tuple[int, int, str]]:
    """Union over annotators of the mentions with a kept identifier type, overlaps merged."""
    spans = sorted(
        (m["start_offset"], m["end_offset"], m["entity_type"])
        for record in annotations
        for m in record["entity_mentions"]
        if m["identifier_type"] in identifier_types
    )
    merged: list[list] = []
    for start, end, kind in spans:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, kind])
    return [(s, e, k) for s, e, k in merged]


@cached_build
def load_tab(
    identifiers: str = "direct", split: str = "test", max_samples: int | None = None, seed: int = 0
) -> Dataset:
    """TAB judgments cut into segments; `identifiers="direct"` or `"direct+quasi"` decides the gold."""
    kept = {"direct": {"DIRECT"}, "direct+quasi": {"DIRECT", "QUASI"}}[identifiers]
    with zipfile.ZipFile(hub_file("ildpil/text-anonymization-benchmark", f"echr_{split}.zip")) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".json"))
        records = json.loads(archive.read(name))
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_doc.setdefault(record["doc_id"], []).append(record)
    rows = []
    for doc_id, annotations in sorted(by_doc.items()):
        text = annotations[0]["text"]
        spans = tab_document_spans(annotations, kept)
        rows.extend(document_rows(doc_id, text, spans, source="tab", source_category=identifiers))
    return finish(rows, f"tab-{identifiers}", max_samples, seed)


########## PII-Bench (Russian) ##########


def pii_bench_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per text, or, for the L-DIALOG rows (a JSON list of chat messages), one row per user message.

    The dialogue rows' offsets index the raw JSON string, where escapes shift
    characters, so each entity is placed in its message by its value instead.
    """
    entities = record.get("entities") or []
    text = record["text"]
    meta = {"source": "pii-bench-ru", "language": "ru", "source_category": record["domain"]}
    if record["domain"] == "L-DIALOG":
        try:
            messages = json.loads(text)
        except ValueError:
            messages = None
        if isinstance(messages, list):
            turns, targets = [], []
            for message in messages:
                content = str(message.get("content") or "")
                values = [(str(e["text"]), str(e["type"])) for e in entities if e.get("text") and e["text"] in content]
                inside = [found(kind, s, e) for s, e, kind in value_spans(content, values)]
                role = str(message.get("role") or "")
                turns.append((role, content, inside))
                targets.append(role == "user")
            return turn_rows(turns, targets, id_prefix=str(record["id"]), **meta)
    spans = [(e["start"], e["end"], e["type"]) for e in entities]
    return [segment_row(str(record["id"]), text, spans, **meta)]


@cached_build
def load_pii_bench_ru(max_samples: int | None = None, seed: int = 0) -> Dataset:
    import pyarrow.parquet as pq

    records = pq.read_table(hub_file("hivetrace/pii-bench", "data/domain-00000-of-00001.parquet")).to_pylist()
    rows = [row for record in records for row in pii_bench_rows(record)]
    return finish(rows, "pii-bench-ru", max_samples, seed)


########## Gretel ##########

GRETEL_LANGUAGE = {"France": "fr", "English": "en", "German": "de", "Spanish": "es", "Swedish": "sv", "Dutch": "nl"}
GRETEL_LANGUAGE["Italian"] = "it"


@cached_build
def load_gretel_finance(max_samples: int | None = None, seed: int = 0) -> Dataset:
    """Gretel synthetic financial documents (test), cut into segments; labels are LLM + GLiNER made (noisy)."""
    import pyarrow.parquet as pq

    path = hub_file("gretelai/synthetic_pii_finance_multilingual", "data/test-00000-of-00001.parquet")
    rows = []
    for index, record in enumerate(pq.read_table(path).to_pylist()):
        spans = [(s["start"], s["end"], s["label"]) for s in json.loads(record["pii_spans"] or "[]")]
        language = GRETEL_LANGUAGE.get(record.get("language") or "", (record.get("language") or "").lower()[:2])
        rows.extend(
            document_rows(
                f"gretel-finance-{index}",
                record["generated_text"],
                spans,
                source="gretel-finance",
                language=language,
                source_category=record.get("document_type") or "",
            )
        )
    return finish(rows, "gretel-finance", max_samples, seed)


def value_spans(text: str, values: list[tuple[str, str]]) -> list[tuple[int, int, str]]:
    """Character spans of annotated values given without offsets: every occurrence of each value."""
    spans = []
    for value, kind in values:
        if value:
            spans.extend((m.start(), m.end(), kind) for m in re.finditer(re.escape(value), text))
    return spans


@cached_build
def load_gretel_en(max_samples: int | None = None, seed: int = 0) -> Dataset:
    import pyarrow.parquet as pq

    path = hub_file("gretelai/gretel-pii-masking-en-v1", "data/test-00000-of-00001.parquet")
    rows = []
    for record in pq.read_table(path).to_pylist():
        entities = ast.literal_eval(record["entities"]) if isinstance(record["entities"], str) else record["entities"]
        values = [(str(e["entity"]), str((e.get("types") or ["pii"])[0])) for e in entities or []]
        rows.append(
            segment_row(
                str(record["uid"]),
                record["text"],
                value_spans(record["text"], values),
                source="gretel-en",
                language="en",
                source_category=record.get("domain") or "",
            )
        )
    return finish(rows, "gretel-en", max_samples, seed)


########## ai4privacy 400k / 500k ##########

AI4PRIVACY = {
    "400k": ("ai4privacy/pii-masking-400k", "validation"),
    "500k": ("ai4privacy/open-pii-masking-500k-ai4privacy", "validation"),
}


@cached_build
def load_ai4privacy(
    version: str = "400k", per_language: int | None = 8000, max_samples: int | None = 50_000, seed: int = 0
) -> Dataset:
    """An ai4privacy validation split, at most `per_language` rows per language; leaked placeholders dropped."""
    from datasets import load_dataset

    repo, split = AI4PRIVACY[version]
    rows = []
    for record in load_dataset(repo, split=split):
        text = record["source_text"]
        if _PLACEHOLDER.search(text):
            continue
        spans = [(m["start"], m["end"], m["label"]) for m in record.get("privacy_mask") or []]
        rows.append(
            segment_row(
                str(record["uid"]),
                text,
                spans,
                source=f"ai4privacy-{version}",
                language=(record.get("language") or "").lower(),
                source_category=record.get("locale") or record.get("region") or "",
            )
        )
    rows = cap_per_language(rows, per_language, seed)
    return finish(rows, f"ai4privacy-{version}", max_samples, seed)


########## SPY ##########

SPY_FILES = {
    "legal": "data/legal_questions_placeholders.parquet",
    "medical": "data/medical_consultations_placeholders.parquet",
}


def spy_document(tokens: list[str], tags: list[str], fake: Any) -> tuple[str, list[tuple[int, int, str]]]:
    """The document with each <TYPE> placeholder filled by a Faker value, and the filled spans."""
    fillers = {
        "NAME": lambda: fake.name(),
        "EMAIL": lambda: fake.email(),
        "PHONE_NUM": lambda: fake.phone_number(),
        "ADDRESS": lambda: fake.address().replace("\n", ", "),
        "URL": lambda: fake.url(),
        "USERNAME": lambda: fake.user_name(),
        "ID_NUM": lambda: fake.ssn(),
    }
    text, spans = "", []
    for token, tag in zip(tokens, tags):
        if tag != "O":
            kind = tag[2:]
            value = fillers.get(kind, fake.word)()
            spans.append((len(text), len(text) + len(value), kind))
            text += value + token[len(token.rstrip()) :]
        else:
            text += token
    return text, spans


@cached_build
def load_spy(
    domains: tuple[str, ...] = ("legal", "medical"), max_samples: int | None = None, seed: int = 0
) -> Dataset:
    """SPY documents (placeholders filled with Faker values seeded by `seed`), cut into segments."""
    import pyarrow.parquet as pq
    from faker import Faker

    fake = Faker("en_US")
    Faker.seed(seed)
    rows = []
    for domain in domains:
        for index, record in enumerate(pq.read_table(hub_file("mks-logic/SPY", SPY_FILES[domain])).to_pylist()):
            text, spans = spy_document(record["tokens"], record["ent_tags"], fake)
            rows.extend(document_rows(f"spy-{domain}-{index}", text, spans, source="spy", source_category=domain))
    return finish(rows, "spy", max_samples, seed)


########## Privy ##########


@cached_build
def load_privy(member: str = "test-small.json", max_samples: int | None = None, seed: int = 0) -> Dataset:
    """Privy protocol traces from the release zip (the `O` filler spans are not PII and are dropped)."""
    with zipfile.ZipFile(hub_file("beki/privy", "privy-dataset.zip")) as archive:
        name = next(n for n in archive.namelist() if n.endswith(member))
        records = json.loads(archive.read(name))
    rows = []
    for index, record in enumerate(records):
        spans = [
            (s["start_position"], s["end_position"], s["entity_type"])
            for s in record.get("spans") or []
            if s["entity_type"] != "O"
        ]
        rows.append(
            segment_row(
                f"privy-{index}",
                record["full_text"],
                spans,
                source="privy",
                language="en",
                source_category=str(record.get("template_id") or ""),
            )
        )
    return finish(rows, "privy", max_samples, seed)


########## E3-JSI, urchade ##########


E3JSI_LANGUAGE = {
    "English": "en",
    "Dutch": "nl",
    "German": "de",
    "Slovene": "sl",
    "French": "fr",
    "Italian": "it",
    "Greek": "el",
}


@cached_build
def load_e3jsi(max_samples: int | None = None, seed: int = 0) -> Dataset:
    """E3-JSI synthetic multilingual PII NER (train is the only split); every annotated type counts."""
    import pyarrow.parquet as pq

    path = hub_file("E3-JSI/synthetic-multi-pii-ner-v1", "data/train-00000-of-00001.parquet")
    rows = []
    for index, record in enumerate(pq.read_table(path).to_pylist()):
        values = [(str(e["entity"]), str((e.get("types") or ["pii"])[0])) for e in record.get("entities") or []]
        rows.append(
            segment_row(
                f"e3jsi-{index}",
                record["text"],
                value_spans(record["text"], values),
                source="e3jsi-multi",
                language=E3JSI_LANGUAGE.get(record.get("language") or "", (record.get("language") or "").lower()),
                source_category=record.get("domain") or "",
            )
        )
    return finish(rows, "e3jsi-multi", max_samples, seed)


def joined_tokens(tokens: list[str], ner: list[list]) -> tuple[str, list[tuple[int, int, str]]]:
    """Space-joined tokens and the character spans of inclusive token ranges."""
    starts, position = [], 0
    for token in tokens:
        starts.append(position)
        position += len(token) + 1
    text = " ".join(tokens)
    spans = [(starts[a], starts[b] + len(tokens[b]), str(kind)) for a, b, kind in ner if 0 <= a <= b < len(tokens)]
    return text, spans


@cached_build
def load_urchade_mistral(max_samples: int | None = 20_000, seed: int = 0) -> Dataset:
    """urchade/synthetic-pii-ner-mistral-v1 (no splits; urchade's GLiNER PII models were trained on it)."""
    with open(hub_file("urchade/synthetic-pii-ner-mistral-v1", "data.json")) as handle:
        records = json.load(handle)
    rows = []
    for index, record in enumerate(records):
        text, spans = joined_tokens(record["tokenized_text"], record["ner"])
        rows.append(segment_row(f"urchade-{index}", text, spans, source="urchade-mistral", language=""))
    return finish(rows, "urchade-mistral", max_samples, seed)


########## SpokenWOZ dev ##########


@cached_build
def load_spokenwoz_dev(max_samples: int | None = None, seed: int = 0) -> Dataset:
    """The 500 SpokenWOZ dev calls (listed in valListFile.json, stored in the train text file)."""
    listing = open(hub_file("ssz1111/SpokenWOZ-Train-Text", "valListFile.json")).read().split()
    wanted = {name.strip() for name in listing if name.strip()}
    with open(hub_file("ssz1111/SpokenWOZ-Train-Text", "data.json")) as handle:
        dialogs = json.load(handle)
    rows = [
        row
        for dialog_id, dialog in sorted(dialogs.items())
        if dialog_id in wanted
        for row in spokenwoz_rows(dialog_id, dialog)
    ]
    return finish(rows, "spokenwoz-dev", max_samples, seed)


__all__ = [
    "chunk_ranges",
    "document_rows",
    "joined_tokens",
    "load_ai4privacy",
    "load_e3jsi",
    "load_gretel_en",
    "load_gretel_finance",
    "load_pii_bench_ru",
    "load_privy",
    "load_spokenwoz_dev",
    "load_spy",
    "load_tab",
    "load_urchade_mistral",
    "pii_bench_rows",
    "spy_document",
    "tab_document_spans",
    "value_spans",
]
