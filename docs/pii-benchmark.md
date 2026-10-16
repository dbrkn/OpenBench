# PII benchmark: Roblox PII classifier v2 vs OpenAI Privacy Filter

One question per row: **does this chunk give out PII, so that masking the whole chunk is right?** That is the
decision a chunk-masking layer over an always-on transcript makes. A row is a turn (or, for document sets, a
document or a sentence); `label` is `unsafe` when it holds PII. Every dataset is built at load time from its
original release (`src/openbench/dataset/pii_benchmarks.py`); nothing is re-hosted.

## Systems

| Alias | Model | Reading |
|---|---|---|
| `roblox-pii-v2` | `Roblox/roblox-pii-classifier-v2` (XLM-R large, PyTorch) | the turn as speaker `t` after up to 16 earlier turns, left-truncated to 512 tokens; PII when the giving head reaches 0.55 (card threshold) |
| `roblox-pii-v2-target-only` | same | the turn alone (how the Roblox card fed the other models) |
| `roblox-pii-v2-any-head` | same | asking 0.60 / giving 0.55 / off-platform 0.10, any firing; only for `roblox-pii-chat` |
| `roblox-pii-v2-coreml` | Sova's W8A16 Core ML bundle (macOS) | as `roblox-pii-v2`; for parity with the shipped asset |
| `openai-privacy-filter` | `openai/privacy-filter` (1.5B total / 50M active, bf16) | the turn alone; PII when any token's probability of not being `O` reaches 0.5; that probability (max over the turn) is the ROC score |
| `openai-privacy-filter-context` | same | the same 16 earlier turns as `speaker: text` lines; only tokens inside the target turn count |
| `constant-pii` | none | masks everything: recall 1, precision = the PII share; the F1 floor |

Privacy Filter is read through the `transformers` token-classification head (per-token argmax and
probabilities), not OpenAI's constrained Viterbi decoder and its operating points, so its spans can be more
fragmented than in the `opf` tooling; the per-turn verdict depends only on whether any token is PII.

## Datasets

| Alias | Rows | PII share | Unit and labeling rule | Licence |
|---|---|---|---|---|
| `roblox-pii-chat` | 39,202 | 37.5% | conversation; all of `t`'s turns; unsafe when any of the three labels is set | Apache-2.0 |
| `roblox-pii-chat-giving` | 34,504 | 28.9% | the safe conversations against those labeled PRIVACY_GIVING_PII | Apache-2.0 |
| `nemotron-pii` | 10,000 | 100% | document; unsafe when any span is tagged (seeded subset of the 100k test) | CC BY 4.0 |
| `nemotron-pii-sentences` | ~31,000 | ~36% | the sentences of 2,000 seeded documents, labeled by span overlap | CC BY 4.0 |
| `openpii` | 20,000 | ~99% | document from OpenPII 1.5M validation (seeded shuffle buffer), 20+ languages | ai4privacy licence ("other"; check before redistributing results) |
| `dialogpii` | 30,820 | 45.7% | turn of a written dialog, 11 languages; unsafe on a direct identifier (below) | CC BY 4.0 |
| `dialogpii-asr` | ~31,000 | ~48% | the same dialogs after TTS -> WhisperX, labels projected | CC BY 4.0 |
| `pii-trace` | 2,250 | 61.1% | user turn of the public 500-chat release (assistant turns are context) | MIT |
| `spokenwoz` | 17,782 | 19.3% | user turn of a real spoken call; unsafe when it informs or edits a profile slot | CC BY-NC 4.0 |
| `abcd` | 9,017 | 21.8% | customer turn; unsafe when it holds a scenario value (below) | MIT |

Every alias has a `-mini` variant: a seeded 2,000-row subsample with the PII share kept. `dialogpii-en` and
`dialogpii-asr-en` are the English slices.

Labeling rules worth knowing before reading a number:

- **Nemotron-PII and OpenPII documents are (almost) all PII.** Precision cannot fall, so F1 there only measures
  recall; the Roblox card's 99.2% / 99.8% on these two are recall numbers, and the card lists both datasets in its
  training-data metadata. `nemotron-pii-sentences` gives them a negative class. Nemotron's labels are broad
  (`company_name`, `occupation`, `date` count as PII).
- **DialogPII** uses TAB's schema, which tags professions, organisations, quantities and dates like "Sundays".
  Only direct identifiers count here: `PERSON`, `PERSON_EMAIL`, `CODE`, `CODE_PHONE`, `CODE_URL`, `LOC_STREET`,
  `LOC_HOUSENUMBER`, `LOC_ZIP` (`pii_types=` widens or narrows it). Names said in greetings ("Good morning,
  Judith") are PII under this rule.
- **PII-TRACE** is labeled in Privacy Filter's own taxonomy (`private_person` ... `secret`, plus `other_pii`), and
  only on user turns; the assistant often repeats the values unlabeled, which is why assistant turns are context
  only.
- **SpokenWOZ** labels come from the dialogue acts: `profile-inform` / `profile-edit` on name, phone number, ID
  number, email or plate number. The values are spelled and split across turns ("h u g h .", "9 1 6 ."), so a turn
  can be PII only in context. `profile-ack` turns (agreeing with the agent's read-back) count as negatives.
- **ABCD** labels are string matches of the scenario's values (customer name, email, phone, username, account and
  order id, street, zip) in customer turns; a value the customer gives in another form is missed (a false negative
  in the labels).
- Rows are turns of the speaker under test, so with context the classifier also sees that speaker's earlier
  turns. Roblox's giving head stays high on turns after a disclosure (`docs/design/00-sova-architecture.md`
  A11 in the Sova repo), which shows up as precision lost to context.

## Running on a GPU machine

The main `uv.lock` pins transformers 4.x and the audio stack; Privacy Filter needs transformers >= 5.6. The PII
benchmark therefore runs in its own text-only environment (no NeMo, pyannote.audio, whisperx; those pipeline
families are skipped with a warning).

```bash
git clone -b berkin/pii-benchmark git@github.com:dbrkn/OpenBench.git && cd OpenBench
scripts/pii/setup_env.sh                      # .venv-pii, prints the GPUs, runs the unit tests
MINI=1 OUT=pii_mini scripts/pii/run.sh        # every pair on the 2,000-row subsets
JOBS=8 scripts/pii/run.sh                     # the full sets; pii_results/report.md
```

`run.sh` builds every dataset once (cached under `~/.cache/openbench/pii/built/`), then runs each
(pipeline, dataset) pair as its own `openbench-cli evaluate` process, `JOBS` at a time, round-robin over the
visible GPUs; a finished pair is skipped on a rerun. `PIPELINES=` and `DATASETS=` narrow it. The report adds the
false-mask rate (the share of PII-free turns that get masked, from `constant-pii`'s precision) and the mean
milliseconds per turn.

A single pair by hand:

```bash
.venv-pii/bin/openbench-cli evaluate -p openai-privacy-filter -d spokenwoz -m recall -m precision -m f1 -m roc_auc
```

Every prediction is kept in `outputs/<date>/<time>/pii_results/<dataset>/predictions/<id>.json` with the raw
head probabilities (Roblox) or the detected spans (Privacy Filter), for error analysis.

## Not included

TAB (court cases; documents need chunking), Earnings-21 (numbers that are not PII; needs entity labels for
names), the Santa Barbara corpus (everyday talk, but it carries real names, so it is not a clean negative set),
the Reddit self-disclosure corpus (on request), Kaggle's PII essays (competition terms), PriMock57 (no PII labels),
and the scam / social-engineering dialogue sets (they label intent to extract, not PII present).
