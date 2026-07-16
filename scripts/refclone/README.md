# Refclone → OpenBench

OpenBench port of the `refclone_study` scripts: take VAD segment clips from
`argmaxinc/force_aligner_speech_regions`, build reference-length prompts + a
fixed target, voice-clone with `argmax-speech-generation-prototype`, score **WER**
and **SIM**.

## What each step does

1. **Build** (`build_dataset.py`) — Download segment wavs/transcripts from HF.
  Glue early segments into short/long **reference** prompts. Glue the last ~60s into a fixed **target** (the text to synthesize + the real audio for SIM).
2. **Generate** — OpenBench runs `tts-cli` voice_clone: “speak the target text in
  the reference speaker’s voice.”
3. **WER** — Transcribe the clone and compare to the target transcript.
4. **SIM** — Compare the clone’s speaker embedding to the **real target audio**
  (same yardstick as the study — not the prompt clip).



## Setup

```bash
# from OpenBench-tts-eval root
python scripts/refclone/build_dataset.py
# smoke-test sized dataset:
python scripts/refclone/build_dataset.py --mini
```

Requires HF read access to the private dataset (`hf auth login`).

## Evaluate

```bash
export REFCLONE_DATASET_PATH="$(pwd)/outputs/refclone_openbench/hf_dataset"

# needs Argmax `tts-cli` (e.g. /Users/rosegranger/Argmax/.venv/bin/tts-cli)
# OpenBench voice_clone defaults (MLX end-to-end):
#   --voice-clone-backend mlx, --code-decoder-backend mlx,
#   --no-chunk, --max-new-tokens 2500, --mlx-max-sequence-length 8192,
#   Base-bf16 MLX talker. Needs ArgmaxPrototypes with --voice-clone-backend
#   (e.g. berkin/mlx-voice-clone). Override with --pipeline-config if needed.
openbench-cli evaluate \
  --pipeline argmax-speech-generation-prototype \
  --dataset refclone-speech-regions-mini \
  -m wer -m sim \
  --pipeline-config cli_path=/Users/rosegranger/Argmax/.venv/bin/tts-cli
```

Needs `tts-cli` from ArgmaxPrototypes/Argmax on PATH (or
`pipeline-config cli_path=...`). For seed-tts-faithful SIM, pass the WavLM
checkpoint via metric kwargs / evaluation config
(`checkpoint=.../wavlm_large_finetune.pth`).

Optional overrides (disable study settings for a short smoke test):

```bash
--pipeline-config no_chunk=false --pipeline-config max_new_tokens=245
```



## Unlike raw seedTTS rows


| Field                    | Meaning here                                        |
| ------------------------ | --------------------------------------------------- |
| `text`                   | Target transcript (synthesize this - WER reference) |
| `audio`                  | Real target wav (SIM yardstick)                     |
| `ref_audio` / `ref_text` | Voice-clone ICL prompt (varies by length)           |


Default sample: `en_US_Southern_Agriculture_1592841_channel1` (overridable with
`--sample-id` or several with `--sample-ids a,b,c`).

## Multiple samples (e.g. 5/13 earlier corrected ones)

```bash
.venv/bin/python scripts/refclone/build_dataset.py \
  --sample-ids \
    en_US_General_Agriculture_1586590_channel1,\
en_US_General_Aviation_1586677_channel1,\
en_US_General_Aviation_1586157_channel1,\
en_US_General_Agriculture_1586674_channel1,\
en_CA_Aviation_1586888_channel1 \
  --out-dir outputs/refclone_openbench
```

Each row’s `audio_name` is `{call_id}_L{tag}` so samples don’t collide.