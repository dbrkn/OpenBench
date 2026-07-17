# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Validate the `voiceclone-eval` dataset registration end-to-end.

Loads the alias through `DatasetRegistry`, materializes samples via
`SpeechGenerationDataset` (column mapping, reference audio decode, ICL
fields), and prints a per-speaker summary. Used by the
`voiceclone-eval` GitHub workflow; also runnable locally:

    uv run python scripts/voiceclone/validate_dataset.py --dataset voiceclone-eval-mini
"""

import argparse
import os
import sys
from collections import Counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="voiceclone-eval-mini", help="Registered dataset alias to validate")
    parser.add_argument("--max-rows", type=int, default=10, help="Max sample rows to print in the table")
    args = parser.parse_args()

    from openbench.dataset import DatasetRegistry
    from openbench.dataset.dataset_speech_generation import SpeechGenerationDataset

    config = DatasetRegistry.get_dataset_config(args.dataset)
    print(f"Alias: {args.dataset}")
    print(f"Config: {config}")

    dataset = SpeechGenerationDataset(config)
    n = len(dataset)
    print(f"Loaded {n} samples")

    speakers: Counter[str] = Counter()
    lines = [
        "| sample | speaker | ref dur (s) | ref words | synth words |",
        "|---|---|---|---|---|",
    ]
    for i in range(n):
        sample = dataset[i]
        ref_text = sample.extra_info.get("ref_text", "")
        assert sample.text.strip(), f"sample {i}: empty target text"
        assert ref_text.strip(), f"sample {i}: missing ref_text"
        assert sample.text != ref_text, f"sample {i}: target text equals ref_text"
        # SIM yardstick: held-out real target recording, distinct from the reference clip.
        sim_audio = sample.extra_info.get("sim_audio", "")
        assert sim_audio and os.path.exists(sim_audio), f"sample {i}: missing sim_audio yardstick"
        assert sample.sample_rate == 16000, f"sample {i}: unexpected sample rate {sample.sample_rate}"
        duration = sample.get_audio_duration()
        assert duration > 1.0, f"sample {i}: reference clip too short ({duration:.2f}s)"
        assert sample.extra_info.get("language") == "en", f"sample {i}: unexpected language"

        # audio_name is "{prompt_utt_id}-{target_utt_id}"; speaker is the recording id.
        speaker = sample.audio_name.split("_region_")[0]
        speakers[speaker] += 1
        if i < args.max_rows:
            lines.append(
                f"| {sample.audio_name} | {speaker} | {duration:.1f} | {len(ref_text.split())} | {len(sample.text.split())} |"
            )

    table = "\n".join(lines)
    speaker_summary = "\n".join(f"- {speaker}: {count}" for speaker, count in sorted(speakers.items()))
    report = (
        f"## voiceclone-eval validation: PASSED\n\n"
        f"**{n} samples, {len(speakers)} speakers** (alias `{args.dataset}`)\n\n"
        f"{table}\n\n"
        f"### Samples per speaker\n{speaker_summary}\n"
    )
    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
