# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Example: compute the Speaker Similarity (SIM) metric on audio pairs.

SIM is the cosine similarity between WavLM-large speaker embeddings of a
generated clip and a reference clip (the metric from
BytedanceSpeech/seed-tts-eval). This script reports the per-pair SIM plus the
aggregate ASV (mean) and ASV-var (variance), reproducing the seed-tts-eval
``cal_sim.sh`` output.

Usage
-----
The fine-tuned speaker-encoder checkpoint (``wavlm_large_finetune.pth``) is
required; download it as described in the seed-tts-eval README.

    # On macOS the s3prl/WavLM download needs a CA bundle:
    export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"

    python examples/speaker_similarity_sim.py \
        --checkpoint /path/to/wavlm_large_finetune.pth \
        --pair generated_1.wav reference_1.wav \
        --pair generated_2.wav reference_2.wav

The first run downloads the WavLM-large upstream (~1.2 GB) via torch.hub; later
runs use the cache.
"""

import argparse

from openbench.metric import SpeakerSimilarity


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Speaker Similarity (SIM) on audio pairs.")
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("GENERATED", "REFERENCE"),
        required=True,
        help="A (generated, reference) audio file pair. Repeat for multiple pairs.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the fine-tuned speaker-encoder checkpoint (e.g. wavlm_large_finetune.pth).",
    )
    parser.add_argument("--model-name", default="wavlm_large", help="Speaker-encoder upstream (default: wavlm_large).")
    parser.add_argument("--use-gpu", action="store_true", help="Run the model on GPU.")
    parser.add_argument("--device", default="cuda:0", help="Torch device when --use-gpu is set.")
    args = parser.parse_args()

    metric = SpeakerSimilarity(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        use_gpu=args.use_gpu,
        device=args.device,
    )

    print(f"{'pair':>5}  {'SIM':>8}  generated  vs  reference")
    for i, (generated, reference) in enumerate(args.pair, start=1):
        sim = metric(generated, reference, uri=f"pair_{i}")
        print(f"{i:>5}  {sim:>8.4f}  {generated}  vs  {reference}")

    print("-" * 40)
    print(f"ASV (mean):     {abs(metric):.4f}")
    print(f"ASV-var:        {metric.variance:.4f}")


if __name__ == "__main__":
    main()
