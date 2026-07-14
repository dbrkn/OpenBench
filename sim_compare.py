"""Controlled SIM-vs-tone experiment harness.

Tests the hypothesis: *a voice clone whose emotional delivery differs from its
reference clip scores a lower speaker-similarity (SIM) than one whose tone
matches.*

It does this the only way that survives the noise floor: hold the reference
clip and the clone mode fixed, vary **only** the emotion of the synthesized
text, and repeat each condition N times so we compare distributions (mean +/-
std) rather than single stochastic samples.

Pipeline per clip:

    text --> [tts-cli voice_clone] --> generated.wav --> [SIM] --> cosine vs reference

Why this file lives in `OpenBench-tts-eval/` and not `scripts/`: the SIM metric
(`openbench.metric.SpeakerSimilarity`) only exists in this fork's environment,
whereas `tts-cli` lives in the root `argmax_prototypes` environment. So we run
SIM in-process (this env) and generate by shelling out to the root env via
`uv run tts-cli` (cwd = repo root).

Example
-------
    cd OpenBench-tts-eval
    uv run python sim_compare.py \
        --sim-checkpoint /ABS/PATH/wavlm_large_finetune.pth \
        --calm-text  "We will win this election." \
        --angry-text "We will win this election, and they will pay for it!" \
        --n 15

Defaults assume the calm Trump reference at ``../refs/trump_calm.wav`` with the
transcript used during the earlier manual run (ICL mode). Pass ``--x-vector-only``
to clone without a transcript.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Repo root = parent of OpenBench-tts-eval/ (this file's dir).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF_AUDIO = REPO_ROOT / "refs" / "trump_calm.wav"
DEFAULT_REF_TEXT = (
    "Thank you class very much, it is a privilege to be here at this forum where"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "generated_audio" / "sim_compare"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repeated, controlled SIM comparison of tone-matched vs tone-mismatched clones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sim-checkpoint",
        required=True,
        help="Path to wavlm_large_finetune.pth (the seed-tts-eval WavLM-large speaker-verification checkpoint).",
    )
    p.add_argument(
        "--ref-audio",
        default=str(DEFAULT_REF_AUDIO),
        help="Reference (prompt) clip: drives the clone AND is the SIM target.",
    )
    p.add_argument(
        "--ref-text",
        default=DEFAULT_REF_TEXT,
        help="Transcript of --ref-audio (required for ICL mode; ignored with --x-vector-only).",
    )
    p.add_argument(
        "--calm-text",
        default="We will win this election.",
        help="Tone-MATCHED text (delivery similar to the calm reference).",
    )
    p.add_argument(
        "--angry-text",
        default="We will win this election, and they will pay for it!",
        help="Tone-MISMATCHED text (angry wording / exclamation).",
    )
    p.add_argument("--n", type=int, default=15, help="Repetitions per condition.")
    p.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Clone from the speaker embedding only (no ICL); --ref-text not needed.",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where generated clips + sim_results.csv are written.",
    )
    p.add_argument(
        "--version-dir",
        default="12hz-0.6b-base",
        help="tts-cli --version-dir (clone assets live under the base variant).",
    )
    p.add_argument(
        "--encoder-variant",
        default="W16A16-10s",
        help="Variant for the speaker/speech encoder voice-clone assets.",
    )
    p.add_argument(
        "--code-decoder-backend",
        default=None,
        choices=["coreml", "mlx"],
        help="tts-cli --code-decoder-backend. Omit for the CLI default (coreml). "
        "Use 'mlx' to match the seedTTS run (needs `uv sync --group mlx` in the root project).",
    )
    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="Re-score clips already present in --output-dir instead of regenerating.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the tts-cli commands that would run, then exit (no generation, no scoring).",
    )
    return p.parse_args()


def build_tts_command(
    args: argparse.Namespace,
    text: str,
    out_dir: Path,
    out_name: str,
) -> list[str]:
    """Assemble a single `uv run tts-cli` voice-clone invocation (root env)."""
    cmd = [
        "uv",
        "run",
        "tts-cli",
        "--mode",
        "voice_clone",
        "--ref-audio",
        str(Path(args.ref_audio).resolve()),
        "--text",
        text,
        "--version-dir",
        args.version_dir,
        "--speaker-encoder-variant",
        args.encoder_variant,
        "--output-dir",
        str(out_dir.resolve()),
        "--output-filename",
        out_name,
    ]
    if args.x_vector_only:
        cmd.append("--x-vector-only")
    else:
        cmd += [
            "--ref-text",
            args.ref_text,
            "--speech-encoder-variant",
            args.encoder_variant,
            "--speech-encoder-rvq-variant",
            args.encoder_variant,
        ]
    if args.code_decoder_backend:
        cmd += ["--code-decoder-backend", args.code_decoder_backend]
    return cmd


def generate_clip(cmd: list[str]) -> bool:
    """Run one generation in the root env. Returns True on success."""
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(
            "\n  generation FAILED (exit %d). Last stderr lines:\n%s\n"
            % (proc.returncode, "\n".join(proc.stderr.strip().splitlines()[-8:]))
        )
        return False
    return True


def summarize(label: str, values: list[float]) -> dict:
    if not values:
        return {"condition": label, "n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "condition": label,
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def cohens_d(a: list[float], b: list[float]) -> float:
    """Effect size (a - b) / pooled_sd. Magnitude guide: ~0.2 small, 0.5 medium, 0.8 large."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    if pooled == 0:
        return float("nan")
    return (statistics.fmean(a) - statistics.fmean(b)) / pooled


CONDITIONS = ("calm", "angry")


def main() -> None:
    args = parse_args()

    ckpt = Path(args.sim_checkpoint)
    if not args.dry_run and not ckpt.is_file():
        raise SystemExit(
            f"SIM checkpoint not found: {ckpt}\n"
            "Download wavlm_large_finetune.pth (the seed-tts-eval WavLM-large "
            "speaker-verification checkpoint) and pass its path via --sim-checkpoint."
        )

    ref_audio = Path(args.ref_audio).resolve()
    if not ref_audio.is_file():
        raise SystemExit(f"Reference audio not found: {ref_audio}")

    out_root = Path(args.output_dir).resolve()
    texts = {"calm": args.calm_text, "angry": args.angry_text}

    mode = "x-vector-only" if args.x_vector_only else "ICL"
    print("=" * 70)
    print("SIM vs tone experiment")
    print(f"  reference : {ref_audio}  (mode: {mode})")
    print(f"  calm  text: {texts['calm']!r}")
    print(f"  angry text: {texts['angry']!r}")
    print(f"  N per cond: {args.n}    output: {out_root}")
    print("=" * 70)

    # --- Dry run: just show the commands -------------------------------------
    if args.dry_run:
        for cond in CONDITIONS:
            cdir = out_root / cond
            cmd = build_tts_command(args, texts[cond], cdir, f"{cond}_00")
            print(f"\n[{cond}] example command (x{args.n}, filename varies):")
            print("  (cwd=%s)" % REPO_ROOT)
            print("  " + " ".join(repr(c) if " " in c else c for c in cmd))
        return

    # --- Generate ------------------------------------------------------------
    clips: dict[str, list[Path]] = {c: [] for c in CONDITIONS}
    for cond in CONDITIONS:
        cdir = out_root / cond
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            name = f"{cond}_{i:02d}"
            wav = cdir / f"{name}.wav"
            if args.skip_generation:
                if wav.is_file():
                    clips[cond].append(wav)
                else:
                    sys.stderr.write(f"  [skip-generation] missing {wav}, skipping\n")
                continue
            print(f"[gen] {cond} {i + 1}/{args.n} ...", flush=True)
            t0 = time.time()
            ok = generate_clip(build_tts_command(args, texts[cond], cdir, name))
            if ok and wav.is_file():
                clips[cond].append(wav)
                print(f"      -> {wav.name} ({time.time() - t0:.1f}s)")

    total = sum(len(v) for v in clips.values())
    if total == 0:
        raise SystemExit("No clips available to score.")

    # --- Score (loads the WavLM model once; first call is slow) --------------
    print("\nLoading SIM model (WavLM-large + ECAPA); first score is slow...")
    from openbench.metric import SpeakerSimilarity

    metric = SpeakerSimilarity(model_name="wavlm_large", checkpoint=str(ckpt))

    scores: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    rows: list[dict] = []
    for cond in CONDITIONS:
        for wav in clips[cond]:
            sim = metric.score(str(wav), str(ref_audio))
            scores[cond].append(sim)
            rows.append({"condition": cond, "clip": wav.name, "sim": f"{sim:.6f}"})
            print(f"  SIM[{cond:5s}] {wav.name}: {sim:.4f}")

    # --- Report --------------------------------------------------------------
    csv_path = out_root / "sim_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "clip", "sim"])
        w.writeheader()
        w.writerows(rows)

    calm_stats = summarize("calm (tone match)", scores["calm"])
    angry_stats = summarize("angry (tone mismatch)", scores["angry"])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for s in (calm_stats, angry_stats):
        if s["n"]:
            print(
                f"  {s['condition']:24s} n={s['n']:2d}  mean={s['mean']:.4f}  "
                f"std={s['std']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}"
            )
        else:
            print(f"  {s['condition']:24s} n=0 (no clips)")

    if scores["calm"] and scores["angry"]:
        gap = calm_stats["mean"] - angry_stats["mean"]
        d = cohens_d(scores["calm"], scores["angry"])
        print(f"\n  calm - angry  = {gap:+.4f}   (positive supports the hypothesis)")
        print(f"  Cohen's d     = {d:+.2f}   (|d|: ~0.2 small, 0.5 medium, 0.8 large)")
        print(
            "\n  Interpretation: a positive gap that is large relative to the stds "
            "(|d| >= ~0.5)\n  is evidence that tone mismatch lowers SIM. If the gap is "
            "tiny vs the\n  stds, the earlier 0.01 seedTTS difference was likely noise."
        )
    print(f"\n  Raw per-clip SIMs written to: {csv_path}")


if __name__ == "__main__":
    main()
