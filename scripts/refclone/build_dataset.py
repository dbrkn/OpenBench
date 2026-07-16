#!/usr/bin/env python3
"""Build an OpenBench speech-generation dataset from force_aligner_speech_regions.

Mirrors the refclone_study build step:

1. Download VAD segment clips + transcripts from the HF repo.
2. For each recording: split into a FIXED generation target (tail) + reference pool (head).
3. Build reference clips at several lengths (joined segment wavs + joined text).
4. Write a HuggingFace `save_to_disk` dataset OpenBench can load, where each
   row is one (sample × reference-length) condition:

     text       = target transcript   (WER reference + TTS prompt)
     audio      = real target wav     (SIM yardstick)
     ref_audio  = reference wav path  (voice-clone ICL prompt)
     ref_text   = reference transcript
     language   = "en"
     audio_name = {short_id}_L{tag}   (unique sample id)

Run (from OpenBench-tts-eval root):

  python scripts/refclone/build_dataset.py
  python scripts/refclone/build_dataset.py --sample-ids id1,id2,id3 --mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import snapshot_download

DEFAULT_SAMPLE = "en_US_Southern_Agriculture_1592841_channel1"
DEFAULT_TARGET_SPEECH_S = 64.0
DEFAULT_GRID = [2.3, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 276.0]
GAP_S = 0.25
REPO_ID = "argmaxinc/force_aligner_speech_regions"


def _dur(row: dict) -> float:
    return float(row["end_sec"]) - float(row["start_sec"])


def _tag_from_wav_s(wav_s: float) -> str:
    return f"{round(wav_s, 1):g}".replace(".", "p")


def _short_id(sample_id: str) -> str:
    """Stable short name for audio_name / folders (last numeric id if present)."""
    parts = sample_id.split("_")
    for p in reversed(parts):
        if p.isdigit():
            return p
    return sample_id.replace("/", "_")[:32]


def _concat_regions(repo_root: Path, regs: list[dict], gap_s: float = GAP_S) -> tuple[np.ndarray, int, float]:
    auds: list[np.ndarray] = []
    sr: int | None = None
    for r in regs:
        a, s = sf.read(str(repo_root / r["file_name"]))
        a = np.asarray(a, dtype=np.float32)
        sr = sr or int(s)
        if int(s) != sr:
            raise ValueError(f"sample-rate mismatch in {r['file_name']}: {s} vs {sr}")
        auds.append(a)
        auds.append(np.zeros(int(gap_s * sr), dtype=np.float32))
    if not auds:
        raise ValueError("no regions to concatenate")
    audio = np.concatenate(auds[:-1])
    speech_s = sum(_dur(r) for r in regs)
    assert sr is not None
    return audio, sr, speech_s


def _load_segments(repo_root: Path, sample_id: str) -> list[dict]:
    meta = repo_root / "segments" / "metadata.jsonl"
    if not meta.exists():
        raise FileNotFoundError(f"missing {meta}; expected flat segments/metadata.jsonl on Hub")
    rows = []
    for line in meta.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("sample_id") == sample_id:
            rows.append(row)
    rows.sort(key=lambda r: int(r["segment_index"]))
    if not rows:
        raise SystemExit(f"no segments for sample_id={sample_id!r} in {REPO_ID}")
    return rows


def _split_head_tail(rows: list[dict], target_speech_s: float) -> tuple[list[dict], list[dict]]:
    if len(rows) < 3:
        raise SystemExit(f"need at least 3 regions, got {len(rows)}")
    cum = 0.0
    cut = len(rows) - 1
    for i in range(len(rows) - 1, 0, -1):
        cum += _dur(rows[i])
        cut = i
        if cum >= target_speech_s:
            break
    cut = max(1, cut)
    head, tail = rows[:cut], rows[cut:]
    if not tail:
        raise SystemExit("empty target split; lower --target-speech-s")
    return head, tail


def _nearest_prefix_n(cum: np.ndarray, L: float) -> int:
    return int(np.argmin(np.abs(np.log(np.maximum(cum, 1e-6)) - np.log(L)))) + 1


def _build_one(
    repo_root: Path,
    sample_id: str,
    out: Path,
    grid: list[float],
    midpoints: list[tuple[int, int]],
    target_speech_s: float,
    mini: bool,
    mini_n: int,
) -> tuple[dict, list[dict]]:
    sid_short = _short_id(sample_id)
    refs_dir = out / "refs" / sid_short
    tgt_dir = out / "target" / sid_short
    refs_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {sample_id} ===", flush=True)
    rows = _load_segments(repo_root, sample_id)
    head, tail = _split_head_tail(rows, target_speech_s)

    t_audio, sr, t_speech = _concat_regions(repo_root, tail)
    target_wav = tgt_dir / "target.wav"
    target_txt = tgt_dir / "target.txt"
    sf.write(str(target_wav), t_audio, sr)
    t_text = " ".join((r.get("transcript") or "").strip() for r in tail).strip()
    target_txt.write_text(t_text + "\n")
    print(
        f"[target] regions {tail[0]['segment_index']}..{tail[-1]['segment_index']}  "
        f"speech {t_speech:.1f}s  wav {len(t_audio)/sr:.1f}s  words {len(t_text.split())}",
        flush=True,
    )

    cum = np.cumsum([_dur(r) for r in head])
    max_speech = float(cum[-1])
    use_grid = [L for L in grid if L <= max_speech * 1.05]
    if max_speech not in use_grid and max_speech > 0:
        use_grid.append(round(max_speech, 1))

    man: dict = {
        "sample_id": sample_id,
        "source_repo": REPO_ID,
        "target": {
            "regions": [int(tail[0]["segment_index"]), int(tail[-1]["segment_index"])],
            "speech_s": round(t_speech, 1),
            "words": len(t_text.split()),
            "wav": str(target_wav),
            "txt": str(target_txt),
        },
        "refs": [],
    }

    seen_tags: set[str] = set()
    for L in use_grid:
        n = _nearest_prefix_n(cum, L)
        regs = head[:n]
        audio, sr, _sp = _concat_regions(repo_root, regs)
        tag = _tag_from_wav_s(len(audio) / sr)
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        wav_path = refs_dir / f"ref_{tag}s.wav"
        txt_path = refs_dir / f"ref_{tag}s.txt"
        sf.write(str(wav_path), audio, sr)
        r_text = " ".join((r.get("transcript") or "").strip() for r in regs).strip()
        txt_path.write_text(r_text + "\n")
        man["refs"].append(
            {
                "L_target": L,
                "n_regions": n,
                "speech_s": round(float(cum[n - 1]), 1),
                "wav_s": round(len(audio) / sr, 1),
                "words": len(r_text.split()),
                "tag": tag,
                "prefix": True,
                "regions": [int(regs[0]["segment_index"]), int(regs[-1]["segment_index"])],
                "wav": str(wav_path),
                "txt": str(txt_path),
            }
        )
        print(
            f"[ref {L:>6g}s] regions {regs[0]['segment_index']}..{regs[-1]['segment_index']}  "
            f"speech {cum[n-1]:6.1f}s  wav {len(audio)/sr:6.1f}s  words {len(r_text.split()):4d}",
            flush=True,
        )

    if midpoints:
        for i, j in midpoints:
            if j > len(head):
                continue
            regs = head[i - 1 : j]
            audio, sr, sp = _concat_regions(repo_root, regs)
            tag = _tag_from_wav_s(len(audio) / sr)
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            wav_path = refs_dir / f"ref_{tag}s.wav"
            txt_path = refs_dir / f"ref_{tag}s.txt"
            sf.write(str(wav_path), audio, sr)
            r_text = " ".join((r.get("transcript") or "").strip() for r in regs).strip()
            txt_path.write_text(r_text + "\n")
            man["refs"].append(
                {
                    "L_target": float(tag.replace("p", ".")),
                    "n_regions": len(regs),
                    "speech_s": round(sp, 1),
                    "wav_s": round(len(audio) / sr, 1),
                    "words": len(r_text.split()),
                    "tag": tag,
                    "prefix": i == 1,
                    "regions": [i, j],
                    "wav": str(wav_path),
                    "txt": str(txt_path),
                }
            )
            print(f"[mid {tag}s] r{i}-r{j}  speech {sp:.1f}s  words {len(r_text.split())}", flush=True)

    man["refs"].sort(key=lambda r: r["speech_s"])
    refs = man["refs"]
    if mini:
        refs = refs[: max(1, mini_n)]

    hf_rows = []
    for ref in refs:
        hf_rows.append(
            {
                "audio": str(target_wav),
                "text": t_text,
                "ref_audio": ref["wav"],
                "ref_text": Path(ref["txt"]).read_text().strip(),
                "language": "en",
                "audio_name": f"{sid_short}_L{ref['tag']}",
                "ref_tag": ref["tag"],
                "ref_speech_s": ref["speech_s"],
                "sample_id": sample_id,
            }
        )
    return man, hf_rows


def build(args: argparse.Namespace) -> Path:
    out = Path(args.out_dir).resolve()
    ds_dir = out / "hf_dataset"
    out.mkdir(parents=True, exist_ok=True)

    sample_ids: list[str] = args.sample_ids
    print(f"[1/3] downloading {REPO_ID} ...", flush=True)
    repo_root = Path(snapshot_download(REPO_ID, repo_type="dataset", token=args.token))

    print(f"[2/3] building {len(sample_ids)} sample(s) ...", flush=True)
    all_mans: list[dict] = []
    all_rows: list[dict] = []
    for sid in sample_ids:
        man, rows = _build_one(
            repo_root,
            sid,
            out,
            args.grid,
            args.midpoints,
            args.target_speech_s,
            args.mini,
            args.mini_n,
        )
        all_mans.append(man)
        all_rows.extend(rows)

    manifest_path = out / "build_manifest.json"
    payload = {"source_repo": REPO_ID, "samples": all_mans} if len(all_mans) > 1 else all_mans[0]
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"[3/3] writing OpenBench hf_dataset ({len(all_rows)} rows) ...", flush=True)
    ds = Dataset.from_list(all_rows).cast_column("audio", Audio(sampling_rate=None))
    DatasetDict({"train": ds}).save_to_disk(str(ds_dir))
    print(f"wrote {len(all_rows)} rows → {ds_dir}", flush=True)
    print(f"manifest → {manifest_path}", flush=True)
    print(
        "\nNext:\n"
        f"  export REFCLONE_DATASET_PATH={ds_dir}\n"
        "  openbench-cli evaluate \\\n"
        "    --pipeline argmax-speech-generation-prototype \\\n"
        "    --dataset refclone-speech-regions \\\n"
        "    --metrics wer sim\n",
        flush=True,
    )
    return ds_dir


def _parse_midpoints(raw: str | None) -> list[tuple[int, int]]:
    if not raw:
        return []
    out = []
    for part in raw.split(","):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-id", default=None, help="single sample (default if --sample-ids unset)")
    ap.add_argument(
        "--sample-ids",
        default=None,
        help="comma-separated sample ids (overrides --sample-id)",
    )
    ap.add_argument("--out-dir", default="outputs/refclone_openbench")
    ap.add_argument("--target-speech-s", type=float, default=DEFAULT_TARGET_SPEECH_S)
    ap.add_argument("--grid", default=",".join(str(x) for x in DEFAULT_GRID))
    ap.add_argument("--midpoints", default="1-2,2-3,3-4")
    ap.add_argument("--no-midpoints", action="store_true")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--mini-n", type=int, default=2)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    if args.sample_ids:
        args.sample_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    else:
        args.sample_ids = [args.sample_id or DEFAULT_SAMPLE]
    args.grid = [float(x) for x in args.grid.split(",") if x.strip()]
    args.midpoints = [] if args.no_midpoints else _parse_midpoints(args.midpoints)
    build(args)


if __name__ == "__main__":
    main()
