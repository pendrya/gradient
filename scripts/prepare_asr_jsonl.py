#!/usr/bin/env python3
"""Build training JSONL for Qwen3-ASR fine-tuning (Kazakh / Russian / code-switched).

Produces the exact format consumed by QwenLM/Qwen3-ASR finetuning/qwen3_asr_sft.py:

    {"audio": "/abs/path.wav", "text": "language None<asr_text>transcript", "prompt": "Terms: ..."}

Why the extra work versus feeding the raw manifest:

  * The official collator uses padding=True with truncation=False, so every batch pads to
    its longest clip. Without duration bucketing one long utterance blows up GPU memory.
    This script drops over-long clips and sorts by duration so buckets stay tight.
  * The collator calls librosa.load per batch. Pre-resampling to 16 kHz mono WAV offline
    removes that CPU bottleneck from the training loop.
  * `prompt` is undocumented in the README but lands in the same system-role slot as the
    inference-time `context=` argument, so populating it trains context-biasing behaviour
    on your own glossary.

Input manifest: TSV or CSV with columns `audio` and `text` (optional `prompt`).

Usage:
    python scripts/prepare_asr_jsonl.py \
        --manifest data/ksc2_train.tsv \
        --out data/train.jsonl \
        --wav-dir data/wav16k \
        --glossary data/glossary.txt
"""

import argparse
import csv
import json
import subprocess
import wave
from pathlib import Path

# Utterances longer than this dominate padded batches; 30 s matches common ASR practice.
DEFAULT_MAX_SEC = 30.0
DEFAULT_MIN_SEC = 0.3

# "language None" tells the model not to learn a language tag from this sample. That is what
# you want for code-switched audio, which cannot carry two tags. See report section 6.
ASR_TEXT_TAG = "<asr_text>"


def duration(path: Path) -> float:
    """Duration in seconds, cheapest method first.

    Spawning ffprobe once per file costs ~10 ms of process overhead; across KSC2's 600k
    utterances that is hours of pure subprocess churn. Header reads avoid it entirely.
    """
    try:
        import soundfile
        info = soundfile.info(str(path))
        return info.frames / info.samplerate
    except ImportError:
        pass
    except RuntimeError:
        pass  # format soundfile cannot open; fall through to ffprobe

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as fh:
                return fh.getnframes() / fh.getframerate()
        except wave.Error:
            pass

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def resample(src: Path, dst: Path) -> None:
    """Convert to 16 kHz mono 16-bit WAV, the format the collator expects. Requires ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-i", str(src), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)],
        check=True,
    )


def read_manifest(path: Path) -> list[dict]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=delimiter))
    missing = {"audio", "text"} - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"manifest is missing required column(s): {sorted(missing)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="TSV/CSV with audio,text[,prompt]")
    ap.add_argument("--out", type=Path, required=True, help="output JSONL")
    ap.add_argument("--wav-dir", type=Path, help="resample into this dir (skip to use audio as-is)")
    ap.add_argument("--language", default="None",
                    help="language tag; 'None' (default) is right for code-switched audio. "
                         "Note: a fine-tuned 'Kazakh' tag still fails SUPPORTED_LANGUAGES "
                         "validation at inference unless you patch qwen_asr/inference/utils.py")
    ap.add_argument("--glossary", type=Path,
                    help="newline-separated terms; injected as the `prompt` field")
    ap.add_argument("--glossary-prefix", default="Technical terms: ",
                    help="framing for the glossary; format materially affects biasing quality")
    ap.add_argument("--max-sec", type=float, default=DEFAULT_MAX_SEC)
    ap.add_argument("--min-sec", type=float, default=DEFAULT_MIN_SEC)
    ap.add_argument("--no-sort", action="store_true", help="skip duration sorting")
    args = ap.parse_args()

    glossary = ""
    if args.glossary:
        terms = [t.strip() for t in args.glossary.read_text(encoding="utf-8").splitlines() if t.strip()]
        glossary = args.glossary_prefix + ", ".join(terms)
        print(f"glossary: {len(terms)} terms, {len(glossary)} chars")

    rows = read_manifest(args.manifest)
    kept, skipped_long, skipped_short, failed = [], 0, 0, 0

    for row in rows:
        src = Path(row["audio"])
        text = (row.get("text") or "").strip()
        if not text:
            failed += 1
            continue
        try:
            if args.wav_dir:
                dst = args.wav_dir / (src.stem + ".wav")
                if not dst.exists():
                    resample(src, dst)
            else:
                dst = src
            secs = duration(dst)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError, OSError):
            failed += 1
            continue

        if secs > args.max_sec:
            skipped_long += 1
            continue
        if secs < args.min_sec:
            skipped_short += 1
            continue

        kept.append({
            "audio": str(dst.resolve()),
            "text": f"language {args.language}{ASR_TEXT_TAG}{text}",
            "prompt": (row.get("prompt") or glossary),
            "_sec": secs,
        })

    if not args.no_sort:
        # Sorting by duration keeps padded batches tight; the Trainer shuffles within the
        # epoch anyway, so this mainly bounds worst-case batch memory.
        kept.sort(key=lambda r: r["_sec"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_sec = sum(r["_sec"] for r in kept)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in kept:
            rec.pop("_sec")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(kept)} utterances ({total_sec / 3600:.1f} h) -> {args.out}")
    print(f"skipped: {skipped_long} too long (>{args.max_sec}s), "
          f"{skipped_short} too short (<{args.min_sec}s), {failed} unreadable")


if __name__ == "__main__":
    main()
