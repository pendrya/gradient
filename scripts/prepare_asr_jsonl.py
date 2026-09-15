#!/usr/bin/env python3
"""Build training JSONL for Qwen3-ASR fine-tuning (Kazakh / Russian / code-switched).

Produces the exact format consumed by QwenLM/Qwen3-ASR finetuning/qwen3_asr_sft.py:

    {"audio": "/abs/path.wav", "text": "language None<asr_text>transcript", "prompt": "Terms: ..."}

Why the extra work versus feeding the raw manifest:

  * The official collator uses padding=True with truncation=False, so every batch pads to
    its longest clip -- and the encoder never truncates to 30s the way Whisper does. With a
    conversational duration spread, random batching pays ~2.8x the real audio in encoder
    compute at batch 32. This script drops over-long clips and emits a `length` field so the
    Trainer can group similar-length samples (see the note it prints on exit: sorting the
    file alone is NOT enough, because the default sampler reshuffles every epoch).
  * The collator calls librosa.load per batch. Pre-resampling to 16 kHz mono WAV offline
    removes that CPU bottleneck from the training loop.
  * `prompt` is undocumented in the README but lands in the same system-role slot as the
    inference-time `context=` argument, so populating it trains context-biasing behaviour
    on your own glossary.
  * --bias-curriculum samples that prompt instead of pasting one static list. A model always
    trained on a perfectly matching bias list learns "on the list => in the audio" and then
    hallucinates terms at inference, when the real list holds hundreds of mostly-absent
    entries. The fix is empty-context dropout plus phonetically plausible distractors, with
    list length varied so the model generalises past one list size.

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
import random
import re
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


def find_terms(text: str, terms: list[str]) -> list[str]:
    """Glossary terms actually present in this transcript (case-insensitive, word-bounded)."""
    low = text.casefold()
    return [t for t in terms if re.search(rf"(?<!\w){re.escape(t.casefold())}(?!\w)", low)]


def sample_prompt(text: str, terms: list[str], rng: random.Random, args) -> str:
    """Build one training-time bias list per the curriculum in the feasibility report, section 6."""
    if rng.random() < args.p_empty:
        return ""  # dropout: force the model back onto acoustic evidence

    true_terms = find_terms(text, terms)
    include_true = bool(true_terms) and rng.random() < args.p_include_true
    shown = list(true_terms) if include_true else []

    # Vary list length so a model trained on 10-term lists does not degrade on 300 at inference.
    target = rng.randint(args.min_terms, args.max_terms)
    pool = [t for t in terms if t not in true_terms]
    rng.shuffle(pool)
    shown += pool[: max(0, target - len(shown))]

    if not shown:
        return ""
    # Shuffle so list position never correlates with whether a term is actually spoken.
    rng.shuffle(shown)
    return args.glossary_prefix + ", ".join(shown)


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
    ap.add_argument("--bias-curriculum", action="store_true",
                    help="sample the `prompt` per utterance (empty / true+distractors / distractors "
                         "only) instead of pasting the whole glossary into every sample")
    ap.add_argument("--p-empty", type=float, default=0.45,
                    help="share of samples with an empty bias list (default 0.45)")
    ap.add_argument("--p-include-true", type=float, default=0.7,
                    help="given a non-empty list, chance of including the terms actually spoken")
    ap.add_argument("--min-terms", type=int, default=5, help="shortest sampled bias list")
    ap.add_argument("--max-terms", type=int, default=50, help="longest sampled bias list")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling")
    ap.add_argument("--max-sec", type=float, default=DEFAULT_MAX_SEC)
    ap.add_argument("--min-sec", type=float, default=DEFAULT_MIN_SEC)
    ap.add_argument("--no-sort", action="store_true", help="skip duration sorting")
    args = ap.parse_args()

    terms: list[str] = []
    glossary = ""
    if args.glossary:
        terms = [t.strip() for t in args.glossary.read_text(encoding="utf-8").splitlines() if t.strip()]
        glossary = args.glossary_prefix + ", ".join(terms)
        print(f"glossary: {len(terms)} terms")
        if args.bias_curriculum:
            print(f"bias curriculum: p_empty={args.p_empty} p_include_true={args.p_include_true} "
                  f"list_len={args.min_terms}-{args.max_terms} seed={args.seed}")
        else:
            print(f"static glossary prompt ({len(glossary)} chars) -- pass --bias-curriculum "
                  "before training biasing behaviour, or the model will learn to over-trust it")
    elif args.bias_curriculum:
        raise SystemExit("--bias-curriculum requires --glossary")

    rng = random.Random(args.seed)

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

        if row.get("prompt"):
            prompt = row["prompt"]  # explicit per-row prompt always wins
        elif args.bias_curriculum:
            prompt = sample_prompt(text, terms, rng, args)
        else:
            prompt = glossary

        kept.append({
            "audio": str(dst.resolve()),
            "text": f"language {args.language}{ASR_TEXT_TAG}{text}",
            "prompt": prompt,
            # Mel frames at hop_length=160 / 16 kHz, i.e. the magnitude the encoder pads to.
            # Consumed by the Trainer's length-grouped sampler.
            "length": int(secs * 100),
            "_sec": secs,
        })

    if not args.no_sort:
        # Sorting by duration keeps padded batches tight; the Trainer shuffles within the
        # epoch anyway, so this mainly bounds worst-case batch memory.
        kept.sort(key=lambda r: r["_sec"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_sec = sum(r["_sec"] for r in kept)
    kept_prompts = [r["prompt"] for r in kept]
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in kept:
            rec.pop("_sec")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(kept)} utterances ({total_sec / 3600:.1f} h) -> {args.out}")
    if args.bias_curriculum and kept_prompts:
        empty = sum(1 for x in kept_prompts if not x)
        lens = [x.count(",") + 1 for x in kept_prompts if x]
        print(f"bias lists: {empty / len(kept_prompts):.0%} empty, "
              f"median length {sorted(lens)[len(lens) // 2] if lens else 0} terms")
    print(f"skipped: {skipped_long} too long (>{args.max_sec}s), "
          f"{skipped_short} too short (<{args.min_sec}s), {failed} unreadable")

    print(
        "\nEach record carries a `length` field (mel frames). To actually benefit from it, the\n"
        "training script needs two one-line patches -- file order alone is overridden because\n"
        "the default sampler reshuffles every epoch:\n"
        "  1. TrainingArguments: train_sampling_strategy='group_by_length',\n"
        "     length_column_name='length'   (older transformers: group_by_length=True)\n"
        "  2. qwen3_asr_sft.py drops unknown columns -- add 'length' to its `keep` set,\n"
        "     or the sampler will never see it.\n"
        "Without both, expect to pay ~2.8x the real audio in padded encoder compute."
    )


if __name__ == "__main__":
    main()
