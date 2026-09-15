#!/usr/bin/env python3
"""Audit transcript conventions across a Kazakh-Russian code-switched corpus.

At 4,000 h the model cannot learn an inconsistent target, so convention drift between
annotators — not data volume — sets the WER ceiling. This is Phase 0 of
docs/qwen3-asr-kk-ru-feasibility.md: run it before training, not after.

It reports, per corpus:
  * script mix          — how much of the corpus is actually code-switched
  * numeral convention  — digits vs spelled-out, the most common inconsistency
  * casing / punctuation — annotation-round drift
  * non-speech markers  — [inaudible] style tags and their variants
  * homoglyphs          — Latin letters hiding inside Cyrillic words, a silent killer
  * duplicates          — copy-paste and pipeline errors

Nothing here is a hard failure; every number is a prompt to go look. Percentages that are
neither ~0 % nor ~100 % are the ones worth investigating.

Usage:
    python scripts/audit_transcripts.py --manifest data/train.tsv [--glossary data/glossary.txt]
    python scripts/audit_transcripts.py --jsonl data/train.jsonl
"""

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# Letters unique to Kazakh Cyrillic (absent from the Russian alphabet).
KK_ONLY = set("әғқңөұүһіІӘҒҚҢӨҰҮҺ")

# There is no converse set: Kazakh Cyrillic contains all 33 Russian letters plus 9 extra, so
# no letter proves Russian. These six are *in* the Kazakh alphabet but appear in native Kazakh
# words essentially never -- they carry Russian borrowings -- so they are a weak positive
# signal for Russian-origin material, not proof. Notably 'ы' is NOT one of them: it is common
# in native Kazakh (бойынша, қызық) and treating it as Russian mislabels ordinary Kazakh.
RU_MARKER = set("цщъьэёЦЩЪЬЭЁ")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")

# Latin characters that are visually identical to Cyrillic ones. A Latin 'o' inside an
# otherwise-Cyrillic word is invisible to a human reviewer and a different token to the model.
HOMOGLYPHS = set("aAcCeEoOpPxXyYBHKMTN3")
MARKER = re.compile(r"[\[\(<]\s*[^\]\)>]{0,40}\s*[\]\)>]")


def classify(text: str) -> str:
    """Label an utterance by the script evidence it carries.

    This is a character heuristic, not language ID. It cannot see a Russian insertion written
    entirely in letters Kazakh shares (which is most of them), so it *undercounts* code-switching.
    Use it to find convention drift, not to report a code-switch rate.
    """
    has_kk = any(c in KK_ONLY for c in text)
    has_ru = any(c in RU_MARKER for c in text)
    has_cyr = bool(CYRILLIC.search(text))
    if has_kk and has_ru:
        return "kazakh + russian-borrowing letters"
    if has_kk:
        return "kazakh (kk-specific letters)"
    if has_ru:
        return "russian-borrowing letters only"
    if has_cyr:
        return "cyrillic, undetermined"
    return "no cyrillic"


def homoglyph_hits(text: str) -> list[str]:
    """Words that are mostly Cyrillic but contain a look-alike Latin character."""
    hits = []
    for word in re.findall(r"\S+", text):
        if CYRILLIC.search(word) and any(c in HOMOGLYPHS for c in word):
            hits.append(word)
    return hits


def pct(n: int, total: int) -> str:
    return f"{n / total * 100:5.1f}%" if total else "  n/a"


def load(args) -> list[str]:
    if args.jsonl:
        out = []
        for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = json.loads(line).get("text", "")
            out.append(text.split("<asr_text>", 1)[-1])  # strip the language prefix
        return out
    path = Path(args.manifest)
    delim = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(encoding="utf-8") as fh:
        return [r["text"] for r in csv.DictReader(fh, delimiter=delim) if r.get("text")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="TSV/CSV with a `text` column")
    src.add_argument("--jsonl", help="training JSONL produced by prepare_asr_jsonl.py")
    ap.add_argument("--glossary", type=Path, help="check custom-vocab spelling consistency")
    ap.add_argument("--examples", type=int, default=3, help="examples to print per finding")
    args = ap.parse_args()

    texts = load(args)
    n = len(texts)
    if not n:
        raise SystemExit("no transcripts found")
    print(f"transcripts: {n}\n")

    print("== script evidence (heuristic, NOT a code-switch rate) ==")
    for label, count in Counter(classify(t) for t in texts).most_common():
        print(f"  {label:38s} {count:8d}  {pct(count, n)}")

    words = [w for t in texts for w in re.findall(r"\S+", t)]
    kk_w = sum(1 for w in words if any(c in KK_ONLY for c in w))
    ru_w = sum(1 for w in words if any(c in RU_MARKER for c in w))
    lat_w = sum(1 for w in words if LATIN.search(w))
    und = len(words) - kk_w - ru_w - lat_w
    print(f"\n  word level: {pct(kk_w, len(words))} carry kk-specific letters, "
          f"{pct(ru_w, len(words))} russian-borrowing letters, {pct(lat_w, len(words))} latin")
    print(f"              {pct(und, len(words))} undetermined (letters both languages share)")
    print("  Kazakh Cyrillic is a superset of Russian, so no letter proves Russian and this")
    print("  undercounts code-switching. For a real rate, run word-level language ID.")

    print("\n== numeral convention ==")
    digits = [t for t in texts if re.search(r"\d", t)]
    print(f"  contain digits                   {len(digits):8d}  {pct(len(digits), n)}")
    print("  -> want ~0% (all spelled out) or ~100%; anything between is annotator drift")

    print("\n== casing ==")
    alpha = [t for t in texts if any(c.isalpha() for c in t)]
    upper = sum(1 for t in alpha if t.lstrip()[:1].isupper())
    allcaps = sum(1 for t in alpha if t.isupper())
    print(f"  starts uppercase                 {upper:8d}  {pct(upper, len(alpha))}")
    print(f"  entirely uppercase               {allcaps:8d}  {pct(allcaps, len(alpha))}")

    print("\n== punctuation ==")
    terminal = sum(1 for t in texts if t.rstrip()[-1:] in ".?!…")
    internal = sum(1 for t in texts if re.search(r"[,;:—–]", t))
    print(f"  ends with terminal punctuation   {terminal:8d}  {pct(terminal, n)}")
    print(f"  contains internal punctuation    {internal:8d}  {pct(internal, n)}")

    print("\n== non-speech markers ==")
    marked = Counter(m.strip() for t in texts for m in MARKER.findall(t))
    if marked:
        total = sum(marked.values())
        print(f"  {total} markers, {len(marked)} distinct forms — variants below suggest drift")
        for form, count in marked.most_common(10):
            print(f"    {form!r:38s} {count:7d}")
    else:
        print("  none found")

    print("\n== homoglyphs (Latin characters inside Cyrillic words) ==")
    bad = Counter(w for t in texts for w in homoglyph_hits(t))
    if bad:
        affected = sum(1 for t in texts if homoglyph_hits(t))
        print(f"  {affected} utterances ({pct(affected, n).strip()}), {len(bad)} distinct words")
        for word, count in bad.most_common(args.examples):
            detail = " ".join(f"{c}={unicodedata.name(c, '?').split()[0]}" for c in word if c in HOMOGLYPHS or c in KK_ONLY)
            print(f"    {word!r:24s} x{count:<6d} {detail}")
        print("  -> these are distinct tokens to the model; normalise before training")
    else:
        print("  none found")

    print("\n== whitespace and duplicates ==")
    ws = sum(1 for t in texts if "  " in t or t != t.strip())
    dupes = Counter(texts)
    repeated = sum(c for c in dupes.values() if c > 1)
    print(f"  irregular whitespace             {ws:8d}  {pct(ws, n)}")
    print(f"  duplicated transcript text       {repeated:8d}  {pct(repeated, n)}")

    if args.glossary:
        print("\n== custom vocabulary ==")
        terms = [t.strip() for t in args.glossary.read_text(encoding="utf-8").splitlines() if t.strip()]
        joined = "\n".join(texts)
        low = joined.casefold()
        missing = []
        for term in terms:
            exact = len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", joined))
            fold = len(re.findall(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", low))
            if fold == 0:
                missing.append(term)
            elif exact != fold:
                print(f"  {term!r}: {fold} occurrences, only {exact} match the glossary casing")
        if missing:
            print(f"  {len(missing)} glossary terms never appear in transcripts: {missing[:args.examples]}")
        print("  -> case variants and absent terms both weaken biasing supervision")

    print("\nNext: double-annotate a 2-5 h sample and measure inter-annotator WER.")
    print("That number is the floor your model cannot beat.")


if __name__ == "__main__":
    main()
