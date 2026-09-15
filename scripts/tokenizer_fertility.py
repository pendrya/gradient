#!/usr/bin/env python3
"""Measure Qwen3-ASR tokenizer fertility on Kazakh vs Russian.

Reproduces the numbers quoted in docs/qwen3-asr-kk-ru-feasibility.md section 3.
Kazakh costs ~1.7x more tokens per word than Russian, which translates directly
into longer decoder sequences, slower training and higher memory per utterance.

Usage:
    pip install tokenizers huggingface_hub requests
    python scripts/tokenizer_fertility.py
"""

import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL = "Qwen/Qwen3-ASR-1.7B-hf"  # -hf variant ships tokenizer.json
ROWS = "https://datasets-server.huggingface.co/rows"

# Kazakh-specific Cyrillic letters absent from the Russian alphabet.
KK_LETTERS = "әғқңөұүһіІ"  # note: һ is Cyrillic shha U+04BB, not Latin h

SAMPLE_WORDS = ["Қазақстан", "Ұлттық", "өңірдегі", "жағдай", "сәлеметсіз"]


def wiki_text(config: str, pages: int = 3) -> list[str]:
    """Pull Wikipedia article text via the HF datasets server (100 rows/page max)."""
    out: list[str] = []
    for page in range(pages):
        resp = requests.get(
            ROWS,
            params={
                "dataset": "wikimedia/wikipedia",
                "config": config,
                "split": "train",
                "offset": page * 100,
                "length": 100,
            },
            timeout=90,
        )
        resp.raise_for_status()
        out += [r["row"]["text"] for r in resp.json()["rows"] if r["row"].get("text")]
    return out


def main() -> None:
    tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
    print(f"model: {MODEL}")
    print(f"vocab size: {tok.get_vocab_size()}\n")

    print("Kazakh-specific letters — single token each means no byte fallback:")
    for ch in KK_LETTERS:
        n = len(tok.encode(ch, add_special_tokens=False).ids)
        print(f"  {ch}  -> {n} token(s){'' if n == 1 else '   <-- fragmented'}")

    print("\nWord-level fragmentation:")
    for word in SAMPLE_WORDS:
        ids = tok.encode(word, add_special_tokens=False).ids
        print(f"  {word:12s} -> {len(ids)} tokens")

    print("\nCorpus fertility (300 Wikipedia articles per language):")
    fertility = {}
    for label, config in [("Kazakh", "20231101.kk"), ("Russian", "20231101.ru")]:
        # Truncate each article so one long page cannot dominate the average.
        text = "\n".join(doc[:4000] for doc in wiki_text(config))
        words = len(text.split())
        tokens = len(tok.encode(text).ids)
        fertility[label] = tokens / words
        print(f"  {label:8s} words={words:7d} tokens={tokens:7d} tok/word={fertility[label]:5.2f}")

    ratio = fertility["Kazakh"] / fertility["Russian"]
    print(f"\nKazakh/Russian fertility ratio: {ratio:.2f}x")
    print("=> budget ~%.0f%% longer decoder sequences for Kazakh than Russian." % ((ratio - 1) * 100))


if __name__ == "__main__":
    main()
