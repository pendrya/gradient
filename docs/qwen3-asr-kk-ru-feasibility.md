# Fine-tuning Qwen3-ASR for Kazakh–Russian code-switching and domain-specific wording

**Feasibility assessment — 2026-09-15**

## Verdict

**Feasible, medium effort.** The request splits into two halves with very different costs:

| Half of the ask | Assessment | Cost |
|---|---|---|
| **Specific wording / terminology** | Largely solvable *without training* — Qwen3-ASR has native context biasing | Days |
| **Kazakh + kk↔ru code-switching** | Needs fine-tuning — Kazakh is **not** a supported language | 2–4 weeks |

The decisive facts: Russian is natively supported and strong (FLEURS WER 5.99), Kazakh is
absent from the model's 30-language list, and ~1,200 h of commercially-licensed Kazakh speech
containing kk-ru code-switching already exists (KSC2, CC-BY-4.0). A published precedent
(Polyglot-Lion) added an equally-unsupported language to this exact model for **~$81 of GPU time**.

The real risk is not the model — it is that **no public benchmark exists for conversational
Kazakh-Russian code-switched speech in your domain**. You will have to build your own eval set,
and that should happen before any training.

---

## 1. What Qwen3-ASR actually is

| | Qwen3-ASR-0.6B | Qwen3-ASR-1.7B |
|---|---|---|
| Total params (measured from safetensors) | **~0.94 B** | **~2.35 B** |
| Composition | Qwen3-0.6B LLM + ~180 M AuT encoder | Qwen3-1.7B LLM + ~300 M AuT encoder |
| bf16 weights on disk | 1.88 GB | 4.70 GB |
| FLEURS Russian WER | 9.91 | **5.99** |
| FLEURS Turkish WER (nearest Turkic) | 16.18 | 9.47 |
| License | Apache-2.0 | Apache-2.0 |

> **Naming caveat:** the size in the name refers only to the LLM decoder. "Qwen3-ASR-1.7B" is a
> **2.35 B-parameter** model once the audio encoder and projector are counted. Budget memory accordingly.

Architecture: an AuT audio encoder (128-dim fbank, 8× downsampling → 12.5 Hz token rate) feeding a
Qwen3 LLM decoder through a projector. Trained in four stages (40 M h of pseudo-labelled audio for
encoder pretraining, omni pretraining, ASR SFT *including context-biasing data*, then RL).

Relevant capabilities: automatic language ID, streaming, up to **1200 s** of audio per ASR call
(auto-chunked at low-energy boundaries), vLLM day-0 support, and a separate `Qwen3-ForcedAligner-0.6B`
for word-level timestamps (11 languages — Kazakh not among them).

### Language support — the core constraint

The model supports 30 languages + 22 Chinese dialects. The list is hard-coded in
`qwen_asr/inference/utils.py:37`:

```
Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian,
Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch,
Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian
```

**Russian: yes. Kazakh: no.** Expect very poor zero-shot Kazakh — for Tamil, similarly unsupported,
the 1.7B model scores **139.96 % WER** on Common Voice (i.e. worse than emitting nothing).

---

## 2. The easy half: domain-specific wording needs no training

Qwen3-ASR accepts free-form biasing text and was explicitly trained to use it
("the model learns to utilize the context tokens inside the system prompt as background knowledge").

```python
model.transcribe("call.wav", context="Technical terms: Қазақстан Халық Банкі, ЖСН, БСН, овердрафт, ...")
```

Reading the implementation confirms the mechanism — `_build_messages`
(`qwen_asr/inference/qwen3_asr.py:448`) drops the string straight into the **system role**:

```python
return [
    {"role": "system", "content": context or ""},
    {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
]
```

Two practical notes:

- **Format matters more than content.** A published 184-run sweep found that wrapping terms as
  `"Technical terms: ..."`, `"Vocabulary: ..."` or `"Proper nouns: ..."` gave up to **2× better WER**
  than simply space-joining the same terms. Tune this before concluding anything.
- **This is the single highest-leverage, lowest-cost experiment available.** Run it first.

### The undocumented lever

The training script accepts a `prompt` field per sample that lands in **the same system-role slot**
as the inference-time `context=`. The README never mentions it; only the argument-validation string does
(`"Needs fields: audio, text, optional prompt"`), and `make_preprocess_fn_prefix_only` reads
`ex.get("prompt", "")` into `build_prefix_messages`.

**Consequence: you can fine-tune the context-biasing behaviour itself on your own glossary** — teaching
the model how *your* terminology should be applied, not just that it exists. This is the strongest
argument for choosing Qwen3-ASR over a Whisper derivative for this use case.

---

## 3. The hard half: adding Kazakh

### Precedent — this has been done on this exact model

**Polyglot-Lion** full-fine-tuned Qwen3-ASR (both sizes) on 968.83 h / 607,839 utterances across
English, Mandarin, Tamil and Malay:

| Benchmark | Qwen3-ASR-1.7B base | After fine-tuning |
|---|---|---|
| Tamil CV (WER) | 139.96 % | **39.19 %** |
| Malay Mesolitica (WER) | 39.00 % | **21.51 %** |
| English LibriSpeech (WER) | 2.31 % | 2.10 % (improved) |
| Mandarin AISHELL-1 (CER) | 1.52 % | 1.45 % (improved) |

Cost: **48 h on a single 48 GB GPU ≈ $81** (vs $18,862 for the 128-GPU baseline they compared against).
Recipe: full FT, AdamW, cosine, peak LR 2e-5, per-device batch 8 × grad-acc 4.

Critically, they found **no catastrophic forgetting** — high-resource languages were preserved or
improved — but only because of **balanced upsampling**: intra-language balancing to the largest dataset
in each group, then inter-language replication to exactly 25 % per language.

### Kazakh should land better than Tamil

Three reasons to expect a better outcome than the Tamil result:

1. **Script is already covered.** Russian is in-domain, so Cyrillic is well-represented — unlike Tamil script.
2. **Turkic transfer exists.** Turkish is supported (9.47 FLEURS WER); Kazakh is typologically close
   (agglutinative, vowel harmony).
3. **4–6× more data available** than Tamil's 215 h.

### Measured: the tokenizer is not a blocker, but it is a tax

I measured this directly against `Qwen/Qwen3-ASR-1.7B-hf` (vocab 151,705) — reproducible via
`scripts/tokenizer_fertility.py`.

**Good news — no vocabulary surgery needed.** Every Kazakh-specific Cyrillic letter
(`ә ғ қ ң ө ұ ү һ і І`) exists as a **single token**. Nothing falls back to multi-byte fragments.

**Bad news — Kazakh fragments badly.** On 300 matched Wikipedia articles per language
(~108 k Kazakh words, ~130 k Russian words):

| Language | tokens/word | Ratio |
|---|---|---|
| Russian | 2.85 | 1.00× |
| **Kazakh** | **4.87** | **1.71×** |

The BPE merges were learned on Russian, so Kazakh-specific letters break merge chains mid-word:

```
Қазақстан  -> 5 tokens
Ұлттық     -> 5 tokens
өңірдегі   -> 7 tokens
```

**Implications:** ~1.7× longer decoder sequences for Kazakh → proportionally slower training and
inference, higher memory per utterance, and longer dependency chains to learn. Expect Kazakh WER to
settle **above** Russian WER even after a successful fine-tune. Vocabulary extension would fix the
fertility but destroys pretrained embedding structure and needs far more data than you have —
**not recommended at this scale.**

---

## 4. Data situation — the strongest part of the case

| Corpus | Hours | License | Notes |
|---|---|---|---|
| **KSC2** (ISSAI) | **~1,200** | **CC-BY-4.0** | 600 k+ utterances; TV, radio, senate, podcasts. **Explicitly contains kk-ru code-switching.** Subsumes KSC + KazakhTTS2. |
| KSD (OpenSLR 140) | 554 | open | 204 k utterances, regional/age diversity |
| KSC (OpenSLR 102) | 332 | open | subsumed by KSC2 |
| KazakhTTS2 | 271 | open | TTS-oriented, clean read speech |
| FLEURS `kk_kz` | ~12 | CC-BY | useful as a *comparable* eval set |
| Common Voice kk | 3.76 (2.39 validated) | CC-0 | too small to matter |

KSC2 is available at `issai/Kazakh_Speech_Corpus_2` on Hugging Face. **CC-BY-4.0 permits commercial
use** but requires attribution in your product.

Russian side: Common Voice ru, Golos, SOVA — ample. You mainly need Russian as *ballast* to prevent
forgetting, not as the primary training signal.

### The actual gap

There is **no public conversational kk-ru code-switching corpus with your domain's wording.**
KSC2 contains code-switched utterances but is broadcast/read-heavy. If your audio is telephony or
spontaneous conversation, this is your dominant source of error — and the main reason to record and
label **10–50 h of your own in-domain audio**. Budget for this; it is not optional.

### Baselines to beat

| System | Kazakh WER | Caveat |
|---|---|---|
| Whisper large-v3 (zero-shot) | 43.20 % | |
| Whisper large-v3-turbo FT on KSC2 | **9.16 %** | strongest open result |
| Whisper-base FT on KSC2 | 15.36 % | |
| ElevenLabs Scribe (commercial) | 3.1 % FLEURS / 5.5 % CV | **vendor-reported, read speech only** |

Treat the Scribe number with care — FLEURS is read speech and is not comparable to spontaneous
code-switched conversation. Do not anchor targets on it.

---

## 5. Engineering reality of the official recipe

The official recipe is `finetuning/qwen3_asr_sft.py` in `QwenLM/Qwen3-ASR`. Having read it, several
things matter that the README does not say.

### Data format

JSONL, one object per line:

```json
{"audio": "/data/wavs/utt0001.wav", "text": "language Kazakh<asr_text>Сәлеметсіз бе", "prompt": "Terms: ЖСН, БСН"}
```

The prefix is masked to `-100`; loss is computed only on the target. Audio is loaded with `librosa` at
16 kHz mono inside the collator.

### It is full fine-tuning only

**No LoRA, no layer freezing, no gradient checkpointing flag, no DeepSpeed/FSDP.** The script hands the
entire model to a plain HF `Trainer`. Memory for the 1.7B variant (2.35 B params, AdamW):

```
bf16 weights          4.7 GB
bf16 gradients        4.7 GB
AdamW fp32 states    18.8 GB   (8 bytes/param)
---------------------------------
static subtotal     ~28.2 GB   + activations
```

The 0.6B variant is ~11.3 GB static.

### Known failure modes — and fixes

There are open OOM reports on 48 GB A40s at the **0.6B** size with batch 8. The causes are visible in
the code:

| Problem | Where | Fix |
|---|---|---|
| `truncation=False` + `padding=True` pads to longest in batch — one 60 s clip explodes a batch of 32 | collator | Cap utterances at ~30 s; **bucket by duration** |
| No duration sorting/bucketing | dataset pipeline | Sort by length, group into buckets |
| README default `--batch_size 32` is unrealistic | README | Use 2–4 + grad-acc on 48 GB |
| No gradient checkpointing exposed | `TrainingArguments` | Add `gradient_checkpointing=True` |
| AdamW fp32 states dominate memory | `TrainingArguments` | `optim="adamw_bnb_8bit"` → 18.8 GB becomes 4.7 GB |
| `librosa.load` per batch in the collator | `load_audio()` | Pre-resample everything to 16 kHz mono WAV offline |
| No LoRA path | script | Wrap with `peft` if you want cheap iteration |

### Inference gotcha after fine-tuning

`SUPPORTED_LANGUAGES` is **validated at inference** (`qwen_asr/inference/utils.py:105`):

```python
if language not in SUPPORTED_LANGUAGES:
    raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")
```

So `model.transcribe(..., language="Kazakh")` will **raise**, even on your fine-tuned checkpoint.
You must either patch that list or pass `language=None`. Also note that `Qwen3ASRModel.from_pretrained`
on a checkpoint dir only works because a callback copies tokenizer/config files into each checkpoint —
if you write your own training loop, replicate that.

---

## 6. Code-switching specifics

The model's native output format is `language X<asr_text>...`, and `merge_languages` merges labels
across chunks — it can legitimately emit `"Chinese,English"`. **Multi-language output is native**, which
is a genuine advantage over Whisper's single-language-token design.

Recommendations for kk-ru:

- **Do not force a language at inference** on code-switched audio. Let LID run.
- **Train with `language None<asr_text>`** (or one consistent tag). A single utterance cannot carry two
  tags, and the README warns that `language None` means the model won't learn LID from that sample —
  which is the right trade here. Polyglot-Lion dropped language tags entirely for exactly this reason,
  relying on implicit acoustic identification, and it worked.
- **Train one bilingual model, not two monolingual ones.** The kk-ru code-switching literature is
  consistent on this: unified bilingual models outperform separate monolingual systems on mixed speech.
- Note that the forced aligner does not cover Kazakh, so **word-level timestamps for Kazakh will not
  work out of the box**. If your product needs them, that is separate work.

---

## 7. Recommended plan

### Phase 0 — Baseline and eval set (days, ~$0) — *do this first*

1. Build a **2–5 h in-domain eval set** with your actual terminology, split three ways:
   Russian-only / Kazakh-only / code-switched.
2. Measure Qwen3-ASR-1.7B zero-shot on all three.
3. Measure reference points: fine-tuned-Whisper-KSC2 (9.16 %) and, if self-hosting is negotiable,
   a commercial API.

**Decision gate:** if your real traffic is mostly Russian and context biasing alone clears your bar,
**stop here** — you will have saved a month.

### Phase 1 — Context biasing only (1–2 weeks, ~$0)

Tune the glossary format (`"Technical terms: ..."` vs alternatives) against the Phase 0 eval set.
This addresses the "specific wording" half of the request with no training.

### Phase 2 — Fine-tune for Kazakh (2–4 weeks)

- Start with **LoRA on the 1.7B** for fast iteration; move to full FT once the data pipeline is proven.
- Mixture: KSC2 (Kazakh) + Russian ballast + your in-domain data, **balanced-upsampled** to roughly equal
  proportions per Polyglot-Lion.
- Use `language None<asr_text>` targets.
- Validate on all three Phase 0 splits every checkpoint — watch for Russian regression.

### Phase 3 — In-domain code-switching

Add your recorded 10–50 h, with the `prompt` field populated with your real glossary so the model learns
*your* biasing behaviour.

### Hardware

Your environment is Paperspace Gradient (`setup.sh` provisions the ASR server on port 5000).

- **1.7B full FT:** A100-80GB comfortably; 48 GB workable with 8-bit optimizer + gradient checkpointing + bucketing.
- **1.7B LoRA:** fits on 24–48 GB.
- **0.6B:** fits most cards; use it to debug the pipeline cheaply.
- Rough estimate for ~1,200 h of audio: **1–3 days per epoch on a single 80 GB card.** (Extrapolated from
  Polyglot-Lion's 48 h for ~969 h — treat as an order-of-magnitude figure, not a quote.)

---

## 8. Risks and honest caveats

| Risk | Severity | Mitigation |
|---|---|---|
| **No public kk-ru conversational benchmark** — you cannot compare against anyone | High | Build your own eval set in Phase 0; it is a prerequisite, not a nicety |
| **KSC2 is broadcast/read-heavy** — domain mismatch if your audio is telephony | High | This is the biggest threat to any WER estimate below. Record in-domain data |
| Catastrophic forgetting of Russian if trained Kazakh-only | Medium | Balanced upsampling; validate Russian every checkpoint |
| Kazakh tokenizer fertility 1.71× | Medium | Accept it; budget compute and expect Kazakh WER > Russian WER |
| Official script lacks LoRA/checkpointing/bucketing | Medium | Patches listed in §5 |
| No Kazakh forced alignment | Low–Medium | Separate work if timestamps are required |
| KSC2 attribution obligation (CC-BY-4.0) | Low | Add the required notice to your product |

### What I would *not* promise

Sub-10 % WER on spontaneous, code-switched, in-domain telephony audio without in-domain training data.
A realistic post-Phase-2 expectation is **Kazakh WER in the 12–20 % range on broadcast-like speech**,
degrading on spontaneous conversation, with Russian holding near its 5.99 baseline if the mixture is
balanced. Phase 0 will replace these guesses with numbers.

---

## 9. Alternatives worth weighing

| Option | For | Against |
|---|---|---|
| **Qwen3-ASR + fine-tune** (this proposal) | Native context biasing; LLM decoder; native multi-language output; Apache-2.0 | Kazakh from scratch; full-FT-only tooling |
| **Whisper-large-v3-turbo FT on KSC2** | Already at 9.16 % Kazakh; checkpoints exist today | No context-biasing mechanism; weaker code-switching; single language token |
| **Commercial API (e.g. Scribe)** | Kazakh supported natively, zero training | No self-hosting; no domain-glossary training; per-minute cost; vendor-reported numbers on read speech |

The differentiator for *your* use case is context biasing plus trainable glossary behaviour. If the
"specific wording" requirement is genuinely central, Qwen3-ASR is the right base. If raw Kazakh accuracy
is all that matters and terminology is secondary, a fine-tuned Whisper is the cheaper path to a good number.

---

## Sources

- [QwenLM/Qwen3-ASR (GitHub)](https://github.com/QwenLM/Qwen3-ASR) — README, `finetuning/qwen3_asr_sft.py`, `qwen_asr/inference/`
- [Qwen3-ASR Technical Report (arXiv 2601.21337)](https://arxiv.org/html/2601.21337v1)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) · [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
- [Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR (arXiv 2603.16184)](https://arxiv.org/html/2603.16184v1)
- [KSC2: An Industrial-Scale Open-Source Kazakh Speech Corpus (Interspeech 2022)](https://www.isca-archive.org/interspeech_2022/mussakhojayeva22_interspeech.html) · [ISSAI corpus page](https://issai.nu.edu.kz/kz-speech-corpus/) · [IS2AI/ISSAI_SAIDA_Kazakh_ASR](https://github.com/IS2AI/ISSAI_SAIDA_Kazakh_ASR)
- [Kazakh Speech Dataset (OpenSLR 140)](https://www.openslr.org/140/) · [Kazakh Speech Corpus (OpenSLR 102)](https://www.openslr.org/102/)
- [Impact of Using a Bilingual Model on Kazakh–Russian Code-Switching Speech](https://ceur-ws.org/Vol-2590/short13.pdf)
- [Evaluating ASR Pipeline Configurations for Kazakh (MDPI Information 17(7):690)](https://www.mdpi.com/2078-2489/17/7/690)
- [abilmansplus/whisper-turbo-ksc2](https://huggingface.co/abilmansplus/whisper-turbo-ksc2) · [akuzdeuov/whisper-base.kk](https://huggingface.co/akuzdeuov/whisper-base.kk)
- [Context-format sweep: Qwen3Plugin issue #321 (TypeWhisper)](https://github.com/TypeWhisper/typewhisper-mac/issues/321) · [Qwen3-ASR fine-tuning OOM discussion #90](https://github.com/QwenLM/Qwen3-ASR/discussions/90)
- [ElevenLabs Kazakh speech-to-text](https://elevenlabs.io/speech-to-text/kazakh) · [Common Voice Kazakh 24.0](https://datacollective.mozillafoundation.org/datasets/cmj8u3pbb00dhnxxbsqe4vbpc)
