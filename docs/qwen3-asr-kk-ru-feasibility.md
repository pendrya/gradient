# Fine-tuning Qwen3-ASR for Kazakh–Russian code-switching and domain-specific wording

**Feasibility assessment — revised 2026-09-15 for a 4,000 h in-domain corpus**

> **Revision note.** The first version of this study assumed training would rest on public corpora
> (KSC2) with 10–50 h of in-domain audio added later, and it recommended starting with LoRA. The
> availability of **4,000+ h of good-quality, in-domain, code-switched audio with custom vocabulary**
> invalidates both. Data is no longer the constraint; **compute, data-pipeline throughput, and
> transcript consistency are.** Recommendations in §5, §7 and §9 changed accordingly.

## Verdict

**Feasible with high confidence.** This is now a well-resourced training project, not a research gamble.

| Half of the ask | Assessment | Cost |
|---|---|---|
| **Specific wording / terminology** | Solvable *without* training via native context biasing; 4,000 h lets you go further and train the biasing behaviour itself | Days (inference) → folded into training |
| **Kazakh + kk↔ru code-switching** | Needs fine-tuning — Kazakh is **not** a supported language — but 4,000 h of in-domain code-switched audio is a strong hand | 4–8 weeks |

For scale: **4,000 h is 4× the entire Polyglot-Lion corpus** (968.83 h across four languages), which
took an unsupported language from 139.96 % to 39.19 % WER on one 48 GB GPU. It is also **3.3× KSC2**,
and unlike KSC2 it is in-domain and genuinely code-switched.

**The three things that now decide the outcome, in order:**

1. **Transcript convention consistency** across 4,000 h — this, not data volume, now caps your WER.
2. **Data-pipeline throughput** — the official recipe wastes roughly an order of magnitude of GPU time.
3. **Context-biasing curriculum** — trained naively, biasing makes hallucination *worse*, not better.

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

**At 4,000 h, use the 1.7B.** The 0.6B is for debugging the pipeline cheaply, not for the production run.
With this much data the smaller model will underfit, and the 1.7B's Russian baseline is 40 % better.

### Language support — the core constraint

The model supports 30 languages + 22 Chinese dialects, hard-coded in `qwen_asr/inference/utils.py:37`:

```
Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian,
Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch,
Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian
```

**Russian: yes. Kazakh: no.** Expect very poor zero-shot Kazakh — for Tamil, similarly unsupported,
the 1.7B model scores **139.96 % WER** on Common Voice (worse than emitting nothing).

---

## 2. Context biasing: free at inference, better when trained

Qwen3-ASR accepts free-form biasing text and was explicitly trained to use it ("the model learns to
utilize the context tokens inside the system prompt as background knowledge").

```python
model.transcribe("call.wav", context="Technical terms: Қазақстан Халық Банкі, ЖСН, БСН, овердрафт, ...")
```

`_build_messages` (`qwen_asr/inference/qwen3_asr.py:448`) drops the string into the **system role**:

```python
return [
    {"role": "system", "content": context or ""},
    {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
]
```

**The training script's `prompt` field lands in that same slot.** It is undocumented in the README —
visible only in an argument-validation string (`"Needs fields: audio, text, optional prompt"`) and in
`make_preprocess_fn_prefix_only`, which reads `ex.get("prompt", "")` into `build_prefix_messages`.

With 4,000 h you can therefore train *how your terminology gets applied*, not merely supply a list at
runtime. **This is the single strongest reason to choose Qwen3-ASR over a fine-tuned Whisper here** —
Whisper has no equivalent mechanism.

### The curriculum matters more than the glossary — see §6

Naive training on this field actively harms you. Details and the recipe are in §6.

---

## 3. Precedent, and why Kazakh should beat Tamil

**Polyglot-Lion** full-fine-tuned Qwen3-ASR on 968.83 h / 607,839 utterances across English, Mandarin,
Tamil and Malay:

| Benchmark | Qwen3-ASR-1.7B base | After fine-tuning |
|---|---|---|
| Tamil CV (WER) | 139.96 % | **39.19 %** |
| Malay Mesolitica (WER) | 39.00 % | **21.51 %** |
| English LibriSpeech (WER) | 2.31 % | 2.10 % (improved) |
| Mandarin AISHELL-1 (CER) | 1.52 % | 1.45 % (improved) |

Cost: **48 h on a single 48 GB GPU ≈ $81**. Recipe: full FT, AdamW, cosine, peak LR 2e-5,
per-device batch 8 × grad-acc 4. No catastrophic forgetting — but only because of balanced upsampling.

Four reasons to expect a materially better result than Tamil's 39.19 %:

1. **Script is already covered.** Russian is in-domain, so Cyrillic is well-represented.
2. **Turkic transfer exists.** Turkish is supported (9.47 FLEURS WER); Kazakh is typologically close.
3. **~19× more target-language audio** than Tamil's 215 h.
4. **In-domain and code-switched**, where Tamil's was generic read speech. This is the biggest factor —
   in-domain data is worth several times its volume in generic data.

---

## 4. Measured: the tokenizer is not a blocker, but it is a tax

Measured directly against `Qwen/Qwen3-ASR-1.7B-hf` (vocab 151,705); reproduce with
`scripts/tokenizer_fertility.py`.

**No vocabulary surgery needed.** Every Kazakh-specific Cyrillic letter (`ә ғ қ ң ө ұ ү һ і І`) exists
as a **single token**. Nothing falls back to multi-byte fragments.

**But Kazakh fragments badly.** On 300 matched Wikipedia articles per language (~108 k Kazakh words,
~130 k Russian words):

| Language | tokens/word | Ratio |
|---|---|---|
| Russian | 2.85 | 1.00× |
| **Kazakh** | **4.87** | **1.71×** |

The BPE merges were learned on Russian, so Kazakh-specific letters break merge chains mid-word
(`Қазақстан` → 5 tokens, `өңірдегі` → 7).

**Implications:** ~1.7× longer decoder sequences for Kazakh → proportionally more decoder compute and
memory, and longer dependency chains to learn. Expect Kazakh WER to settle **above** Russian WER even
after a successful fine-tune.

**At 4,000 h, is vocabulary extension now worth it?** Still no. It is the kind of intervention that
becomes tempting at this data scale, but adding Kazakh-specific merges reinitialises embeddings and
discards the pretrained structure that makes Cyrillic work today — while the fertility tax is a
constant-factor compute cost, not an accuracy ceiling. Revisit only if you plateau and have ruled out
data quality first.

---

## 5. Engineering at 4,000 h — where this project actually gets hard

The official recipe is `finetuning/qwen3_asr_sft.py` in `QwenLM/Qwen3-ASR`. At 50 h it is adequate.
At 4,000 h its shortcuts become the dominant cost.

### Corpus physics

| Quantity | Estimate |
|---|---|
| Audio as 16 kHz mono 16-bit WAV | **~461 GB** (115.2 MB per hour) |
| Same as FLAC (`soundfile` reads it natively) | **~230–280 GB** |
| Utterances at ~8 s mean | **~1.8 M** |
| Mean tokens/utterance (100 audio + ~75 Kazakh text + prompt) | ~205 |

Storage alone is a planning item on Paperspace `/notebooks` persistent volumes. **Store FLAC, not WAV** —
it halves footprint and IO at no accuracy cost.

### It is full fine-tuning only

**No LoRA, no layer freezing, no gradient checkpointing, no DeepSpeed/FSDP.** The script hands the whole
model to a plain HF `Trainer`. Memory for the 1.7B variant (2.35 B params, AdamW):

```
bf16 weights          4.7 GB
bf16 gradients        4.7 GB
AdamW fp32 states    18.8 GB   (8 bytes/param)
---------------------------------
static subtotal     ~28.2 GB   + activations
```

**Revised from v1: use full fine-tuning, not LoRA.** LoRA was the right call for a 50 h in-domain set.
At 4,000 h it is the wrong tool twice over — it would underfit this much data, and you are teaching a
*new language*, which requires moving the audio encoder's representations for unseen phonology, not
just adapting attention projections. Keep LoRA only for same-day smoke tests.

### Compute budget — and the order of magnitude the pipeline is wasting

Two independent estimates, which disagree by ~10×:

**(a) Extrapolating the measured Polyglot-Lion anchor.** Their balanced-upsampled corpus is ~1,039 h
per epoch, trained in 48 h on one 48 GB card — roughly 20× realtime, assuming one epoch (the paper does
not state the epoch count). Scaling to 4,000 h: **~190 GPU-hours per epoch.**

**(b) First-principles FLOPs floor.** At 6ND with N = 2.35 B and ~205 tokens/utterance over 1.8 M
utterances ≈ 5.2 × 10¹⁸ FLOPs/epoch, plus ~50 % encoder overhead. On an A100-80GB at a realistic 35 %
MFU (~110 TFLOPS effective): **~20–30 GPU-hours per epoch.**

**That gap is the data pipeline, and most of it is recoverable.** The official loop is input-bound, not
compute-bound: `librosa.load` runs per batch inside the collator, `load_dataset(...).map(num_proc=1)`
preprocesses ~1.8 M rows single-threaded, and `padding=True` with `truncation=False` pads every batch to
its longest clip with no duration bucketing.

**Plan for 50–100 GPU-hours per epoch with a fixed pipeline; 200+ if you run the official one as-is.**
At 2–3 epochs that is the difference between a long weekend on 8 GPUs and several weeks. Treat these as
planning estimates, not quotes — your first 200 h pilot run (Phase 1) will replace them with a measured
figure.

### Fixes, in descending order of payoff at this scale

| Problem | Where | Fix |
|---|---|---|
| `librosa.load` per batch in the collator | `load_audio()` | Pre-convert to 16 kHz mono FLAC; load with `soundfile`; raise `--num_workers` to 8–16 |
| No duration bucketing; pads to longest in batch | collator | **Sort/bucket by duration** — reclaims 20–40 % of wasted compute outright |
| `.map(num_proc=1)` over ~1.8 M rows | dataset pipeline | Raise `num_proc`; cache the processed dataset to disk once |
| No gradient checkpointing exposed | `TrainingArguments` | `gradient_checkpointing=True` |
| AdamW fp32 states dominate memory | `TrainingArguments` | `optim="adamw_bnb_8bit"` → 18.8 GB becomes 4.7 GB |
| Plain DDP replicates 28 GB of state per GPU | `torchrun` path | Add **DeepSpeed ZeRO-2** (shards optimizer + gradients) via `TrainingArguments(deepspeed=...)`; the script hardcodes its args and does not expose it |
| README default `--batch_size 32` | README | Unrealistic; size to your card after bucketing |

**Multi-GPU is now required.** At 50–100 GPU-hours/epoch, a single card means weeks of wall clock.
The script supports `torchrun` DDP, but DDP replicates full optimizer state on every GPU — fine on
80 GB cards, tight on 48 GB. ZeRO-2 is the sweet spot for a 2.35 B model and needs a small patch.

### Inference gotcha after fine-tuning

`SUPPORTED_LANGUAGES` is **validated at inference** (`qwen_asr/inference/utils.py:105`):

```python
if language not in SUPPORTED_LANGUAGES:
    raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")
```

`model.transcribe(..., language="Kazakh")` **raises**, even on your fine-tuned checkpoint. Patch the
list or pass `language=None`. Also note `Qwen3ASRModel.from_pretrained` works on a checkpoint dir only
because a callback copies tokenizer/config files into it — replicate that if you write your own loop.

---

## 6. Code-switching and the context-biasing curriculum

### Code-switching

The model's native output format is `language X<asr_text>...`, and `merge_languages` merges labels
across chunks — it can legitimately emit `"Chinese,English"`. **Multi-language output is native**, a
real advantage over Whisper's single-language-token design.

- **Train with `language None<asr_text>`.** A single utterance cannot carry two tags, and the README
  warns `language None` means the model won't learn LID from that sample — which is correct here.
  Polyglot-Lion dropped language tags entirely for this reason and it worked.
- **Do not force a language at inference** on code-switched audio.
- **Train one bilingual model, not two monolingual ones.** The kk-ru literature is consistent that
  unified bilingual models beat separate monolingual systems on mixed speech.
- The forced aligner does not cover Kazakh, so **word-level Kazakh timestamps will not work out of the
  box**. Separate work if your product needs them.

**You cannot measure your own code-switch rate by character class.** Kazakh Cyrillic is a *superset*
of Russian — all 33 Russian letters plus 9 extra (`ә ғ қ ң ө ұ ү һ і`) — so no letter proves Russian,
and `ы`, `ь`, `ъ`, `э` are ordinary Kazakh despite intuition. Only `ц щ ъ ь э ё` are weak positive
signals, being near-absent from native Kazakh vocabulary and arriving with Russian borrowings. Any
Russian insertion written in shared letters is invisible to character heuristics, which therefore
**undercount code-switching**. Use word-level language ID if you need a real number.

### Context biasing: train it with dropout and distractors, or it will hallucinate

This is the part most likely to go wrong, and it is not intuitive.

> "If a model is always trained with a perfectly matching bias list, it learns that if a word is on the
> list, it is likely in the audio." In deployment the list holds hundreds of terms, of which **none**
> may be present — and the model inserts them anyway.

The established mitigation is to **deactivate biasing for a large fraction of training samples**:
frequently supplying an empty bias list forces the model back onto its acoustic evidence, so biasing
acts "as a helpful hint rather than a crutch." Reported effect where applied: **43.3 % reduction in
rare-word error** while preserving general accuracy.

Concretely, when generating the `prompt` field:

| Share of samples | `prompt` contents |
|---|---|
| ~40–50 % | **Empty** — prevents over-reliance |
| ~30–40 % | True terms present in the utterance **+ 5–20 distractor terms** that are not |
| ~10–20 % | **Distractors only**, no true term — teaches the model to decline the hint |

Draw distractors from your own vocabulary, preferring phonetically similar terms — the literature
specifically recommends "related" negatives as distractors so the model learns "subtle differences to
better discriminate between similar phrases," plus false-positive negatives that teach it *not* to
substitute a biasing phrase over a contextually correct word.

Also vary list **length** during training (5 to a few hundred terms). A model trained only on 10-term
lists degrades when handed 300 at inference. Keep the framing string fixed to whatever you deploy —
a 184-run sweep found `"Technical terms: ..."` and similar framings beat space-joined terms by ~2× WER.

`scripts/prepare_asr_jsonl.py --bias-curriculum` implements this sampling and prints the realised
distribution (share of empty lists, median list length) so the curriculum is verifiable rather than
assumed. Without the flag it pastes one static glossary into every sample — the exact failure mode
described above.

---

## 7. Recommended plan

### Phase 0 — Data audit and splits (1–2 weeks) — *the highest-value phase*

With data volume solved, **transcript consistency is now your WER ceiling.** 4,000 h almost certainly
means multiple annotators over an extended period, and the model cannot learn an inconsistent target.

Audit and normalise, in this order:

1. **Russian-inside-Kazakh orthography.** Are Russian loanwords/insertions transcribed in Russian
   spelling, Kazakh phonetic spelling, or inconsistently? This is the central convention question for a
   code-switched corpus and the easiest to have gotten wrong at scale.
2. **Numerals, dates, currency, amounts.** Digits or words? Qwen3-ASR's own ITN behaviour is
   undocumented, so whatever you pick, you are teaching it — but it must be *consistent*.
3. **Casing and punctuation.** Consistent, or a mix of styles from different annotation rounds?
4. **Non-speech markers.** `[inaudible]`, hesitations, fillers, overlap tags — consistent, and do you
   want them emitted at inference?
5. **Custom-vocabulary spelling.** One canonical spelling per term, or drift across annotators?

`scripts/audit_transcripts.py` automates the mechanical half of items 1–5 — script evidence, numeral
and casing distributions, marker variants, duplicates, glossary casing drift, and **homoglyphs**
(Latin `e` inside a Cyrillic word is invisible to a reviewer and a different token to the model).
Percentages that are neither ~0 % nor ~100 % are the ones to investigate.

Then **quantify your noise floor**: double-annotate a 2–5 h sample and measure inter-annotator WER.
That number is the floor your model cannot beat. If it is 8 %, do not plan for 5 %.

**Build splits that don't leak:** hold out by **speaker *and* recording session**, never randomly.
Random utterance splits on conversational data leak speaker and session characteristics and will
overstate your result badly. Target ~20–40 h test, ~10–20 h dev. Add two extra slices:

- a **code-switch-dense** slice (utterances with the most language alternations), and
- an **unseen-terminology** slice, whose custom terms are absent from the training glossary, to
  measure whether biasing *generalises* or merely memorises.

Finally, baseline Qwen3-ASR-1.7B zero-shot on these splits. Expect poor Kazakh; record it anyway as
your improvement denominator.

### Phase 1 — Pipeline proof on a 200 h subset (~1 week)

Do not launch a 4,000 h run against an unvalidated loop. On a 200 h stratified subset:

- Convert to 16 kHz mono FLAC; build JSONL with `scripts/prepare_asr_jsonl.py`.
- Apply the §5 fixes: bucketing, workers, gradient checkpointing, 8-bit optimizer, ZeRO-2.
- Train the **0.6B** first purely to prove the loop, then the 1.7B.
- **Measure actual throughput** (hours-of-audio per GPU-hour) and replace the §5 estimates with it.
- Confirm loss decreases and Kazakh output is well-formed Cyrillic.

Exit criterion: measured throughput within ~2× of the first-principles floor. If you are at 20×,
the pipeline is still input-bound — fix it here, not after burning 200 GPU-hours.

### Phase 2 — Full training run (2–4 weeks)

- **Full fine-tune the 1.7B** on all 4,000 h. LR 2e-5 peak, cosine, warmup ~2 %.
- Start at **2 epochs**; extend only if dev WER is still improving.
- **Decide on KSC2 by ablation, not assumption.** Your data is in-domain and code-switched; KSC2 is
  broadcast/read Kazakh. It may add general Kazakh robustness or may just dilute. Run one arm with and
  one without on the 200 h pilot, and let the dev set decide. Your hours are worth more than its hours.
- **Watch Russian for regression** every checkpoint. 4,000 h of full FT will specialise the model; that
  is fine for a dedicated system, but if you also need general-purpose Russian ASR, keep a Russian
  ballast slice in the mixture or accept that you are building a specialist.
- Keep `language None<asr_text>` targets throughout.

### Phase 3 — Context-biasing curriculum (1–2 weeks)

Regenerate the JSONL with `scripts/prepare_asr_jsonl.py --bias-curriculum`, which implements the §6
sampling (empty-context dropout, true terms plus distractors, varied list length, shuffled order) and
reports the resulting distribution so you can verify it. Then continue training. Evaluate on
the **unseen-terminology** slice and explicitly measure **hallucination rate with an all-distractor
list** — that is the failure mode this curriculum exists to prevent.

### Hardware

Your environment is Paperspace Gradient (`setup.sh` provisions the ASR server on port 5000).

| Setup | Viability at 4,000 h |
|---|---|
| 1 × 48 GB (A6000-class) | Pilot only. Needs 8-bit optim + checkpointing; weeks of wall clock for the full run |
| 1 × A100-80GB | Workable but slow — plan multiple weeks |
| **4–8 × A100/H100-80GB + ZeRO-2** | **Recommended.** Brings a full run into days |

Budget in **GPU-hours** (~50–100/epoch fixed pipeline, ~190+/epoch unfixed) and multiply by your
provider's rate, rather than anchoring on Polyglot-Lion's $81 — that figure was for a corpus a quarter
this size.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Transcript convention inconsistency across 4,000 h** — now the #1 risk and your WER ceiling | **High** | Phase 0 audit + normalisation; measure inter-annotator WER as the floor |
| **Split leakage** from random rather than speaker/session-disjoint splits | **High** | Split by speaker *and* session; a leaked split makes every later decision wrong |
| **Biasing over-reliance / term hallucination** | **High** | §6 curriculum: empty-context dropout + distractors; measure hallucination on all-distractor lists |
| Pipeline input-bound, wasting ~10× GPU time | Medium–High | Phase 1 throughput gate before the full run |
| Official script lacks ZeRO/FSDP, bucketing, checkpointing | Medium | Patches in §5 |
| Russian regression from heavy specialisation | Medium | Russian ballast; validate every checkpoint |
| Kazakh tokenizer fertility 1.71× | Medium | Accept it; budget compute; expect Kazakh WER > Russian WER |
| Storage/IO at ~461 GB WAV | Medium | Store FLAC (~230–280 GB) |
| No Kazakh forced alignment | Low–Medium | Separate work if timestamps required |

### What I would and would not promise

The earlier estimate of **12–20 % Kazakh WER was conditioned on training mostly on out-of-domain KSC2**
and no longer applies. With 4,000 h of in-domain, code-switched, good-quality audio, the realistic
landing zone is meaningfully better — plausibly **high single digits to low teens on your own
conversational test set**, with Russian holding near its 5.99 baseline.

That range is an expectation, not a commitment, and it is bounded by two things outside the model's
control: your **inter-annotator agreement** (Phase 0 will give you the actual floor) and the intrinsic
difficulty of dense code-switching. Spontaneous, overlapping, telephone-bandwidth speech will land
worse than clean single-speaker segments regardless of data volume. Phase 0 and the Phase 1 pilot will
replace this estimate with measurements — treat any number before then, including this one, as a prior.

---

## 9. Alternatives

| Option | For | Against |
|---|---|---|
| **Qwen3-ASR-1.7B + full FT** (recommended) | Native context biasing *and* trainable biasing behaviour; native multi-language output; Apache-2.0; 4,000 h is ample for full FT | Kazakh from scratch; tooling needs the §5 patches |
| Whisper-large-v3-turbo FT on your data | Mature ASR fine-tuning ecosystem | **No context-biasing mechanism** — cannot satisfy the custom-vocab requirement as directly; single language token is a poor fit for code-switching |
| Commercial API | Kazakh supported natively, zero training | Cannot train on your 4,000 h or your glossary — which is the entire value of your dataset |

With 4,000 h in hand, the calculus has shifted decisively toward Qwen3-ASR. Whisper's head start on
Kazakh (9.16 % WER fine-tuned on KSC2) mattered when you had no Kazakh data of your own; you now have
3.3× KSC2, in-domain. What Whisper still cannot do is learn your terminology-biasing behaviour, and a
commercial API cannot use your data at all.

**Reference points for context only** (read speech — not comparable to conversational code-switched audio):
Whisper large-v3 zero-shot Kazakh 43.20 %; Whisper-large-v3-turbo FT on KSC2 9.16 %; ElevenLabs Scribe
3.1 % FLEURS / 5.5 % CV (vendor-reported).

### Public corpora — now optional

| Corpus | Hours | License | Role now |
|---|---|---|---|
| KSC2 (ISSAI) | ~1,200 | CC-BY-4.0 | Optional ballast; **decide by ablation** (§7 Phase 2) |
| KSD (OpenSLR 140) | 554 | open | Optional |
| FLEURS `kk_kz` | ~12 | CC-BY | Useful as a *comparable, publishable* eval point |
| Common Voice kk | 3.76 | CC-0 | Negligible |

If you do use KSC2, its CC-BY-4.0 terms permit commercial use but require attribution in your product.

---

## Sources

- [QwenLM/Qwen3-ASR (GitHub)](https://github.com/QwenLM/Qwen3-ASR) — README, `finetuning/qwen3_asr_sft.py`, `qwen_asr/inference/`
- [Qwen3-ASR Technical Report (arXiv 2601.21337)](https://arxiv.org/html/2601.21337v1)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) · [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
- [Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR (arXiv 2603.16184)](https://arxiv.org/html/2603.16184v1)
- [Contextual Speech Recognition with Difficult Negative Training Examples (arXiv 1810.12170)](https://arxiv.org/pdf/1810.12170) · [Wiki-En-ASR-Adapt (arXiv 2309.17267)](https://arxiv.org/pdf/2309.17267) · [Contextual Biasing for LLM-Based ASR with Hotword Retrieval and RL (arXiv 2512.21828)](https://arxiv.org/pdf/2512.21828)
- [Context-format sweep: Qwen3Plugin issue #321 (TypeWhisper)](https://github.com/TypeWhisper/typewhisper-mac/issues/321) · [Qwen3-ASR fine-tuning OOM discussion #90](https://github.com/QwenLM/Qwen3-ASR/discussions/90)
- [FSDP vs DeepSpeed (HF Accelerate)](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed) · [DeepSpeed with HF](https://huggingface.co/docs/peft/en/accelerate/deepspeed)
- [KSC2 (Interspeech 2022)](https://www.isca-archive.org/interspeech_2022/mussakhojayeva22_interspeech.html) · [ISSAI corpus page](https://issai.nu.edu.kz/kz-speech-corpus/) · [OpenSLR 140](https://www.openslr.org/140/)
- [Impact of Using a Bilingual Model on Kazakh–Russian Code-Switching Speech](https://ceur-ws.org/Vol-2590/short13.pdf) · [abilmansplus/whisper-turbo-ksc2](https://huggingface.co/abilmansplus/whisper-turbo-ksc2)
