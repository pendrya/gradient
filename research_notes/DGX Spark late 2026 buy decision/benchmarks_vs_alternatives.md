# DGX Spark vs alternatives: measured local-LLM inference, fine-tuning and image/video benchmarks (state as of 20 Sep 2026)

Scope note: numbers below are dated and tagged with software/quant wherever the source gave them. "Vendor claim" means NVIDIA/AMD/Apple-published; everything else is a third-party or community measurement. Aggregator sites (tokenstead, localaimaster, presenc, modelfit, llmcheck, runaihome, etc.) are flagged as such; several of them mix measured and estimated figures and are cited only where nothing better exists. Hardware constants used throughout: DGX Spark (GB10) = 128 GB LPDDR5X at 273 GB/s; Strix Halo (Ryzen AI Max+ 395) = 128 GB at ~256 GB/s nominal (~215 GB/s measured); M3 Ultra = up to 512 GB at ~800-819 GB/s; M4 Max = 546 GB/s; M5 Max = 614 GB/s; RTX 5090 = 32 GB GDDR7 at ~1.79 TB/s; RTX PRO 6000 Blackwell = 96 GB GDDR7.

## Key Question 1: For the best ~100-250B MoE models, what decode and prefill tok/s does DGX Spark achieve today, and how does that compare with M3 Ultra, M5-gen Macs, Strix Halo, RTX PRO 6000 and 5090?

### Takeaway
On gpt-oss-120b (MXFP4) a single Spark delivers ~55-61 tok/s decode and ~1,600-2,000 tok/s prefill in llama.cpp (Oct-Nov 2025 builds), which is roughly tied with M3 Ultra on decode (~60 tok/s) and ~1.2-1.8x faster than Strix Halo on decode but 2-5x faster on prefill; a single RTX PRO 6000 is ~3-4x faster than all three on both axes but costs ~$16k in Sept 2026, and a 5090 cannot hold the model (~30 tok/s with CPU expert offload at zero context). For 235B-class models a single Spark cannot hold Q4 weights, and 2x Spark results range from ~12 tok/s (llama.cpp RPC, Dec 2025) to ~25 tok/s (vLLM AWQ) and, on 2026-era "Flash" MoE models with speculative decoding, 40-72 tok/s single-stream.

### Cited Findings

**DGX Spark (single node), llama.cpp, ggerganov's official benchmark thread**
- gpt-oss-120b MXFP4: pp2048 1,956 t/s, tg32 60.57 t/s; at d4096 pp 1,637 / tg 54.14; at d32768 pp 1,027 / tg 40.55 (build 7db35a7, Nov 2, 2025) — [llama.cpp discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- gpt-oss-20b MXFP4: pp2048 1,480 t/s, tg32 70.22 t/s; d32768 pp 1,078 / tg 45.69 — [llama.cpp discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- Qwen3-Coder-30B-A3B Q8_0: pp2048 1,384 t/s, tg32 58.95 t/s; d32768 pp 609 / tg 24.96 — [llama.cpp discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- Gemma 3 4B QAT: pp2048 3,851 / tg32 263.89; Qwen2.5-Coder-7B: pp2048 2,732 / tg32 184.08 — [llama.cpp discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- In the same thread, an AMD AI MAX+ 395 (ROCm) posted gpt-oss-120b pp2048 ~1,000 t/s / tg32 47.49 t/s, and an NVIDIA AGX Thor posted pp2048 967 / tg32 42 — [llama.cpp discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- Note on inconsistency: NVIDIA's own Oct 24, 2025 blog quotes gpt-oss-120b at 55.37 tok/s in llama.cpp, slightly below ggerganov's 60.57 — [NVIDIA developer blog (vendor claim)](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks)

**DGX Spark (single node), other frameworks**
- Ollama v0.12.6, firmware 580.95.05, Oct 23, 2025: gpt-oss-120b MXFP4 prefill 1,169 / decode 41.14; gpt-oss-20b 3,224 / 58.27; Gemma 3 27B q4_K_M 834 / 10.83, q8_0 585 / 7.21; Llama 3.1 70B q4_K_M 1,911 / 4.42; Qwen3 32B q4_K_M 705 / 9.41 — [Ollama blog](https://ollama.com/blog/nvidia-spark-performance)
- LMSYS/SGLang review (Oct 13, 2025): gpt-oss-20b Ollama batch 1 = 2,053 prefill / 49.7 decode; Llama 3.1 70B FP8 SGLang = 803 prefill / 2.7 decode; Llama 3.1 8B SGLang batch 1 = 7,991 / 20.5, batch 32 = 7,949 / 368; DeepSeek-R1-14B FP8 batch 8 = 2,074 / 83.5; EAGLE3 speculative decoding gave "up to 2x" end-to-end — [LMSYS blog](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- LMSYS caveat: "Software support for the DGX Spark is still in its early stages, the benchmark results ... may become outdated" — [LMSYS blog](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- ServeTheHome launch review (Oct 14, 2025) measured only 14.5 tok/s on gpt-oss-120b (framework not stated in the extract; far below llama.cpp numbers from the same week) — [ServeTheHome](https://www.servethehome.com/nvidia-dgx-spark-review-the-gb10-machine-is-so-freaking-cool/4/)
- vLLM 26.03, Apr 21, 2026: gpt-oss-120b MXFP4 single-stream 33.53 tok/s (structured-extraction workload, ~1,500-token prompts); Nemotron Super 49B NVFP4 single-stream 5.79 tok/s — [Dendro Logic](https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/)
- May 31, 2026 practical guide: gpt-oss-120b on SGLang "~50 tokens/s"; dense ~32B ~9-10 tok/s; small-active MoE dropped from ~20-25 tok/s at 16K context to ~15-17 tok/s at 32K — [Corti](https://corti.com/running-gpt-oss-120b-on-a-single-nvidia-dgx-spark-a-practical-guide/)
- Aggregated 2026 figures (unverified aggregator): gpt-oss-120b FP8 41.9 tok/s / 498 ms TTFT; vLLM MXFP4 single node 58.82 tok/s; 2 nodes 75.96 tok/s — [tokenstead (aggregator)](https://tokenstead.ai/guides/dgx-spark-benchmarks-2026)
- GLM-5.3-Flash (320B MoE) on ONE Spark using EXL3 2.05-bit (85 GB) + DFlash2 K7 speculation: 64 tok/s structured, 25 tok/s open-ended prose, ~182 tok/s aggregate at 4 streams, 262K context; community reports crashes/hangs and a 20-30% quality hit from 2-bit; previous single-Spark GLM-5.3-Flash results were "mid-20s to low-30s" (Sep 3, 2026) — [NVIDIA forums](https://forums.developer.nvidia.com/t/60-tok-s-glm-5-3-flash-on-a-single-dgx-spark/382140)
- Community benchmark index (2026, vLLM 0.24-0.26): Qwen 3.6 35B-A3B NVFP4 106.5 tok/s single-stream (Unsloth Fast, k=3 spec decode gave +57%); Nemotron 3 Nano 30B-A3B 55-61 tok/s across 2K-131K context; Qwen AgentWorld 35B-A3B BF16 30.4 tok/s — [howtospark.com](https://howtospark.com/)

**2x DGX Spark**
- Qwen3-235B-A22B UD-Q4_K_XL (134 GB) via llama.cpp RPC over TCP/IP: ~12.5 tok/s decode, 37.7 tok/s prefill at 2K context; single-Spark baseline ~1.8 tok/s (offloaded); vLLM+Ray and TRT-LLM NVFP4 attempts failed (missing SM121 GEMM kernels); author estimates the RPC workaround loses 2-3x vs native NCCL TP (Dec 17, 2025; CUDA 13, NCCL 2.28.9) — [NVIDIA forums report](https://forums.developer.nvidia.com/t/dgx-spark-multi-node-llm-inference-report-for-qwen3-235b-model/355126)
- Another user reported ~25 tok/s on 2 nodes with Qwen3-VL-235B-A22B AWQ in vLLM — [NVIDIA forums (search snippet, not independently verified)](https://forums.developer.nvidia.com/t/question-on-inference-performance-results-of-qwen3-235b-a22b-on-2x-dgx-spark/355053)
- NVIDIA vendor claim: Qwen3-235B NVFP4 on dual Spark in TRT-LLM = 11.73 tok/s (Oct 24, 2025) — [NVIDIA developer blog (vendor claim)](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks)
- NVIDIA CES 2026 vendor claim: Qwen-235B on dual Spark "2.6x" faster with NVFP4 + Eagle3 speculative decoding vs FP8 (Jan 5, 2026) — [NVIDIA developer blog (vendor claim)](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark)
- DeepSeek V4 Flash on 2x Spark (ConnectX-7 direct, vLLM/DSpark, Aug 12, 2026): single-request decode 52.9 tok/s at 256-token prompt, ~45-53 tok/s from 2K-131K; aggregate at 6 concurrent = 115.4 (256-tok prompt), 97.2 (2K), 57.4 (8K), 23.4 (32K), 6.5 (131K); TTFT at 131K = 89.4 s; code-gen C6 aggregate 110.5 tok/s — [aussielunix gist](https://gist.github.com/aussielunix/cc6630820513c57f63fd1f4563b2aa14)
- 2-node Spark lab (200 GbE RoCE, re-measured Sep 5, 2026): single-user tok/s short-prompt / code / agentic-4k / agentic-33k = DeepSeek-V4-Flash 72.4 / 59.2 / 41.8 / 22.5; Qwen3.8-Flash-Next (NVFP4 + NEXTN spec) 42.6 / 39.0 / 32.1 / 21.0; GLM-5.3-Flash (DFlash2 spec) 41.1 / 22.5 / 24.9 / 17.7. Concurrent (6 or 8 streams): DeepSeek 124.5 / 99.5 / 74.8 / 29.2; Qwen 131.1 / 116.6 / 99.3 / 28.6; GLM - / 77.3 / 51.8 / 23.7. Spec decoding gave 1.4-2.9x; run-to-run spread up to 2.02x on DeepSeek code — [WeZZard/dgx-spark-bench](https://github.com/WeZZard/dgx-spark-bench)
- GLM-5.2 (753B MoE) on dual Spark TP2 at 2-bit experts + NVFP4 attention: 25.79 tok/s, 96K context (community index) — [howtospark.com](https://howtospark.com/)
- 4x Spark: GLM-5.3-Flash NVFP4 at TP4 over a switchless RoCE ring, 36 tok/s, 262K context, 1.26M-token FP8 KV pool (repo README) — [tonyd2wild GitHub](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)

**AMD Strix Halo (Ryzen AI Max+ 395, 128 GB)**
- gpt-oss-120b ROCm in ggerganov's thread: pp2048 ~1,000 t/s / tg32 47.49 t/s (see above) — [llama.cpp discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- Minisforum MS-S1, Vulkan (RADV), 2026: gpt-oss-120b pp2048 447 t/s, pp32768 266 t/s, tg128 34.1 t/s; gpt-oss-20b pp2048 1,275 t/s, tg128 46.5 t/s; ~100 W sustained, ~10 W idle — [akehir.com](https://akehir.com/blog/strix-halo-kubernetes-llm-gpt-oss)
- Other Strix Halo gpt-oss-120b reports: 51.1 tok/s decode / 174 tok/s prefill (~59 GB used); "55 tok/s"; "~340 tok/s prefill vs Spark ~1,700" (search snippets from strixhaloguide.com / runaihome / llmrequirements aggregators; backend/version not stated) — [strixhaloguide](https://strixhaloguide.com/evidence/); [llmrequirements (aggregator)](https://llmrequirements.com/dgx-spark-vs-strix-halo)
- Level1Techs Strix Halo thread (ROCm TheRock 7.0, llama.cpp b5863, Jul 2025): Qwen3-30B-A3B pp512 605 / tg128 72.0; Llama 70B pp 94.7 / tg 5.0; Llama 2 7B Q4_0 ~1,000 / 46; no gpt-oss-120b, no 2026 updates — [Level1Techs forum](https://forum.level1techs.com/t/strix-halo-ryzen-ai-max-395-llm-benchmark-results/233796)
- The kyuz0 Strix Halo grid is now marked "Legacy results: these benchmarks are no longer updated" and redirects to local-llm-benchmarks.dev — [kyuz0 grid](https://kyuz0.github.io/amd-strix-halo-toolboxes/)
- Notebookcheck (Nov 10, 2025) relayed GMKtec's manufacturer-run test claiming the $2,199 EVO-X2 beat the Spark on token generation and first-response latency on Llama 3.3 70B, Qwen3 Coder, gpt-oss-20b, Qwen3 0.6B, but published no numbers — [Notebookcheck (vendor-run test)](https://www.notebookcheck.net/Nvidia-DGX-Spark-vs-AMD-Strix-Halo-4-000-AI-supercomputer-challenged-by-2-199-mini-PC-with-better-real-time-performance.1159829.0.html)
- Tom's Hardware's own review headline says the Spark "beats out AMD's Ryzen AI Max+ 395" (full page content not retrievable) — [Tom's Hardware review p.4](https://www.tomshardware.com/pc-components/gpus/nvidia-dgx-spark-review/4)

**Apple M3 Ultra / M5 Max / M5 Ultra**
- M3 Ultra, gpt-oss-120b 8-bit MLX (mlx-lm 0.27.0, Sep 6, 2025): 1,000 tok/s prefill and 60 tok/s decode when working; first long-context query sometimes ran ~7x slower with the GPU at 10 W / 50% utilization (workaround: warm-up query) — [mlx-lm issue #432](https://github.com/ml-explore/mlx-lm/issues/432)
- M3 Ultra 512 GB (80-core GPU), DeepSeek-R1-0528, ~11K-token prompt, May 2025: llama.cpp q4_K_M = 76.8 tok/s prefill / 4.26 decode (~9 min total); MLX 4-bit = 189.5 prefill / 11.15 decode (~4 min), 422 GB peak memory — [someoddcodeguy, dev.to](https://dev.to/someoddcodeguy/running-deepseek-r1-0528-q4km-and-mlx-4-bit-on-a-mac-studio-m3-le4)
- M3 Ultra 512 GB owners report Qwen3-235B-A22B at 16 tok/s (llama.cpp) to 24 tok/s (MLX); DeepSeek R1/V3 671B 4-bit ~6-18 tok/s depending on framework (aggregator summaries; quant not specified) — [vettedconsumer (aggregator)](https://vettedconsumer.com/which-mac-for-local-llms-2026-buyers-guide/); [willitrunai (aggregator)](https://willitrunai.com/blog/qwen-3-5-mlx-apple-silicon-guide)
- HN thread "maxed out Mac Studio gets 60-100 tok/s on 120B models" contains a measured M1 Ultra 128 GB gpt-oss-120b run: prefill 392 tok/s at 65,536 tokens (~167 s), decode 65.5 tok/s; commenters note prefill on Mac is "10s of times slower" than a high-end GPU for long prompts — [Hacker News](https://news.ycombinator.com/item?id=45303242)
- llama.cpp Apple-silicon table (7B test model): M3 Ultra 80c/800 GB/s Q4_0 pp512 1,471 / tg128 92.1; M4 Max 546 GB/s Q4_0 886 / 83.1; M5 Max 40c/614 GB/s Q4_0 pp 3,220 / tg 119.9 (commit c1d0e7a, Aug 25, 2026) — i.e., M5's prefill is ~3.4x M4 Max and ~2.2x M3 Ultra on this small model — [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
- M5 Max 128 GB (MacBook Pro), llmcheck leaderboard aggregate (Aug 2026): Llama 3.3 70B Q4_K_M MLX 15 tok/s; DeepSeek-R1-70B Ollama 11 tok/s; 32B-class A3B MoE ~80-82 tok/s; Llama 3.1 8B 138 tok/s (aggregator; some listed model names such as "Llama 5 70B" could not be verified) — [localaimaster (aggregator)](https://localaimaster.com/blog/apple-m5-for-ai-guide)
- M5 Ultra Mac Studio: no independent LLM benchmarks exist yet; the machine does not reach customers until Sep 22, 2026; published figures (e.g., gpt-oss-120b ~43 tok/s, DeepSeek-R1 671B ~6 tok/s, 70B Q4 20-25 tok/s) are bandwidth-based estimates — [Pinggy blog (estimates)](https://pinggy.io/blog/self_hosting_llms_on_512gb_m5_ultra_mac_studio/); [modelfit (estimates)](https://modelfit.io/blog/best-llm-mac-studio-m5-ultra-512gb/)
- Aggregated M5 Max vs Spark vs 5090 (not author-measured): 70B Q4 decode Spark 35-45 vs M5 Max 25-32 tps [flag: Spark 70B decode is inconsistent with Ollama's 4.4 tok/s measured above]; prefill for 8B: 5090 ~7,200, Spark ~1,200, M5 Max ~400 tok/s — [presenc.ai (aggregator; internally inconsistent)](https://presenc.ai/research/dgx-spark-vs-m5-max-vs-rtx-5090-throughput-2026)

**RTX PRO 6000 Blackwell (96 GB) and RTX 5090 (32 GB)**
- LMSYS: gpt-oss-20b prefill/decode = RTX PRO 6000 10,108 / 215 tps; RTX 5090 8,519 / 205 tps; Spark ~4x slower (Oct 2025) — [LMSYS blog](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- RTX PRO 6000 Server Edition, vLLM, 50 concurrent prompts (Aug 25, 2026): gpt-oss-120b 8-bit 1,779 tok/s aggregate, 342 ms TTFT, 32 ms TPOT; gpt-oss-20b 4,378 tok/s; DeepSeek-R1-Qwen-32B 16-bit 966 tok/s (1,655 at 300 concurrent) — [Database Mart](https://www.databasemart.com/blog/vllm-gpu-benchmark-pro6000)
- RTX PRO 6000 gpt-oss-120b single-stream: "193.30 tok/s tg128 at Q8_0 in llama.cpp", Q4_K_M weights ~59.4 GB (aggregator; not independently verified) — [modelfit (aggregator)](https://modelfit.io/gpu/rtx-6000-pro/)
- Llama 3.3 70B AWQ INT4 batched: one PRO 6000 8,425 tok/s aggregate vs one RTX 5090 4,570 tok/s (aggregator citing vendor/partner data) — [runaihome (aggregator)](https://www.runaihome.com/blog/rtx-pro-6000-blackwell-local-ai-2026/)
- RTX 5090 with gpt-oss-120b (59 GB MXFP4 does not fit in 32 GB): with llama.cpp `--n-cpu-moe 21` about 30 tok/s at zero context; another user went from 3.4 to >8 tok/s using MoE offload — [Hardware Corner](https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/); [llama.cpp guide #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)
- Spark vs 5090, Ollama (CUDA 12.6, Jan 17, 2026): Qwen2.5-7B Spark 46-49 vs 5090 220 tok/s; Qwen2.5-72B / Llama 3.2 90B / DeepSeek-R1-70B on Spark 4.4-4.7 tok/s, could not load on the 5090 — [ProX PC](https://www.proxpc.com/blogs/nvidia-dgx-spark-gb10-performance-test-vs-5090-llm-image-and-video-generation)

**Multi-GPU consumer and used datacenter cards**
- Single RTX 3090 at 250 W running Qwen3.8-27B (dense, requant) in patched vLLM: 127 tok/s single-user, ~1,035 tok/s at 64 concurrent, 150-262K context — [syv-ai/HyperQwen](https://github.com/syv-ai/qwen38-27b-rtx3090)
- No measured 2-4x RTX 3090/4090/5090 numbers for gpt-oss-120b or Qwen3-235B surfaced in searches (see Gaps).

### Inferences
- Decode on all three 128 GB unified-memory boxes is bandwidth-bound, so gpt-oss-120b lands in the same 35-60 tok/s band regardless of vendor; the differentiator is prefill, where Spark's Blackwell tensor cores (and FP4/NVFP4) give ~2-5x over Strix Halo and roughly parity-to-2x over M3 Ultra MLX depending on prompt length.
- Long-context degradation is steep everywhere: Spark gpt-oss-120b decode falls from 60.6 to 40.6 tok/s at 32K depth, and 2x Spark DeepSeek-V4-Flash falls from ~53 to ~17 tok/s single-stream at 32K with 89 s TTFT at 131K.
- For 235B-class Q4 models the honest single-Spark answer is "does not fit"; the 2x Spark path went from ~12 tok/s (llama.cpp RPC, Dec 2025) to 40-72 tok/s on 2026 Flash-style MoEs largely thanks to NVFP4 plus speculative decoding rather than raw hardware.
- Because M5 Max's llama.cpp prefill jumped ~3.4x over M4 Max on the same 7B test, an M5 Ultra (shipping Sep 22, 2026) could plausibly erase Spark's prefill edge while keeping ~3x its bandwidth; this is not yet measured.

### Gaps
- No independent M5 Ultra Mac Studio LLM benchmarks exist as of Sep 20, 2026 (ships Sep 22; 512 GB SKU "late October").
- No comparable gpt-oss-120b MXFP4 numbers on M3 Ultra from a controlled llama.cpp/MLX run with stated prompt length (only the 1,000/60 figure from a bug report and an M1 Ultra HN datapoint).
- Could not retrieve GLM-4.5-Air numbers on Spark, Mac or Strix Halo; 2026 community attention has moved to GLM-5.x Flash.
- No measured Qwen3-235B numbers on 2x Spark using native NCCL TP with NVFP4 (the working measurements are llama.cpp RPC or AWQ vLLM).
- No multi-GPU consumer rig (2-4x 3090/4090/5090) measurements for gpt-oss-120b / Qwen3-235B were found in this pass; no A100/H100 gpt-oss-120b single-card numbers were found.
- The Level1Techs and Hardware Canucks/Alex Ziskind video comparisons could not be fetched as text; only summaries ("inference is low but fine-tuning is fast", "TRT-LLM most promising", "1 PFLOP only in 500 ms bursts") surfaced — [Level1Techs](https://www.level1techs.com/node/3305).

## Key Question 2: Where does Spark win (prefill/compute, CUDA ecosystem, FP4, concurrency/batching, fine-tuning, image/video) and where does it lose (decode bandwidth, price, power/thermal)?

### Takeaway
Spark wins on prefill/compute (1.6-2k tok/s on gpt-oss-120b), batched serving (862 tok/s aggregate at 256 concurrent on gpt-oss-120b in vLLM), fine-tuning of models that do not fit in 32 GB (gpt-oss-120b LoRA in 17.8 min where a 5090 failed), NVFP4/CUDA tooling, and quiet ~60-200 W operation; it loses on single-stream decode (273 GB/s), on diffusion/video generation (3-5x slower than a 5090), and on price versus Strix Halo (~2x the cost for ~1.2-1.8x decode).

### Cited Findings

**Concurrency / batching (Spark)**
- vLLM 26.03, gpt-oss-120b MXFP4: aggregate 33.5 tok/s at c=1, 252 at c=32, 565 at c=128, 862.8 at c=256 (3.6 tok/s per stream at peak); Nemotron Super 49B NVFP4 695 tok/s at c=256; Nemotron Nano 9B (Mamba-2 hybrid) plateaus at ~156 tok/s above c=8; production extraction pipeline hit ~3,900 records/hour (Apr 21, 2026) — [Dendro Logic](https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/)
- SGLang Llama 3.1 8B: 20.5 tok/s at batch 1 to 368 tok/s at batch 32 with prefill flat at ~7,950 tok/s (Oct 2025) — [LMSYS blog](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- 2x Spark scaling efficiency at 6 concurrent = 41% vs c=1 on DeepSeek-V4-Flash (Aug 2026) — [aussielunix gist](https://gist.github.com/aussielunix/cc6630820513c57f63fd1f4563b2aa14)

**Fine-tuning**
- Measured (Jan 17, 2026, CUDA 12.6): gpt-oss-20b fine-tune Spark 4.62 min vs 5090 3.45 min; gpt-oss-120b fine-tune Spark 17.82 min vs 5090 failed (OOM) — [ProX PC](https://www.proxpc.com/blogs/nvidia-dgx-spark-gb10-performance-test-vs-5090-llm-image-and-video-generation)
- Vendor claims (Oct 24, 2025, PyTorch): Llama 3.2 3B full FT 82,739 tok/s; Llama 3.1 8B LoRA 53,658 tok/s; Llama 3.3 70B QLoRA 5,079 tok/s — [NVIDIA developer blog (vendor claim)](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks)
- Unsloth/NVIDIA (Oct 23, 2025): on an RTX 5090 (Alpaca, bs 2, GA 4, rank 32, QLoRA all-linear) Unsloth is 2x faster than HF+FA2 with >70% less VRAM and 12x longer context; Unsloth supports fine-tuning up to ~200B parameters on Spark (e.g., gpt-oss-120b) — no Spark tok/s tables published — [NVIDIA developer blog (vendor claim)](https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/); [Unsloth on Spark](https://build.nvidia.com/spark/unsloth)
- Aggregator claim: "RTX 5090 is 4x faster and ~30% cheaper for fine-tuning smaller models" but OOMs sooner — [float16.cloud (aggregator)](https://float16.cloud/en-en/ai-benchmark/dgx-spark-vs-rtx-5090/)
- Kaitchup's "use it for fine-tuning" piece gives no quantitative fine-tuning benchmarks; it argues the 128 GB matters for <~8B models with long context/large batch — [Kaitchup](https://kaitchup.substack.com/p/dgx-spark-use-it-for-fine-tuning)
- Level1Techs (Wendell): "for inference performance it's low but fine tuning and other tasks are very fast"; 1 PFLOP FP4 only achievable in ~500 ms bursts — [Level1Techs](https://www.level1techs.com/node/3305)

**Image / video generation**
- Measured (Jan 2026): FLUX.1-dev image Spark 237 s / 156 s (two runs) vs 5090 50 s / 56.6 s; Hunyuan Video 1.5 FP16 Spark 3,606 s vs 5090 1,310 s — [ProX PC](https://www.proxpc.com/blogs/nvidia-dgx-spark-gb10-performance-test-vs-5090-llm-image-and-video-generation)
- Vendor claims: Flux.1 12B FP4 "23 images/minute" (Oct 2025 blog) vs "a 1K image every 2.6 seconds" (Jan 2026 update) [note: 23/min = 2.6 s/image, consistent]; SDXL 1.0 BF16 7 images/min; MacBook Pro M4 Max + Spark 4K FLUX.1-dev + Wan 2.2 pipeline ~8 min to ~1 min (NVFP4/NVFP8 + RTX Video Super Resolution); Stable Diffusion 3.5 Large ~1.4x since launch — [NVIDIA blog (vendor claim)](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks); [StorageReview relaying NVIDIA CES 2026 claims](https://www.storagereview.com/news/nvidia-dgx-spark-achieves-2-5x-performance-and-8x-video-speed-in-ces-2026-enterprise-update)
- Community: FLUX.1-dev (95 GB unquantized) and Qwen-Image (63 GB) run on Spark without VRAM limits "though they run slowly at times"; a Wan2GP fork targets Spark for Wan 2.1/2.2, LTX-2, Hunyuan — [haruni.net](https://www.haruni.net/en/blog/dgx-spark); [Wan2GP-DGX-SPARK](https://github.com/rongxike/Wan2GP-DGX-SPARK)

**Power, thermals, noise**
- ServeTheHome (Oct 14, 2025): idle 40-45 W, CPU load 120-130 W, CPU+GPU "just under 200 W", AI inference 60-90 W; fans "never hit 40 dBA when we were not stress testing" — [ServeTheHome](https://www.servethehome.com/nvidia-dgx-spark-review-the-gb10-machine-is-so-freaking-cool/4/)
- ProX PC: Spark "under 100 W" vs 5090 system "800-900 W from the wall" during their tests — [ProX PC](https://www.proxpc.com/blogs/nvidia-dgx-spark-gb10-performance-test-vs-5090-llm-image-and-video-generation)
- Early owners (incl. John Carmack) reported the Spark capping at ~100 W and delivering ~half the quoted performance, plus crash/shutdown reports on the NVIDIA forums (Oct 2025) — [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/users-question-dgx-spark-performance); [NVIDIA forums](https://forums.developer.nvidia.com/t/only-getting-half-the-advertised-performance-and-capping-at-100w/352253)
- A later firmware update cut idle power ~32% (about 37 W to 25 W with a display connected) via ConnectX NIC hot-plug detection — [Tom's Hardware (headline; body not retrievable)](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-dgx-spark-update-cuts-idle-power-by-32-percent-or-more-hot-plug-detection-on-connectx-nic-makes-for-a-more-efficient-ai-workstation)
- Strix Halo (MS-S1) ~100 W sustained / ~10 W idle under llama.cpp — [akehir.com](https://akehir.com/blog/strix-halo-kubernetes-llm-gpt-oss)
- M3 Ultra ran DeepSeek-R1 671B "entirely in memory using less than 200 W" (reviewer report) — [TechRadar](https://www.techradar.com/pro/apple-mac-studio-m3-ultra-workstation-can-run-deepseek-r1-671b-ai-model-entirely-in-memory-using-less-than-200w-reviewer-finds)

**Ecosystem / software**
- 2x Spark Qwen3-235B report: vLLM+Ray and TRT-LLM NVFP4 both failed in Dec 2025; only llama.cpp RPC worked — [NVIDIA forums](https://forums.developer.nvidia.com/t/dgx-spark-multi-node-llm-inference-report-for-qwen3-235b-model/355126)
- Spark supports TRT-LLM, vLLM, SGLang, llama.cpp, PyTorch, JAX, ComfyUI, NeMo, Unsloth per NVIDIA — [StorageReview](https://www.storagereview.com/news/nvidia-dgx-spark-achieves-2-5x-performance-and-8x-video-speed-in-ces-2026-enterprise-update)
- mlx-lm on M3 Ultra showed first-query prefill stalls (7x slower) in Sep 2025 — [mlx-lm issue #432](https://github.com/ml-explore/mlx-lm/issues/432)

### Inferences
- Batched throughput is Spark's clearest quantitative win over Macs/Strix Halo: gpt-oss-120b goes from ~34 to ~863 tok/s aggregate (26x) as concurrency rises to 256, something MLX/llama.cpp on Mac cannot approach; but per-stream speed collapses to 3.6 tok/s, so this benefits pipelines, not chat.
- Fine-tuning verdict is capacity-driven: a 5090 is ~25-35% faster on a 20B QLoRA but cannot run a 120B one; RTX PRO 6000 would beat both but at ~3.4x Spark's price.
- Diffusion/video is a compute-bound workload where Spark's ~1/4-1/5 of a 5090 shows plainly (FLUX ~3-4.7x slower, Hunyuan ~2.75x slower); NVFP4 checkpoints narrow but do not close this gap.
- Power/noise: Spark at 60-90 W during inference and sub-40 dBA is competitive with Strix Halo (~100 W) and Mac Studio (<200 W for 671B), and an order of magnitude below a 5090 workstation.

### Gaps
- No apples-to-apples fine-tuning tok/s between Spark, 5090, RTX PRO 6000 and Mac (MLX LoRA) from a single tester; NVIDIA's tok/s claims have no third-party reproduction found.
- No dBA measurement for Spark under sustained GPU load (STH gives only "<40 dBA when not stress testing"); no Tom's Hardware power/noise table retrievable.
- No measured Wan 2.2 numbers on Spark vs 5090 from an independent tester (only NVIDIA's pipeline claim and a Hunyuan datapoint).

## Key Question 3: What does a 2x Spark cluster deliver in practice for DeepSeek / Qwen3-235B / Kimi K2 at 4-bit versus a single 512 GB Mac Studio?

### Takeaway
A 2x Spark cluster (256 GB, ~$9.4k) gets Qwen3-235B Q4 to ~12-25 tok/s (llama.cpp RPC / vLLM AWQ, Dec 2025) and 2026 Flash-class MoEs (DeepSeek-V4-Flash, GLM-5.3-Flash, Qwen3.8-Flash) to 40-72 tok/s single-stream with 100-130 tok/s aggregate, but true 671B DeepSeek-V3/R1 at 4-bit (~380-420 GB) and Kimi K2 do not fit in 256 GB; a 512 GB M3 Ultra runs DeepSeek-R1 671B at 4-11 tok/s decode with only 77-190 tok/s prefill, and the 512 GB SKU was withdrawn in March 2026 with the M5 Ultra replacement priced TBD for late October.

### Cited Findings
- Qwen3-235B Q4_K_XL on 2x Spark via llama.cpp RPC: 12.5 tok/s decode, 37.7 tok/s prefill; native vLLM/TRT-LLM paths failed at the time (Dec 2025) — [NVIDIA forums](https://forums.developer.nvidia.com/t/dgx-spark-multi-node-llm-inference-report-for-qwen3-235b-model/355126)
- Qwen3-VL-235B AWQ on 2x Spark vLLM ~25 tok/s (user report) — [NVIDIA forums (unverified)](https://forums.developer.nvidia.com/t/question-on-inference-performance-results-of-qwen3-235b-a22b-on-2x-dgx-spark/355053)
- NVIDIA: Qwen3-235B NVFP4 TRT-LLM dual Spark 11.73 tok/s (Oct 2025 claim), then "2.6x" with NVFP4 + Eagle3 (Jan 2026 claim) — [NVIDIA (vendor)](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks); [NVIDIA (vendor)](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark)
- DeepSeek-V4-Flash on 2x Spark: ~53 tok/s single-stream (short prompt), 115 tok/s aggregate at 6 streams, 89 s TTFT at 131K (Aug 2026) — [aussielunix gist](https://gist.github.com/aussielunix/cc6630820513c57f63fd1f4563b2aa14)
- 2x Spark, Sep 5, 2026: DeepSeek-V4-Flash 72.4 / Qwen3.8-Flash-Next 42.6 / GLM-5.3-Flash 41.1 tok/s single-user short-prompt; 22.5 / 21.0 / 17.7 at 33K agentic context; concurrent 124.5 / 131.1 / (n/a) — [WeZZard/dgx-spark-bench](https://github.com/WeZZard/dgx-spark-bench)
- GLM-5.2 753B on 2x Spark at 2-bit experts: 25.8 tok/s at 96K context; 3x Spark serves REAP-pruned GLM-5.2 469B NVFP4 with vLLM pipeline parallel at 256K context — [howtospark.com](https://howtospark.com/); [bird/GLM-spark](https://github.com/bird/GLM-spark)
- NVIDIA/LMSYS: two Sparks over dual QSFP (200 Gb/s aggregate) handle "models with up to 405 billion parameters in FP4" — [LMSYS blog](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- M3 Ultra 512 GB, DeepSeek-R1-0528: MLX 4-bit 189.5 tok/s prefill / 11.15 decode (422 GB peak); llama.cpp q4_K_M 76.8 / 4.26 (May 2025) — [someoddcodeguy](https://dev.to/someoddcodeguy/running-deepseek-r1-0528-q4km-and-mlx-4-bit-on-a-mac-studio-m3-le4)
- M3 Ultra 512 GB: Qwen3-235B-A22B 16 tok/s llama.cpp / 24 tok/s MLX; DeepSeek R1/V3 671B 4-bit 6-18 tok/s (owner reports via aggregator) — [vettedconsumer (aggregator)](https://vettedconsumer.com/which-mac-for-local-llms-2026-buyers-guide/)
- Apple removed the 512 GB Mac Studio option (previously a $4,000 upgrade) on Mar 5, 2026 amid the DRAM shortage and raised the 256 GB upgrade to $2,000 — [MacRumors](https://www.macrumors.com/2026/03/05/mac-studio-no-512gb-ram-upgrade/); [Tom's Hardware](https://www.tomshardware.com/tech-industry/apple-pulls-512-mac-studio-upgrade-option)
- M5 Ultra Mac Studio (pre-orders Aug 25, 2026): base $5,499; 36-core CPU/80-core GPU +$1,300; 96 GB to 256 GB +$4,000; 512 GB "coming late October", price unannounced; max current config $18,299 — [AppleInsider](https://appleinsider.com/articles/26/08/25/you-can-spend-18299-on-a-mac-studio-today-or-more-in-october)

### Inferences
- At ~$9.4k for two Sparks (256 GB) versus roughly $11-12k for an M5 Ultra 256 GB (base $5,499 + $4,000 memory + likely GPU upgrade), neither can hold full 671B DeepSeek V3/R1 or Kimi K2 (1T) at 4-bit; the 2x Spark cluster's practical ceiling is ~400B FP4 or 2-bit tricks (GLM-5.2 753B at 25.8 tok/s with quality loss).
- For models that fit both (Qwen3-235B Q4), a 512 GB M3 Ultra's 24 tok/s MLX beat the Dec-2025 2x Spark llama.cpp path (12.5 tok/s), while the 2026 vLLM/NVFP4/speculative path on 2x Spark now reaches 40-72 tok/s on Flash-class MoEs, which the Mac has no published equivalent for.
- The Mac's weakness is prefill (77-190 tok/s on 671B; ~1,000 on 120B MLX), so agentic/long-context work on a 512 GB Mac means minutes of TTFT; 2x Spark also degrades badly at 131K (89 s TTFT) but is far better at 4-32K.
- Cluster overhead is real: 41% scaling efficiency at 6 streams and run-to-run spread up to 2x on DeepSeek code workloads indicate immature multi-node tuning even in Sep 2026.

### Gaps
- No Kimi K2 numbers on either platform were found.
- No measured M3 Ultra 512 GB numbers for 2026 Flash-class models (DeepSeek-V4-Flash, GLM-5.3-Flash) to pair with the 2x Spark results.
- No pricing for the M5 Ultra 512 GB SKU (announced for late October 2026).

## Key Question 4: What are the price/performance numbers (USD per decode tok/s, USD per GB of memory) across these options at Sept 2026 prices?

### Takeaway
At Sept 2026 prices (Spark $4,699; RTX PRO 6000 $16,000 MSRP; Strix Halo $2,000-2,350; M3 Ultra 256 GB ~$6,000; M5 Ultra 256 GB ~$9,500-10,800), Strix Halo is the cheapest per decode tok/s and per GB (~$46/tok/s, ~$17/GB on gpt-oss-120b), Spark is ~$78/tok/s and ~$37/GB, the Macs are ~$100+/tok/s but ~$23-40/GB with far more capacity, and the RTX PRO 6000 is ~$83/tok/s (fastest absolute) but ~$167/GB.

### Cited Findings
- DGX Spark MSRP raised from $3,999 to $4,699 (+$700) citing memory supply — [TechPowerUp](https://www.techpowerup.com/346833/nvidia-raises-dgx-spark-pricing-to-usd-4-700); [OC3D](https://overclock3d.net/news/systems/nvidia-raises-dgx-spark-price-by-700-due-to-memory-supply-constraints/)
- RTX PRO 6000 Blackwell MSRP now $16,000 (was $13,250 earlier in 2026; launched at ~$8,435-8,565 in Apr 2025) — [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-doubles-rtx-pro-6000-blackwells-msrp-to-a-staggering-usd16-000-96gb-card-started-pre-orders-below-usd8-000-last-year); [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-raises-rtx-pro-6000-blackwell-gpu-pricing-to-usd13-250-55-percent-increase-over-msrp-in-a-years-time)
- Mac Studio M3 Ultra 256 GB upgrade now $2,000 (+$400); 512 GB option withdrawn Mar 2026 — [Tom's Hardware](https://www.tomshardware.com/tech-industry/apple-pulls-512-mac-studio-upgrade-option)
- M5 Ultra Mac Studio base $5,499; 256 GB +$4,000; 512 GB TBD — [AppleInsider](https://appleinsider.com/articles/26/08/25/you-can-spend-18299-on-a-mac-studio-today-or-more-in-october)
- M5 Max 128 GB MacBook Pro from ~$3,899 before memory upgrade (aggregator) — [localaimaster (aggregator)](https://localaimaster.com/blog/apple-m5-for-ai-guide)
- Strix Halo: GMKtec EVO-X2 top model $2,199 (Nov 2025); "$1,999.99 list" price floor and "~$2,348" cited by aggregators — [Notebookcheck](https://www.notebookcheck.net/Nvidia-DGX-Spark-vs-AMD-Strix-Halo-4-000-AI-supercomputer-challenged-by-2-199-mini-PC-with-better-real-time-performance.1159829.0.html); [aimultiple (aggregator)](https://aimultiple.com/dgx-spark-alternatives)
- RTX 5090 "~$2,500 + host" (aggregator) — [presenc.ai (aggregator)](https://presenc.ai/research/dgx-spark-vs-m5-max-vs-rtx-5090-throughput-2026)
- Used A100 80 GB: $4,000-9,000 (Alibaba buying guide) vs $12,000-18,000 (Hashrate Index) — conflicting; used H100 $18,000-22,000 (2026) — [Alibaba guide](https://electronics.alibaba.com/buyingguides/nvidia-a100-80gb-price-guide-2026); [Hashrate Index](https://hashrateindex.com/blog/used-gpu-market-pricing-deprecation-secondary-ai/)
- Medusa Halo (Ryzen AI Max 500, Strix Halo successor): leaks point to 384-bit LPDDR6 at ~512-691 GB/s (roughly 2-2.7x Strix Halo); no shipping product or LLM benchmarks as of Sep 2026 — [Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/amds-future-medusa-halo-apus-could-use-lpddr6-ram-new-leak-suggests-ryzen-ai-max-500-series-could-have-80-percent-more-memory-bandwidth); [Hardware Corner](https://www.hardware-corner.net/amd-medusa-halo-local-llm-20250823/)

### Inferences (computed from the cited prices and gpt-oss-120b decode numbers above; per-tok/s uses best measured single-stream llama.cpp/MLX decode)
- DGX Spark: $4,699 / 60.6 tok/s = ~$78 per tok/s; $4,699 / 128 GB = ~$37/GB. 2x Spark: $9,398 / 256 GB = ~$37/GB; ~$130-235 per tok/s on Flash-class MoEs (40-72 tok/s).
- Strix Halo (GMKtec EVO-X2 $2,199): / 47.5 tok/s (ROCm) = ~$46 per tok/s; ~$17/GB. With Vulkan's 34 tok/s the figure is ~$65 per tok/s.
- Mac Studio M3 Ultra 256 GB (~$3,999 base + $2,000 = ~$6,000 assumed; the base price was not re-verified in this pass): / 60 tok/s (MLX 8-bit) = ~$100 per tok/s; ~$23/GB. M5 Ultra 256 GB (~$9,500-10,800 depending on GPU tier): ~$37-42/GB; per-tok/s unknown until benchmarks land.
- M5 Max 128 GB MacBook Pro (~$4,700-5,400 once memory is configured; the exact configured price was not verified): ~$37-42/GB; no measured gpt-oss-120b decode found.
- RTX PRO 6000 ($16,000 card alone): / ~193 tok/s = ~$83 per tok/s; ~$167/GB; add $1.5-3k for a host.
- RTX 5090 (~$2,500 card + host): gpt-oss-120b only via CPU offload at ~30 tok/s → >$83 per tok/s and cannot hold the model in VRAM; per-GB ~$78/GB of VRAM.
- Used A100 80 GB at $4,000-9,000 would be $50-113/GB with 2 TB/s HBM but no FP4/FP8 (Ampere) and a server chassis requirement; no gpt-oss-120b measurement was found to compute per-tok/s.

### Gaps
- No single source gave a September-2026 street price for a base M3 Ultra Mac Studio, a configured M5 Max 128 GB MacBook Pro, or a Framework Desktop 128 GB; the per-GB figures for Apple above are estimates from the cited upgrade prices.
- Used A100/H100 price sources conflict by ~2x and none give gpt-oss-120b tok/s.
- No Medusa Halo pricing or benchmarks exist yet.

## Key Question 5: Has Spark's performance improved materially through software updates since launch (before/after numbers)?

### Takeaway
Yes, but unevenly: NVIDIA claims 2.5x overall since launch (Jan 2026, vendor) and the most visible measured gains are in multi-node serving (Qwen3-235B-class from ~12 tok/s in Dec 2025 to 40-72 tok/s on 2026 Flash MoEs via NVFP4 + speculative decoding) and speculative-decoding uplifts of 1.4-2.9x; single-stream gpt-oss-120b llama.cpp decode has stayed in the 55-61 tok/s band since Oct-Nov 2025 because it is bandwidth-bound.

### Cited Findings
- Launch week (Oct 2025) measurements ranged widely by framework: 14.5 tok/s (ServeTheHome), 41.1 tok/s (Ollama v0.12.6), 55.4 tok/s (NVIDIA llama.cpp), 60.6 tok/s (ggerganov llama.cpp) on gpt-oss-120b — [ServeTheHome](https://www.servethehome.com/nvidia-dgx-spark-review-the-gb10-machine-is-so-freaking-cool/4/); [Ollama](https://ollama.com/blog/nvidia-spark-performance); [NVIDIA](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks); [llama.cpp #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- llama.cpp thread updates: CUDA spin-scheduling change (Oct 15, 2025) measurably improved decode on Blackwell; Linux 6.17.1 with NO_PAGE_MAPCOUNT cut model load from 1 min 44 s to 22 s (Nov 2, 2025) — [llama.cpp #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- NVIDIA CES 2026 (Jan 5, 2026, vendor claims): "2.5x" overall since launch via TRT-LLM, quantization and decoding optimizations; Qwen-235B >2x / 2.6x with NVFP4 + Eagle3; Qwen3-30B ~1.4x; SD3.5 Large ~1.4x; llama.cpp MoE "average 35% uplift"; idle-power firmware fix — [StorageReview](https://www.storagereview.com/news/nvidia-dgx-spark-achieves-2-5x-performance-and-8x-video-speed-in-ces-2026-enterprise-update); [NVIDIA blog](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark)
- Aggregator summary of the same period: gpt-oss-120b "~36 tok/s (FP8) or 49.7 tok/s (post-CES 2026 optimizations)" (unverified) — [tokenstead (aggregator)](https://tokenstead.ai/guides/dgx-spark-benchmarks-2026)
- Multi-node before/after: Qwen3-235B Q4 12.5 tok/s via llama.cpp RPC with vLLM/TRT-LLM failing (Dec 17, 2025) versus DeepSeek-V4-Flash 52.9-72.4 tok/s and Qwen3.8-Flash 42.6 tok/s single-user on 2x Spark with vLLM/NCCL-RoCE (Aug-Sep 2026) — [NVIDIA forums](https://forums.developer.nvidia.com/t/dgx-spark-multi-node-llm-inference-report-for-qwen3-235b-model/355126); [aussielunix gist](https://gist.github.com/aussielunix/cc6630820513c57f63fd1f4563b2aa14); [WeZZard](https://github.com/WeZZard/dgx-spark-bench)
- Speculative decoding on Spark: GLM DFlash2 1.57-2.87x; Qwen NEXTN 1.42-1.94x; Qwen 3.6 35B-A3B +57% with k=3; costs KV capacity (GLM 816k → 163k tokens) — [WeZZard](https://github.com/WeZZard/dgx-spark-bench); [howtospark.com](https://howtospark.com/)
- Single-Spark GLM-5.3-Flash moved from "mid-20s to low-30s" tok/s to 64 tok/s structured with EXL3 2-bit + DFlash2 (Sep 2026), with stability and quality caveats — [NVIDIA forums](https://forums.developer.nvidia.com/t/60-tok-s-glm-5-3-flash-on-a-single-dgx-spark/382140)
- Early complaints of ~100 W cap and "half the quoted performance" (Carmack, Oct 2025) and later idle-power firmware fix (~37 W → 25 W) — [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/users-question-dgx-spark-performance); [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-dgx-spark-update-cuts-idle-power-by-32-percent-or-more-hot-plug-detection-on-connectx-nic-makes-for-a-more-efficient-ai-workstation)
- Benchmark caveats seen repeatedly: run-to-run spread up to 2x on DeepSeek code (acceptance rate 28-83%), prefix cache toggles, `--ignore-eos` fixed 512-token outputs, spec-decode gains that vanish on open-ended prose (GLM 64 → 25 tok/s), and 2-bit quality losses — [WeZZard](https://github.com/WeZZard/dgx-spark-bench); [NVIDIA forums](https://forums.developer.nvidia.com/t/60-tok-s-glm-5-3-flash-on-a-single-dgx-spark/382140)

### Inferences
- The "2.5x" claim is best read as applying to TRT-LLM/NVFP4/speculative paths on specific models; nothing in the measured record suggests bandwidth-bound single-stream llama.cpp decode on gpt-oss-120b has improved more than ~10-15% since Nov 2025.
- The largest real-world change is that multi-node vLLM/TRT-LLM on Spark went from non-functional (Dec 2025) to routinely benchmarked (Aug-Sep 2026), which is what makes a 2x Spark purchase defensible in late 2026 where it was not at launch.
- Many of the headline 2026 numbers depend on speculative decoding on structured/code outputs; buyers should expect ~35-50% lower throughput on prose and ~50-60% lower at 32K+ contexts.

### Gaps
- No controlled before/after re-run of ggerganov's exact llama.cpp benchmark table on a 2026 build was found (the #16578 thread's last extracted datapoints are Nov 2025).
- NVIDIA's "35% llama.cpp MoE uplift" and "2.5x overall" have no third-party reproduction located in this pass.
- No Phoronix, Ars Technica, or The Register follow-up benchmark for Spark in 2026 was located.
