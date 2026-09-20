# NVIDIA DGX Spark (GB10) — Hardware Market Position, Pricing, Roadmap and Alternatives as of 20 September 2026

Scope note: all claims are dated. "Confirmed" = vendor statement or primary documentation; "reported" = press/review measurement; "rumored" = leak/analyst. Prices are US unless stated. Several retailer pages (Micro Center, Tom's Hardware paywalled articles, CNBC) blocked fetches; where a number comes only from a search snippet rather than a fetched page it is flagged.

## Baseline: DGX Spark specs and known bottlenecks (context for every question below)

### Takeaway
DGX Spark is a 128 GB LPDDR5X / ~273 GB/s unified-memory box with a 6,144-CUDA-core (48 SM) Blackwell GPU that is architecturally consumer-Blackwell (sm_121), not datacenter Blackwell; the memory bus is the hard ceiling for single-stream decode of dense models, while MoE models and batched serving fare much better.

### Cited Findings
- GB10 has 48 SMs, 6,144 CUDA cores, 1 PFLOP sparse FP4, 5th-gen tensor cores with FP4/FP6; LPDDR5X shared CPU/GPU at 273 GB/s; compute capability 12.1 (sm_121). It lacks datacenter-Blackwell features (no TMEM, no tcgen05, no WGMMA, no multi-SM cooperative execution, 128 KB vs 228 KB shared memory per SM); its tensor path uses extended `mma.sync`, so FlashMLA/FlashInfer/FlashAttention need separate SM12x implementations (19 Feb 2026) — [Backend.AI](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell)
- NVIDIA's compute-capability split: sm_100 for B200/GB200, sm_120 for RTX Blackwell cards, sm_121 for GB10; DGX Spark is ARM64-only, runs DGX OS 7.5.0 (Ubuntu 24.04-based) with CUDA 13.0 — [Kubesimplify](https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists); [vLLM issue #36821](https://github.com/vllm-project/vllm/issues/36821)
- Bandwidth math: dense 70B FP8 ceiling is 273 ÷ 70 ≈ 3.9 tok/s; measured batch-1 decode: Llama 3.3 70B FP8 ~2.7–3.0 tok/s, Llama 70B Q4 ~6.8 tok/s, Qwen 32B FP8 ~8.5 tok/s, gpt-oss-120b MXFP4 "tens of tok/s" (only ~3–4 GB active); prefill ~4x faster than Mac Studio M3 Ultra (24 Jun 2026) — [Enverge](https://spark.enverge.ai/blog/dgx-spark-prefill-vs-decode)
- Usable memory on a single node is 121.69 GiB (4 Sep 2026) — [Unsloth PR #10280](https://github.com/unslothai/unsloth/pull/10280)
- Reserved display memory is BIOS-configurable to 2 GB or 4 GB as of the July 2026 DGX OS release — [NVIDIA DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
- Two-node clustering is over ConnectX-7 200GbE links — [Unsloth PR #10280](https://github.com/unslothai/unsloth/pull/10280); ConnectX-7 hot-plug power management (18 W savings) added in the January 2026 DGX OS release — [NVIDIA release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)

### Inferences
- Because sm_121 is not sm_100, any kernel written for "Blackwell" datacenter parts does not automatically run well on Spark; the software-maturity story (below) follows directly from this.

### Gaps
- No independent measurement of sustained FLOPS vs the 1 PFLOP sparse-FP4 headline was fetched in this pass (Carmack's "about half the quoted performance" claim, Oct 2025, is cited under Q6).

## Q1. Price and availability in September 2026; discounted or refreshed since the October 2025 launch?

### Takeaway
Not discounted — the opposite: NVIDIA raised the Founders Edition MSRP from $3,999 to $4,699 in late February 2026 citing memory supply constraints, and there has been no hardware refresh of GB10. OEM GB10 boxes (notably ASUS Ascent GX10) undercut the FE, with the ASUS US list price at $3,999 in August 2026 and lower-storage SKUs reported near $3,000. Stock is described as constrained but obtainable through NVIDIA Marketplace, Micro Center, PNY and Amazon.

### Cited Findings
- NVIDIA announced 25 Feb 2026 that DGX Spark (Founders Edition) MSRP moved from $3,999 to $4,699 (+$700, +17.5%), "reflects industry wide memory supply constraints", no hardware change, applies globally, existing orders keep old pricing; OEM GB10 pricing "should be directed to the OEM" — [NVIDIA Developer Forums price change announcement](https://forums.developer.nvidia.com/t/2-23-2026-price-change-announcement/361713); also [TechPowerUp](https://www.techpowerup.com/346833/nvidia-raises-dgx-spark-pricing-to-usd-4-700)
- NVIDIA Marketplace listed $4,699 as of July 2026; available via NVIDIA Marketplace, Micro Center, PNY, Amazon; "stock is constrained" (17 Jul 2026) — [Enverge where-to-buy](https://spark.enverge.ai/blog/where-to-buy-nvidia-dgx-spark)
- Launch-period availability (Oct 2025): sold out at NVIDIA's store; Micro Center had stock at 29 of 31 stores, one-per-household limit, no reservations — [Computerworld](https://www.computerworld.com/article/4072897/nvidias-dgx-spark-desktop-supercomputer-is-on-sale-now-but-hard-to-find)
- ASUS lists a US starting price of $3,999 for the Ascent GX10 (128 GB) as of 18 Aug 2026 (search snippet; ASUS eShop page not fetched) — [ASUS eShop](https://eshop.asus.com/us/ascent-gx10.html)
- Central Computers lists DGX Spark-class GB10 variants from $2,999.99 (1 TB) to $3,999.99 (4 TB); Walmart listed ASUS GX10 at $3,999.99 while other listings were $5,999–$6,999 (search snippets, dates not confirmed) — [Central Computers](https://www.centralcomputer.com/blog/post/dgx-spark-which-should-i-choose); [BetterClaw](https://www.betterclaw.io/blog/dgx-spark-alternative)
- OEM GB10 variants on the market: Acer Veriton GN100, ASUS Ascent GX10, Dell Pro Max with GB10, Gigabyte AI TOP ATOM, HP ZGX Nano AI Station, Lenovo ThinkStation PGX SFF, MSI EdgeXpert-11SUS/01SKUS; each typically 1/2/4 TB storage; all 128 GB — no 64 GB or reduced-memory DGX Spark/GB10 variant exists (15 Aug 2026) — [pi3g](https://pi3g.com/how-to-get-nvidia-dgx-spark-gb10-discounts/)
- Discount channels: NVIDIA Inception (startups, up to 64 FE + 64 OEM units lifetime), EDU discount for qualified institutions, reseller volume discounts (>10 units) — [pi3g](https://pi3g.com/how-to-get-nvidia-dgx-spark-gb10-discounts/)
- Rental alternative: hosted DGX Spark from $0.75/h single, $1.65/h for a 2-node pair — [Enverge pricing](https://spark.enverge.ai/blog/nvidia-dgx-spark-price)
- A 64 GB configuration DOES exist in the Windows sibling product line: ASUS ProArt GR1X (RTX Spark, N1X) is offered "up to 128 GB unified memory; 64 GB variant also available", 10 GbE, Windows, no price or date yet — [ASUS ProArt GR1X](https://www.asus.com/displays-desktops/mini-pcs/proart-mini-pc-series/proart-gr1x-mini-pc/) (see Q2 for details)

### Inferences
- Real September 2026 street price for 128 GB GB10 hardware: ~$3,000–$3,999 for OEM boxes (storage-dependent) vs $4,699 FE; the FE premium buys the 4 TB SSD and NVIDIA branding, not different silicon.
- The February price hike and the RAM-driven price increases across Apple, AMD Strix Halo and discrete GPUs (see Q5) mean a price cut before end-2026 is unlikely absent a memory-market reversal; no source predicts one.

### Gaps
- Could not fetch live Micro Center listings (HTTP 403) to confirm today's in-store price/stock for FE or ASUS GX10.
- Individual current prices for Dell/HP/Lenovo/MSI/Acer/Gigabyte GB10 SKUs were not found from primary pages.
- No refurbished/used-market pricing found.

## Q2. Credible successor for 2027 that would make a late-2026 purchase regret-inducing?

### Takeaway
Confirmed: NVIDIA's Computex 2026 roadmap (2 Jun 2026) shows a Vera Rubin Spark with LPDDR6 in the 2027–2028 window and Rosa Feynman Spark in 2029–2030; no firm date, price, memory capacity or bandwidth has been given. The nearer-term change is the Windows "RTX Spark" (N1X) launching October 2026: the same GB10-class silicon (6,144 CUDA cores, up to 128 GB LPDDR5X, ~1 PFLOP FP4) in cheaper Windows laptops/mini PCs without ConnectX-7. It does not raise bandwidth, so it does not obsolete DGX Spark for LLM decode, but it may undercut it on price.

### Cited Findings
- Roadmap slide, Computex/GTC Taipei keynote (2 Jun 2026): Grace Blackwell Spark — 2026 — LPDDR5X; Vera Rubin Spark — 2027–2028 — LPDDR6; Rosa Feynman Spark — 2029–2030 — memory not specified; applies to "laptops & compact desktops" — [Fudzilla](https://fudzilla.com/nvidia-shows-full-long-term-rtx-spark-roadmap-at-computex-gtc-keynote-including-future-rosa-feynman-planned-for-2029-2030/); corroborated by [VideoCardz](https://videocardz.com/newz/nvidia-confirms-rtx-spark-roadmap-with-rubin-in-2027-and-rosa-feynman-in-2029) and [Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/nvidia-unveils-dgx-sparrk-roadmap-for-laptops-and-desktop-pcs-at-computex-2026-three-generations-outlined-rubin-followed-by-rosa-feynman) (both paywalled/truncated on fetch; headlines only)
- Rosa Feynman Spark is described as using "HBM-Next" in one summary (search snippet, unverified) — [Let's Data Science](https://letsdatascience.com/news/nvidia-expands-rtx-spark-roadmap-through-2030-d7a616d8)
- NVIDIA developer-forum thread "DGX Spark 2 release date?" (8 Aug 2026): no NVIDIA staff statement; users cite the roadmap and characterize the successor as "2+ years from now, late 2028 or later"; one reading places Vera Rubin Spark in 2028 and Rosa Feynman in 2030 — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/dgx-spark-2-release-date-will-there-be-one/379601). Note: this conflicts with headlines saying "Rubin in 2027"; the slide itself gives a 2027–2028 range, so both readings are consistent with it.
- RTX Spark (N1X) unveiled 1 Jun 2026: 20-core MediaTek-co-designed Arm CPU, 6,144 CUDA cores, up to 128 GB LPDDR5X, "1 petaflops FP4", NVLink-C2C ~600 GB/s CPU–GPU link (not DRAM bandwidth); "essentially a modified version of the DGX Spark launched last year, but with Windows support"; devices "expected to hit shelves in Fall 2026" from ASUS, Dell, HP, Lenovo, Microsoft Surface, MSI — [Notebookcheck](https://www.notebookcheck.net/Nvidia-N1X-officially-confirmed-to-arrive-as-the-RTX-Spark.1312010.0.html)
- Second N1X variant: 18-core CPU / 5,120 CUDA cores; laptop TDP 45–80 W (search snippet) — [VideoCardz](https://videocardz.com/newz/nvidia-confirms-rtx-spark-n1x-6144-and-5120-cuda-core-specs-launching-next-month)
- RTX Spark DRAM bandwidth is "unconfirmed by NVIDIA"; RTX Spark and DGX Spark are "two different chips" despite proximity, and for dense-70B batch-1 both land at roughly 4 tok/s (3 Jul 2026, updated 5 Sep 2026) — [LeCompute](https://lecompute.fr/en/silicon/rtx-spark-vs-dgx-spark/)
- RTX Spark SFF mini PCs shown at Computex 2026 (ASUS ProArt, Dell XPS RTX Spark Desktop, Lenovo SFF, MSI EdgeMesa N AI+): "the biggest difference ... is the lack of a ConnectX-7 NIC and QSFP ports"; Windows; configs may go "as little as 16 GB" up to 128 GB; no pricing announced (5 Jun 2026) — [ServeTheHome](https://www.servethehome.com/scoping-out-rtx-spark-sff-mini-pcs-at-computex-2026/)
- Additional RTX Spark desktops: Acer SFF RTX Spark (IFA 2026, "design" not retail product), Microsoft Surface RTX Spark Dev Box (US, "later this year"), ASUS ProArt GR1X 150×150×51 mm with up to 128 GB — [VideoCardz Acer](https://videocardz.com/newz/acer-unveils-compact-rtx-spark-pc-with-128gb-unified-memory); [VideoCardz Surface](https://videocardz.com/newz/microsoft-shows-mini-surface-pc-with-nvidia-rtx-spark-and-128gb-memory); [VideoCardz ASUS GR1X](https://videocardz.com/newz/asus-details-proart-gr1x-mini-pc-with-nvidia-rtx-spark-128gb-unified-memory-and-10gbe)
- Timing: launch October 2026, "wider availability won't happen until early 2027" (leak, 25 Apr 2026) — [Notebookcheck](https://www.notebookcheck.net/Nvidia-N1X-leak-points-to-limited-2026-availability.1282855.0.html)
- Pricing (rumored, analyst): N1X systems "won't be priced below $2,899" per Morgan Stanley estimates; 128 GB pricing not announced — [VideoCardz](https://videocardz.com/newz/nvidia-rtx-spark-laptops-may-start-above-1799-n1x-systems-reportedly-above-2899); [Wccftech](https://wccftech.com/laptops-and-pcs-powered-by-nvidia-rtx-spark-n1x-variant-cant-be-priced-below-2900/)
- DGX Station (GB300, 72-core Grace + Blackwell Ultra, up to 20 PFLOPS FP4, trillion-parameter models) is the step-up tier; "DGX Station for Windows" partners ASUS, Dell, Gigabyte, HP, MSI, Supermicro "later in 2026" — [Fudzilla](https://fudzilla.com/nvidia-shows-full-long-term-rtx-spark-roadmap-at-computex-gtc-keynote-including-future-rosa-feynman-planned-for-2029-2030/); listed at ~$94,930 starting, partners only, no Founders Edition — [Tom's Hardware](https://www.tomshardware.com/desktops/nvidias-gb300-powered-dgx-station-desktop-tower-listed-for-nearly-usd100-000-online-enterprise-ai-powerhouse-now-available-to-buy-for-mere-mortals-with-lots-of-cash); [ServeTheHome](https://www.servethehome.com/nvidia-dgx-station-systems-available-at-last-gb300-gb200-workstations-for-your-desktop/)
- HN commenter speculation (June 2026): Vera Rubin DGX Spark "late 2027/early 2028"; counter-view that current hardware stays useful for years — [Hacker News](https://news.ycombinator.com/item?id=48638270) (opinion, not sourced)

### Inferences
- The only confirmed successor is Vera Rubin Spark, with LPDDR6 as the only disclosed spec. LPDDR6 at 14.4 Gbps on a 256-bit-class bus is in the ~460–690 GB/s range in AMD Medusa Halo leaks (Q3), so a 2–2.5x bandwidth uplift is the plausible expectation, but NVIDIA has published no figure. Earliest plausible availability is 2027; "2028" is at least as likely given the slide's range and the forum consensus.
- The October 2026 RTX Spark launch is the regret risk most relevant to a late-2026 buyer: same-class silicon in Windows boxes that may cost ~$1,000–1,800 less for 128 GB, but with no ConnectX-7 (no low-latency 2-node clustering) and Windows-on-Arm rather than DGX OS/Linux tooling. Whether Linux will be officially supported on RTX Spark boxes is not documented.
- A user who wants Linux + CUDA + 2-node clustering today has no confirmed cheaper path before 2027; a user who can tolerate Windows and single-node should wait for October RTX Spark pricing.

### Gaps
- No NVIDIA statement on Vera Rubin Spark memory capacity, bandwidth, price or exact quarter.
- No confirmation that RTX Spark mini PCs will ship a 128 GB Linux-supported configuration or the price of the 128 GB tier.
- Whether the "RTX Spark" die is identical to GB10 (same package/bins) vs a distinct N1X die is not authoritatively documented; sources say "modified"/"rebadged".

## Q3. How does 273 GB/s constrain decode vs Mac Studio (M3 Ultra 819 GB/s, M5 generation), Strix Halo, and discrete GPUs? Concrete numbers.

### Takeaway
Batch-1 decode is memory-bandwidth-bound: on Spark a dense 70B model runs ~3 tok/s at FP8 and ~6–7 tok/s at Q4, while MoE models with small active parameter counts run 50–120 tok/s. Apple's M5 Ultra (shipping 22 Sep 2026) raises Mac bandwidth to 1.2 TB/s — ~4.4x Spark — and the new M5 Max is 460–614 GB/s; Strix Halo is ~215–256 GB/s (slightly below Spark); discrete GPUs have far more bandwidth but only 32–96 GB. Spark's compensating advantages are prefill (~4x M3 Ultra), CUDA, and batched throughput.

### Cited Findings
DGX Spark single node:
- Dense: Llama 3.3 70B FP8 ~2.7–3.0 tok/s; Llama 70B Q4 ~6.8 tok/s; Qwen 32B FP8 ~8.5 tok/s (24 Jun 2026) — [Enverge](https://spark.enverge.ai/blog/dgx-spark-prefill-vs-decode)
- gpt-oss-120b MXFP4: 52.37 tok/s output on SGLang (SM121 analysis) — [spark-vllm-mxfp4-docker](https://github.com/christopherowen/spark-vllm-mxfp4-docker/blob/main/docs/analysis/SGLANG_ANALYSIS.md); 68.9 tok/s single-stream with Eagle3 speculative decoding (64.3 plain) and 300 tok/s aggregate at 30 users on patched vLLM + SM121 CUTLASS MXFP4 kernels, up from 52.6/244 on the prior SGLang stack (2026) — [gptoss-spark](https://github.com/luka-loehr/gptoss-spark)
- gpt-oss-120b on two Sparks, optimized SGLang: 75 tok/s single-request; 660.93 tok/s aggregate at 2.57 req/s (Dec 2025) — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/setting-up-vllm-sglang-or-tensorrt-on-two-dgx-sparks/353338)
- Qwen 3.6 35B-A3B NVFP4 (10 GB): 117–121 tok/s single-stream at 16K context and 114–121.5 tok/s at 1M context on the Rust "Atlas" engine (18 May 2026, updated 19 Sep 2026) — [Flowtivity](https://flowtivity.ai/blog/120-tok-s-1m-context-private-ai-dgx-spark/)
- NVIDIA CES 2026 software update: Qwen-235B >2.5x faster than launch via TRT-LLM + NVFP4; Qwen3 30B and SD3.5 >30% faster (Jan 2026) — [StorageReview](https://www.storagereview.com/news/nvidia-dgx-spark-achieves-2-5x-performance-and-8x-video-speed-in-ces-2026-enterprise-update); [The Register](https://www.theregister.com/2026/01/05/nvidia_dgx_spark_speed/)
- Prefill/TTFT: Spark ~4x faster prefill than M3 Ultra — [Enverge](https://spark.enverge.ai/blog/dgx-spark-prefill-vs-decode); EXO measured Llama-3.1-8B FP16 with 8,192-token prompt: Spark prefill 1.47 s vs M3 Ultra 5.57 s, but Spark decode of 32 tokens 2.87 s vs M3 Ultra 0.85 s (~3.4x faster decode on Mac); Spark ~100 TFLOPS FP16 vs M3 Ultra ~26 TFLOPS — [EXO Labs](https://blog.exolabs.net/nvidia-dgx-spark/)
- LMSYS launch review: Spark "not the fastest or most cost-effective choice for brute force inference"; shines for prototyping and batched serving of smaller models (13 Oct 2025) — [LMSYS](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)

Apple:
- M5 Ultra announced 25 Aug 2026: up to 512 GB unified memory, 1.2 TB/s bandwidth ("50% higher than M3 Ultra"), up to 80 GPU cores with a Neural Accelerator per core, 32-core Neural Engine; ships in new Mac Studio — [Apple Newsroom](https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/)
- Mac Studio page: M5 Max "up to 128 GB unified memory", "up to 614 GB/s"; M5 Ultra "up to 512 GB" (512 GB "coming late October"), 1.2 TB/s; available 22 Sep 2026 — [Apple Mac Studio](https://www.apple.com/mac-studio/). Macworld reports M5 Max at "460 GB/s" — [Macworld](https://www.macworld.com/article/3220024/apple-announces-the-m5-ultra-mac-studio-with-up-to-512gb-of-ram.html). Discrepancy: likely two M5 Max bins; not resolved.
- M3 Ultra Mac Studio: 819 GB/s, 512 GB option discontinued and 256 GB upgrade raised from $1,600 to $2,000 (6 Mar 2026) — [Notebookcheck](https://www.notebookcheck.net/Apple-increases-RAM-upgrade-pricing-for-M3-Ultra-Mac-Studio-retires-512-GB-option.1244246.0.html); 512 GB M3 Ultra had been $9,499 — [Hacker News](https://news.ycombinator.com/item?id=47296302)

AMD Strix Halo (Ryzen AI Max+ 395, 128 GB):
- Dense 70B Q4 ~5 tok/s; gpt-oss-120b ~31 tok/s @ ~120 W (ServeTheHome on Beelink GTR9 Pro, cross-checked by StorageReview on HP Z2 Mini G1a); Qwen3-30B-A3B ~70–100 tok/s; 7–13B dense ~30–45 tok/s; real bandwidth "~215 GB/s" (5 Jul 2026, updated 13 Sep 2026) — [DataHardware](https://datahardware.ai/blog/strix-halo-tokens-per-second-2026)
- Other reports: gpt-oss-120b ~55 tok/s on Vulkan with 96 GB allocation; 51.1 tok/s TG / 174 tok/s PP (search snippets, config-dependent) — [strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide); [carteakey](https://carteakey.dev/blog/optimizing%20gpt-oss-120b-local%20inference/)
- Medusa Halo (Ryzen AI MAX 500, rumored): LPDDR6, ~460 GB/s at 256-bit (+80%) up to ~512–691 GB/s at 14.4 Gbps, 24 Zen 6 cores; expected "next year" (2027) per leaker "Gray"; AMD has not confirmed (11 Feb 2026) — [HotHardware](https://hothardware.com/news/amd-medusa-halo-lpddr6-leak); [Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/amds-future-medusa-halo-apus-could-use-lpddr6-ram-new-leak-suggests-ryzen-ai-max-500-series-could-have-80-percent-more-memory-bandwidth) (paywalled on fetch); another summary places it "2027 to 2028" — [TechPowerUp](https://www.techpowerup.com/346164/amd-medusa-halo-apu-to-use-lpddr6-memory)

Discrete GPUs:
- RTX 5090 32 GB street: best US price $5,199 (Tom's tracker, Sep 2026); average $4,626 with range $4,199.99 (30 Jul) to $7,369.90 (12 Sep 2026); launch MSRP $1,999 — [videocardprices](https://videocardprices.com/card/nvidia-rtx-5090/); [ai.rs](https://ai.rs/ai-for-business/rtx-5090-price-climb-september-2026)
- RTX PRO 6000 Blackwell 96 GB: NVIDIA MSRP now $16,000 (from $8,565 at Mar 2025 launch; interim $13,250), lowest US retailer $15,599, Amazon $19,999 (15 Sep 2026); GDDR7 supply cited — [Thunder Compute](https://www.thundercompute.com/blog/nvidia-rtx-pro-6000-pricing); [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-doubles-rtx-pro-6000-blackwells-msrp-to-a-staggering-usd16-000-96gb-card-started-pre-orders-below-usd8-000-last-year); [TechPowerUp](https://www.techpowerup.com/351549/nvidia-rtx-pro-6000-blackwell-96-gb-gpu-now-costs-usd-16-000)
- Reported "RTX 60 delayed to 2028" (search snippet, secondary site, unverified) — [Tech Insider](https://tech-insider.org/rtx-5090-price-4329-rtx-60-delay-2028-2026/)

### Inferences
- Bandwidth ratio → expected batch-1 dense decode ratio: M5 Ultra ≈ 4.4x Spark, M3 Ultra ≈ 3x, M5 Max ≈ 1.7–2.2x, Strix Halo ≈ 0.8–0.9x. For gpt-oss-120b-class MoE, Spark (~52–69 tok/s) vs Strix Halo (~31–55 tok/s) is consistent with that ratio plus Spark's better kernels.
- Spark's prefill lead (~4x M3 Ultra, and 6x-class FLOPS advantage) matters for long-context/agentic use where TTFT dominates; Apple's M5 Neural Accelerators (per-core) may narrow this, but no M5 Ultra LLM prefill benchmark was found.
- The RTX PRO 6000 at $16,000 is now ~3.4x a DGX Spark FE for 96 GB; multi-GPU builds for ≥128 GB (2x PRO 6000 ≈ $32k, or 4x 5090 ≈ $18–20k for 128 GB plus a workstation host) are an order of magnitude above Spark and only make sense for throughput, not capacity.

### Gaps
- No published tok/s benchmarks for M5 Ultra or M5 Max Mac Studio (ships 22 Sep 2026) — only bandwidth spec; treat any decode number as extrapolation.
- No fetched head-to-head table (same model/quant/framework) across Spark, M3 Ultra, Strix Halo; numbers above come from different labs and stacks.
- RTX PRO 6000 / RTX 5090 memory-bandwidth figures were not verified in this pass (widely quoted as ~1.8 TB/s GDDR7 for both, but no primary source fetched).
- Tom's Hardware's DGX Spark vs Ryzen AI Max+ 395 review could not be fetched (truncated/paywalled).

## Q4. What does 2x Spark clustering actually deliver?

### Takeaway
2-node clustering is real and supported by vLLM, SGLang, TensorRT-LLM, llama.cpp RPC, EXO, NVIDIA's Sync Cluster Assistant and (since Sep 2026) Unsloth for training. Tensor parallelism roughly doubles decode for models that must be split (2.09x measured), pipeline parallelism adds capacity but no decode speed (1.08x), and the main value is running ~200–240 GB models such as DeepSeek V4 Flash at ~45–53 tok/s single-stream and ~110–126 tok/s aggregate with 1M context.

### Cited Findings
- Unsloth PR (4 Sep 2026): two Sparks over ConnectX-7 200GbE; training via `torch.distributed.pipelining` with DualPipeV at 1.96x (4,204 vs 2,149 tok/s); inference TP decode 2.09x (median TPOT 332.7 ms → 162.4 ms); PP 1.08x ("cannot accelerate decode (bytes-per-token unchanged)"); Llama-3.3-70B training fits at 68.6 GiB per node; "models fitting on one Spark often run slower when split" — [Unsloth PR #10280](https://github.com/unslothai/unsloth/pull/10280)
- DeepSeek V4 Flash on 2x Spark (vLLM / "DSpark" stack, direct ConnectX-7): 256-token prompt C1 decode 52.87 tok/s, C6 aggregate 115.40 tok/s; 131K-token C1 decode 47.18 tok/s, TTFT 89.36 s; code-gen C1 47.04 tok/s, C6 110.50 tok/s (updated 12 Aug 2026) — [aussielunix gist](https://gist.github.com/aussielunix/cc6630820513c57f63fd1f4563b2aa14); ~46 tok/s single-stream and ~107 tok/s at 6 concurrent, 41 tok/s at 1M context — [Flowtivity](https://flowtivity.ai/blog/deepseek-v4-flash-1m-context-dual-dgx-spark/)
- GLM 5.3 Flash on a documented 2-Spark recipe: 29–70 tok/s with 1M context (search snippet) — [Flowtivity](https://flowtivity.ai/blog/deepseek-v4-flash-1m-context-dual-dgx-spark/)
- gpt-oss-120b on 2 Sparks (Dec 2025): vLLM 598 tok/s aggregate / 100 ms TTFT; SGLang 661 tok/s / 92 ms TTFT / 75 tok/s single-request; TensorRT-LLM 128.6 tok/s (underperforming); SGLang needed `--disable-cuda-graph` until a patch; vLLM load ~12 min vs SGLang 3–4 min — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/setting-up-vllm-sglang-or-tensorrt-on-two-dgx-sparks/353338)
- StorageReview cluster review (11 May 2026) on Dell/Gigabyte/HP GB10 pairs (PP=2, batch 64): gpt-oss-120b 464–505 tok/s aggregate; Llama-3.1-8B-FP4 1,413–1,459 tok/s; no single-node baseline; frames the cluster as "a development and learning platform" — [StorageReview](https://www.storagereview.com/review/nvidia-dgx-spark-cluster-review-distributed-inference-on-dell-gigabyte-and-hp)
- NVIDIA added "Sync Cluster Assistant" and NCCL 2.30u1 with three-system ring topology in the June 2026 DGX OS release — [NVIDIA release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
- EXO 1.0 can pool a Spark (prefill) with an M3 Ultra (decode) over 10 GbE: combined 2.32 s vs 4.34 s Spark-alone on an 8K-prompt Llama-8B test — [EXO Labs](https://blog.exolabs.net/nvidia-dgx-spark/)
- NanoChat depth-20 pretraining on 2 Sparks: ~1,890 tok/s, ~653 M tokens over four days (search snippet) — [Pith/arXiv 2608.07226](https://pith.science/paper/2608.07226)
- Hosted 2x NVLinked/ConnectX pair rentable at $1.65/h — [Enverge](https://spark.enverge.ai/blog/nvidia-dgx-spark-price)

### Inferences
- A 2x Spark pair (~$9,400 FE, ~$6,000–8,000 OEM) yields ~256 GB (≈243 GiB usable) at the same 273 GB/s per node; it enables ~230B-class MoE models at ~45–75 tok/s but does not lift dense-70B decode above ~6–13 tok/s.
- The 2-node path competes directly with a 256 GB M5 Ultra Mac Studio ($9,499) which has 4.4x the bandwidth in one box but no CUDA; for MoE decode the Mac should win single-stream, while the Spark pair keeps prefill and concurrency advantages (no M5 Ultra data yet).

### Gaps
- No apples-to-apples single-node vs dual-node number for the same model on the same stack other than the Unsloth TP/PP figures.
- Real-world reliability of long-running 2-node jobs (NCCL hangs, ConnectX link issues) was not surveyed.

## Q5. Cost per GB of usable model memory and per unit of decode throughput across alternatives

### Takeaway
On $/GB, Strix Halo boxes (~$16–26/GB) beat OEM GB10 (~$23–31/GB), DGX Spark FE (~$37/GB) and M5 Ultra 256 GB (~$37/GB); discrete GPUs are ~$145–170/GB. On $/tok/s for MoE decode, Strix Halo and OEM GB10 are close; the Mac's advantage is bandwidth per dollar at ≥256 GB, where nothing else competes below ~$30k. All prices are inflated by the 2026 memory shortage.

### Cited Findings
- DGX Spark FE $4,699 / 128 GB — [NVIDIA forums](https://forums.developer.nvidia.com/t/2-23-2026-price-change-announcement/361713); ASUS GX10 $3,999 (128 GB) — [ASUS eShop](https://eshop.asus.com/us/ascent-gx10.html); OEM 1 TB SKUs from $2,999.99 — [Central Computers](https://www.centralcomputer.com/blog/post/dgx-spark-which-should-i-choose)
- Strix Halo 128 GB mini PCs: GMKtec EVO X2 ~$1,499, Corsair AI Workstation 300 ~$2,299, Framework ~$2,499, Acemagic Tank M1A Pro+ $3,299, AMD "Ryzen AI Halo Developer Platform" $3,999; one source says boxes that were $1,500–1,800 six months earlier are now "$3,000+" (search snippets; individual pages not fetched) — [TerminalBytes](https://terminalbytes.com/best-mini-pc-for-local-llm-2026/); [TweakTown](https://www.tweaktown.com/news/112183/amds-ryzen-ai-halo-ai-mini-pc-launches-in-the-us-with-128gb-memory-and-a-dollars3999-price-tag/index.html); [TechPowerUp](https://www.techpowerup.com/349943/pre-orders-for-usd-4000-amd-ryzen-ai-halo-mini-pc-dev-kits-go-live); [Fenado](https://fenado.ai/articles/2900-price-point-highlights-value-of-128gb-amd-strix-halo-ai-mini-pcs)
- Mac Studio M5 Ultra: $5,499 base (96 GB); 256 GB is +$4,000 → $9,499 (30-core CPU/64-core GPU) or $10,799 with 36-core CPU/80-core GPU; 512 GB price not published, "coming late October"; max 256 GB/16 TB config $18,299 (25 Aug 2026) — [AppleInsider](https://appleinsider.com/articles/26/08/25/you-can-spend-18299-on-a-mac-studio-today-or-more-in-october); [Macworld](https://www.macworld.com/article/2973459/2026-mac-studio-m5-release-date-specs-price-rumors.html)
- Mac Studio M5 Max starts $2,499 (base memory not stated; up to 128 GB) — [Macworld](https://www.macworld.com/article/3220024/apple-announces-the-m5-ultra-mac-studio-with-up-to-512gb-of-ram.html)
- RTX PRO 6000 96 GB $16,000 MSRP / $15,599 low — [Thunder Compute](https://www.thundercompute.com/blog/nvidia-rtx-pro-6000-pricing); RTX 5090 32 GB ~$4,626 avg — [videocardprices](https://videocardprices.com/card/nvidia-rtx-5090/)
- DGX Station GB300 ~$94,930+ — [Tom's Hardware](https://www.tomshardware.com/desktops/nvidias-gb300-powered-dgx-station-desktop-tower-listed-for-nearly-usd100-000-online-enterprise-ai-powerhouse-now-available-to-buy-for-mere-mortals-with-lots-of-cash)
- Decode reference points used below: Spark gpt-oss-120b 52–69 tok/s — [gptoss-spark](https://github.com/luka-loehr/gptoss-spark); Strix Halo 31–55 tok/s — [DataHardware](https://datahardware.ai/blog/strix-halo-tokens-per-second-2026); Spark dense 70B Q4 6.8 tok/s — [Enverge](https://spark.enverge.ai/blog/dgx-spark-prefill-vs-decode); Strix dense 70B Q4 ~5 tok/s — [DataHardware](https://datahardware.ai/blog/strix-halo-tokens-per-second-2026)

### Inferences (derived table; all arithmetic mine, Sep 2026 prices)
| System | Price | Memory | $/GB | Bandwidth | $/(GB/s) | gpt-oss-120b batch-1 | $ per tok/s |
|---|---|---|---|---|---|---|---|
| DGX Spark FE (4 TB) | $4,699 | 128 GB (~122 GiB usable) | $36.7 | 273 GB/s | $17.2 | 52–69 tok/s | $68–90 |
| ASUS Ascent GX10 | $3,999 | 128 GB | $31.2 | 273 GB/s | $14.6 | 52–69 | $58–77 |
| OEM GB10 1 TB (Central Computers) | $2,999 | 128 GB | $23.4 | 273 GB/s | $11.0 | 52–69 | $43–58 |
| 2x DGX Spark FE | $9,398 | 256 GB | $36.7 | 2x273 (per node) | — | ~75 (2-node SGLang) | $125 |
| Strix Halo 128 GB mini PC | $2,000–3,300 | 128 GB (~96–110 GB GPU-allocatable) | $15.6–25.8 | ~215–256 GB/s | $8–15 | 31–55 | $36–106 |
| Mac Studio M5 Ultra 96 GB | $5,499 | 96 GB | $57.3 | 1,200 GB/s | $4.6 | no data | — |
| Mac Studio M5 Ultra 256 GB | $9,499 | 256 GB | $37.1 | 1,200 GB/s | $7.9 | no data | — |
| Mac Studio M3 Ultra 256 GB | ~$9,499 (2025 512 GB price; 256 GB now +$2,000 over base) | 256 GB | ~$25–37 | 819 GB/s | ~$8–12 | no fetched number | — |
| RTX PRO 6000 (card only) | $16,000 | 96 GB | $167 | not verified | — | no data | — |
| RTX 5090 (card only) | ~$4,600 | 32 GB | $144 | not verified | — | cannot hold 120B | — |
| DGX Station GB300 | ~$95,000 | ~784 GB (not verified) | — | — | — | — | — |

- Strix Halo has the best $/GB and roughly matches OEM GB10 on $/tok/s for MoE models; DGX Spark's premium buys CUDA, ~4x prefill, ConnectX-7 clustering, and better-optimized MoE kernels.
- For the specific goal "best possible open-weight models locally", memory capacity is the gating factor: 128 GB boxes are limited to ~120B-class at 4-bit; 256 GB (2x Spark or M5 Ultra 256 GB) reaches ~235–250B-class MoE at 4-bit; 512 GB (M5 Ultra, late Oct 2026, price unknown) is the only sub-$30k route to ~600B–1T-class models at 4-bit. At the 256 GB tier the M5 Ultra costs the same as 2x Spark FE with 4.4x the bandwidth.

### Gaps
- M5 Ultra 512 GB price unknown until late October 2026.
- No Strix Halo price from a fetched retailer page; the range spans $1,499–$3,999 across sources of differing dates.
- No decode benchmark for M5 Ultra / M5 Max; Mac $/tok/s cannot be computed yet.
- DGX Station GB300 memory total and discrete-GPU bandwidth not verified from primary sources.

## Q6. Reliability, thermal, firmware and driver issues in the first year; software support lifecycle

### Takeaway
The first year saw well-documented launch problems (100 W power cap, overheating, spontaneous reboots, a 30 W USB-PD controller defect requiring RMA), a January 2026 software update that reviewers say transformed performance, and a mid-2026 EC firmware regression that broke fan curves on some (mostly OEM) units. sm_121 software support was "severely lacking" in January 2026 with NVIDIA staff citing non-existent versions; by mid-2026 FlashInfer/vLLM/SGLang had SM121 paths but with caveats. NVIDIA has published no end-of-support date for DGX OS on GB10, and monthly-ish releases continued through July 2026 (DGX OS 7.5.0, driver 580.159.03, CUDA 13.0.2, kernel 6.17).

### Cited Findings
Launch-period hardware/power issues (Oct–Nov 2025):
- John Carmack (27 Oct 2025): Spark "maxing out at only 100 watts ... less than half of the rated 240 watts, and it only seems to be delivering about half the quoted performance"; "gets quite hot"; saw a report of spontaneous rebooting — [Carmack on X](https://x.com/ID_AA_Carmack/status/1982831774850748825); coverage: [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/users-question-dgx-spark-performance); [Slashdot](https://hardware.slashdot.org/story/25/10/29/035247/early-reports-indicate-nvidia-dgx-spark-may-be-suffering-from-thermal-issues)
- ServeTheHome measured just under 200 W combined CPU+GPU on a retail unit and never reached 240 W (search snippet) — [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/users-question-dgx-spark-performance)
- Diagnostic taxonomy: (1) 30 W PD-controller defect (hardware, RMA), (2) 100 W thermal throttling (protection), (3) 5 W driver bug (software); firmware pushed late April 2026 addressed USB PD Controller and Embedded Controller stability — [ai-muninn](https://ai-muninn.com/en/blog/dgx-spark-30w-power-safety-mode)
- "Keeps rebooting every 20–30 minutes" (9 Nov 2025): causes traced to kernel builds 1014/1016 (1013 fine), blacklisted `sbsa_gwdt` watchdog module, missing linux-generic metapackage, container memory exhaustion; NVIDIA's guidance was to reflash from recovery media and contact Enterprise Support; 6+ users with the same symptom — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/dgx-spark-keeps-rebooting-every-20-30-minutes/350692)
- Driver-load failures / DGX Dashboard updates stuck at "Downloading and installing update" — [NVIDIA forums](https://forums.developer.nvidia.com/t/dgx-spark-unable-to-load-nvidia-drivers-dgx-dashboard-updates-stuck/352122); [NVIDIA forums](https://forums.developer.nvidia.com/t/dgx-spark-nvidia-driver-issue/351828)

Mid-2026 thermal/firmware regression:
- EC firmware 0x03000508 broke the fan curve (ACPI zones 96–97 °C, fans "virtually inaudible under load"); rollback to 0x02004e18 via `fwupdmgr downgrade` restored ~32 °C idle / 35–37 °C load with zero throttling; NVIDIA support "have not been able to reproduce this on an FE DGX Spark" (suggesting OEM units), RMAs approved for failing Field Diagnostics; 2–3 users confirmed, mixed results (16 Jul 2026) — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/nvidia-dgx-spark-gb10-thermal-throttling-fan-curve-fix-via-ec-firmware-rollback/377069); related: [throttling after EC/UEFI updates](https://forums.developer.nvidia.com/t/dgx-spark-gb10-thermal-throttling-after-ec-uefi-updates-acpi-zones-96-97c-fans-not-ramping/377044); [hard-freezes under sustained inference, support portal unavailable (Aug 2026)](https://forums.developer.nvidia.com/t/dgx-spark-hard-freezes-under-sustained-few-minutes-inference-powerstress-thermal-failure-support-portal-unavailable/379195); ASUS GX10 hard power-offs reproduced — [ai-muninn](https://ai-muninn.com/en/blog/gx10-thermal-hard-poweroff)

Software support / sm_121 maturity:
- "SM121 Software Support is Severely Lacking – Official Roadmap Needed" (15–16 Jan 2026): gaps in PyTorch (no CUDA 13 ARM64 wheels on PyPI), Triton (SM121 treated as SM80), FlashInfer (compile failures), CUTLASS FP8 dispatch, MoE kernels, vLLM (NGC container stuck at 0.11, `--enforce-eager` costing 20–30%), SGLang (unofficial branch); NVIDIA staff (Johnny_nv) initially cited non-existent versions (PyTorch 2.11, FlashInfer 0.5.8, CUTLASS 4.4.x) then corrected; NVFP4 "supported" but no documented production path; a leading community contributor (eugr) later joined NVIDIA's Spark team (23 Jul 2026) — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663)
- vLLM aarch64 sm_121 support issue and FlashInfer SM121 audit remain open tracking items — [vLLM #36821](https://github.com/vllm-project/vllm/issues/36821); [FlashInfer #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170)
- CES 2026 update: NVIDIA claimed up to 2.5x over launch (Qwen-235B via TRT-LLM + NVFP4) — [The Register](https://www.theregister.com/on-prem/2026/01/05/nvidia-says-dgx-spark-is-now-25x-faster-than-at-launch/2472637); a 4-month owner review titled "I was ready to return my DGX Spark, then NVIDIA's January update changed everything" (Medium; page could not be fetched, HTTP 403) — [Data Science Collective](https://medium.com/data-science-collective/i-was-ready-to-return-my-dgx-spark-then-nvidias-january-update-changed-everything-e67699155a45)
- DGX OS release cadence: Nov 2025 (6.14 HWE kernel, unified-memory reporting fix), Jan 2026 (ConnectX-7 hot-plug, BT audio, multi-monitor), Apr 2026 (air-gapped/enterprise mgmt, BT keyboard fix), Jun 2026 (OTA no longer mandatory at setup, Sync Cluster Assistant, NCCL 2.30u1), Jul 2026 (OOM handling for unified memory, display-reserve 2/4 GB, hot-plug display fix); current DGX OS 7.5.0 / driver 580.159.03 / CUDA 13.0.2 / kernel 6.17; "No end-of-support statements are provided" — [NVIDIA DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html); known-issues page — [NVIDIA](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)
- Long-term positioning: "DGX Spark is brilliant for the right job and brutally wrong for the wrong one"; not the most cost-effective for brute-force inference (search snippet summarizing LMSYS/StorageReview) — [IntuitionLabs](https://intuitionlabs.ai/articles/nvidia-dgx-spark-review); [Creative Strategies](https://creativestrategies.com/research/nvidias-dgx-spark-review/)

### Inferences
- The reliability picture is "some batches, some OEM units", not universal; FE units appear less affected by the July 2026 EC regression per NVIDIA's own statement, and the 30 W PD defect was an RMA-class hardware fault.
- NVIDIA's continued monthly DGX OS releases through July 2026 plus the roadmap commitment to further Spark generations imply GB10 will stay supported through at least the Vera Rubin transition; there is no published lifecycle date either way, so treat multi-year CUDA support as likely but unconfirmed.
- Because sm_121 shares its instruction path with sm_120 (RTX 50-series) and RTX Spark (N1X) now expands the installed base into millions of Windows PCs from October 2026, the ecosystem incentive to keep SM12x kernels current has strengthened materially relative to the January 2026 situation.

### Gaps
- No official NVIDIA DGX OS / CUDA end-of-support date for GB10.
- The Medium "4 months later" review and the Tom's Hardware Carmack article could not be fetched; contents are known only from titles/snippets.
- No quantitative failure-rate data; only forum anecdotes (6+ reboot reports, 2–3 EC firmware reports).
- Whether the July 2026 EC firmware regression was fixed by a later EC release was not established.
