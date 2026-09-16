# Retail Shelf / Product-Facing Analytics — Vendor & Pricing Landscape

**Date:** September 2026
**Scope:** Enterprise & SaaS solutions for shelf layout, product facings, share-of-shelf,
planogram compliance and on-shelf availability (OSA). Ordered cheapest → most expensive.

> **Pricing confidence.** This category is ~90% "contact sales". Figures below are tagged:
> **[P] = publicly published** by the vendor or a pricing directory;
> **[R] = reported** in press/analyst coverage (may be dated);
> **[E] = estimate** based on typical deal shapes in the category — directional, not quotable.
> Treat [E] as a budgeting starting point, not a quote.

---

## 1. First, split the category — it is four different purchases

People say "shelf analytics" and mean one of four things with wildly different price points:

| Layer | Question it answers | Buyer | Price order of magnitude |
|---|---|---|---|
| **A. Space planning / planogram authoring** | *What layout should the shelf have?* | Category management | $0 – $10k/yr (SMB) → $50k–$500k/yr (enterprise suites) |
| **B. Shelf execution measurement (photo/IR)** | *Does the real shelf match the plan? What's my share of facings?* | CPG field sales / trade marketing | $30–$150/user/mo → $25k–$500k/yr |
| **C. Continuous autonomous sensing** | *What is the shelf doing right now, all day, every day?* | Retailer store ops | $500–$4,000 **per store per month** |
| **D. Third-party / syndicated shelf data** | *What's happening in stores I don't visit?* | Brand insights teams | $20–$50 per store visit → 6-figure syndicated contracts |

Most "cheap vs expensive" confusion comes from comparing a $199/mo planogram tool (Layer A)
to a $3,000/store/month robot (Layer C). They are not substitutes.

---

## 2. The price ladder (cheapest → most expensive)

### Tier 0 — DIY / near-free
- **Spreadsheets + manual store walks.** Still the default at most regional chains.
- **Extractify** — free tier for lightweight testing; turns shelf photos into detections /
  share-of-shelf exports. Good for a proof of value before spending. [P: free tier]
- **Open-source CV** (YOLO/DETR + SKU embedding). Real cost is the SKU catalogue and
  annotation pipeline, not the model. Budget 3–6 engineer-months before first useful output. [E]

### Tier 1 — SMB planogram / space-planning SaaS — **$0 – $7,500/yr**
| Vendor | Published price | Notes |
|---|---|---|
| **Shelf Logic** | ProView from **$14.99/mo** [P] | Cheapest credible entry; desktop planogram lineage |
| **DotActiv** | Free (40 SKUs) / **$800** Lite / **$2,000** Pro / **$4,500** Enterprise / **$7,500** Enterprise AI — **per licence per year** [P] | Rare vendor that publishes a full ladder |
| **Quant Retail** | from ~**$770–$1,320/yr** [P] | Planograms + store-level task execution |
| **PlanoHero** | **$199/mo** Light (5k SKUs, 100 planograms, unlimited users) / **$299/mo** Pro / Enterprise custom [P] | Flat-rate, not per-seat — unusually buyer-friendly |
| **LEAFIO Shelf Efficiency** | from ~**$1,000** company-wide subscription [P] | Adds AI shelf-space optimisation tied to demand |

**What you get:** planogram authoring, shelf/fixture modelling, facings allocation,
days-of-supply-driven space, store clustering (higher tiers), PDF/mobile planogram
distribution, and *self-reported* compliance (staff tick a box or upload a photo).
**What you don't get:** automated verification. Nobody is counting your facings for you.

### Tier 2 — Retail execution apps with IR bundled — **~$30–$150 / user / month**
| Vendor | Published price | Notes |
|---|---|---|
| **GoSpotCheck by FORM** | entry ~**$35–$40/user/mo** [P] | Task/audit workflow; Trax IR available as the premium engine |
| **Repsly (+ ShelfScan)** | from **$29/mo** entry, commonly ~**$89/user/mo** [P] | IR bundled rather than bought separately; aimed at smaller CPG |
| **Salesforce Consumer Goods Cloud** | **$100/user/mo** (Retail Execution – Merchandiser, Enterprise, billed annually) [P] | Plus implementation; IR via partner |
| **Trax Connector for Salesforce** | **$50/user/year** [P] | Connector only — requires Trax CPG licences underneath |

**Rule of thumb:** a 50-rep field team lands at **$30k–$90k/yr** for the app layer alone,
before image-recognition volume fees and before implementation. [E]

### Tier 3 — Standalone shelf image-recognition platforms (mid-market) — **~$25k – $150k/yr** [E]
Vendors: **ParallelDots ShelfWatch**, **Infilect InfiViz**, **Vispera**, **Neurolabs ZIA**,
**Snap2Insight**, **Vision Group Store360**, **Storesight (Shelfgram + Field Agent)**.

- All custom-quoted. Typical commercial shapes seen in this tier: **per image/scan**
  (~$0.02–$0.15 at volume, higher at low volume), **per store-visit**, or a **platform fee +
  volume tier**. [E]
- Deployment timelines are the honest differentiator, and these *are* published:
  Store360 <4 weeks, ShelfWatch 4–8 weeks, InfiViz <1 week (vendor claim),
  Repsly ShelfScan 8–12 weeks, FORM+Trax 3–6 months. [P]
- Neurolabs' pitch is synthetic training data — cost doesn't scale with SKU count the way
  annotation-based IR does; no public numbers, but it's the structural argument to press
  vendors on if you have a long tail of SKUs. [P: claim only]

**What you get:** phone photo → stitched shelf panorama → SKU-level detection → facings count,
share of shelf, OSA/void detection, planogram compliance %, price-tag compliance,
promo/display execution, per-visit scorecards, rep feedback in-app (seconds, not overnight).

### Tier 4 — Enterprise IR + syndicated shelf data — **~$250k – $2M+/yr** [E]
Vendors: **Trax Retail**, **NIQ (incl. Spaceman for planning)**, **Circana**, **Acosta/Pensa**,
**Pensa Systems**, **Blue Yonder Space Planning**, **Relex**, **Symphony/SymphonyAI**.

- Pricing is global-contract shaped: country coverage × store universe × SKU library ×
  refresh frequency. Enterprise space-planning suites (Blue Yonder, NIQ Spaceman) publish
  nothing at all and scale hard on seat count — a 6-person space team and a 60-supplier
  network are different purchases. [P: "no public number"]
- This is where "Tier-1 incumbent name for procurement" is the actual product being bought.
  Expect 3–6 month implementations.

### Tier 5 — Continuous sensing: fixed shelf cameras / ESL-integrated vision — **~$500 – $2,500 / store / month** [E]
Vendors: **Focal Systems**, **VusionGroup Captana (ShelfEye)**, **Scandit ShelfView**,
**Standard AI**, **Trigo** (checkout-first, shelf as by-product).

Hardware density is the cost driver, and it *is* documented:
- Focal: ~**400 cameras** to cover a 30,000 sq ft store, each camera ~8 ft of linear shelf;
  vendor owns maintenance. [P]
- Monoprix/Captana: **120 ShelfEye cameras per store**, ~12,000 across the first 100 stores. [P]
- Reported pilot economics for shelf monitoring: **$50k–$150k per store for a pilot**,
  12–16 weeks to first store, payback typically 12–18 months, 2–4% sales lift from
  reduced out-of-stocks. [R — one source, treat the pilot figure as an upper bound that
  likely bundles programme/integration cost, not steady-state per-store run rate]
- Usually sold as-a-service (capex folded into the monthly) or capex + SaaS. Ask which.

### Tier 6 — Autonomous robots — **~$2,000 – $4,000 / store / month** [R]
- **Simbe Robotics (Tally 3.0)**: Robot-as-a-Service, **zero upfront**, monthly fee varying by
  store size, SKU count, scan frequency, and whether you take CV only or CV+RFID. The
  **$2,000–$4,000/store/month** figure is from 2020 press — directionally still the band
  people quote, but verify. [R] Vendor markets ~$200k+/store/yr of recovered value.
- Note the graveyard: Bossa Nova was pulled by Walmart. Robots win on aisle coverage without
  shelf-edge hardware install; they lose on scan frequency (a few passes/day vs. continuous)
  and on store-associate goodwill. Scandit's answer is hybrid — robots + fixed cams + phones.

### Cross-cutting — buying the data instead of the system
- **Crowdsourced audits** (Field Agent/Storesight, Gigwalk, Observa, Premise):
  **$20–$50 per store visit** to the buyer; gig worker payout $3–$20/task. [P]
- Cheapest way to get real shelf photos at national scale without deploying anything.
  Weakness: no guaranteed cadence, uneven photo quality, no store-system integration.
- Sensible pattern: crowdsourced photos **into** a Tier-3 IR engine. Often beats a Tier-4
  contract for a challenger brand. [E]

---

## 3. Worked budget scenarios [E]

**A. Regional chain, 40 stores, wants better layouts + basic compliance**
PlanoHero Pro or DotActiv Pro: **$2.4k–$4k/yr** + staff photo self-audits. Total < $10k/yr.

**B. Mid-size CPG, 25 reps, 1,500 stores covered by visit**
Retail execution app ($89/user/mo ≈ **$27k/yr**) + IR volume (25 reps × 8 stores/day × 10
images ≈ 500k images/yr @ $0.05 ≈ **$25k/yr**) + onboarding **$10k–$25k** one-off.
**Year 1 ≈ $60k–$80k.**

**C. Global CPG, enterprise programme**
Trax/NIQ-class contract, multi-country: **$400k–$1.5M/yr**, 3–6 month implementation,
plus internal data-ops headcount. The SKU library and master-data cleanup is the real project.

**D. Grocer, 200 stores, continuous OSA**
Fixed cameras at ~$1,200/store/mo ≈ **$2.9M/yr**; robots at ~$3,000/store/mo ≈ **$7.2M/yr**.
Justified only against a quantified OSA loss — usually needs 1–2% sales lift to clear.

---

## 4. Hidden costs to put in the model

1. **SKU library / model onboarding** — the classic overrun. Annotation-based IR charges
   (or delays) per new SKU; pack-shot collection is your job. Ask for the per-new-SKU SLA
   *in days and dollars*.
2. **Per-image overage** — negotiate the tier ceiling, not the headline rate.
3. **Master data** — planogram files, fixture dimensions, product dimensions. If this is
   dirty, no vendor saves you.
4. **Store networking & power** for fixed cameras; retrofit labour per fixture.
5. **Integration** to ERP/replenishment/task management — otherwise you buy dashboards, not
   actions. Value comes from the work order, not the chart.
6. **Change management** — a compliance number nobody is accountable for produces zero lift.

## 5. Negotiation levers that actually work here

- Pay per **outcome unit** (store-visit, audited store-week) rather than per image.
- Cap SKU-onboarding fees for the contract term; make new-SKU turnaround an SLA with credits.
- Insist on a **paid pilot with an exit**, 10–20 stores, with an accuracy acceptance test on
  *your* shelves (detection recall on your categories, not the vendor's demo deck).
- Ask for accuracy measured as **facings-level recall/precision**, not "99.7% accuracy" —
  that number is meaningless without the denominator.
- For Tier 5/6, demand the capex/opex split in writing and what happens to hardware at term end.

## 6. Market context

- Shelf image-recognition AI: **$1.82B (2025) → $2.3B (2026)**, ~26% CAGR, ~$5.9B by 2030.
- On-shelf availability market: **~$6.85B (2026) → $13.6B (2035)**, ~8.7% CAGR.
- Image recognition in CPG: **~$4.84B (2026) → $23B (2035)**, ~18.7% CAGR.
- Smart shelves: ~$7.1B by 2026 (older MarketsandMarkets forecast).
(Analyst figures differ on scope definitions; use for direction only.)

## 7. If the question is build vs. buy

Building is defensible when: you own the stores (so you control camera placement and master
data), your SKU set is narrow/stable, and you want the data inside your own replenishment
loop. It is not defensible for a brand needing coverage across retailers it doesn't control —
there you're buying distribution of field labour, not a model.

The genuinely hard parts, in order: (1) SKU-level fine-grained recognition across pack
variants, (2) keeping the catalogue current as packaging churns, (3) panorama stitching and
shelf-geometry reconstruction to get *facings* rather than *detections*, (4) turning a
compliance delta into a task someone actually completes.

---

## Sources

- Simbe Robotics FAQ / RaaS pricing — https://www.simberobotics.com/faqs , https://thespoon.tech/simbe-robotics-announces-new-tally-3-0-shelf-scanning-robot/
- DotActiv pricing — https://dotactiv.com/pricing
- PlanoHero pricing — https://planohero.com/en/pricing/
- Shelf Logic — https://shelflogic.com/
- Quant Retail — https://www.quantretail.com/en/planogram-software
- LEAFIO — https://www.leafio.ai/planogram-software/
- Repsly pricing — https://www.capterra.com/p/212159/Repsly/ , https://www.softwaresuggest.com/repsly
- GoSpotCheck by FORM — https://www.itqlick.com/compare/repsly/gospotcheck-software
- Salesforce Consumer Goods Cloud pricing — https://www.salesforce.com/consumer-goods/retail-execution-software/pricing/
- Trax Connector (AppExchange) — https://appexchange.salesforce.com/appxListingDetail?listingId=a0N4V00000IEUbpUAH
- Focal Systems shelf cameras — https://focal.systems/solutions
- VusionGroup Captana / Monoprix — https://www.vusion.com/insights/monoprix-deepens-the-digitization-of-its-stores-with-the-captana-sensor-cloud-solution-from-ses-imagotag/
- Scandit ShelfView — https://www.scandit.com/products/shelfview/
- Pensa Systems — https://pensasystems.com/
- Platform comparison & implementation timelines — https://visiongroupretail.com/blog/ai-image-recognition-platforms-retail
- Planogram software buyer's guide — https://www.cpgscout.ai/planogram-software
- Pilot cost / payback figures — https://brainxtech.com/blog/computer-vision-in-retail-applications-2026/
- Crowdsourced audit economics — https://www.fieldagentcanada.com/blog/why-crowdsourced-audits-beat-traditional-agencies
- Market sizing — https://www.researchandmarkets.com/reports/6215383/shelf-image-recognition-artificial-intelligence , https://www.businessresearchinsights.com/market-reports/on-shelf-availability-market-121311
