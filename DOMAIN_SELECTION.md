# Knowledge Area #001 — Domain Selection

Produced in response to `PROJECT_INITIATION.md` §5, which requires the first
knowledge area to be chosen from a shortlist "based on commercial and technical
potential, not simply how interesting the subject is".

## 1. Scoring model

Each candidate is scored 1–5 against seven weighted criteria. The weights encode
what actually determines whether a knowledge area can become a *product* rather
than a demo.

| # | Criterion | Weight | What a 5 looks like |
|---|-----------|--------|---------------------|
| C1 | **Source accessibility & licensing** | 0.20 | Authoritative primary sources are public, machine-fetchable and legally redistributable as citations |
| C2 | **Answer verifiability** | 0.15 | Questions have checkable answers anchored to a specific clause, figure or incident |
| C3 | **Willingness to pay** | 0.20 | Identifiable buyers with budget who currently pay humans for this |
| C4 | **Cost of being wrong** | 0.10 | Errors cause death, write-offs or regulatory action — so evidence is valued |
| C5 | **Update velocity** | 0.10 | The corpus changes often enough to justify a subscription, not a one-off purchase |
| C6 | **Moat vs. a general chatbot** | 0.15 | Knowledge is fragmented across documents a frontier model answers vaguely |
| C7 | **Weekend feasibility** | 0.10 | No licensing negotiation, no OCR-heavy scans, no proprietary data feed |

C1 is weighted jointly highest on purpose. Success criterion #1 is *"a domain can
be ingested largely automatically"*. A domain whose valuable material sits behind
a licence fails the very first criterion regardless of how lucrative it is.

## 2. Shortlist scores

| Candidate | C1 | C2 | C3 | C4 | C5 | C6 | C7 | **Weighted** |
|-----------|----|----|----|----|----|----|----|--------------|
| **Battery Failure Intelligence** | 5 | 5 | 4 | 5 | 5 | 4 | 5 | **4.65** |
| Industrial Fire & Explosion Intelligence | 4 | 4 | 4 | 5 | 3 | 4 | 4 | 4.00 |
| UK Structured Finance / CLO Intelligence | 2 | 4 | 5 | 5 | 4 | 4 | 3 | 3.80 |
| Corrosion Intelligence | 3 | 4 | 4 | 4 | 2 | 4 | 4 | 3.60 |
| Telecom Infrastructure Intelligence | 4 | 4 | 3 | 3 | 4 | 3 | 4 | 3.55 |
| Food Shelf-Life Intelligence | 4 | 4 | 3 | 4 | 3 | 3 | 4 | 3.55 |
| Engineering Failure Intelligence | 4 | 3 | 3 | 4 | 3 | 3 | 2 | 3.20 |
| Mine-to-Market Intelligence | 3 | 3 | 4 | 3 | 4 | 3 | 2 | 3.20 |
| Extreme-Environment Materials Intelligence | 4 | 4 | 2 | 3 | 2 | 4 | 3 | 3.20 |
| African Construction Materials Intelligence | 2 | 3 | 3 | 4 | 2 | 5 | 2 | 3.00 |
| Water-Scarcity Engineering Intelligence | 4 | 3 | 2 | 3 | 3 | 3 | 3 | 3.00 |

## 3. Selected: Lithium-Ion Battery Failure Intelligence

**`kb-001-battery-failure`** — scoped deliberately narrowly to *lithium-ion cell
and system failure*: failure mechanisms, thermal runaway initiation and
propagation, vent-gas composition, early detection, suppression and cooling,
BESS and EV incident precedent, and the transport/storage/installation rules that
govern them.

It is **not** "batteries in general". It excludes cell chemistry R&D, battery
manufacturing economics, and state-of-charge estimation, because a knowledge area
with fuzzy edges cannot be evaluated honestly or told when it is out of scope.

### Why it wins

**Sources are public and primary.** Accident investigation bodies (NTSB, AAIB),
regulators (EU Battery Regulation 2023/1542 on EUR-Lex, UK OPSS on gov.uk, FAA,
UNECE R100, UN 38.3 via the UN Model Regulations), national labs (Sandia, NREL,
PNNL, DOE), standards bodies' public-facing material (UL 9540A method summaries,
NFPA 855), and a genuinely large open-access literature (arXiv, MDPI, Journal of
Power Sources preprints). None of it requires a licence negotiation to *cite*.

**Answers are checkable.** "What onset temperature does the literature report for
separator melt in a polyethylene separator?" and "Does NFPA 855 permit indoor
installation of a 600 kWh ESS in a residential occupancy?" have answers that
resolve to a clause, a figure or a measured range. That is what makes an
evaluation suite meaningful rather than vibes-based.

**There are buyers.** BESS insurers and loss adjusters, fire engineers writing
basis-of-design documents, grid-scale storage developers and O&M teams, EV fleet
and bus operators, waste and recycling operators, marine/aviation cargo handlers,
and fire services. Several of these already pay consultants day rates for answers
that are, fundamentally, document retrieval plus judgement.

**The consequences are severe.** Battery fires kill people, destroy eight-figure
assets and trigger regulatory action. This is precisely the market that will pay
for *"here is the clause, here is the incident report"* over a confident paragraph
from a general assistant.

**It changes constantly.** New incidents, new revisions of NFPA 855 and IEC 62619,
the phased EU Battery Regulation timetable. A knowledge system with versioning and
freshness metrics has recurring value here.

**A general chatbot is genuinely weak here.** The knowledge lives in PDFs of
incident reports and standards clauses. Ask a frontier model for a specific
UL 9540A test threshold and it will often produce a plausible number with no
source. That gap is the product.

### Honest risks

* **Standards text is copyrighted.** NFPA and UL full texts are not
  redistributable. The system ingests public summaries, scope statements,
  regulatory adoptions and secondary literature, and cites the standard as a
  pointer rather than reproducing it. `known_limitations` in the manifest states
  this explicitly, and the public evidence page repeats it.
* **Safety-critical advice liability.** Every answer must carry the standing
  limitation that it is an evidence-retrieval tool, not a basis-of-design
  authority. This is enforced in the system prompt and asserted by a test.
* **Numeric ranges vary by cell format and chemistry.** Answers that state a
  single number without qualifying chemistry/format are a failure mode the
  red-team suite explicitly targets.

## 4. Deferred, with reasons

**UK Structured Finance / CLO Intelligence** has the highest revenue-per-seat on
the shortlist and should be built — but not first. Its commercially decisive
material (rating agency criteria in full, LMA documentation, offering circulars,
Intex/Moody's Analytics loan-level data) is licensed. Building KA-001 on a corpus
we cannot legally ingest would fail success criterion #1 and would make the whole
factory look like it works when it does not. It is scheduled as **KA-003**, after
the pipeline has proven itself on a domain where nothing is blocked, and after the
licensing position on public regulatory material (FCA Handbook, EU Securitisation
Regulation, BoE market notices) has been confirmed.

**Industrial Fire & Explosion Intelligence** scored second and shares much of
KA-001's source base (CSB, HSE). It is the designated **KA-002**, because
criterion #10 — *"a second knowledge area can subsequently be created using
substantially the same pipeline"* — is tested most cheaply by a domain that is
adjacent but genuinely distinct in ontology, sources and evaluation set.

## 5. Decision

| | |
|---|---|
| **KA-001** | `kb-001-battery-failure` — Lithium-Ion Battery Failure Intelligence |
| **KA-002** | Industrial Fire & Explosion Intelligence (proves repeatability) |
| **KA-003** | UK Structured Finance / CLO Intelligence (pending licensing review) |
