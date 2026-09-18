# Knowledge Foundry — Project Initiation Document

## 1. Mission

Build a one-person-operated AI knowledge laboratory that creates highly specialised,
evidence-backed AI systems for narrow domains.

The product is not another general-purpose chatbot.

The core proposition:

> Turn a well-defined body of specialist knowledge into a continuously tested AI
> knowledge system that people can interrogate, challenge and verify.

Each knowledge area becomes an independent Specialist Intelligence System.

Examples could eventually include:

* Battery Failure Intelligence
* Industrial Fire & Explosion Intelligence
* Corrosion Intelligence
* Telecom Infrastructure Intelligence
* African Construction Materials Intelligence
* Engineering Failure Intelligence
* Water-Scarcity Engineering Intelligence
* Mine-to-Market Intelligence
* Food Shelf-Life Intelligence
* Extreme-Environment Materials Intelligence

Do not assume these are the final domains. The system should make creating and
evaluating new domains repeatable.

## 2. The fundamental idea

Do not create 2,000 separately hosted LLMs.

Instead:

```
                    KNOWLEDGE FOUNDRY
                           |
             +-------------+-------------+
             |                           |
      Knowledge Factory            Shared AI Infrastructure
             |                           |
      +------+------+             +------+------+
      v      v      v             v      v      v
   Domain A Domain B Domain C   LLM/RAG  Eval  Agents
      |      |      |
      v      v      v
 Specialist Specialist Specialist
 Intelligence Intelligence Intelligence
```

A knowledge area should consist primarily of:

* curated source material
* structured knowledge
* metadata
* provenance
* retrieval indexes
* domain ontology
* evaluation questions
* expert challenges
* system prompts/instructions
* version history

The underlying model can be shared.

## 3. Important architectural principle

Do not fine-tune a separate LLM for every knowledge area initially.

The initial architecture should favour:

> Base model + specialist knowledge + retrieval + structured reasoning + evaluation

rather than:

> Base model + expensive domain-specific fine-tuning

Fine-tuning can be introduced later where experimentation demonstrates that it
materially improves performance.

The knowledge should remain independently inspectable and updateable.

## 4. Role of frontier models

Use frontier models as research and knowledge-engineering workers, not necessarily
as the production inference engine.

A frontier model should be capable of helping the lab:

1. discover candidate knowledge domains
2. identify authoritative sources
3. collect and classify source material
4. extract structured information
5. propose an ontology
6. identify relationships between concepts
7. generate candidate questions
8. generate evaluation datasets
9. identify contradictions
10. propose missing knowledge
11. perform adversarial testing
12. assist with updates

The production specialist system can subsequently use a cheaper/open model where
appropriate.

The objective is to make the knowledge layer the valuable asset, rather than
permanently depending on a particular frontier model.

## 5. First experiment

Do not attempt to build the entire platform first.

Build one complete knowledge area end-to-end.

The coding agent should initially select a suitable candidate domain from a
generated shortlist.

One candidate already discussed is:

* UK Structured Finance / CLO Intelligence

However, the lab should also investigate less obvious domains such as:

* Battery Failure Intelligence
* Industrial Fire & Explosion Intelligence
* Corrosion Intelligence
* Telecom Infrastructure Intelligence
* African Construction Materials Intelligence
* Engineering Failure Intelligence
* Water-Scarcity Engineering Intelligence
* Mine-to-Market Intelligence
* Food Shelf-Life Intelligence
* Extreme-Environment Materials Intelligence

The selection must be based on commercial and technical potential, not simply how
interesting the subject is.

## 6. Knowledge Area specification

Every knowledge area should have a standard manifest.

Example:

```yaml
knowledge_area:
  id: battery-failure-intelligence
  name: Battery Failure Intelligence
  description:
  domain:
  version:
  owner:
  created:
  last_updated:
sources:
  authoritative:
  academic:
  regulatory:
  industry:
  datasets:
knowledge_model:
  ontology:
  entities:
  relationships:
retrieval:
  semantic:
  keyword:
  structured:
evaluation:
  question_count:
  factual_tests:
  reasoning_tests:
  citation_tests:
  adversarial_tests:
governance:
  review_status:
  known_limitations:
  confidence:
```

The exact schema can evolve.

## 7. Source ingestion

Build a reusable ingestion pipeline.

It should eventually support:

* web pages
* PDFs
* academic papers
* books/documents where legally available
* standards
* technical manuals
* datasets
* structured databases
* expert contributions

Every piece of knowledge needs provenance.

At minimum:

```
Source
|-- URL / identifier
|-- publisher
|-- publication date
|-- retrieval date
|-- document version
|-- authority/type
|-- extracted content
`-- claims derived from source
```

The system must be able to answer:

> "Where did this answer come from?"

## 8. Knowledge representation

Do not rely exclusively on vector embeddings.

Investigate a hybrid architecture combining:

**Semantic retrieval** — vector embeddings for conceptual similarity.

**Keyword retrieval** — BM25/full-text search for exact technical terminology.

**Structured retrieval** — metadata, entities, dates, classifications and relationships.

**Knowledge graph / ontology** — where useful, represent `Entity -> Relationship -> Entity`.

For example:

```
Battery
   | susceptible_to
Thermal Runaway
   | triggered_by
Internal Short Circuit
   | detected_by
Temperature Anomaly
```

The architecture should remain modular so the knowledge representation can evolve.

## 9. Specialist AI

Each knowledge area gets its own specialist agent configuration.

It should:

1. understand the domain
2. retrieve relevant evidence
3. reason over that evidence
4. answer the user's question
5. cite supporting sources
6. distinguish evidence from inference
7. acknowledge insufficient evidence
8. avoid fabricating information

Critical rule:

> If the system cannot establish sufficient evidence, it should say so.

A confident unsupported answer is a failure.

## 10. Evidence-first responses

Responses should ideally expose:

```
ANSWER
Reasoning / explanation
SOURCES
[1] Source...
[2] Source...
[3] Source...
EVIDENCE CONFIDENCE
...
LIMITATIONS
...
```

The precise UX can evolve. The user should be able to inspect the evidence
supporting an answer.

## 11. Evaluation is a core product feature

Do not treat evaluation as an afterthought.

Every knowledge area must have an evaluation suite.

Initially create perhaps:

* 100-500 curated questions
* factual questions
* multi-document questions
* reasoning questions
* deliberately ambiguous questions
* impossible/unanswerable questions
* citation tests
* temporal/freshness tests
* adversarial questions

Eventually scale this much further.

## 12. Automated daily health check

Each knowledge area should be evaluated automatically.

```
                NIGHTLY EVALUATION
                       |
          +------------+------------+
          v            v            v
      Retrieval     Answer       Citation
       accuracy     accuracy      accuracy
          |            |            |
          +------------+------------+
                       v
                 REGRESSION TEST
                       |
              +--------+--------+
              v                 v
            PASS              FAIL
              |                 |
           Publish          Investigate /
           version          rollback
```

Track metrics such as:

* retrieval precision/recall
* answer correctness
* citation correctness
* unsupported-claim rate
* hallucination rate
* source coverage
* freshness
* contradiction rate
* regression rate

Do not invent a single meaningless "AI accuracy" number.

## 13. Red-team system

Create an automated adversarial evaluator.

A second AI should continuously attempt to break the specialist system by
generating questions such as:

* questions outside the knowledge base
* misleading questions
* questions with false premises
* contradictory source scenarios
* edge cases
* highly specific technical questions
* questions requiring multiple sources
* questions where the correct answer is "unknown"

The evaluator should check whether the system:

* hallucinated
* cited irrelevant evidence
* misunderstood the source
* made unsupported deductions
* failed to recognise uncertainty

## 14. Expert validation

The lab does not need the founder to possess a PhD in every knowledge area.

Instead:

> The system earns credibility through transparent evidence and external challenge.

For each domain, recruit a small number of knowledgeable people to challenge the
system. They should be able to:

* ask questions
* flag incorrect answers
* identify missing knowledge
* challenge sources
* submit corrections
* propose evaluation questions

Expert feedback becomes another dataset for improving the knowledge system.

## 15. Public evidence page

Every mature knowledge area should eventually have a public page showing:
About, Sources, Coverage, Evaluation, Updates, Known limitations, Challenges, Version.

For example:

```
Knowledge Area v0.7.2
Last evaluated: 18 September 2026
Sources: 1,842
Evaluation questions: 1,200
Latest regression status: PASS
```

This becomes a major part of the differentiation.

## 16. "Try to break it"

The public experience should encourage people to challenge the system.

> Don't trust our AI. Test it.

> Try to break this knowledge system.

The goal is to make expert scrutiny part of the product rather than something hidden.

## 17. Infrastructure

The initial system must be extremely cheap.

Target: prototype the entire lab for less than ~$100/month.

Do not architect for a massive data centre. Use:

* one developer machine/local Mac Studio where practical
* inexpensive VPS/container infrastructure
* object storage where required
* shared inference
* containerised knowledge areas
* automated deployment

A knowledge area should not require one dedicated VPS.

Infrastructure cost should scale primarily with usage and data, not the number of
knowledge areas.

## 18. Multi-tenancy

Design from the beginning so that a Knowledge Area can be `public`, `private` or
`enterprise`. A company could eventually deploy its own private knowledge area
using the same architecture.

## 19. API

The long-term product should expose an API.

```
POST /knowledge/{knowledge_area}/ask
```

Returning:

```json
{
  "answer": "...",
  "sources": [],
  "confidence": "...",
  "knowledge_version": "...",
  "evaluation_status": "passed"
}
```

Potential future customers: software companies, researchers, enterprises,
consultants, universities, specialist professionals, AI applications.

The API should not be the priority for the first weekend.

## 20. Business model

Potential revenue streams: consumer/professional subscription, API usage,
enterprise knowledge systems, data/knowledge products, expert/research products.

The first objective is not revenue. The first objective is proving:

> A narrow knowledge domain can be converted into a demonstrably useful,
> continuously tested AI system.

## 21. The knowledge factory

Once Knowledge Area #001 works, the real objective is to make the creation process
repeatable.

```
CREATE DOMAIN -> RESEARCH DOMAIN -> DISCOVER SOURCES -> INGEST -> CLEAN ->
STRUCTURE -> BUILD ONTOLOGY -> INDEX -> GENERATE EVALUATIONS -> BUILD SPECIALIST ->
RED TEAM -> EXPERT REVIEW -> PUBLISH -> CONTINUOUS UPDATE
```

The founder should increasingly act as editor/orchestrator, not manually perform
every step.

## 22. Weekend MVP

**Saturday morning** — repository structure, knowledge-area schema, source
ingestion, document storage, metadata, basic retrieval, LLM interface.

**Saturday afternoon** — specialist agent, citation generation, source viewer,
initial knowledge base.

**Saturday evening** — first 100+ evaluation questions, automated evaluator, basic
scoring dashboard.

**Sunday morning** — red-team evaluator, regression testing, answer/citation
validation, knowledge versioning.

**Sunday afternoon** — simple public UI, public evidence page, "Ask the
specialist", evaluation results, source inspection.

**Sunday evening** — invite 5-10 domain experts/practitioners, technically
knowledgeable people and sceptics. Ask them one thing: *try to break it*.

## 23. What NOT to build initially

Do not spend the weekend building: mobile apps, complex user accounts, elaborate
branding, payments, 2,000 domains, custom foundation models, distributed GPU
infrastructure, complicated knowledge graphs, enterprise SSO, elaborate agent
orchestration.

The first milestone is: **one specialist knowledge system that can survive serious
questioning.**

## 24. Success criteria for Knowledge Area #001

1. A domain can be ingested largely automatically.
2. The system answers genuinely useful questions.
3. Answers contain traceable evidence.
4. Unsupported claims are detected.
5. Evaluation is automated.
6. Regression testing works.
7. Knowledge can be updated without rebuilding the whole system.
8. Experts can challenge it.
9. Failures can be captured and converted into new tests.
10. A second knowledge area can subsequently be created using substantially the
    same pipeline.

Criterion #10 is particularly important.

We're not building one clever chatbot. We're building the factory that makes
specialist knowledge systems.

## 25. Guiding philosophy

The lab should never claim: *"Our AI is smarter."*

It should demonstrate:

> "Here is what it knows, here is where it came from, here is how we test it,
> here is where it failed, and here is what changed."

That is the foundation of the project.

## 26. First coding-agent instruction

Read `PROJECT_INITIATION.md` completely. Do not immediately start coding. First
inspect the repository and produce `IMPLEMENTATION_PLAN.md` describing the
architecture, technology choices, directory structure, data model, ingestion
pipeline, retrieval architecture, evaluation framework, red-team framework,
deployment model and weekend MVP sequence.

Do not over-engineer. The objective is to prove the complete Knowledge Foundry loop
with one knowledge area at minimal cost.

Where there are architectural choices, favour simple, replaceable components. Avoid
creating dependencies on a single proprietary model or infrastructure provider.

After producing the implementation plan, begin implementing the MVP in small,
testable increments. Every major component must have tests.

The final MVP must demonstrate: source ingestion -> knowledge representation ->
retrieval -> specialist answer -> citations -> automated evaluation -> adversarial
testing -> versioning -> public evidence.

Do not build a generic chatbot. Build the first instance of a repeatable
specialist-knowledge-system factory.

---

Repository name: `knowledge-foundry`

Initial milestone: **Knowledge Area #001 — End-to-End Specialist Intelligence System.**
