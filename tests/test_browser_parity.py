"""Two implementations of one engine must not drift.

`engine.js` answers in the reader's browser; `foundry.specialist` answers in
the nightly. They are the same system or they are not, and "not" is the failure
that matters: a page answering where the pipeline would abstain publishes a
system whose evaluation results describe something else.

So parity is tested rather than hoped for. Every primitive is compared against
real corpus text, and every gate verdict against the shipped evaluation suite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from foundry.browser_index import render_browser_index
from foundry.specialist import Specialist
from foundry.text import content_terms, numbers_with_units, sentences

ENGINE = Path(__file__).resolve().parents[1] / "src" / "foundry" / "web" / "static" / "engine.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed; parity cannot be checked"
)


def _node(script: str, payload: dict) -> dict:
    """Run a snippet against the real engine and hand back its JSON."""
    result = subprocess.run(
        ["node", "-e", f"const ENGINE={json.dumps(str(ENGINE))};\n{script}"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(f"node failed:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout)


# -- the primitives every later comparison rests on ----------------------

_PRIMITIVES = """
const e = require(ENGINE);
let raw = ""; process.stdin.on("data", (d) => (raw += d));
process.stdin.on("end", () => {
  const { samples } = JSON.parse(raw);
  process.stdout.write(JSON.stringify({
    terms: samples.map((s) => [...e.contentTerms(s)].sort()),
    sentences: samples.map((s) => e.sentences(s)),
    numbers: samples.map((s) => [...e.numbersWithUnits(s)].sort()),
  }));
});
"""


def test_the_text_primitives_agree_on_real_corpus_text(built):
    """Tokenising, stemming and unit parsing drive retrieval, the gates and
    grounding alike. If these disagree, nothing above them can agree."""
    _manifest, store = built
    samples = [row["text"][:600] for row in store.conn.execute("SELECT text FROM chunks")]
    samples += [
        "What's the battery capacity for a Tesla 3?",
        "Exceeds 80 C. [1] Next claim. [2]",
        "A cell vents at 12 bar and 80 °C, per UL 9540A.",
        "the operator's duty and the batteries' casings",
        "Limited to 100 watt hours, or 2.5 kg of material.",
    ]
    assert samples, "the fixture corpus is empty"

    js = _node(_PRIMITIVES, {"samples": samples})
    for i, sample in enumerate(samples):
        assert js["terms"][i] == sorted(content_terms(sample)), f"terms differ on {sample[:60]!r}"
        assert js["sentences"][i] == sentences(sample), f"sentences differ on {sample[:60]!r}"
        assert js["numbers"][i] == sorted(numbers_with_units(sample)), f"numbers differ: {sample[:60]!r}"


# -- retrieval and the gates, over the shipped evaluation suite ----------

# Statuses alone are too coarse to protect anything. A threshold or a formula
# can drift a long way before a verdict flips, and the first question whose
# verdict it flips is a reader's, not a test's. So the gates' own numbers are
# compared, which is what those verdicts are made of.
_VERDICTS = """
const e = require(ENGINE);
let raw = ""; process.stdin.on("data", (d) => (raw += d));
process.stdin.on("end", () => {
  const { corpus, questions } = JSON.parse(raw);
  const idx = e.buildIndex(corpus);
  process.stdout.write(JSON.stringify(questions.map((q) => {
    const evidence = e.retrieve(idx, q);
    const a = e.ask(idx, q);
    const subject = e.coversTheSubject(corpus, q, evidence);
    const scope = e.scopeCheck(corpus, q);
    return {
      status: a.status,
      answer: a.answer,
      evidence: evidence.map((x) => x.chunk_id),
      scope_in: scope.inScore, scope_out: scope.outScore,
      missing_share: subject.share, missing: subject.missing,
      citation_validity: a.citation_validity, grounded_ratio: a.grounded_ratio,
      confidence_score: a.confidence_score,
    };
  })));
});
"""


def _browser_equivalent(manifest):
    """The configuration the browser can actually run.

    The semantic retriever's projection matrix is the vocabulary by the
    embedding dimension -- tens of megabytes, which is not a page -- so it and
    the structured and graph retrievers stay behind. Python is configured the
    same way for the comparison: parity means "the same engine", not "a
    different engine that happens to agree".
    """
    manifest.retrieval.semantic_enabled = False
    manifest.retrieval.structured_enabled = False
    manifest.retrieval.graph_enabled = False
    return manifest


def test_retrieval_and_every_gate_agree_question_for_question(built):
    manifest, store = built
    _browser_equivalent(manifest)

    questions = []
    for suite in manifest.evaluation.suites:
        path = Path(manifest.resolve(suite))
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        questions.extend(q["question"] for q in data.get("questions", []))
    questions += [
        "What is the resale value of a widget in Lagos in 2031?",
        "How should I treat a bacterial infection?",
        "What is the limit?",
        "",
    ]
    assert questions

    payload, _report = render_browser_index(manifest, store)
    js = _node(_VERDICTS, {"corpus": json.loads(payload), "questions": questions})

    specialist = Specialist(manifest, store)
    mismatches = []
    for question, actual in zip(questions, js):
        label = f"{question[:60]!r}"
        expected = specialist.ask(question)
        if actual["status"] != expected.status:
            mismatches.append(f"{label}: status python={expected.status} js={actual['status']}")
            continue
        if actual["answer"].strip() != expected.answer.strip():
            mismatches.append(f"{label}: answer text differs")
            continue
        if not question.strip():
            continue

        evidence = specialist.retriever.retrieve(question).evidence
        if actual["evidence"] != [e.chunk_id for e in evidence]:
            mismatches.append(f"{label}: evidence ordering differs")
            continue

        _in_scope, in_score, out_score = specialist.scope_check(question)
        _ok, missing, share = specialist.covers_the_subject(question, evidence)
        for name, want, got in (
            ("scope_in", in_score, actual["scope_in"]),
            ("scope_out", out_score, actual["scope_out"]),
            ("missing_share", share, actual["missing_share"]),
            ("citation_validity", expected.citation_validity, actual["citation_validity"]),
            ("grounded_ratio", expected.grounded_ratio, actual["grounded_ratio"]),
            ("confidence_score", expected.confidence_score, actual["confidence_score"]),
        ):
            if abs(float(want) - float(got)) > 1e-6:
                mismatches.append(f"{label}: {name} python={want:.6f} js={got:.6f}")
        if sorted(missing) != sorted(actual["missing"]):
            mismatches.append(f"{label}: missing terms {sorted(missing)} vs {sorted(actual['missing'])}")

    assert not mismatches, "engines disagree:\n" + "\n".join(mismatches)


def test_the_index_carries_what_the_gates_need(built):
    """A threshold missing from the export silently becomes a different gate."""
    manifest, store = built
    payload, _report = render_browser_index(manifest, store)
    corpus = json.loads(payload)

    for key in ("min_evidence_score", "min_chunks", "max_missing_subject_weight",
                "min_question_coverage", "min_sentence_support", "max_unsupported_ratio",
                "top_k", "candidate_k"):
        assert key in corpus["thresholds"], f"{key} missing from the browser index"
    assert corpus["rerank"]["authority_weight"] == manifest.retrieval.authority_weight
    assert corpus["fusion_k"] == manifest.retrieval.fusion_k
    # Provenance, or the answer cannot be checked and the page is a chatbot.
    for source in corpus["sources"].values():
        assert source["uri"] and source["publisher"]


# -- the same comparison, against a real knowledge area ------------------
#
# The synthetic fixture is four passages and three questions. It is enough to
# catch a broken port and demonstrably not enough to catch a drifting one: a
# changed BM25 constant, a changed length normalisation and a changed prose
# filter all survived it. A real corpus with a real suite catches all three,
# because the paths exist there to be exercised.
#
# It needs a built corpus, which a cold checkout does not have. Rather than
# skip quietly in the one place depth is available, the nightly builds before
# it tests, so this runs there.


def _built_knowledge_areas():
    from foundry.manifest import Manifest, list_knowledge_areas
    from foundry.storage import Store

    out = []
    for ka_id in list_knowledge_areas():
        try:
            manifest = Manifest.load(ka_id)
            with Store(ka_id) as store:
                if store.counts()["chunks"] > 0:
                    out.append(manifest.id)
        except Exception:
            continue
    return out


@pytest.mark.parametrize("ka_id", _built_knowledge_areas() or ["none-built"])
def test_the_engines_agree_on_a_real_corpus(ka_id, monkeypatch):
    from foundry.manifest import Manifest
    from foundry.storage import Store

    if ka_id == "none-built":
        pytest.skip("no knowledge area is built; run `foundry build <ka>` for depth here")

    monkeypatch.delenv("FOUNDRY_ROOT", raising=False)
    monkeypatch.delenv("FOUNDRY_VAR", raising=False)
    manifest = _browser_equivalent(Manifest.load(ka_id))

    questions = []
    for suite in manifest.evaluation.suites:
        path = Path(manifest.resolve(suite))
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        questions.extend(q["question"] for q in data.get("questions", []))
    if not questions:
        pytest.skip(f"{ka_id} ships no evaluation questions")

    # The suite is all questions the corpus was built to answer, so none of it
    # lands in the band where the subject gate decides anything -- a mutation
    # moving that threshold from 0.6 to 0.99 passed unnoticed. These are
    # deliberately off-corpus, which is the only way to exercise it.
    questions += [
        "What is the resale value of a used e-bike in Lagos?",
        "How many goals did Arsenal score last season?",
        "What is the 2031 wholesale price index for cobalt in Chile?",
        "Who won the mayoral election in Bristol?",
    ]

    with Store(ka_id) as store:
        payload, _report = render_browser_index(manifest, store)
        js = _node(_VERDICTS, {"corpus": json.loads(payload), "questions": questions})

        specialist = Specialist(manifest, store)
        mismatches = []
        for question, actual in zip(questions, js):
            expected = specialist.ask(question)
            evidence = specialist.retriever.retrieve(question).evidence
            if actual["status"] != expected.status:
                mismatches.append(
                    f"{question[:70]!r}: python={expected.status} js={actual['status']}"
                )
                continue
            if actual["evidence"] != [e.chunk_id for e in evidence]:
                mismatches.append(f"{question[:70]!r}: evidence ordering differs")
                continue
            if actual["answer"].strip() != expected.answer.strip():
                mismatches.append(f"{question[:70]!r}: answer text differs")
                continue
            _ok, missing, share = specialist.covers_the_subject(question, evidence)
            if sorted(missing) != sorted(actual["missing"]):
                mismatches.append(f"{question[:70]!r}: missing terms differ")
                continue
            for name, want, got in (
                ("missing_share", share, actual["missing_share"]),
                ("grounded_ratio", expected.grounded_ratio, actual["grounded_ratio"]),
                ("citation_validity", expected.citation_validity, actual["citation_validity"]),
                ("confidence_score", expected.confidence_score, actual["confidence_score"]),
            ):
                if abs(float(want) - float(got)) > 1e-6:
                    mismatches.append(f"{question[:70]!r}: {name} {want:.6f} vs {got:.6f}")

    assert not mismatches, (
        f"{ka_id}: {len(mismatches)} of {len(questions)} disagree:\n" + "\n".join(mismatches[:15])
    )
