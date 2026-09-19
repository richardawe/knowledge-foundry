/*
 * The ask box, answering in the page (§16).
 *
 * The corpus arrives as one JSON file and `engine.js` runs retrieval, the
 * gates, sentence selection and the grounding check on the reader's own
 * machine. No instance to find, no key, no round trip, no wait -- and it works
 * offline, on a fork, and with the repository private.
 *
 * `tests/test_browser_parity.py` asserts this engine returns the same verdict,
 * the same evidence in the same order and the same answer text as the Python
 * pipeline, over every question in the shipped evaluation suites. So what a
 * reader sees here is what the nightly graded, not an approximation of it.
 *
 * Challenging an answer is the one thing that leaves the page: it opens the
 * issue form, where a workflow records it as a durable challenge that can be
 * promoted into the evaluation suite. Fast path for asking, durable path for
 * disagreeing.
 */
(function () {
  "use strict";

  var root = document.getElementById("ask-app");
  if (!root) return;

  var area = root.dataset.knowledgeArea;
  var corpusUrl = root.dataset.corpus;
  var repo = root.dataset.repo || "";
  var statusEl = document.getElementById("ask-status");
  var formEl = document.getElementById("ask-form");
  var questionEl = document.getElementById("ask-question");
  var outputEl = document.getElementById("ask-output");
  var index = null;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function setStatus(kind, html) {
    if (!statusEl) return;
    statusEl.className = "small " + (kind === "error" ? "banner" : "muted");
    statusEl.innerHTML = html;
  }

  function challengeUrl(question, answer) {
    var base = repo + "/issues/new?template=ask.yml";
    return base +
      "&knowledge-area=" + encodeURIComponent(area) +
      "&question=" + encodeURIComponent(question) +
      "&problem=" + encodeURIComponent(
        "The in-page engine answered with status '" + answer.status + "'. ");
  }

  function renderSources(sources) {
    var cited = sources.filter(function (s) { return s.cited; });
    if (!cited.length) return "";
    var rows = cited.map(function (s) {
      var title = esc(s.source_title || s.source_id);
      var link = s.uri ? '<a href="' + esc(s.uri) + '" rel="noreferrer">' + title + "</a>" : title;
      var detail = [s.publisher, s.published_at].filter(Boolean).map(esc).join(" — ");
      return "<li><code>[" + s.rank + "]</code> " + link +
        (detail ? ' <span class="muted small">' + detail + "</span>" : "") +
        ' <span class="tag">authority ' + esc(s.authority) + "/5</span></li>";
    });
    return "<h3>Sources</h3><ul class='tight'>" + rows.join("") + "</ul>";
  }

  function render(question, a) {
    var pct = function (x) { return Math.round(x * 100) + "%"; };
    var head = a.status === "answered"
      ? ""
      : "<div class='banner'><strong>The system declined to answer.</strong> " +
        "Status <code>" + esc(a.status) + "</code>. That is a designed outcome, not an error — " +
        "several gates can end a question without an answer.</div>";

    var limitations = (a.limitations || []).length
      ? "<h3>Limitations</h3><ul class='tight'>" +
        a.limitations.map(function (l) { return "<li>" + esc(l) + "</li>"; }).join("") + "</ul>"
      : "";

    var unsupported = (a.unsupported_claims || []).length
      ? "<h3>Claims that could not be traced to the evidence</h3><ul class='tight'>" +
        a.unsupported_claims.map(function (u) { return "<li>" + esc(u.sentence) + "</li>"; }).join("") +
        "</ul>"
      : "";

    outputEl.innerHTML =
      head +
      "<h2>Answer</h2><p>" + esc(a.answer) + "</p>" +
      (a.reasoning ? "<p class='small muted'>" + esc(a.reasoning) + "</p>" : "") +
      renderSources(a.sources || []) +
      unsupported +
      "<h3>Evidence confidence</h3><p><span class='tag'>" + esc(a.confidence) + "</span> " +
      "<span class='small muted'>score " + a.confidence_score.toFixed(2) +
      " · citations valid " + pct(a.citation_validity) +
      " · claims grounded " + pct(a.grounded_ratio) + "</span></p>" +
      limitations +
      "<p class='small muted'>Answered in this page by <code>" + esc(area) + "</code> " +
      esc(a.knowledge_version) + " (evaluation: " + esc(a.evaluation_status) + "), " +
      "retrievers: " + esc((a.retrievers || []).join(", ")) + ". No model was involved: this " +
      "engine selects and cites stored sentences and cannot state anything the corpus does " +
      "not already say.</p>" +
      (repo
        ? "<p><a class='button' href='" + esc(challengeUrl(question, a)) + "' rel='noreferrer'>" +
          "This is wrong — tell us why &rarr;</a></p>" +
          "<p class='small muted'>That opens an issue. A workflow records it as a challenge " +
          "with the answer it disputes, and an accepted challenge is promoted into the " +
          "evaluation suite — so the same mistake cannot be made twice without a test failing.</p>"
        : "");
    outputEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function answer(question) {
    setStatus("muted", "Answering…");
    try {
      var result = window.FoundryEngine.ask(index, question);
      render(question, result);
      setStatus("muted", "Answered in this page, from " + index.corpus.chunks.length +
        " stored passages. Nothing left your browser.");
    } catch (err) {
      setStatus("error", "The engine failed on that question: " + esc(err && err.message));
    }
  }

  setStatus("muted", "Loading the corpus…");
  fetch(corpusUrl)
    .then(function (r) {
      if (!r.ok) throw new Error("corpus unavailable (" + r.status + ")");
      return r.json();
    })
    .then(function (corpus) {
      index = window.FoundryEngine.buildIndex(corpus);
      if (formEl) formEl.hidden = false;
      setStatus("muted", "Ready — " + corpus.chunks.length + " passages, " +
        Object.keys(corpus.sources).length + " sources, answering in this page.");
    })
    .catch(function (err) {
      setStatus("error", "Could not load the corpus (" + esc(err.message) + "). " +
        "The recorded answers are still on the " +
        "<a href='../transcript/'>transcript</a> page.");
    });

  if (formEl) {
    formEl.addEventListener("submit", function (e) {
      e.preventDefault();
      var q = (questionEl.value || "").trim();
      if (q && index) answer(q);
    });
  }
})();
