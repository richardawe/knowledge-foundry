/*
 * Ask client for a published evidence page.
 *
 * This does NOT reimplement anything. Retrieval, the gates, generation and
 * grounding validation all run in the Python instance it talks to -- normally
 * the reader's own machine, where Ollama is. The page is the interface; the
 * engine stays where the knowledge and the model are.
 *
 * Backend resolution, in order:
 *   1. ?api=  in the URL          (share a link to a specific instance)
 *   2. what the reader last used  (localStorage)
 *   3. the endpoint baked in at export time (--api-base)
 *   4. http://localhost:8000      (the default `make serve` address)
 *
 * Browsers treat http://localhost as a trustworthy origin, so an HTTPS page
 * may call it without a mixed-content error. A tunnelled HTTPS endpoint works
 * the same way for readers who are not running it themselves.
 */
(function () {
  "use strict";

  var root = document.getElementById("ask-app");
  if (!root) return;

  var area = root.dataset.knowledgeArea;
  var fallbackBase = root.dataset.apiBase || "http://localhost:8000";
  var STORAGE_KEY = "foundry.apiBase";

  function stored(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }
  function store(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* private mode */ }
  }

  function resolveBase() {
    var fromQuery = new URLSearchParams(window.location.search).get("api");
    if (fromQuery) { store(STORAGE_KEY, fromQuery); return fromQuery; }
    return stored(STORAGE_KEY) || fallbackBase;
  }

  var base = resolveBase().replace(/\/+$/, "");
  var transcriptUrl = root.dataset.transcript || "";
  var recorded = null;   // the prepared transcript index, once loaded
  var mode = "probing";  // probing | live | recorded | none
  var statusEl = document.getElementById("ask-status");
  var formEl = document.getElementById("ask-form");
  var offlineEl = document.getElementById("ask-offline");
  var outputEl = document.getElementById("ask-output");
  var endpointEl = document.getElementById("ask-endpoint");
  var questionEl = document.getElementById("ask-question");
  var buttonEl = document.getElementById("ask-submit");
  var modeEl = document.getElementById("ask-mode");

  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function pct(value) {
    return value === null || value === undefined ? "–" : Math.round(value * 100) + "%";
  }

  function setStatus(state, message) {
    statusEl.className = "small " + state;
    statusEl.innerHTML = message;
  }

  function showConnected(info) {
    var areas = (info && info.knowledge_areas) || [];
    var hasArea = areas.length === 0 || areas.indexOf(area) !== -1;
    if (!hasArea) {
      setStatus("hold",
        "Connected to <code>" + esc(base) + "</code>, but it does not serve <code>" +
        esc(area) + "</code>. Build it there first: <code>make build KA=" + esc(area) + "</code>.");
      return;
    }
    var version = info && info.version ? " (v" + esc(info.version) + ")" : "";
    setStatus("pass", "Connected to <code>" + esc(base) + "</code>" + version +
      " — questions are answered live by that instance.");
    mode = "live";
    formEl.hidden = false;
    offlineEl.hidden = true;
    if (modeEl) modeEl.hidden = true;
    if (questionEl) {
      questionEl.placeholder =
        "Ask anything — retrieval and the model run on that instance.";
    }
  }

  function showOffline(reason) {
    var tried = candidates().map(function (c) { return "<code>" + esc(c) + "</code>"; });
    setStatus("muted",
      "No instance reachable (" + esc(reason) + "). Tried " + tried.join(", ") + ".");
    offlineEl.hidden = false;

    // Nothing is answering, but the answers it has already given are published
    // right here. Search those rather than leaving the box dead.
    loadTranscript().then(function () {
      mode = "recorded";
      formEl.hidden = false;
      // The box works now, so the setup instructions stop being the headline
      // and move behind a link.
      offlineEl.hidden = true;
      if (questionEl) {
        questionEl.placeholder =
          "Search " + recorded.entries.length + " recorded answers…";
      }
      if (modeEl) {
        modeEl.hidden = false;
        modeEl.className = "small hold";
        modeEl.innerHTML =
          "Searching <strong>" + recorded.entries.length + " recorded answers</strong> " +
          "instead — every one produced by the full pipeline, with its evidence. " +
          '<a href="#" id="ask-show-setup">How to ask new questions live →</a>';
        var toggle = document.getElementById("ask-show-setup");
        if (toggle) {
          toggle.addEventListener("click", function (event) {
            event.preventDefault();
            offlineEl.hidden = !offlineEl.hidden;
          });
        }
      }
    }).catch(function () {
      mode = "none";
      formEl.hidden = true;
      if (modeEl) modeEl.hidden = true;
    });
  }

  // Ports the project realistically listens on: `make serve`, a common
  // alternative, and the Hugging Face Spaces default.
  var CANDIDATE_PORTS = [8000, 8080, 7860];

  function candidates() {
    var list = [base];
    if (/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(base)) {
      var host = base.replace(/:\d+$/, "");
      CANDIDATE_PORTS.forEach(function (port) {
        var candidate = host + ":" + port;
        if (list.indexOf(candidate) === -1) list.push(candidate);
      });
    }
    return list;
  }

  function withTimeout(ms) {
    var controller = new AbortController();
    setTimeout(function () { controller.abort(); }, ms);
    return controller.signal;
  }

  /*
   * A browser cannot distinguish "nothing is listening" from "something is
   * listening but rejected the cross-origin request" — both surface as the
   * same opaque TypeError. So when the normal request fails, probe again with
   * mode:"no-cors": that resolves whenever *anything* answered, and rejects
   * only when nothing did. The difference tells the reader whether to start
   * the server or to update it.
   */
  function probeOne(candidate) {
    return fetch(candidate + "/healthz", { signal: withTimeout(3500) })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (info) { return { state: "ok", base: candidate, info: info }; })
      .catch(function () {
        return fetch(candidate + "/healthz", { mode: "no-cors", signal: withTimeout(3500) })
          .then(function () { return { state: "blocked", base: candidate }; })
          .catch(function () { return { state: "absent", base: candidate }; });
      });
  }

  function isInsecureLocal(candidate) {
    return window.location.protocol === "https:" && candidate.indexOf("http://") === 0;
  }

  function showBlocked(candidate) {
    var safari = /^((?!chrome|android|crios|fxios).)*safari/i.test(navigator.userAgent);
    var message =
      "An instance is answering at <code>" + esc(candidate) + "</code>, but this page " +
      "cannot read its replies.";
    if (safari && isInsecureLocal(candidate)) {
      message += " Safari blocks an HTTPS page from calling <code>http://localhost</code>. " +
        "Open the instance directly instead — it serves this same page: " +
        '<a href="' + esc(candidate) + "/k/" + esc(area) + '/ask">' +
        esc(candidate) + "/k/" + esc(area) + "/ask</a>.";
    } else {
      message += " That usually means it is running a build from before " +
        "cross-origin support was added. Update and restart it:" +
        "<br><code>git pull &amp;&amp; make serve</code>";
    }
    setStatus("hold", message);
    formEl.hidden = true;
    offlineEl.hidden = false;
    loadTranscript().then(function () {
      mode = "recorded";
      formEl.hidden = false;
      if (modeEl) {
        modeEl.hidden = false;
        modeEl.className = "small hold";
        modeEl.innerHTML = "Meanwhile, searching <strong>" +
          recorded.entries.length + " recorded answers</strong>.";
      }
    }).catch(function () { mode = "none"; });
  }

  function probe() {
    mode = "probing";
    if (modeEl) modeEl.hidden = true;
    setStatus("muted", "Looking for an instance at <code>" + esc(base) + "</code>…");
    if (endpointEl) endpointEl.value = base;

    Promise.all(candidates().map(probeOne)).then(function (results) {
      var ok = results.filter(function (r) { return r.state === "ok"; })[0];
      if (ok) {
        base = ok.base;
        store(STORAGE_KEY, base);
        if (endpointEl) endpointEl.value = base;
        showConnected(ok.info);
        return;
      }
      var blocked = results.filter(function (r) { return r.state === "blocked"; })[0];
      if (blocked) { showBlocked(blocked.base); return; }
      showOffline("nothing is listening");
    });
  }

  /*
   * Recorded answers.
   *
   * With no instance reachable the box is not inert: it searches the answers
   * the system has already given, published as static JSON by the machine that
   * ran them. That is the same shape as the sibling localtest project — the
   * model runs on the Mac, the results are published as files.
   *
   * This matches a typed question against roughly a hundred RECORDED
   * QUESTIONS. It does not retrieve over the corpus: the 1,267 passages are
   * never downloaded, nothing is scored against them, and nothing here can
   * produce an answer the system has not already given. Every word shown was
   * generated by the real pipeline, gates and validation included.
   */
  var STOPWORDS = {};
  ("a an the and or but if then than that this these those there here is are was " +
   "were be been being am do does did doing done have has had having will would " +
   "shall should can could may might must of in on at to for from by with without " +
   "within into onto over under as it its their they them you your we our us i " +
   "what which who whom whose when where why how not no all any some more most " +
   "about between during above below out off again once each few very"
  ).split(" ").forEach(function (w) { STOPWORDS[w] = true; });

  // Mirrors the server-side stemmer closely enough that "gases" and "gas", or
  // "venting" and "vents", land on the same key.
  function stem(token) {
    if (token.length > 4 && /ies$/.test(token)) return token.slice(0, -3) + "y";
    if (token.length > 5 && /ing$/.test(token)) token = token.slice(0, -3);
    else if (token.length > 4 && /ed$/.test(token)) token = token.slice(0, -2);
    else if (token.length > 3 && /s$/.test(token) && !/ss$/.test(token)) {
      token = token.slice(0, -1);
    }
    if (token.length > 3 && /e$/.test(token)) token = token.slice(0, -1);
    return token;
  }

  function terms(text) {
    var found = String(text || "").toLowerCase().match(/[a-z0-9][a-z0-9'\-]*/g) || [];
    var out = [];
    for (var i = 0; i < found.length; i++) {
      var t = found[i];
      if (t.length > 2 && !STOPWORDS[t]) out.push(stem(t));
    }
    return out;
  }

  function prepare(data) {
    var entries = (data.entries || []).map(function (entry) {
      return { entry: entry, terms: terms(entry.question) };
    });
    // Weight by rarity across the recorded questions, so a term every question
    // shares ("battery") counts for far less than one that pins the topic down.
    var df = {};
    entries.forEach(function (row) {
      var seen = {};
      row.terms.forEach(function (t) {
        if (!seen[t]) { seen[t] = true; df[t] = (df[t] || 0) + 1; }
      });
    });
    var n = Math.max(1, entries.length);
    var idf = {};
    Object.keys(df).forEach(function (t) { idf[t] = Math.log(1 + n / df[t]); });
    return { meta: data, entries: entries, idf: idf, unseen: Math.log(1 + n) };
  }

  function weightOf(term) {
    return recorded.idf[term] === undefined ? recorded.unseen : recorded.idf[term];
  }

  function searchRecorded(question) {
    var wanted = terms(question);
    if (!wanted.length) return [];
    var total = 0;
    wanted.forEach(function (t) { total += weightOf(t); });
    if (!total) return [];

    return recorded.entries.map(function (row) {
      var present = {};
      row.terms.forEach(function (t) { present[t] = true; });
      var matched = 0;
      var seen = {};
      wanted.forEach(function (t) {
        if (present[t] && !seen[t]) { seen[t] = true; matched += weightOf(t); }
      });
      return { entry: row.entry, score: matched / total };
    }).filter(function (r) {
      return r.score >= 0.34;
    }).sort(function (a, b) {
      return b.score - a.score;
    }).slice(0, 4);
  }

  function loadTranscript() {
    if (!transcriptUrl) return Promise.reject(new Error("no transcript"));
    return fetch(transcriptUrl, { signal: withTimeout(8000) })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        recorded = prepare(data);
        if (!recorded.entries.length) throw new Error("transcript is empty");
        return recorded;
      });
  }

  function renderRecorded(question, matches) {
    if (!matches.length) {
      outputEl.innerHTML =
        '<div class="banner"><strong>Not asked yet.</strong> ' +
        "Nothing in the recorded answers is close enough to that question to " +
        "show honestly. Browse them all in the " +
        '<a href="' + esc(transcriptLink()) + '">transcript</a>, or run an ' +
        "instance to ask it live.</div>";
      return;
    }

    var best = matches[0];
    var entry = best.entry;
    var sources = (entry.cited || []).map(function (s) {
      var copy = {};
      Object.keys(s).forEach(function (k) { copy[k] = s[k]; });
      copy.cited = true;
      return copy;
    });

    var html =
      '<div class="banner"><strong>Recorded answer, not a live one.</strong> ' +
      "Nothing is running to answer your question, so this is the closest " +
      "question the system has already been asked" +
      (recorded.meta.provider
        ? ", answered by <code>" + esc(recorded.meta.provider) + "</code>"
        : "") +
      " on run <code>" + esc(recorded.meta.run_id || "?") + "</code>" +
      (recorded.meta.version ? " against <code>" + esc(recorded.meta.version) + "</code>" : "") +
      ".</div>";

    html += '<h2>Recorded answer</h2><p class="small">' +
      '<span class="tag">recorded</span> ' +
      '<span class="tag">' + esc(entry.id) + "</span> " +
      '<span class="tag">' + esc(entry.type) + "</span> " +
      '<span class="tag">' + esc(entry.status) + "</span> " +
      '<span class="tag">confidence: ' + esc(entry.confidence) + "</span> " +
      '<span class="tag">citations valid: ' + pct(entry.citation_validity) + "</span> " +
      '<span class="tag">claims grounded: ' + pct(entry.grounded_ratio) + "</span> " +
      (entry.passed
        ? '<span class="pass">passed its test</span>'
        : '<span class="fail">FAILS its test</span>') +
      "</p>";

    html += '<p class="small muted">It was asked: <strong>' + esc(entry.question) +
      "</strong><br>You asked: " + esc(question) +
      " — matched at " + pct(best.score) + ".</p>";

    html += '<div class="panel">' + esc(entry.answer) + "</div>";
    if (!entry.passed) {
      html += '<p class="small fail">This answer fails its own evaluation. ' +
        "It is shown anyway, because hiding the failures would defeat the point.</p>";
    }
    html += renderSources(sources);

    if (matches.length > 1) {
      html += "<h3>Other recorded questions near yours</h3><ul class=\"tight small\">";
      matches.slice(1).forEach(function (m) {
        html += '<li><a href="' + esc(transcriptLink()) + "#" + esc(m.entry.id) + '">' +
          esc(m.entry.question) + "</a> — " + pct(m.score) + " match, " +
          (m.entry.passed ? "passed" : '<span class="fail">failed</span>') + "</li>";
      });
      html += "</ul>";
    }

    html += '<p class="small muted">To ask a question nobody has asked yet, run an ' +
      "instance — see the box above.</p>";
    outputEl.innerHTML = html;
    outputEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function transcriptLink() {
    return transcriptUrl
      .replace("/api/knowledge/", "/k/")
      .replace(".transcript.json", "/transcript/");
  }

  function renderSources(sources) {
    if (!sources || !sources.length) {
      return '<p class="small muted">No sources cited — the system declined to answer.</p>';
    }
    var rows = sources.map(function (s) {
      var link = s.uri && s.uri.indexOf("inline:") !== 0
        ? '<br><a class="muted" href="' + esc(s.uri) + '" rel="nofollow noopener">' + esc(s.uri) + "</a>"
        : "";
      return '<tr><td class="small">[' + esc(s.rank) + "]" +
        (s.cited ? ' <span class="tag">cited</span>' : "") + "</td>" +
        '<td class="small"><strong>' + esc(s.source_title) + "</strong><br>" +
        '<span class="muted">' + esc(s.publisher) +
        (s.published_at ? " · " + esc(s.published_at) : "") +
        " · authority " + esc(s.authority) + "/5</span>" + link + "</td>" +
        '<td class="small">' + (s.section ? "<em>" + esc(s.section) + "</em><br>" : "") +
        esc(s.text) + "</td></tr>";
    }).join("");
    return '<h3>Evidence</h3><p class="small muted">Passages marked <strong>cited</strong> ' +
      "were used in the answer above.</p>" +
      "<table><tr><th>#</th><th>Source</th><th>Passage</th></tr>" + rows + "</table>";
  }

  function renderList(title, items, className) {
    if (!items || !items.length) return "";
    return "<h3>" + title + '</h3><ul class="tight small ' + (className || "") + '">' +
      items.map(function (i) { return "<li>" + esc(i) + "</li>"; }).join("") + "</ul>";
  }

  function render(answer) {
    var chips = [
      answer.status, "confidence: " + answer.confidence,
      "citations valid: " + pct(answer.citation_validity),
      "claims grounded: " + pct(answer.grounded_ratio),
      answer.knowledge_version,
    ];
    var model = answer.llm && answer.llm.provider
      ? '<span class="tag">' + esc(answer.llm.provider) +
        (answer.llm.model ? " · " + esc(answer.llm.model) : "") + "</span>"
      : "";

    var html = "<h2>Answer</h2><p class=\"small\">" +
      chips.map(function (c) { return '<span class="tag">' + esc(c) + "</span>"; }).join(" ") +
      " " + model + "</p>" +
      '<div class="panel">' + esc(answer.answer) + "</div>";

    if (answer.reasoning) {
      html += '<h3>Reasoning</h3><p class="small">' + esc(answer.reasoning) + "</p>";
    }
    if (answer.contradictions && answer.contradictions.length) {
      html += renderList("Sources disagree",
        answer.contradictions.map(function (c) { return c.description; }));
    }
    if (answer.unsupported_claims && answer.unsupported_claims.length) {
      html += "<h3>Unsupported claims detected</h3>" +
        '<p class="small muted">Found by automated validation after generation, and shown ' +
        "rather than removed.</p><table><tr><th>Sentence</th><th>Why it failed</th></tr>" +
        answer.unsupported_claims.map(function (c) {
          return '<tr><td class="small">' + esc(c.sentence) + "</td>" +
            '<td class="small fail">' + esc(c.reason) + "</td></tr>";
        }).join("") + "</table>";
    }
    html += renderSources(answer.sources);
    html += renderList("Limitations", answer.limitations);
    outputEl.innerHTML = html;
    outputEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function ask(question) {
    buttonEl.disabled = true;
    buttonEl.textContent = "Asking…";
    outputEl.innerHTML = '<p class="small muted">Retrieving evidence and generating an ' +
      "answer on " + esc(base) + "…</p>";

    fetch(base + "/knowledge/" + encodeURIComponent(area) + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    })
      .then(function (r) {
        return r.json().then(function (body) {
          if (!r.ok) throw new Error(body.error || "HTTP " + r.status);
          return body;
        });
      })
      .then(render)
      .catch(function (err) {
        outputEl.innerHTML = '<p class="small fail">Could not reach the instance: ' +
          esc(err.message) + "</p>";
      })
      .finally(function () {
        buttonEl.disabled = false;
        buttonEl.textContent = "Ask";
      });
  }

  formEl.addEventListener("submit", function (event) {
    event.preventDefault();
    var question = questionEl.value.trim();
    if (!question) return;
    if (mode === "recorded") {
      renderRecorded(question, searchRecorded(question));
    } else {
      ask(question);
    }
  });

  var endpointForm = document.getElementById("ask-endpoint-form");
  if (endpointForm) {
    endpointForm.addEventListener("submit", function (event) {
      event.preventDefault();
      base = endpointEl.value.trim().replace(/\/+$/, "") || fallbackBase;
      store(STORAGE_KEY, base);
      outputEl.innerHTML = "";
      probe();
    });
  }

  probe();
})();
