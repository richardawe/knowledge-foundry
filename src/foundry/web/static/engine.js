/*
 * The answering engine, in the browser (§16).
 *
 * Without a model, answering is arithmetic over stored text. Nothing here
 * calls anything: the corpus arrives as one JSON file and the reader's own
 * machine runs retrieval, the gates, sentence selection and the grounding
 * check. No server, no key, no wait.
 *
 * This is a second implementation of behaviour that already exists in Python,
 * which is a real risk and is treated as one. A browser that answers where the
 * nightly abstains would be publishing a different system than the one the
 * evaluation results describe. `tests/test_browser_parity.py` runs the real
 * evaluation questions through both engines and asserts the verdicts agree, so
 * drift fails the build rather than reaching a reader.
 *
 * Every primitive below is a deliberate port of `foundry.text`. Where a choice
 * looks odd, it is odd in Python too and the comment there explains why; the
 * rule is that this file copies that behaviour rather than improving on it.
 */
"use strict";

const STOPWORDS = new Set(["a","about","above","again","all","also","am","an","and","any","are","as","at","be","been","being","below","between","both","but","by","can","could","did","do","does","doing","done","down","during","e.g","each","etc","few","for","from","further","had","has","have","having","he","hence","her","here","his","how","however","i","i.e","if","in","into","is","it","it's","its","may","might","more","most","must","no","nor","not","of","off","on","once","onto","or","other","our","out","over","per","shall","she","should","so","some","such","than","that","the","their","them","then","there","therefore","these","they","this","those","thus","to","under","up","us","very","via","was","we","were","what","when","where","which","who","whom","whose","why","will","with","within","without","would","you","your"]);

const WORD_RE = /[A-Za-z][A-Za-z0-9\-/+']*|\d+(?:[.,]\d+)*/g;
const SENTENCE_SPLIT_RE = /(?<=[.!?])\s+(?=[A-Z\[""'(])/;
const CITATION_RE = /\[(\d+)\]/g;
const CITATION_ONLY_RE = /^(?:\[\d+\]\s*)+$/;
const LEADING_CITATION_RE = /^((?:\[\d+\]\s*)+)([\s\S]*)$/;

function tokenize(text) {
  return (String(text || "").match(WORD_RE) || []).map((t) => t.toLowerCase());
}

/* "what's" -> "what", so it is filtered as a stopword instead of stemming to
 * "what'" -- a term in no document, which any rarity weighting then treats as
 * the most important thing in the question. */
function depossess(token) {
  for (const tail of ["'s", "’s", "s'", "s’"]) {
    if (token.length > tail.length + 1 && token.endsWith(tail)) {
      return token.slice(0, -tail.length) + (tail[0] === "s" ? "s" : "");
    }
  }
  return token.replace(/['’]+$/, "");
}

/* A compact suffix stripper, not Porter. Over-stems slightly ("fire" -> "fir"),
 * which costs precision and buys recall, and is applied to both sides of every
 * comparison so its errors are consistent rather than biased. */
function stem(token) {
  if (!token || !/^[a-z]/i.test(token[0])) return token;
  if (token.length > 4 && token.endsWith("ies")) return token.slice(0, -3) + "y";
  if (token.length > 4 && token.endsWith("sses")) {
    token = token.slice(0, -2);
  } else if (token.length > 5 && token.endsWith("ing")) {
    let stripped = token.slice(0, -3);
    if (stripped.length > 2 && stripped[stripped.length - 1] === stripped[stripped.length - 2]) {
      stripped = stripped.slice(0, -1);
    }
    token = stripped;
  } else if (token.length > 4 && token.endsWith("ed")) {
    token = token.slice(0, -2);
  } else if (
    token.length > 3 && token.endsWith("s") &&
    !(token.endsWith("ss") || token.endsWith("us") || token.endsWith("is"))
  ) {
    token = token.slice(0, -1);
  }
  if (token.length > 3 && token.endsWith("e")) token = token.slice(0, -1);
  return token;
}

function contentTerms(text, stemmed = true) {
  const out = new Set();
  const raw = new Set();
  for (const tok of tokenize(text)) {
    const t = depossess(tok);
    if (!STOPWORDS.has(t) && t.length > 2) raw.add(t);
  }
  if (!stemmed) return raw;
  for (const t of raw) out.add(stem(t));
  return out;
}

/* Asymmetric on purpose: grounding asks "is this sentence's content present in
 * the evidence", not "are the two similar". */
function coverage(needle, haystack) {
  if (!needle.size) return 1.0;
  let hit = 0;
  for (const t of needle) if (haystack.has(t)) hit += 1;
  return hit / needle.size;
}

/* Citation-only fragments are folded back into the sentence they belong to:
 * "...exceeds 80 C. [1]" must not split into a claim with no citation and a
 * citation with no claim, or grounding grades every cited sentence against
 * nothing. */
function sentences(text) {
  const out = [];
  for (const block of String(text || "").split("\n")) {
    const trimmed = block.trim();
    if (!trimmed) continue;
    for (let part of trimmed.split(SENTENCE_SPLIT_RE)) {
      part = (part || "").trim();
      if (!part) continue;
      if (out.length && CITATION_ONLY_RE.test(part)) {
        out[out.length - 1] = out[out.length - 1] + " " + part;
        continue;
      }
      /* "...80 C. [1] Venting follows." -- the marker trails the claim it
       * supports, so it belongs to the sentence before it, not the one after. */
      const leading = LEADING_CITATION_RE.exec(part);
      if (out.length && leading) {
        out[out.length - 1] = out[out.length - 1] + " " + leading[1].trim();
        const remainder = leading[2].trim();
        if (remainder) out.push(remainder);
        continue;
      }
      out.push(part);
    }
  }
  return out;
}

const UNIT_ALIASES = {
  "%": "%",
  "a": "a",
  "ah": "ah",
  "amp": "a",
  "amp hour": "ah",
  "amp hours": "ah",
  "ampere": "a",
  "ampere hour": "ah",
  "ampere hours": "ah",
  "amperes": "a",
  "amps": "a",
  "bar": "bar",
  "c": "°c",
  "celsius": "°c",
  "centimetre": "cm",
  "centimetres": "cm",
  "cm": "cm",
  "deg c": "°c",
  "degc": "°c",
  "degree celsius": "°c",
  "degrees celsius": "°c",
  "degrees fahrenheit": "°f",
  "f": "°f",
  "fahrenheit": "°f",
  "g": "g",
  "gram": "g",
  "gramme": "g",
  "grammes": "g",
  "grams": "g",
  "h": "h",
  "hour": "h",
  "hours": "h",
  "hr": "h",
  "k": "k",
  "kelvin": "k",
  "kg": "kg",
  "kilogram": "kg",
  "kilograms": "kg",
  "kilometre": "km",
  "kilometres": "km",
  "kilowatt": "kw",
  "kilowatt hour": "kwh",
  "kilowatt hours": "kwh",
  "kilowatts": "kw",
  "km": "km",
  "kpa": "kpa",
  "kw": "kw",
  "kwh": "kwh",
  "m": "m",
  "megawatt": "mw",
  "megawatt hour": "mwh",
  "megawatt hours": "mwh",
  "megawatts": "mw",
  "meter": "m",
  "meters": "m",
  "metre": "m",
  "metres": "m",
  "mg": "mg",
  "micrometre": "µm",
  "micron": "µm",
  "microns": "µm",
  "milligram": "mg",
  "milligrams": "mg",
  "millimeter": "mm",
  "millimeters": "mm",
  "millimetre": "mm",
  "millimetres": "mm",
  "min": "min",
  "minute": "min",
  "minutes": "min",
  "mm": "mm",
  "mpa": "mpa",
  "mw": "mw",
  "mwh": "mwh",
  "pa": "pa",
  "pct": "%",
  "percent": "%",
  "ppb": "ppb",
  "ppm": "ppm",
  "psi": "psi",
  "s": "s",
  "sec": "s",
  "second": "s",
  "seconds": "s",
  "v": "v",
  "volt": "v",
  "volts": "v",
  "w": "w",
  "watt": "w",
  "watt hour": "wh",
  "watt hours": "wh",
  "watt-hour": "wh",
  "watt-hours": "wh",
  "watthour": "wh",
  "watthours": "wh",
  "watts": "w",
  "wh": "wh",
  "°c": "°c",
  "°f": "°f",
  "µm": "µm"
};
const NUMBER_RE = /(?<![\w.])(\d+(?:[.,]\d+)?)\s*(°\s?[CF]|µ?[A-Za-z]+(?:[- ][A-Za-z]+)?)?/g;

/* Resolve a unit spelling to its canonical key, or "" if it is not a unit.
 * The pattern captures up to two words so "watt hours" and "degrees celsius"
 * resolve; when the second word is just the next word of the sentence
 * ("grams of", "m between"), fall back to the first token. */
function canonicalUnit(raw) {
  if (!raw) return "";
  const cleaned = raw.trim().toLowerCase().replace(/\s+/g, " ").replace("\u00b0 ", "\u00b0");
  if (Object.prototype.hasOwnProperty.call(UNIT_ALIASES, cleaned)) return UNIT_ALIASES[cleaned];
  const first = cleaned.split(" ")[0].split("-")[0];
  return Object.prototype.hasOwnProperty.call(UNIT_ALIASES, first) ? UNIT_ALIASES[first] : "";
}

/* The cheapest effective hallucination trap in a domain full of thresholds: if
 * the answer states a number, that number had better appear in the evidence it
 * cites. Integers render without a trailing .0 so "80" and "80.0" agree. */
function numbersWithUnits(text) {
  const out = new Set();
  for (const m of String(text || "").matchAll(NUMBER_RE)) {
    const value = m[1].replace(/,/g, "");
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) continue;
    const unit = canonicalUnit(m[2]);
    const rendered = Number.isInteger(numeric) ? String(numeric) : String(numeric);
    out.add(`${rendered}${unit}`);
  }
  return out;
}

/* ---------------------------------------------------------------------------
 * Retrieval.
 *
 * Okapi BM25 over the passages, with the section heading weighted below the
 * body -- the same shape as the FTS5 ranking the Python keyword retriever
 * uses. The numbers will not match SQLite's implementation exactly, so parity
 * is asserted on verdicts and cited sources rather than on scores.
 * ------------------------------------------------------------------------- */

const BM25_K1 = 1.2;
const BM25_B = 0.75;
const SECTION_WEIGHT = 0.6;

function buildIndex(corpus) {
  const docs = corpus.chunks.map((c) => {
    const bodyTokens = tokenize(c.text).filter((t) => !STOPWORDS.has(t) && t.length > 1);
    const sectionTokens = tokenize(c.section).filter((t) => !STOPWORDS.has(t) && t.length > 1);
    const tf = new Map();
    for (const t of bodyTokens) tf.set(t, (tf.get(t) || 0) + 1);
    for (const t of sectionTokens) tf.set(t, (tf.get(t) || 0) + SECTION_WEIGHT);
    return { chunk: c, tf, length: bodyTokens.length + SECTION_WEIGHT * sectionTokens.length };
  });

  const df = new Map();
  for (const d of docs) for (const t of d.tf.keys()) df.set(t, (df.get(t) || 0) + 1);
  const avgLen = docs.reduce((a, d) => a + d.length, 0) / Math.max(1, docs.length);
  const byId = new Map(corpus.chunks.map((c) => [c.id, c]));
  return { corpus, docs, df, avgLen, n: docs.length, byId };
}

function search(index, query, limit) {
  const terms = tokenize(query).filter((t) => !STOPWORDS.has(t) && t.length > 1);
  if (!terms.length) return [];
  const scored = [];
  for (const d of index.docs) {
    let score = 0;
    for (const t of terms) {
      const f = d.tf.get(t);
      if (!f) continue;
      const n = index.df.get(t) || 0;
      const idf = Math.log(1 + (index.n - n + 0.5) / (n + 0.5));
      score += idf * ((f * (BM25_K1 + 1)) / (f + BM25_K1 * (1 - BM25_B + BM25_B * (d.length / index.avgLen))));
    }
    if (score > 0) scored.push({ chunk: d.chunk, score });
  }
  scored.sort((a, b) => b.score - a.score || (a.chunk.id < b.chunk.id ? -1 : 1));
  return scored.slice(0, limit);
}

/* Reciprocal rank fusion, then a single authority/recency rerank.
 *
 * RRF reads only ranks, so it is scale-free; the rerank is applied once to the
 * fused ranking rather than inside the retriever, because authority is a
 * property of the source and not of the match. The browser fuses one list,
 * which is a no-op in shape but keeps the arithmetic identical to Python's --
 * and the rerank is emphatically not a no-op: a regulator's text outranks an
 * encyclopaedia entry on a tie, and dropping it was the last thing keeping the
 * two engines' orderings apart.
 */
function reciprocalRankFusion(candidates, weight, k) {
  const fused = new Map();
  candidates.forEach((c, i) => {
    fused.set(c.chunk.id, (fused.get(c.chunk.id) || 0) + weight / (k + i + 1));
  });
  return fused;
}

/* 1.0 for undated sources; decays smoothly with age, never to zero. */
function recencyFactor(publishedAt, halfLifeDays) {
  if (!publishedAt || halfLifeDays <= 0) return 1.0;
  const text = String(publishedAt);
  let when = null;
  for (const re of [/^(\d{4})-(\d{2})-(\d{2})$/, /^(\d{4})-(\d{2})$/, /^(\d{4})$/]) {
    const m = re.exec(text);
    if (m) { when = new Date(Date.UTC(+m[1], m[2] ? +m[2] - 1 : 0, m[3] ? +m[3] : 1)); break; }
  }
  if (!when || Number.isNaN(when.getTime())) return 1.0;
  const ageDays = Math.max(0, Math.floor((Date.now() - when.getTime()) / 86400000));
  return 0.5 + 0.5 * Math.exp((-Math.log(2) * ageDays) / halfLifeDays);
}

function rerank(index, fused, topK) {
  const cfg = index.corpus.rerank || { authority_weight: 0.25, recency_half_life_days: 1460 };
  const adjusted = [];
  for (const [chunkId, score] of fused) {
    const chunk = index.byId.get(chunkId);
    if (!chunk) continue;
    const src = index.corpus.sources[chunk.source_id] || {};
    const authority = Number(src.authority || 0);
    const authorityFactor = 1.0 + cfg.authority_weight * ((authority - 3) / 2.0);
    const recency = recencyFactor(src.published_at, cfg.recency_half_life_days);
    const recencyF = 1.0 + cfg.authority_weight * (recency - 1.0);
    adjusted.push([chunkId, score * authorityFactor * recencyF]);
  }
  adjusted.sort((a, b) => b[1] - a[1]);
  return adjusted.slice(0, topK);
}

/* Evidence carries its provenance, because a passage without its source is not
 * evidence and an answer built from it could not be checked. */
function retrieve(index, query) {
  const corpus = index.corpus;
  const candidateK = corpus.thresholds.candidate_k || 60;
  const hits = search(index, query, candidateK);
  const fused = reciprocalRankFusion(hits, corpus.fusion_weights.keyword, corpus.fusion_k || 60);
  const ranked = rerank(index, fused, corpus.thresholds.top_k || 8);
  return ranked.map(([chunkId, score], i) => {
    const chunk = index.byId.get(chunkId);
    const src = corpus.sources[chunk.source_id] || {};
    return {
      rank: i + 1, chunk_id: chunk.id, text: chunk.text, section: chunk.section, score,
      source_id: chunk.source_id, source_title: src.title || chunk.source_id,
      publisher: src.publisher || "", uri: src.uri || "", source_type: src.source_type || "",
      authority: src.authority || 0, published_at: src.published_at || "",
      licence: src.licence || "",
    };
  });
}

/* ---------------------------------------------------------------------------
 * The gates. Three of five can end a turn without an answer, and that is the
 * design: a system that cannot establish sufficient evidence must say so.
 * ------------------------------------------------------------------------- */

/* Deliberately conservative: refuse only when the question clearly matches a
 * declared out-of-scope topic AND matches it better than anything in scope. */
function scopeCheck(corpus, question) {
  const out = corpus.scope.out_of_scope || [];
  if (!out.length) return { inScope: true, inScore: 0, outScore: 0 };
  const terms = contentTerms(question);
  if (!terms.size) return { inScope: true, inScore: 0, outScore: 0 };
  const best = (topics) =>
    topics.reduce((m, t) => Math.max(m, coverage(contentTerms(t), terms)), 0);
  const outScore = best(out);
  const inScore = (corpus.scope.in_scope || []).length ? best(corpus.scope.in_scope) : 0;
  return { inScope: !(outScore >= 0.5 && outScore > inScore), inScore, outScore };
}

/* A request to hand back a source's own words, rather than an answer drawn
 * from them. Plain English: a property of the request, not of any domain. */
const VERBATIM_CUES = ["word for word", "verbatim", "reproduce", "quote", "exact text",
  "exact wording", "full text", "recite", "transcribe", "in full", "copy of the",
  "read out the"];

/* Gate 1b: is this a request to quote a source the licence forbids storing?
 *
 * Decided from the register rather than the evidence, and before retrieval,
 * because what may be reproduced is a property of the source and not of
 * whatever happened to match. Asking the evidence instead only ever finds the
 * secondary sources that mention the thing, and answering from those reads as
 * though the system had complied. */
function reproductionRefused(corpus, question) {
  const lowered = String(question || "").toLowerCase();
  if (!VERBATIM_CUES.some((cue) => lowered.includes(cue))) return null;
  const terms = contentTerms(question);
  if (!terms.size) return null;

  for (const source of Object.values(corpus.sources)) {
    if (source.mirrored !== false) continue;
    const titleTerms = contentTerms(source.title);
    for (const t of contentTerms(String(source.id).replace(/-/g, " "))) titleTerms.add(t);
    // Only a designation identifies a standard. Any-two-shared-words matched
    // a held source on generic vocabulary and refused a question the licence
    // permits, which is the more damaging direction of the two.
    const shared = [...terms].filter((t) => titleTerms.has(t));
    if (shared.some((t) => /\d/.test(t))) {
      return { title: source.title, licence: source.licence || "" };
    }
  }
  return null;
}

/* Fewer than three content terms and the question cannot have specified a
 * chemistry, a format or a condition. */
function specificEnough(question) {
  const n = contentTerms(question).size;
  return n >= 3 ? { ok: true, reason: "" } : { ok: false, reason: `it names only ${n} specific term(s)` };
}

function termWeights(questionTerms, passageTermSets) {
  const total = Math.max(1, passageTermSets.length);
  const w = new Map();
  for (const term of questionTerms) {
    const df = passageTermSets.reduce((a, p) => a + (p.has(term) ? 1 : 0), 0);
    w.set(term, Math.log(1 + total / (1 + df)));
  }
  return w;
}

/* Gate 4: does the evidence mention what the question is ABOUT? A term absent
 * from every passage carries the most weight, because it is the part of the
 * question the corpus cannot speak to. */
function coversTheSubject(corpus, question, evidence) {
  const terms = contentTerms(question);
  if (!terms.size || !evidence.length) return { ok: true, missing: [], share: 0 };
  const passages = evidence.map((e) => contentTerms(e.text));
  const w = termWeights(terms, passages);
  const totalWeight = [...w.values()].reduce((a, b) => a + b, 0);
  if (totalWeight <= 0) return { ok: true, missing: [], share: 0 };
  const missing = [...terms].filter((t) => !passages.some((p) => p.has(t))).sort();
  const share = missing.reduce((a, t) => a + w.get(t), 0) / totalWeight;
  return { ok: share < corpus.thresholds.max_missing_subject_weight, missing, share };
}

/* Gate 5: does the ANSWER address the question, or merely its topic? */
function answersTheQuestion(corpus, question, answerText, evidence) {
  const terms = contentTerms(question);
  const answerTerms = contentTerms(answerText);
  if (!terms.size || !answerTerms.size) return { ok: true, unaddressed: [], addressed: 0 };
  const passages = evidence.length ? evidence.map((e) => contentTerms(e.text)) : [answerTerms];
  const w = termWeights(terms, passages);
  const totalWeight = [...w.values()].reduce((a, b) => a + b, 0);
  if (totalWeight <= 0) return { ok: true, unaddressed: [], addressed: 0 };
  const unaddressed = [...terms].filter((t) => !answerTerms.has(t)).sort();
  const addressed = [...terms].filter((t) => answerTerms.has(t))
    .reduce((a, t) => a + w.get(t), 0) / totalWeight;
  return { ok: addressed >= corpus.thresholds.min_question_coverage, unaddressed, addressed };
}

/* ---------------------------------------------------------------------------
 * Answering: select and cite, never compose. Nothing here can state anything
 * that is not already a sentence in the corpus, which is why it cannot
 * hallucinate and why its prose is stitched rather than written.
 * ------------------------------------------------------------------------- */

const NUMERIC_CUES = ["what temperature", "how hot", "how many", "how much", "how long",
  "what pressure", "at what", "how often", "what is the limit", "threshold"];
const REQUIREMENT_CUES = ["must", "shall", "required", "permitted", "allowed", "mandatory",
  "obligation", "compliant", "legal"];
const NON_PROSE_RE = /(journal of|proceedings of|\bvol\.|\bpp\.|\bdoi:|et al\.,|\bissn\b|\bisbn\b|^fig\.|^figure \d|^table \d|^appendix\b|^abstract$|^\s*\[\d+\])/i;

/* "Is this a statement, or is it furniture?" Without it the engine answers
 * "What does the standard test?" with the page's own heading -- true, cited
 * and useless. */
function looksLikeProse(sentence) {
  sentence = sentence.trim();
  if (sentence.length < 40) return false;
  if (!(sentence.endsWith(".") || sentence.endsWith("?") || sentence.endsWith("!"))) return false;
  const words = sentence.split(/\s+/);
  if (words.length < 7) return false;
  const first = sentence[0];
  if (!(/[A-Z]/.test(first) || /[0-9]/.test(first))) return false;
  if (NON_PROSE_RE.test(sentence)) return false;
  const alpha = words.filter((w) => /^[A-Za-z]/.test(w));
  if (alpha.length && alpha.filter((w) => /^[A-Z]/.test(w)).length / alpha.length > 0.6) return false;
  return true;
}

/* A marker inside a selected sentence belongs to the source's own
 * bibliography, not to this answer's evidence list. Left in place it is
 * indistinguishable from a citation this engine made, and the validator
 * resolves it against ranks that do not exist. */
const SOURCE_FOOTNOTE_RE = /\s*\[\d+\]/g;

function stripSourceFootnotes(sentence) {
  return sentence.replace(SOURCE_FOOTNOTE_RE, "").trim();
}

const MIN_SENTENCE_SCORE = 0.30;
const MAX_SENTENCES = 5;

function scoreSentence(sentence, questionTerms, question, weights) {
  const sentenceTerms = contentTerms(sentence);
  if (!sentenceTerms.size) return 0;
  const matched = [...questionTerms].filter((t) => sentenceTerms.has(t));
  if (!matched.length) return 0;
  let totalWeight = 0;
  for (const t of questionTerms) totalWeight += weights.has(t) ? weights.get(t) : 1.0;
  if (totalWeight <= 0) return 0;
  let score = matched.reduce((a, t) => a + (weights.has(t) ? weights.get(t) : 1.0), 0) / totalWeight;
  score *= 1.0 / (1.0 + 0.003 * Math.max(0, sentence.length - 160));
  const lowered = question.toLowerCase();
  if (NUMERIC_CUES.some((c) => lowered.includes(c)) && numbersWithUnits(sentence).size) score *= 1.6;
  if (REQUIREMENT_CUES.some((c) => lowered.includes(c)) &&
      [" shall ", " must ", "required", "prohibited"].some((w) => sentence.toLowerCase().includes(w))) {
    score *= 1.4;
  }
  return score;
}

function extractiveAnswer(question, evidence) {
  const questionTerms = contentTerms(question);
  const passages = evidence.map((e) => contentTerms(e.text));
  const weights = termWeights(questionTerms, passages);

  const scored = [];
  for (const item of evidence) {
    // Stripped before splitting: a source writes "...oxygen).[1] A fire..."
    // with no space after the stop, so the splitter does not break there and
    // three claims would be emitted under one citation.
    sentences(stripSourceFootnotes(item.text)).forEach((sentence, position) => {
      sentence = sentence.trim();
      if (!looksLikeProse(sentence)) return;
      const score = scoreSentence(sentence, questionTerms, question, weights);
      if (score >= MIN_SENTENCE_SCORE) scored.push({ score, rank: item.rank, position, sentence });
    });
  }
  if (!scored.length) return { text: "", abstained: true };

  scored.sort((a, b) => b.score - a.score || a.rank - b.rank || a.position - b.position);

  /* Cap per-source contributions so one long passage cannot monopolise the
   * answer; a multi-document question should read from several sources. */
  const selected = [];
  const perRank = new Map();
  const seen = new Set();
  for (const s of scored) {
    const key = s.sentence.toLowerCase();
    if (seen.has(key)) continue;
    if ((perRank.get(s.rank) || 0) >= 2 && selected.length >= 2) continue;
    seen.add(key);
    perRank.set(s.rank, (perRank.get(s.rank) || 0) + 1);
    selected.push(s);
    if (selected.length >= MAX_SENTENCES) break;
  }
  selected.sort((a, b) => a.rank - b.rank || a.position - b.position);

  const body = selected.map((s) => `${stripSourceFootnotes(s.sentence)} [${s.rank}]`).join(" ");
  const cited = [...new Set(selected.map((s) => s.rank))].sort((a, b) => a - b);
  const reasoning = "This answer is assembled from the cited passages (" +
    cited.map((r) => `[${r}]`).join(", ") + ") without inference beyond them.";
  return { text: body, reasoning, abstained: false };
}

/* ---------------------------------------------------------------------------
 * Validation: the part that does not trust what produced the answer.
 * ------------------------------------------------------------------------- */

const META_RE = /\b(evidence (is|was) (not )?(sufficient|insufficient)|this answer is|the sources? (do not|does not|available)|i (cannot|can't|could not)|no (relevant )?evidence|not enough (evidence|information)|out of scope|assembled from the cited|without inference beyond|based on the cited (passages|evidence|sources))\b/i;
const SECTION_HEADING_RE = /^(ANSWER|REASONING|SOURCES|LIMITATIONS|CONFIDENCE)\b/i;
const QUOTED_RE = /"([^"]{4,120})"/g;
const APPARATUS_TERMS = new Set(["above","according","answer","below","citation","citations","cite",
  "cited","directly","document","drawn","evidence","excerpt","extract","given","passage","passages",
  "provided","question","quote","quoted","reference","referenced","section","shown","source",
  "sources","state","stated","states","taken","text"]);
const APPARATUS_SHARE = 0.5;

function stripCitations(text) {
  return String(text || "").replace(/\[(\d+)\]/g, " ");
}

function extractCitations(text) {
  return [...String(text || "").matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
}

function quotedSpans(text) {
  return new Set([...String(text || "").matchAll(QUOTED_RE)].map((m) => m[1].trim().toLowerCase()));
}

/* A sentence about the answer rather than about the world.
 *
 * The second test is statistical, and exists because a sentence that narrates
 * its own citations asserts nothing but fails grounding when graded as a
 * claim -- and on a short answer that is enough to drag a correct answer over
 * the unsupported threshold. Carrying a figure or a quotation exempts it:
 * "The source states the limit is 100 Wh" is still checked. */
function isMeta(sentence) {
  sentence = sentence.trim();
  if (META_RE.test(sentence) || SECTION_HEADING_RE.test(sentence)) return true;
  // Citation markers are numerals; counting "[1]" as a stated figure would
  // make every cited sentence look like it carries one.
  const bare = stripCitations(sentence);
  if (numbersWithUnits(bare).size || quotedSpans(bare).size) return false;
  const terms = contentTerms(bare);
  if (terms.size < 2) return false;
  let apparatus = 0;
  for (const t of terms) if (APPARATUS_TERMS.has(t) || APPARATUS_TERMS.has(t + "e")) apparatus += 1;
  return apparatus / terms.size >= APPARATUS_SHARE;
}

function stripTrailingUnit(n) {
  return n.replace(/[a-z\u00b0%]+$/, "");
}

/* Check an answer against the evidence it was given.
 *
 * A faithful port, and it has to be: an earlier looser version agreed with
 * Python on the questions that happened to come up and diverged the moment a
 * different sentence was selected. Citations carrying to following uncited
 * sentences, the three-term floor, and the all-evidence fallback for
 * connective prose are each load-bearing. */
function validateAnswer(answerText, evidence, minSupport) {
  const byRank = new Map(evidence.map((e) => [e.rank, e]));
  const allTerms = new Set();
  for (const item of evidence) for (const t of contentTerms(item.text)) allTerms.add(t);

  const checks = [];
  const citedRanks = new Set();
  let totalCitations = 0;
  let invalidCitations = 0;
  let carried = [];

  for (const raw of sentences(answerText)) {
    const stripped = raw.trim();
    if (!stripped || isMeta(stripped)) continue;

    const citations = extractCitations(stripped);
    totalCitations += citations.length;
    for (const c of citations) {
      if (byRank.has(c)) citedRanks.add(c);
      else invalidCitations += 1;
    }

    const bare = stripCitations(stripped).trim();
    const terms = contentTerms(bare);
    if (terms.size < 3) continue;      // too short to carry a checkable claim

    let effective = citations.filter((c) => byRank.has(c));
    if (!effective.length) effective = carried.filter((c) => byRank.has(c));
    if (citations.length) carried = citations;

    if (!effective.length) {
      // No citation, here or inherited. Connective prose built from evidence
      // vocabulary passes; the cited sentences around it carry the weight.
      const overlap = coverage(terms, allTerms);
      checks.push({ sentence: bare, supported: overlap >= minSupport, support: overlap });
      continue;
    }

    const citedTerms = new Set();
    let citedText = "";
    for (const rank of effective) {
      const item = byRank.get(rank);
      for (const t of contentTerms(item.text)) citedTerms.add(t);
      citedText += " " + item.text.toLowerCase();
    }

    const support = coverage(terms, citedTerms);
    let supported = support >= minSupport;

    // Numeric fidelity: a stated number must appear in the cited evidence.
    const stated = numbersWithUnits(bare);
    if (stated.size) {
      const evidenceNumbers = numbersWithUnits(citedText);
      const bareEvidence = new Set([...evidenceNumbers].map(stripTrailingUnit));
      for (const n of stated) {
        if (!evidenceNumbers.has(n) && !bareEvidence.has(stripTrailingUnit(n))) supported = false;
      }
    }
    // Quotation fidelity: a quoted span claims to be verbatim.
    for (const span of quotedSpans(bare)) {
      if (!citedText.includes(span)) supported = false;
    }

    checks.push({ sentence: bare, supported, support });
  }

  const claims = checks.length;
  const unsupported = checks.filter((c) => !c.supported);
  return {
    citedRanks,
    citationValidity: totalCitations ? (totalCitations - invalidCitations) / totalCitations : 1.0,
    groundedRatio: claims ? (claims - unsupported.length) / claims : 1.0,
    unsupportedRatio: claims ? unsupported.length / claims : 0.0,
    unsupported,
  };
}

/* ---------------------------------------------------------------------------
 * ask(): the gates in order, and the answer record they produce.
 *
 * Mirrors `Specialist.prepare` + `Specialist.judge`. Three of the five gates
 * can end the turn without an answer, and every one of those outcomes is a
 * success state rather than a failure.
 * ------------------------------------------------------------------------- */

const ANSWERED = "answered";
const INSUFFICIENT_EVIDENCE = "insufficient_evidence";
const OUT_OF_SCOPE = "out_of_scope";
const UNSUPPORTED = "unsupported";
const AMBIGUOUS = "ambiguous";
const IRRELEVANT = "irrelevant";
const ERROR = "error";

const TEXT = {
  insufficient: "The evidence in this knowledge area does not establish an answer to this " +
    "question. Rather than guess, the system is declining to answer.",
  outOfScope: (name, inScope) => `This question is outside the declared scope of ${name}. ` +
    `This knowledge area covers: ${inScope}.`,
  unsupported: "A draft answer was generated but failed automated grounding validation: too " +
    "much of it could not be traced to the retrieved evidence. It has been withheld rather " +
    "than published.",
  ambiguous: (reason) => "This question is under-specified, so any single figure would be " +
    `misleading: ${reason}. In this domain the answer depends on cell chemistry, cell ` +
    "format, state of charge, age and the test conditions. Please specify what you mean, " +
    "and the answer can be given against evidence rather than guessed.",
  /* The extractive engine's own abstention, which is what the reader is shown
   * when no sentence clears the selection threshold -- distinct from the
   * sufficiency gate's message, and Python returns the provider's words. */
  providerAbstention: "The available evidence is not sufficient to answer this question.",
  noReproduction: (title, licence) => `This asks for the text of ${title}, which ` +
    "this knowledge area registers as a pointer and does not store: " +
    `${licence || "its licence"}. Nothing of its wording is held here, so nothing ` +
    "of its wording can be returned. Other sources that describe it can be cited, " +
    "and the register records where the original can be obtained -- ask what it " +
    "requires rather than what it says, and the question becomes answerable.",
  noSubject: (terms) => `This knowledge area has nothing on: ${terms}. Passages were ` +
    "retrieved that share the question's general vocabulary, but none of them mentions what " +
    "was actually asked about, so any answer built from them would be about something else. " +
    "Rather than return that, the system is declining and naming the gap.",
  irrelevant: (terms) => "A draft answer was generated, correctly cited and fully traceable " +
    `to the evidence -- and it did not answer the question. It never engaged with: ${terms}. ` +
    "An answer about the right topic but the wrong subject is harder to catch than a wrong " +
    "one and no more useful, so it has been withheld.",
};

function limitationsFor(corpus, evidence, citedRanks, status) {
  const out = [];
  if (corpus.governance.disclaimer) out.push(corpus.governance.disclaimer);
  for (const l of corpus.governance.known_limitations || []) out.push(l);
  if (status === ANSWERED) {
    const sources = new Set(evidence.filter((e) => citedRanks.has(e.rank)).map((e) => e.source_id));
    if (sources.size === 1) {
      out.push("This answer rests on a single source; it has not been corroborated.");
    }
    if (evidence.some((e) => citedRanks.has(e.rank) && !e.published_at)) {
      out.push("Some cited sources carry no publication date, so their currency is unknown.");
    }
  }
  return out;
}

function confidenceFor(evidence, validation) {
  if (!evidence.length) return { label: "none", score: 0 };
  const cited = evidence.filter((e) => validation.citedRanks.has(e.rank));
  const pool = cited.length ? cited : evidence;
  const authority = pool.reduce((a, e) => a + (e.authority || 0), 0) / pool.length / 5.0;
  const corroboration = Math.min(1.0, new Set(pool.map((e) => e.source_id)).size / 2.0);
  const score = 0.35 * validation.groundedRatio + 0.25 * validation.citationValidity +
    0.20 * authority + 0.10 * corroboration + 0.10 * 0.5;
  if (score >= 0.75) return { label: "high", score };
  if (score >= 0.55) return { label: "medium", score };
  return { label: "low", score };
}

function ask(index, question) {
  const corpus = index.corpus;
  const base = {
    question, knowledge_area: corpus.knowledge_area,
    knowledge_version: corpus.version, evaluation_status: corpus.evaluation_status,
    engine: "browser", retrievers: corpus.retrievers,
    sources: [], limitations: [], unsupported_claims: [],
    citation_validity: 1, grounded_ratio: 1, confidence: "none", confidence_score: 0,
    reasoning: "",
  };
  question = String(question || "").trim();
  if (!question) return { ...base, status: ERROR, answer: "No question was provided." };

  // Gate 1 -- scope. Free, and runs before retrieval.
  const scope = scopeCheck(corpus, question);
  if (!scope.inScope) {
    return { ...base, status: OUT_OF_SCOPE,
      answer: TEXT.outOfScope(corpus.name, (corpus.scope.in_scope || []).join("; ") || "see the manifest"),
      limitations: limitationsFor(corpus, [], new Set(), OUT_OF_SCOPE) };
  }

  // Gate 1b -- reproduction, before retrieval for the same reason Python does
  // it there: the register decides, not the evidence.
  const refused = reproductionRefused(corpus, question);
  if (refused) {
    return { ...base, status: INSUFFICIENT_EVIDENCE,
      answer: TEXT.noReproduction(refused.title, refused.licence),
      limitations: limitationsFor(corpus, [], new Set(), INSUFFICIENT_EVIDENCE) };
  }

  const evidence = retrieve(index, question);
  const sourcesBlock = (citedRanks) => evidence.map((e) => ({ ...e, cited: citedRanks.has(e.rank) }));

  // Gate 2 -- specificity, before sufficiency, so an under-specified question
  // is named as such rather than reported as a gap in the corpus.
  const spec = specificEnough(question);
  if (!spec.ok) {
    return { ...base, status: AMBIGUOUS, answer: TEXT.ambiguous(spec.reason),
      sources: sourcesBlock(new Set()),
      limitations: limitationsFor(corpus, evidence, new Set(), AMBIGUOUS) };
  }

  // Gate 3 -- sufficiency.
  const topScore = evidence.length ? evidence[0].score : 0;
  if (evidence.length < corpus.thresholds.min_chunks || !evidence.length) {
    return { ...base, status: INSUFFICIENT_EVIDENCE, answer: TEXT.insufficient,
      sources: sourcesBlock(new Set()),
      limitations: limitationsFor(corpus, evidence, new Set(), INSUFFICIENT_EVIDENCE) };
  }

  // Gate 4 -- subject coverage.
  const subject = coversTheSubject(corpus, question, evidence);
  if (!subject.ok) {
    return { ...base, status: INSUFFICIENT_EVIDENCE,
      answer: TEXT.noSubject(subject.missing.join(", ") || "the subject of the question"),
      sources: sourcesBlock(new Set()), missing_terms: subject.missing,
      limitations: limitationsFor(corpus, evidence, new Set(), INSUFFICIENT_EVIDENCE) };
  }

  const draft = extractiveAnswer(question, evidence);
  if (draft.abstained || !draft.text) {
    return { ...base, status: INSUFFICIENT_EVIDENCE, answer: TEXT.providerAbstention,
      sources: sourcesBlock(new Set()),
      limitations: limitationsFor(corpus, evidence, new Set(), INSUFFICIENT_EVIDENCE) };
  }

  // Gate 6 -- grounding, which can downgrade a fluent answer to an abstention.
  const validation = validateAnswer(draft.text, evidence, corpus.thresholds.min_sentence_support);
  // Gate 5 -- responsiveness.
  const responsive = answersTheQuestion(corpus, question, draft.text, evidence);

  let status = ANSWERED;
  let body = draft.text;
  if (validation.unsupportedRatio > corpus.thresholds.max_unsupported_ratio) {
    status = UNSUPPORTED; body = TEXT.unsupported;
  } else if (!responsive.ok) {
    status = IRRELEVANT; body = TEXT.irrelevant(responsive.unaddressed.join(", ") || "what was asked about");
  }

  const conf = status === ANSWERED ? confidenceFor(evidence, validation) : { label: "none", score: 0 };
  return {
    ...base, status, answer: body,
    reasoning: status === ANSWERED ? draft.reasoning : "",
    sources: sourcesBlock(validation.citedRanks),
    citation_validity: validation.citationValidity,
    grounded_ratio: validation.groundedRatio,
    unsupported_claims: validation.unsupported.map((u) => ({ sentence: u.sentence })),
    confidence: conf.label, confidence_score: conf.score,
    limitations: limitationsFor(corpus, evidence, validation.citedRanks, status),
  };
}

const API = {
  STOPWORDS, tokenize, depossess, stem, contentTerms, coverage, sentences,
  numbersWithUnits, canonicalUnit, buildIndex, search, retrieve, ask,
  scopeCheck, specificEnough, coversTheSubject, answersTheQuestion,
  reproductionRefused,
  looksLikeProse, extractiveAnswer, validateAnswer,
};

if (typeof module !== "undefined" && module.exports) module.exports = API;
if (typeof window !== "undefined") window.FoundryEngine = API;
