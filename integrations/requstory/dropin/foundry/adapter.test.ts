// node --test (after compiling), or with tsx: npx tsx --test foundry/adapter.test.ts
import { test } from "node:test";
import * as assert from "node:assert/strict";
import { edgeFor, mapToClaims, viewOf } from "./adapter";
import type { CheckedClaim } from "./types";

const claim: CheckedClaim = {
  id: "x", text: "Session replay", verdict: "corroborated", note: "",
  support: [
    { source_id: "news-01", source_type: "press", visibility: "claimant_only", title: "t", uri: "u", quote: "Analysts say…", support: 1 },
    { source_id: "site-01", source_type: "self_published", visibility: "public", title: "t", uri: "u", quote: "We offer…", support: 1 },
  ],
};

test("a public viewer never sees third-party evidence", () => {
  const view = viewOf(claim, "public");
  assert.ok(view.support.every((s) => s.visibility === "public"));
  assert.equal(view.verdict, "self_reported"); // corroborated only by the press, which they may not see
});

test("the claimant sees everything", () => {
  const view = viewOf(claim, "claimant");
  assert.equal(view.verdict, "corroborated");
  assert.equal(view.support.length, 2);
});

test("map lines become claims with stable ids", () => {
  const claims = mapToClaims({
    summary: "S", bands: [{ band: "B", units: [{ name: "U", found_on_website: true }] }],
    journey: [{ stage: "J", quote: "“Book a demo”" }], findings: [{ title: "F" }],
  });
  assert.deepEqual(claims.map((c) => c.id), ["summary", "unit:0:0", "journey:0", "finding:0"]);
  assert.equal(claims[2].text, "Book a demo");
  assert.equal(edgeFor("label_only"), "amber");
});
