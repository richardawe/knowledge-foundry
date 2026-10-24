// Requstory map <-> foundry claims.
//
// `RequstoryMap` mirrors what a published company page shows (bands of units,
// a customer journey, research findings). If Requstory's internal map type is
// shaped differently, change `mapToClaims` only — the rest of this file works
// on claim ids, not on the map's shape.
import type { CheckedClaim, ResearchResult, Support, Verdict, Visibility } from "./types";

export interface RequstoryMap {
  summary?: string;
  bands: { band: string; units: { id?: string; name: string; found_on_website: boolean }[] }[];
  journey: { stage: string; quote?: string | null }[];
  findings: { id?: string; title: string; description?: string }[];
}

/** Every checkable line of a map, with an id Requstory can find it by again. */
export function mapToClaims(map: RequstoryMap) {
  const claims: { id: string; text: string; section: string; origin: string }[] = [];
  if (map.summary) claims.push({ id: "summary", text: map.summary, section: "Summary", origin: "from the website" });
  map.bands.forEach((band, b) => band.units.forEach((unit, u) => claims.push({
    id: unit.id ?? `unit:${b}:${u}`, text: unit.name, section: band.band,
    origin: unit.found_on_website ? "found on the website" : "typical for the industry; not confirmed",
  })));
  map.journey.forEach((stage, j) => {
    if (stage.quote) claims.push({ id: `journey:${j}`, text: stage.quote.replace(/^[“"']+|[”"']+$/g, ""),
                                    section: "Customer journey", origin: `quoted at '${stage.stage}'` });
  });
  map.findings.forEach((finding, f) => claims.push({
    id: finding.id ?? `finding:${f}`, text: finding.title, section: "What research found", origin: "research",
  }));
  return claims;
}

export type Viewer = "public" | "claimant";

/** What a viewer may see of one checked line. Third-party evidence never reaches
 *  a public viewer; a verdict that rests only on it is shown as the next-best
 *  verdict the public evidence supports. */
export function viewOf(claim: CheckedClaim, viewer: Viewer): { verdict: Verdict; support: Support[] } {
  const allowed = (v: Visibility) => viewer === "claimant" || v === "public";
  const support = claim.support.filter((s) => allowed(s.visibility));
  if (claim.verdict === "corroborated" && !support.some((s) => s.source_type !== "self_published")) {
    return { verdict: support.length ? "self_reported" : "not_found", support };
  }
  return { verdict: claim.verdict, support };
}

/** How a map line should be drawn. Green only when evidence the viewer may see backs it. */
export function edgeFor(verdict: Verdict): "green" | "amber" | "grey" {
  return verdict === "corroborated" || verdict === "self_reported" ? "green"
       : verdict === "label_only" ? "amber" : "grey";
}

/** The facts a claim-flow screen needs: what to confirm, what was found. */
export function claimFlowPrompts(result: ResearchResult) {
  const held = Object.values(result.identity).flatMap((r) =>
    (r.entities ?? []).filter((e) => !e.corroborated).map((e) => ({ register: r.register, ...e })));
  return {
    confirmIdentity: held,       // "Is this you?" — confirmed ones become evidence on the next run
    registers: result.registers, // public: legal name, number, office, officers, status
    site: result.site,           // public: contact routes, sitemap, llms.txt, pages read
  };
}

export function byId(result: ResearchResult): Map<string, CheckedClaim> {
  return new Map(result.claims.map((c) => [c.id, c]));
}
