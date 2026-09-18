"""Graph retrieval over the ontology (§8).

Entity -> relationship -> entity, one or two hops, expressed as a table and a
couple of joins rather than a graph database (§23 explicitly rules out
"complicated knowledge graphs").

Its job is query expansion with domain structure. Asked "what detects thermal
runaway", it walks ``thermal_runaway --detected_by--> {off_gas_detection,
temperature_anomaly}`` and searches for those terms, which no amount of lexical
or vector similarity would have surfaced from the question's own wording.
"""

from __future__ import annotations

import json
from typing import Sequence

from ..storage import Store
from ..text import tokenize
from .base import Candidate
from .keyword import KeywordRetriever


class GraphRetriever:
    name = "graph"

    def __init__(self, store: Store, max_hops: int = 2) -> None:
        self.store = store
        self.max_hops = max(1, max_hops)
        self._keyword = KeywordRetriever(store)
        self._entities: list[tuple[str, str, list[str]]] | None = None

    def _entity_index(self) -> list[tuple[str, str, list[str]]]:
        """(id, name, surface forms) for every ontology entity."""
        if self._entities is None:
            index = []
            for row in self.store.entities():
                aliases = json.loads(row["aliases"] or "[]")
                surfaces = [row["name"].lower()] + [a.lower() for a in aliases]
                index.append((row["id"], row["name"], surfaces))
            self._entities = index
        return self._entities

    def match_entities(self, query: str) -> list[str]:
        """Entities mentioned in the query, by name or alias."""
        lowered = " " + " ".join(tokenize(query)) + " "
        hits = []
        for entity_id, _name, surfaces in self._entity_index():
            for surface in surfaces:
                needle = " " + " ".join(tokenize(surface)) + " "
                if needle.strip() and needle in lowered:
                    hits.append(entity_id)
                    break
        return hits

    def expand(self, entity_ids: Sequence[str]) -> list[tuple[str, str, str]]:
        """Walk outward up to ``max_hops``, returning (name, predicate, direction)."""
        seen = set(entity_ids)
        frontier = list(entity_ids)
        expansions: list[tuple[str, str, str]] = []

        for _hop in range(self.max_hops):
            if not frontier:
                break
            placeholders = ", ".join("?" * len(frontier))
            rows = self.store.conn.execute(
                f"""
                SELECT t.predicate, t.subject_id, t.object_id,
                       es.name AS subject_name, eo.name AS object_name
                FROM triples t
                JOIN entities es ON es.id = t.subject_id
                JOIN entities eo ON eo.id = t.object_id
                WHERE t.subject_id IN ({placeholders}) OR t.object_id IN ({placeholders})
                """,
                (*frontier, *frontier),
            ).fetchall()
            next_frontier: list[str] = []
            for row in rows:
                for other_id, other_name, direction in (
                    (row["object_id"], row["object_name"], "out"),
                    (row["subject_id"], row["subject_name"], "in"),
                ):
                    if other_id in seen:
                        continue
                    seen.add(other_id)
                    next_frontier.append(other_id)
                    expansions.append((other_name, row["predicate"], direction))
            frontier = next_frontier
        return expansions

    def search(self, query: str, limit: int = 40) -> Sequence[Candidate]:
        matched = self.match_entities(query)
        if not matched:
            return []
        expansions = self.expand(matched)
        if not expansions:
            return []

        # Search on the expanded vocabulary, not the original question: the
        # graph's contribution is the terms the question did not contain.
        expansion_query = " ".join(name for name, _, _ in expansions[:20])
        candidates = self._keyword.search(expansion_query, limit=limit)

        terms = ", ".join(f"{name} ({predicate})" for name, predicate, _ in expansions[:5])
        out = []
        for candidate in candidates:
            out.append(
                Candidate(
                    chunk_id=candidate.chunk_id,
                    score=candidate.score,
                    retriever=self.name,
                    text=candidate.text,
                    section=candidate.section,
                    source_id=candidate.source_id,
                    explain=f"via ontology: {terms}",
                )
            )
        return out
