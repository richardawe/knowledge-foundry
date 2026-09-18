"""Source ingestion: fetch, cache, extract, clean, chunk, index (§7)."""

from .pipeline import IngestReport, ingest_knowledge_area, load_source_register

__all__ = ["IngestReport", "ingest_knowledge_area", "load_source_register"]
