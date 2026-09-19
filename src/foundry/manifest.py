"""Knowledge-area manifest: the schema every knowledge area must satisfy.

Implements PROJECT_INITIATION.md §6. The manifest is the contract between the
factory and a knowledge area -- it is the only thing the factory needs in order
to build, evaluate, version and serve a domain it has never seen before.

Validation is strict and the errors are specific, because a manifest typo that
silently degrades retrieval is far more expensive than a loud failure at load.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from . import paths

VISIBILITIES = ("public", "private", "enterprise")
REVIEW_STATUSES = ("draft", "in_review", "reviewed", "published", "deprecated")
CONFIDENCE_LEVELS = ("low", "medium", "high")


class ManifestError(ValueError):
    """Raised when a manifest is missing or internally inconsistent."""


def _require(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ManifestError(f"{where}: missing required field '{key}'")
    return mapping[key]


def _one_of(value: Any, allowed: tuple, where: str) -> Any:
    if value not in allowed:
        raise ManifestError(f"{where}: {value!r} is not one of {allowed}")
    return value


@dataclass
class ChunkConfig:
    target_chars: int = 1200
    overlap_chars: int = 150
    min_chars: int = 200

    def validate(self) -> None:
        if self.target_chars < 200:
            raise ManifestError("retrieval.chunk.target_chars must be >= 200")
        if self.overlap_chars >= self.target_chars:
            raise ManifestError("retrieval.chunk.overlap_chars must be < target_chars")


@dataclass
class RetrievalConfig:
    """Hybrid retrieval settings (§8).

    Weights live here rather than in code so a knowledge area can be tuned
    without a deploy -- and so that any change to them shows up in the nightly
    retrieval metrics.
    """

    top_k: int = 8
    candidate_k: int = 40
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    keyword_enabled: bool = True
    semantic_enabled: bool = True
    structured_enabled: bool = True
    graph_enabled: bool = True
    graph_max_hops: int = 2
    semantic_provider: str = "lsa"
    semantic_model: str | None = None
    semantic_dim: int = 192
    fusion_k: int = 60
    fusion_weights: dict[str, float] = field(
        default_factory=lambda: {
            "keyword": 1.0,
            "semantic": 1.0,
            "structured": 0.6,
            "graph": 0.5,
        }
    )
    authority_weight: float = 0.15
    recency_half_life_days: int = 1460
    # Extra query cues mapping a question to a source type, e.g.
    # {"standard": ["nfpa", "iec"]}. Domain vocabulary, so it lives here.
    structured_cues: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalConfig":
        data = data or {}
        chunk = ChunkConfig(**(data.get("chunk") or {}))
        semantic = data.get("semantic") or {}
        keyword = data.get("keyword") or {}
        structured = data.get("structured") or {}
        graph = data.get("graph") or {}
        fusion = data.get("fusion") or {}
        rerank = data.get("rerank") or {}
        cfg = cls(
            top_k=int(data.get("top_k", 8)),
            candidate_k=int(data.get("candidate_k", 40)),
            chunk=chunk,
            keyword_enabled=bool(keyword.get("enabled", True)),
            semantic_enabled=bool(semantic.get("enabled", True)),
            structured_enabled=bool(structured.get("enabled", True)),
            graph_enabled=bool(graph.get("enabled", True)),
            graph_max_hops=int(graph.get("max_hops", 2)),
            semantic_provider=semantic.get("provider", "lsa"),
            semantic_model=semantic.get("model"),
            semantic_dim=int(semantic.get("dim", 192)),
            fusion_k=int(fusion.get("k", 60)),
            fusion_weights={**cls().fusion_weights, **(fusion.get("weights") or {})},
            authority_weight=float(rerank.get("authority_weight", 0.15)),
            recency_half_life_days=int(rerank.get("recency_half_life_days", 1460)),
            structured_cues={
                str(k): [str(v) for v in vals]
                for k, vals in (structured.get("cues") or {}).items()
            },
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.chunk.validate()
        if self.top_k < 1:
            raise ManifestError("retrieval.top_k must be >= 1")
        if self.candidate_k < self.top_k:
            raise ManifestError("retrieval.candidate_k must be >= top_k")
        if not any(
            (
                self.keyword_enabled,
                self.semantic_enabled,
                self.structured_enabled,
                self.graph_enabled,
            )
        ):
            raise ManifestError("retrieval: at least one retriever must be enabled")
        for name, weight in self.fusion_weights.items():
            if weight < 0:
                raise ManifestError(f"retrieval.fusion.weights.{name} must be >= 0")


@dataclass
class SpecialistConfig:
    """Specialist agent configuration (§9)."""

    prompt_path: str = "prompts/system.md"
    llm_provider: str = "local_extractive"
    llm_model: str | None = None
    # When the named provider is unreachable, answer with the local provider
    # and record that it happened, rather than failing the whole run.
    llm_fallback: bool = True
    temperature: float = 0.0
    max_tokens: int = 1200
    # Sufficiency gate: below this, the agent abstains rather than guesses.
    min_evidence_score: float = 0.10
    min_chunks: int = 1
    # Subject coverage: abstain when question terms absent from every retrieved
    # passage carry at least this share of the question's weight. 1.0 disables
    # the gate, which is what a knowledge area wanting the old behaviour sets.
    max_missing_subject_weight: float = 0.5
    # Responsiveness: the answer must engage with at least this share of the
    # question's weighted content. 0.0 disables the gate.
    min_question_coverage: float = 0.3
    # Grounding: a sentence is supported if this much of its content terms
    # appear in a cited chunk.
    min_sentence_support: float = 0.34
    # If more than this fraction of claim sentences are unsupported, the answer
    # is downgraded to an abstention rather than published.
    max_unsupported_ratio: float = 0.34

    @classmethod
    def from_dict(cls, data: dict) -> "SpecialistConfig":
        data = data or {}
        llm = data.get("llm") or {}
        suff = data.get("sufficiency") or {}
        ground = data.get("grounding") or {}
        cfg = cls(
            prompt_path=data.get("prompt", "prompts/system.md"),
            llm_provider=llm.get("provider", "local_extractive"),
            llm_model=llm.get("model"),
            llm_fallback=bool(llm.get("fallback", True)),
            temperature=float(llm.get("temperature", 0.0)),
            max_tokens=int(llm.get("max_tokens", 1200)),
            min_evidence_score=float(suff.get("min_evidence_score", 0.10)),
            min_chunks=int(suff.get("min_chunks", 1)),
            max_missing_subject_weight=float(
                suff.get("max_missing_subject_weight", 0.5)
            ),
            min_question_coverage=float(
                (data.get("responsiveness") or {}).get("min_question_coverage", 0.3)
            ),
            min_sentence_support=float(ground.get("min_sentence_support", 0.34)),
            max_unsupported_ratio=float(ground.get("max_unsupported_ratio", 0.34)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not 0.0 <= self.min_sentence_support <= 1.0:
            raise ManifestError("specialist.grounding.min_sentence_support must be in [0,1]")
        if not 0.0 <= self.max_unsupported_ratio <= 1.0:
            raise ManifestError("specialist.grounding.max_unsupported_ratio must be in [0,1]")
        if self.min_chunks < 0:
            raise ManifestError("specialist.sufficiency.min_chunks must be >= 0")
        if not 0.0 <= self.min_question_coverage <= 1.0:
            raise ManifestError(
                "specialist.responsiveness.min_question_coverage must be in [0,1]"
            )
        if not 0.0 < self.max_missing_subject_weight <= 1.0:
            raise ManifestError(
                "specialist.sufficiency.max_missing_subject_weight must be in (0,1]"
            )


@dataclass
class EvaluationConfig:
    """Evaluation and regression settings (§11, §12)."""

    suites: list[str] = field(default_factory=list)
    # A build must clear these absolute floors to be publishable.
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "answer_correctness": 0.60,
            "citation_validity": 0.95,
            "unsupported_claim_rate": 0.15,
            "abstention_correctness": 0.70,
        }
    )
    # And must not regress against the last published version by more than this.
    tolerances: dict[str, float] = field(default_factory=lambda: {"default": 0.02})

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationConfig":
        data = data or {}
        base = cls()
        return cls(
            suites=list(data.get("suites") or []),
            thresholds={**base.thresholds, **(data.get("thresholds") or {})},
            tolerances={**base.tolerances, **(data.get("tolerances") or {})},
        )

    def tolerance_for(self, metric: str) -> float:
        return float(self.tolerances.get(metric, self.tolerances.get("default", 0.02)))


@dataclass
class GovernanceConfig:
    """Governance (§6) -- and the honesty surface of the public page (§15)."""

    review_status: str = "draft"
    confidence: str = "low"
    known_limitations: list[str] = field(default_factory=list)
    disclaimer: str = ""
    reviewers: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "GovernanceConfig":
        data = data or {}
        cfg = cls(
            review_status=data.get("review_status", "draft"),
            confidence=data.get("confidence", "low"),
            known_limitations=list(data.get("known_limitations") or []),
            disclaimer=data.get("disclaimer", ""),
            reviewers=list(data.get("reviewers") or []),
        )
        _one_of(cfg.review_status, REVIEW_STATUSES, "governance.review_status")
        _one_of(cfg.confidence, CONFIDENCE_LEVELS, "governance.confidence")
        return cfg


@dataclass
class Scope:
    """Declared boundaries of the knowledge area.

    Used by the specialist's scope gate to refuse out-of-scope questions before
    retrieval -- the cheapest defence against §13's "questions outside the
    knowledge base".
    """

    in_scope: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Scope":
        data = data or {}
        return cls(
            in_scope=list(data.get("in_scope") or []),
            out_of_scope=list(data.get("out_of_scope") or []),
        )


@dataclass
class Manifest:
    """A complete knowledge-area definition."""

    id: str
    name: str
    description: str
    domain: str
    version: str
    owner: str
    created: str
    last_updated: str
    visibility: str = "public"
    tenant: str = "public"
    scope: Scope = field(default_factory=Scope)
    sources_register: str = "sources.yaml"
    ontology_path: str | None = "ontology.yaml"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    specialist: SpecialistConfig = field(default_factory=SpecialistConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    directory: Path | None = None
    raw: dict = field(default_factory=dict, repr=False)

    # -- loading ---------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict, directory: Path | None = None) -> "Manifest":
        if not isinstance(data, dict):
            raise ManifestError("manifest must be a mapping")
        ka = data.get("knowledge_area")
        if not isinstance(ka, dict):
            raise ManifestError("manifest: missing top-level 'knowledge_area' block")

        for key in ("id", "name", "description", "domain", "version", "owner"):
            _require(ka, key, "knowledge_area")

        visibility = _one_of(
            ka.get("visibility", "public"), VISIBILITIES, "knowledge_area.visibility"
        )
        sources = data.get("sources") or {}
        knowledge_model = data.get("knowledge_model") or {}
        today = _dt.date.today().isoformat()

        manifest = cls(
            id=str(ka["id"]),
            name=str(ka["name"]),
            description=str(ka["description"]),
            domain=str(ka["domain"]),
            version=str(ka["version"]),
            owner=str(ka["owner"]),
            created=str(ka.get("created", today)),
            last_updated=str(ka.get("last_updated", today)),
            visibility=visibility,
            tenant=str(ka.get("tenant", "public")),
            scope=Scope.from_dict(data.get("scope")),
            sources_register=sources.get("register", "sources.yaml"),
            ontology_path=knowledge_model.get("ontology", "ontology.yaml"),
            retrieval=RetrievalConfig.from_dict(data.get("retrieval")),
            specialist=SpecialistConfig.from_dict(data.get("specialist")),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation")),
            governance=GovernanceConfig.from_dict(data.get("governance")),
            directory=directory,
            raw=data,
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, ka_id_or_path: str | Path) -> "Manifest":
        """Load by knowledge-area id or by path to a directory or manifest file."""
        candidate = Path(ka_id_or_path)
        if candidate.is_file():
            path, directory = candidate, candidate.parent
        elif candidate.is_dir():
            path, directory = candidate / "manifest.yaml", candidate
        else:
            directory = paths.knowledge_area_dir(str(ka_id_or_path))
            path = directory / "manifest.yaml"
        if not path.is_file():
            raise ManifestError(f"no manifest found for {ka_id_or_path!r} (looked in {path})")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data, directory=directory)

    def validate(self) -> None:
        if not self.id.replace("-", "").replace("_", "").isalnum():
            raise ManifestError(
                f"knowledge_area.id {self.id!r} must be alphanumeric with - or _"
            )
        if self.directory is not None and self.directory.name != self.id:
            raise ManifestError(
                f"knowledge_area.id {self.id!r} does not match its directory "
                f"{self.directory.name!r}; the id is the primary key everywhere"
            )
        self.retrieval.validate()
        self.specialist.validate()

    # -- derived ---------------------------------------------------------

    def resolve(self, relative: str) -> Path:
        base = self.directory or paths.knowledge_area_dir(self.id)
        return base / relative

    @property
    def sources_path(self) -> Path:
        return self.resolve(self.sources_register)

    @property
    def system_prompt_path(self) -> Path:
        return self.resolve(self.specialist.prompt_path)

    def system_prompt(self) -> str:
        path = self.system_prompt_path
        if not path.is_file():
            raise ManifestError(f"system prompt not found at {path}")
        return path.read_text(encoding="utf-8").strip()

    def content_hash(self) -> str:
        """Stable hash of the configuration that affects answers.

        Feeds the version identity (§10 of the plan): rebuilding identical
        inputs must yield an identical version.
        """
        payload = {
            "id": self.id,
            "scope": asdict(self.scope),
            "retrieval": asdict(self.retrieval),
            "specialist": asdict(self.specialist),
            "evaluation": asdict(self.evaluation),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "version": self.version,
            "owner": self.owner,
            "visibility": self.visibility,
            "tenant": self.tenant,
            "review_status": self.governance.review_status,
            "confidence": self.governance.confidence,
        }


def list_knowledge_areas() -> list[str]:
    root = paths.knowledge_areas_dir()
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and (d / "manifest.yaml").is_file()
    )
