"""``foundry`` -- the command line for the knowledge factory.

Every stage of the loop in PROJECT_INITIATION.md §21 is one subcommand, and
they compose: ``build`` is ``ingest`` then ``index`` then ``version``. Nothing
here is knowledge-area specific.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import paths
from .manifest import Manifest, ManifestError, list_knowledge_areas


def _load(ka_id: str) -> Manifest:
    try:
        return Manifest.load(ka_id)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _store(manifest: Manifest):
    from .storage import Store

    return Store(manifest.id)


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))


# -- commands -----------------------------------------------------------


def cmd_list(args) -> int:
    ids = list_knowledge_areas()
    if not ids:
        print("no knowledge areas found in", paths.knowledge_areas_dir())
        return 0
    for ka_id in ids:
        manifest = Manifest.load(ka_id)
        from .storage import Store

        with Store(manifest.id) as store:
            counts = store.counts()
            version = store.get_meta("current_version", "unbuilt")
            status = store.get_meta("evaluation_status", "unknown")
        print(
            f"{manifest.id:<28} {version:<10} {manifest.governance.review_status:<9} "
            f"eval={status:<8} sources={counts['sources']:<4} chunks={counts['chunks']}"
        )
    return 0


def cmd_ingest(args) -> int:
    from .ingest import ingest_knowledge_area

    manifest = _load(args.knowledge_area)
    with _store(manifest) as store:
        report = ingest_knowledge_area(
            manifest,
            store,
            force=args.force,
            rebuild=args.rebuild,
            offline=args.offline,
            only=args.only.split(",") if args.only else None,
        )
    data = report.as_dict()
    if args.json:
        _emit(data, True)
    else:
        print(
            f"ingested {manifest.id}: {report.indexed} indexed, "
            f"{report.unchanged} unchanged, {report.skipped_fresh} fresh, "
            f"{report.pointers} pointers, {len(report.failures)} failed"
        )
        print(f"  documents={report.documents} chunks={report.chunks} claims={report.claims}")
        for failure in report.failures:
            print(f"  FAILED {failure['source']} at {failure['stage']}: {failure['error']}")
    return 1 if report.failures and args.strict else 0


def cmd_index(args) -> int:
    from .retrieval import build_semantic_index

    manifest = _load(args.knowledge_area)
    with _store(manifest) as store:
        report = build_semantic_index(manifest, store)
    if args.json:
        _emit(report.as_dict(), True)
    else:
        print(
            f"indexed {manifest.id}: {report.chunks_encoded} chunks "
            f"with {report.model} (dim {report.dim})"
            + (f" [{report.skipped}]" if report.skipped else "")
        )
    return 0


def cmd_build(args) -> int:
    """Ingest, index and cut a draft version -- the whole build in one command."""
    from .ingest import ingest_knowledge_area
    from .retrieval import build_semantic_index
    from .versioning import cut_version

    manifest = _load(args.knowledge_area)
    with _store(manifest) as store:
        ingest = ingest_knowledge_area(
            manifest, store, force=args.force, rebuild=args.rebuild, offline=args.offline
        )
        index = build_semantic_index(manifest, store)
        version = cut_version(manifest, store, notes=args.notes or "")
    if args.json:
        _emit(
            {"ingest": ingest.as_dict(), "index": index.as_dict(), "version": version.as_dict()},
            True,
        )
    else:
        print(
            f"built {manifest.id} -> {version.version} "
            f"({ingest.indexed} sources indexed, {index.chunks_encoded} chunks embedded)"
        )
        for failure in ingest.failures:
            print(f"  FAILED {failure['source']}: {failure['error']}")
    return 1 if ingest.failures and args.strict else 0


def cmd_ask(args) -> int:
    from .specialist import Specialist

    manifest = _load(args.knowledge_area)
    with _store(manifest) as store:
        answer = Specialist(manifest, store).ask(args.question, top_k=args.top_k)
    if args.json:
        _emit(answer.as_dict(), True)
    else:
        print(answer.render())
    return 0


def cmd_eval(args) -> int:
    from .evaluation import run_evaluation

    manifest = _load(args.knowledge_area)
    with _store(manifest) as store:
        report = run_evaluation(manifest, store, suites=args.suite, limit=args.limit)
    if args.json:
        _emit(report.as_dict(), True)
    else:
        print(report.render())
    return 0 if report.passed else 1


def cmd_redteam(args) -> int:
    from .redteam import run_redteam

    manifest = _load(args.knowledge_area)
    with _store(manifest) as store:
        report = run_redteam(manifest, store, count=args.count, promote=args.promote)
    if args.json:
        _emit(report.as_dict(), True)
    else:
        print(report.render())
    return 0


def cmd_version(args) -> int:
    from .versioning import list_versions, publish_version, rollback_version

    manifest = _load(args.knowledge_area)
    with _store(manifest) as store:
        if args.action == "list":
            rows = list_versions(store)
            for row in rows:
                marker = "*" if row["status"] == "published" else " "
                print(f"{marker} {row['version']:<12} {row['status']:<12} {row['created_at']}")
            return 0
        if args.action == "publish":
            result = publish_version(manifest, store, version=args.version, force=args.force)
            print(result.render())
            return 0 if result.published else 1
        if args.action == "rollback":
            result = rollback_version(store, version=args.version)
            print(result)
            return 0
    return 0


def cmd_challenge(args) -> int:
    from .feedback import add_challenge, list_challenges, promote_challenge

    manifest = _load(args.knowledge_area)
    with _store(manifest) as store:
        if args.action == "list":
            for row in list_challenges(store, status=args.status):
                print(f"{row['id']:<14} {row['status']:<10} {row['question'][:70]}")
            return 0
        if args.action == "add":
            challenge = add_challenge(
                manifest, store, question=args.question,
                claimed_problem=args.problem or "", submitted_by=args.by or "cli",
                expected=args.expected or "",
            )
            print(f"recorded challenge {challenge}")
            return 0
        if args.action == "promote":
            path = promote_challenge(manifest, store, args.id, expected=args.expected)
            print(f"promoted {args.id} into {path}")
            return 0
    return 0


def cmd_stats(args) -> int:
    manifest = _load(args.knowledge_area)
    from .evidence import knowledge_area_report

    with _store(manifest) as store:
        report = knowledge_area_report(manifest, store)
    if args.json:
        _emit(report, True)
    else:
        print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_serve(args) -> int:
    from .web import serve

    serve(host=args.host, port=args.port)
    return 0


# -- parser -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundry",
        description="Knowledge Foundry: build, test and serve specialist knowledge systems.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list knowledge areas").set_defaults(func=cmd_list)

    def add_ka(p):
        p.add_argument("knowledge_area", help="knowledge area id or path")
        p.add_argument("--json", action="store_true", help="emit JSON")
        return p

    ingest = add_ka(sub.add_parser("ingest", help="fetch and index sources"))
    ingest.add_argument("--force", action="store_true", help="re-fetch ignoring the refresh window")
    ingest.add_argument("--rebuild", action="store_true", help="re-extract even if content is unchanged")
    ingest.add_argument("--offline", action="store_true", help="skip sources needing the network")
    ingest.add_argument("--only", help="comma-separated source ids")
    ingest.add_argument("--strict", action="store_true", help="exit non-zero if any source fails")
    ingest.set_defaults(func=cmd_ingest)

    index = add_ka(sub.add_parser("index", help="build the semantic index"))
    index.set_defaults(func=cmd_index)

    build = add_ka(sub.add_parser("build", help="ingest + index + cut a draft version"))
    build.add_argument("--force", action="store_true")
    build.add_argument("--rebuild", action="store_true")
    build.add_argument("--offline", action="store_true")
    build.add_argument("--strict", action="store_true")
    build.add_argument("--notes", help="version notes")
    build.set_defaults(func=cmd_build)

    ask = add_ka(sub.add_parser("ask", help="ask the specialist"))
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    ask.set_defaults(func=cmd_ask)

    evaluate = add_ka(sub.add_parser("eval", help="run the evaluation suites"))
    evaluate.add_argument("--suite", action="append", help="suite file (repeatable)")
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.set_defaults(func=cmd_eval)

    redteam = add_ka(sub.add_parser("redteam", help="run adversarial evaluation"))
    redteam.add_argument("--count", type=int, default=40)
    redteam.add_argument("--promote", action="store_true", help="write failures into the regression suite")
    redteam.set_defaults(func=cmd_redteam)

    version = add_ka(sub.add_parser("version", help="list, publish or roll back versions"))
    version.add_argument("action", choices=["list", "publish", "rollback"])
    version.add_argument("--version", help="explicit version")
    version.add_argument("--force", action="store_true", help="publish despite a failing gate")
    version.set_defaults(func=cmd_version)

    challenge = add_ka(sub.add_parser("challenge", help="expert challenges (§14)"))
    challenge.add_argument("action", choices=["list", "add", "promote"])
    challenge.add_argument("--id")
    challenge.add_argument("--question")
    challenge.add_argument("--problem")
    challenge.add_argument("--expected")
    challenge.add_argument("--by")
    challenge.add_argument("--status")
    challenge.set_defaults(func=cmd_challenge)

    stats = add_ka(sub.add_parser("stats", help="knowledge area statistics"))
    stats.set_defaults(func=cmd_stats)

    serve = sub.add_parser("serve", help="serve the public evidence pages and ask API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
