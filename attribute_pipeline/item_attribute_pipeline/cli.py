from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import PipelineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract, normalize, embed, and cluster recommendation attributes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract-normalize", help="Run LLM extraction and phrase normalization."
    )
    extract.add_argument("--input", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--limit", type=int)
    extract.add_argument(
        "--workers",
        "--batch-size",
        dest="workers",
        type=int,
        default=1,
        help="Maximum number of concurrent DeepSeek API requests.",
    )
    extract.add_argument("--resume", action="store_true")
    extract.add_argument("--allow-partial", action="store_true")
    extract.add_argument("--max-retries", type=int, default=3)

    embed = subparsers.add_parser("embed", help="Embed normalized attributes with BGE.")
    embed.add_argument("--output", required=True, type=Path)
    embed.add_argument("--batch-size", type=int, default=64)
    embed.add_argument("--bge-model")
    embed.add_argument("--bge-model-path", type=Path)

    cluster = subparsers.add_parser(
        "cluster", help="Cluster embedded phrases and build sparse item mappings."
    )
    cluster.add_argument("--output", required=True, type=Path)
    cluster.add_argument("--threshold", type=float)

    implicit = subparsers.add_parser(
        "implicit",
        help="Find semantically similar non-explicit attributes for each item.",
    )
    implicit.add_argument("--input", required=True, type=Path)
    implicit.add_argument("--output", required=True, type=Path)
    implicit.add_argument("--top-k", type=int, default=10)
    implicit.add_argument("--batch-size", type=int, default=64)
    implicit.add_argument("--bge-model")
    implicit.add_argument("--bge-model-path", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "extract-normalize":
            from .extract import extract_and_normalize

            extract_and_normalize(
                input_path=args.input,
                output_dir=args.output,
                limit=args.limit,
                resume=args.resume,
                workers=args.workers,
                allow_partial=args.allow_partial,
                max_retries=args.max_retries,
            )
        elif args.command == "embed":
            from .embed import embed_attributes

            embed_attributes(
                output_dir=args.output,
                batch_size=args.batch_size,
                model_name=args.bge_model,
                model_path=args.bge_model_path,
            )
        elif args.command == "cluster":
            from .cluster import cluster_attributes

            cluster_attributes(
                output_dir=args.output,
                threshold=args.threshold,
            )
        elif args.command == "implicit":
            from .implicit import build_implicit_attributes

            build_implicit_attributes(
                input_path=args.input,
                output_dir=args.output,
                top_k=args.top_k,
                batch_size=args.batch_size,
                model_name=args.bge_model,
                model_path=args.bge_model_path,
            )
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0
