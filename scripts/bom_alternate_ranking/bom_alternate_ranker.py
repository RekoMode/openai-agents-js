#!/usr/bin/env python3
"""BOM alternate-part ranker.

Pipeline:
  1. fetch   - pull spec/datasheet text for every part in a BOM CSV
  2. embed   - embed each part's text with sentence-transformers
  3. rank    - for every part, rank all other parts by cosine similarity
  4. evaluate - score the top-k rankings against a ground-truth answer key

`run-all` chains all four steps. Each step also has its own subcommand so
you can inspect/cache intermediate output.

BOM CSV schema (data/sample_bom.csv shows an example):
    part_number,description,category

Answer key CSV schema (data/sample_answer_key.csv shows an example):
    target_part,rank,alternate_part
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from embedder import embed_parts
from evaluator import evaluate, format_report, load_answer_key
from fetchers import CompositeFetcher, LocalFileFetcher, WebSearchFetcher
from ranker import rank_alternates

logger = logging.getLogger("bom_alternate_ranker")

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_bom(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"BOM file {path} has no rows")
    if "part_number" not in rows[0]:
        raise ValueError("BOM CSV must have a 'part_number' column")
    return rows


def build_fetcher(args: argparse.Namespace) -> CompositeFetcher:
    fetchers = []
    if args.local_datasheet_dir:
        fetchers.append(LocalFileFetcher(Path(args.local_datasheet_dir)))
    if args.fetch_mode == "web":
        fetchers.append(WebSearchFetcher())
    return CompositeFetcher(fetchers=fetchers, cache_dir=Path(args.cache_dir))


def step_fetch(args: argparse.Namespace) -> Dict[str, str]:
    bom = load_bom(Path(args.bom))
    fetcher = build_fetcher(args)
    texts: Dict[str, str] = {}
    for row in bom:
        part = row["part_number"].strip()
        description = row.get("description", "")
        text = fetcher.fetch(part, description, refresh=args.refresh)
        texts[part] = text
        logger.info("fetched %d chars for %s", len(text), part)
    return texts


def step_embed(texts: Dict[str, str], model_name: str):
    logger.info("embedding %d parts with %s", len(texts), model_name)
    return embed_parts(texts, model_name=model_name)


def step_rank(embeddings, top_k: int) -> Dict[str, List[str]]:
    ranked = rank_alternates(embeddings, top_k=top_k)
    return {part: [alt for alt, _score in alts] for part, alts in ranked.items()}, ranked


def write_rankings_csv(ranked_with_scores, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["part_number", "rank", "alternate_part", "cosine_similarity"])
        for part, alts in ranked_with_scores.items():
            for rank, (alt, score) in enumerate(alts, start=1):
                writer.writerow([part, rank, alt, f"{score:.4f}"])


def cmd_fetch(args: argparse.Namespace) -> None:
    texts = step_fetch(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(texts, indent=2), encoding="utf-8")
    print(f"Wrote spec text for {len(texts)} parts to {out_path}")


def cmd_embed(args: argparse.Namespace) -> None:
    texts = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    embeddings = step_embed(texts, args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez(out_path, **{part: vec for part, vec in embeddings.items()})
    print(f"Wrote embeddings for {len(embeddings)} parts to {out_path}")


def cmd_rank(args: argparse.Namespace) -> None:
    import numpy as np

    data = np.load(args.embeddings)
    embeddings = {part: data[part] for part in data.files}
    _, ranked_with_scores = step_rank(embeddings, args.top_k)
    write_rankings_csv(ranked_with_scores, Path(args.out))
    print(f"Wrote top-{args.top_k} rankings for {len(ranked_with_scores)} parts to {args.out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    predicted: Dict[str, List[str]] = {}
    with open(args.rankings, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            predicted.setdefault(row["part_number"], []).append(row["alternate_part"])

    answer_key = load_answer_key(Path(args.answer_key))
    report = evaluate(predicted, answer_key, top_k=args.top_k)
    print(format_report(report, top_k=args.top_k))


def cmd_run_all(args: argparse.Namespace) -> None:
    texts = step_fetch(args)
    embeddings = step_embed(texts, args.model)
    _, ranked_with_scores = step_rank(embeddings, args.top_k)

    out_path = Path(args.out)
    write_rankings_csv(ranked_with_scores, out_path)
    print(f"Wrote top-{args.top_k} rankings for {len(ranked_with_scores)} parts to {out_path}\n")

    if args.answer_key:
        predicted = {part: [alt for alt, _ in alts] for part, alts in ranked_with_scores.items()}
        answer_key = load_answer_key(Path(args.answer_key))
        report = evaluate(predicted, answer_key, top_k=args.top_k)
        print(format_report(report, top_k=args.top_k))


def add_fetch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bom", required=True, help="Path to BOM CSV (part_number, description, ...)")
    p.add_argument(
        "--fetch-mode",
        choices=["local", "web"],
        default="local",
        help="'local' only uses --local-datasheet-dir; 'web' also tries a best-effort web search fallback",
    )
    p.add_argument(
        "--local-datasheet-dir",
        default=None,
        help="Directory of <part_number>.txt files with pre-fetched spec/datasheet text",
    )
    p.add_argument("--cache-dir", default="data/datasheet_cache", help="Where resolved text is cached")
    p.add_argument("--refresh", action="store_true", help="Ignore the cache and re-fetch")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch spec/datasheet text for every part in the BOM")
    add_fetch_args(p_fetch)
    p_fetch.add_argument("--out", default="data/part_texts.json")
    p_fetch.set_defaults(func=cmd_fetch)

    p_embed = sub.add_parser("embed", help="Embed part texts with sentence-transformers")
    p_embed.add_argument("--texts", default="data/part_texts.json")
    p_embed.add_argument("--model", default=DEFAULT_MODEL)
    p_embed.add_argument("--out", default="data/part_embeddings.npz")
    p_embed.set_defaults(func=cmd_embed)

    p_rank = sub.add_parser("rank", help="Rank alternates by cosine similarity")
    p_rank.add_argument("--embeddings", default="data/part_embeddings.npz")
    p_rank.add_argument("--top-k", type=int, default=5)
    p_rank.add_argument("--out", default="data/rankings.csv")
    p_rank.set_defaults(func=cmd_rank)

    p_eval = sub.add_parser("evaluate", help="Score rankings against a ground-truth answer key")
    p_eval.add_argument("--rankings", default="data/rankings.csv")
    p_eval.add_argument("--answer-key", required=True)
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.set_defaults(func=cmd_evaluate)

    p_all = sub.add_parser("run-all", help="Run fetch -> embed -> rank -> evaluate end to end")
    add_fetch_args(p_all)
    p_all.add_argument("--model", default=DEFAULT_MODEL)
    p_all.add_argument("--top-k", type=int, default=5)
    p_all.add_argument("--out", default="data/rankings.csv")
    p_all.add_argument("--answer-key", default=None, help="If given, also run the evaluate step")
    p_all.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
