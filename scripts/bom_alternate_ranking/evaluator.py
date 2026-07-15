"""Compares predicted top-k alternate rankings against a ground-truth answer key."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


def load_answer_key(path: Path) -> Dict[str, List[str]]:
    """Load a ground-truth answer key CSV with columns:
    target_part, rank, alternate_part

    Rows are ordered by `rank` (ascending) for each target_part. A target
    part may have fewer than top_k ground-truth alternates.
    """
    rows: Dict[str, List[Tuple[int, str]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = row["target_part"].strip()
            alt = row["alternate_part"].strip()
            rank = int(row["rank"])
            rows.setdefault(target, []).append((rank, alt))

    answer_key: Dict[str, List[str]] = {}
    for target, entries in rows.items():
        entries.sort(key=lambda pair: pair[0])
        answer_key[target] = [alt for _, alt in entries]
    return answer_key


@dataclass
class PartScore:
    target_part: str
    predicted: List[str]
    ground_truth: List[str]
    precision_at_k: float
    recall_at_k: float
    hit_at_1: bool
    reciprocal_rank: float


@dataclass
class EvaluationReport:
    per_part: List[PartScore] = field(default_factory=list)

    @property
    def mean_precision_at_k(self) -> float:
        return _mean(s.precision_at_k for s in self.per_part)

    @property
    def mean_recall_at_k(self) -> float:
        return _mean(s.recall_at_k for s in self.per_part)

    @property
    def hit_rate_at_1(self) -> float:
        return _mean(1.0 if s.hit_at_1 else 0.0 for s in self.per_part)

    @property
    def mean_reciprocal_rank(self) -> float:
        return _mean(s.reciprocal_rank for s in self.per_part)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate(
    predicted_rankings: Dict[str, List[str]],
    answer_key: Dict[str, List[str]],
    top_k: int = 5,
) -> EvaluationReport:
    """Score predicted top-k alternate rankings against ground truth.

    Only targets present in the answer key are scored (targets without
    ground truth can't be judged). Metrics per target:
      - precision@k: |predicted[:k] ∩ ground_truth| / k
      - recall@k:    |predicted[:k] ∩ ground_truth| / |ground_truth|
      - hit@1:       whether predicted[0] is any correct alternate
      - reciprocal_rank: 1 / (rank of first correct alternate in predicted list)
    """
    report = EvaluationReport()
    for target, ground_truth in answer_key.items():
        predicted = predicted_rankings.get(target, [])[:top_k]
        gt_set = set(ground_truth)
        overlap = [p for p in predicted if p in gt_set]

        precision = len(overlap) / top_k if top_k else 0.0
        recall = len(overlap) / len(gt_set) if gt_set else 0.0
        hit_at_1 = bool(predicted) and predicted[0] in gt_set

        reciprocal_rank = 0.0
        for rank, part in enumerate(predicted, start=1):
            if part in gt_set:
                reciprocal_rank = 1.0 / rank
                break

        report.per_part.append(
            PartScore(
                target_part=target,
                predicted=predicted,
                ground_truth=ground_truth,
                precision_at_k=precision,
                recall_at_k=recall,
                hit_at_1=hit_at_1,
                reciprocal_rank=reciprocal_rank,
            )
        )
    return report


def format_report(report: EvaluationReport, top_k: int = 5) -> str:
    lines = []
    lines.append(f"Evaluated {len(report.per_part)} target part(s) against the answer key\n")
    header = f"{'target_part':<20} {'precision@' + str(top_k):>12} {'recall@' + str(top_k):>10} {'hit@1':>6} {'RR':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in report.per_part:
        lines.append(
            f"{s.target_part:<20} {s.precision_at_k:>12.2f} {s.recall_at_k:>10.2f} "
            f"{('yes' if s.hit_at_1 else 'no'):>6} {s.reciprocal_rank:>6.2f}"
        )
    lines.append("")
    lines.append(f"Mean precision@{top_k}: {report.mean_precision_at_k:.3f}")
    lines.append(f"Mean recall@{top_k}:    {report.mean_recall_at_k:.3f}")
    lines.append(f"Hit rate@1:          {report.hit_rate_at_1:.3f}")
    lines.append(f"Mean reciprocal rank: {report.mean_reciprocal_rank:.3f}")
    return "\n".join(lines)
