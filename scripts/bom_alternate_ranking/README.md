# BOM alternate-part ranker

Given a benchmark BOM, ranks candidate alternate parts for each line item by
semantic similarity of their spec/datasheet text, and scores the ranking
against a ground-truth answer key.

Pipeline: **fetch → embed → rank → evaluate** (see `bom_alternate_ranker.py --help`).

## Setup

```bash
pip install -r requirements.txt
```

`sentence-transformers` pulls in `torch`; the first `embed` run downloads the
`all-MiniLM-L6-v2` model weights from Hugging Face Hub, so it needs outbound
network access to `huggingface.co` the first time (cached under `~/.cache`
after that).

## Input schemas

**BOM CSV** (`--bom`):

```
part_number,description,category
LM358,Dual general-purpose operational amplifier ...,op-amp
```

Only `part_number` is required; `description` is used as fetch-query context
and as a fallback embedding text if no datasheet text is found.

**Answer key CSV** (`--answer-key`), ground-truth alternates per target part,
ranked best-first:

```
target_part,rank,alternate_part
LM358,1,LM2904
LM358,2,LM358A
```

A target doesn't need exactly 5 rows — `evaluate` scores against however many
ground-truth alternates you provide.

## Getting spec/datasheet text (step 1: fetch)

Two fetch modes, chained by `CompositeFetcher` (`fetchers.py`) with on-disk
caching so re-runs don't re-fetch:

- `--fetch-mode local --local-datasheet-dir <dir>` (default, recommended):
  reads `<dir>/<part_number>.txt`. Point this at your own pre-collected
  datasheet text. `data/sample_local_datasheets/` has a worked example.
- `--fetch-mode web`: additionally tries a best-effort, no-API-key web
  search + HTML/PDF text extraction (`WebSearchFetcher`). This is fragile
  by nature (search markup changes, sites block scrapers, some hosts may be
  blocked by your network egress policy) — always used as a fallback behind
  local files, never as the only source you rely on for accuracy numbers.

If both fetchers come up empty for a part, the BOM `description` column is
used as a last resort so the pipeline never breaks on a missing datasheet.

To use a real parts-data provider instead (Octopart, Digi-Key, Mouser, ...),
implement a class with `fetch(part_number, description) -> str | None` and
add it to the `CompositeFetcher(fetchers=[...])` list in
`bom_alternate_ranker.py::build_fetcher` ahead of `WebSearchFetcher`.

## Usage

Run the whole pipeline against the bundled sample data:

```bash
python3 bom_alternate_ranker.py run-all \
  --bom data/sample_bom.csv \
  --fetch-mode local \
  --local-datasheet-dir data/sample_local_datasheets \
  --top-k 5 \
  --out data/rankings.csv \
  --answer-key data/sample_answer_key.csv
```

Or run each step separately (useful for inspecting/reusing intermediate output):

```bash
python3 bom_alternate_ranker.py fetch --bom data/sample_bom.csv \
  --local-datasheet-dir data/sample_local_datasheets --out data/part_texts.json

python3 bom_alternate_ranker.py embed --texts data/part_texts.json \
  --model all-MiniLM-L6-v2 --out data/part_embeddings.npz

python3 bom_alternate_ranker.py rank --embeddings data/part_embeddings.npz \
  --top-k 5 --out data/rankings.csv

python3 bom_alternate_ranker.py evaluate --rankings data/rankings.csv \
  --answer-key data/sample_answer_key.csv --top-k 5
```

For your real BOM, swap in `--bom your_bom.csv --answer-key your_answer_key.csv`
and either point `--local-datasheet-dir` at your own datasheet text files or
add `--fetch-mode web`.

## Output

`rank` writes a long-format CSV: `part_number,rank,alternate_part,cosine_similarity`
(top 5 per part_number by default).

`evaluate` reports, per target part and averaged across all targets in the
answer key:

- **precision@k** — fraction of the predicted top-k that are correct alternates
- **recall@k** — fraction of the ground-truth alternates that appear in the predicted top-k
- **hit@1** — whether the single top prediction is a correct alternate
- **reciprocal rank** — 1 / (rank of the first correct alternate found)

## Notes on this repo

This script lives under `scripts/` as a standalone Python utility — it's
independent of the rest of the `openai-agents-js` TypeScript SDK in this
repository.
