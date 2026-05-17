# nark-bench IR Corpora

This directory holds named corpora that the bench harness runs Task A
(classical IR) against. Each subdirectory is one corpus, with its own
notes, queries, and methodology README.

## Existing corpora

- **`synthetic-tiny/`** — 12 hand-written notes across 3 topic clusters
  (chess, cooking, electronics) and 10 queries with binary gold labels.
  Designed as a smoke fixture: large enough to exercise BM25 ranking,
  small enough to commit and iterate against on every PR.
- **`nark-self/`** — scaffold for ~150 anonymized real notes from Sean's
  vault and ~50 hand-labeled queries. Currently empty (the scrub work
  is manual and pending). See `nark-self/README.md` for methodology.

## Directory layout (contract for any new corpus)

```
<corpus-name>/
├── README.md             # methodology, anonymization (if applicable), provenance
├── corpus/
│   ├── n01.md            # one markdown note per file
│   ├── n02.md            # ...
│   └── ...
└── queries.jsonl         # one JSON object per line
```

The bench tracks each document by file stem (e.g. `n01` for `corpus/n01.md`).
The nark adapter assigns its own UUIDs internally and maintains a mapping
back to bench IDs — transparent to corpus authors. Other adapters use the
file stem directly.

## Frontmatter requirements

Every `.md` file in `corpus/` must have YAML frontmatter with these
**seven required fields** (nark's `Frontmatter` struct rejects anything
missing):

```yaml
---
title: Sicilian Defense opening principles
author: bench
domain: games
intent: reference
kind: note
status: active
tags: [chess, opening]
---
```

Valid `status` values: `active`, `draft`, `deprecated`, `retracted`.
Other fields are free-form strings (no enum validation in nark for
domain/kind/intent — they're treated as taxonomy labels).

Body comes after the frontmatter block. There is no upper bound; nark
truncates to ~8192 tokens during embedding.

## queries.jsonl format

One JSON object per line, no leading or trailing whitespace:

```jsonl
{"query_id": "q01", "query": "sicilian defense move order", "relevant": ["n01", "n02"], "class": "single_hop"}
{"query_id": "q02", "query": "najdorf variation idea", "relevant": ["n02"], "class": "single_hop"}
```

Fields:
- `query_id` (string, required) — used in result JSON's `errors` array if a
  query fails (e.g. `"search:q07"`).
- `query` (string, required) — the natural-language query sent to each
  adapter's `search()` method.
- `relevant` (array of strings, required) — file stems (without `.md`) of
  notes that are considered correct answers. Empty arrays are tolerated
  (by convention recall = 1.0 for queries with no relevant docs).
- `class` (string, optional) — query class label. Surfaces as a per-class
  breakdown in result JSON's `ir_per_class` map. Each corpus defines its
  own class scheme.

## How to invoke

```bash
# From the workspace root
cargo build -p nark --release
cargo run -p nark-bench --release -- run \
  --task ir \
  --systems fts5,nark,vector \
  --corpus <corpus-name> \
  --output bench/results/main
```

Produces one result JSON per `(system, corpus, config)` combination at
`bench/results/main/ir-<system>-<corpus>-<config>.json`. Today `config`
is always `default`; future ablations (e.g. nark-body-only) will vary it.

## Sizing guidance

| Corpus | Notes | Queries | Use case |
|---|---|---|---|
| synthetic-tiny | 12 | 10 | Smoke fixture; CI regression gate; iterate fast |
| nark-self (target) | ~150 | ~50 | Realistic regression detection |
| LongMemEval (future Phase 2) | 500 questions × variable | n/a | Public scoreboard comparability |

Larger corpora improve statistical confidence in absolute scores. For
regression detection (i.e. "did my last change make search worse?"),
even a 12-note smoke fixture catches most pipeline-level regressions.

## Adding a new corpus

1. Create `bench/datasets/ir/<name>/`.
2. Create `corpus/` subdirectory; drop in `.md` files with valid frontmatter.
3. Create `queries.jsonl` with at least one query.
4. Create a per-corpus `README.md` documenting methodology and provenance.
5. Run the bench with `--corpus <name> --output bench/results/main` to
   produce the initial baseline.
6. Run `bench/scripts/regression-check.sh bench/results/main` to confirm
   the new files are valid (bootstrap mode kicks in for the new files).
7. Commit the baseline JSONs alongside the corpus content.

For Sean specifically: nark-self is the target real-vault corpus.
Methodology and scrub procedure in `nark-self/README.md`.
