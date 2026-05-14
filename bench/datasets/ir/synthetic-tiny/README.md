# synthetic-tiny — IR fixture

12 notes across 3 topic clusters (chess, breadmaking, electronics). 10 queries
with binary gold-relevance labels. Designed as a smoke fixture for the bench
harness — large enough to exercise BM25 ranking, small enough to commit and
iterate against on every PR.

## Layout
- `corpus/n01.md ... n12.md` — markdown notes with YAML frontmatter
- `queries.jsonl` — one JSON object per line: `{query_id, query, relevant: [...], class}`

## Query classes
- `single_hop` — exactly one obvious answer note shares keywords with the query
- `multi_hop` — two notes are relevant, often only one shares keywords directly
- `synonym` — the relevant note uses different wording than the query

These classes are surfaced in result JSON as per-class breakdowns so we can
tell whether a regression hurt one type of retrieval more than others.
