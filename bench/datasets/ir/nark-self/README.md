# nark-self IR Corpus — Methodology

**Status:** Empty as of Phase 1c. Awaiting Sean's one-time anonymization
scrub of ~150 real notes plus ~50 hand-labeled queries.

Running `cargo run -p nark-bench -- run --task ir --systems fts5,nark,vector --corpus nark-self --output bench/results/main`
will fail until `queries.jsonl` exists and `corpus/` contains at least
one `.md` file. The failure mode is "failed to read queries file" —
that's the intended signal that this corpus is not yet populated.

## Purpose

Measure nark's hybrid search pipeline against a corpus that reflects
how Sean *actually uses nark*, not the synthetic smoke fixture.

synthetic-tiny is small, synonym-heavy, and demonstrably favors pure
embedding (both nark and vector hit recall@5 = 1.0 on it). nark-self
should expose cases where nark's pipeline (BM25 + cosine + graph +
engagement blend) pulls ahead of raw vector — particularly on
synthesis queries and queries that reference versioned content.

## Source

Notes drawn from `~/.ark/objects/md/` at a documented point in time.
Filter criteria when doing the scrub:

- Research / reference / how-to notes from approximately the last 6-12 months
- Exclude any note tagged with personally-sensitive domains
  (e.g. `tags: [personal-finance, family, medical]`)
- Aim for diversity across domains so per-class breakdowns are
  meaningful

Record the source `~/.ark` commit SHA (if version-tracked) or the
date range in this README once the scrub completes.

## Anonymization checklist

The bar: **would you be comfortable if this corpus were public on GitHub?**

For each note:

- [ ] Real names of people → pseudonyms (Alice, Bob, Carol, ...)
- [ ] Specific company names → "Corp A", "Corp B", ...
- [ ] URLs → redact to bare domain or remove
- [ ] Specific dollar amounts → replace with `$X` or `$XXX` (order of magnitude OK)
- [ ] Specific dates → year-month only (`2026-05` not `2026-05-17`)
- [ ] Email addresses → remove or replace with `name@example.com`
- [ ] API keys, tokens, credentials → remove entirely (even if test-only)
- [ ] Internal codenames → replace with descriptive but generic equivalents
- [ ] Verbatim quotes from non-public sources → paraphrase
- [ ] Frontmatter `author` field → set to `bench` (matches synthetic-tiny convention)

After anonymizing, do a pass with fresh eyes to catch anything missed.

## Sizing target

- ~150 notes (range 100–200 is fine for v1)
- ~50 queries (range 30–80 is fine for v1)

Smaller is fine to start. Add more queries over time as you notice
real-world questions where you'd want to test how nark handles them.

## Query construction

Write queries reflecting how you *actually search* nark. The goal is
ecological validity, not coverage of every nark feature.

Sources of good real queries:
- Replay your own session history if any is preserved
- Note questions you've asked Claude/Codex while researching in nark
- The kinds of queries you'd issue if dropped into a vault cold

Avoid:
- Crafting queries specifically to test a feature (that's synthetic-tiny's
  job — it has classes for `single_hop` / `multi_hop` / `synonym`
  to make pipeline-component-level diagnosis possible)
- Verbatim copies of note titles (too easy; doesn't measure retrieval)

## Proposed query classes for nark-self

Each query gets an optional `class` field. Suggested taxonomy:

- **`factual`** — direct fact lookup; one note clearly answers
- **`synthesis`** — combining 2+ notes; exercises nark's hybrid
- **`temporal`** — asks about something that changed; exercises nark's
  versioning (currently surfaces latest version, but queries about
  history would benefit from richer query parsing in future phases)
- **`recall`** — find a half-remembered note by partial description;
  exercises the cosine pathway most heavily

Refine the taxonomy as you write queries. The class names are not
enforced; the bench just produces per-class breakdowns based on
whatever class strings appear in `queries.jsonl`.

## Validation checklist

Before committing the corpus, confirm:

- [ ] Every `corpus/*.md` has valid frontmatter (parses without errors
      via `nark write <file>` against a scratch vault)
- [ ] `queries.jsonl` parses as JSONL (one valid JSON object per line,
      no trailing whitespace)
- [ ] Every entry in any query's `relevant` array corresponds to an
      actual file in `corpus/`
- [ ] No personally-sensitive content remains (final pass)
- [ ] File count and query count match the target ranges above

## Running

Once the corpus is populated:

```bash
cargo run -p nark-bench --release -- run \
  --task ir \
  --systems fts5,nark,vector \
  --corpus nark-self \
  --output bench/results/main
```

Produces three new baseline files at
`bench/results/main/ir-{fts5,nark,vector}-nark-self-default.json`.
Run `bench/scripts/regression-check.sh bench/results/main` to confirm
the regression script accepts them (bootstrap mode on first run).

Commit the new baselines alongside the corpus content.

## CI status

**Not in CI Lane 1.** The PR check workflow runs `--corpus synthetic-tiny`
only. nark-self runs are intentionally manual until the corpus has stable
baselines worth gating on.

Future work (deferred): the CLI doesn't yet accept comma-separated
corpus values, so adding nark-self to CI requires either (a) running
the bench twice with different `--corpus` flags, or (b) adding
multi-corpus support to the runner. Cross that bridge when nark-self
has settled.
