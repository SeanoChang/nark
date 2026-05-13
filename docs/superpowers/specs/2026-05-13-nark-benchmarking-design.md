# nark — Benchmarking & Comparative Positioning Design

**Date:** 2026-05-13
**Status:** Design — awaiting user review before plan
**Author:** Sean (with Claude as collaborator)

---

## 1. Goal & framing

Build a benchmark suite and comparison story for **nark** — a local-first, content-addressed, structured knowledge vault for AI agents — so that:

- **Internal iteration is honest.** Every change to `src/registry/search.rs` produces a measurable delta in retrieval quality. Regressions are gated at PR time.
- **Public positioning is credible.** When a power user evaluates nark against mem0 / Letta / Zep / Graphiti / Cognee / LangMem, they see a numerical comparison they can reproduce — including the places nark does not win.
- **The unique-quadrant claim is quantified.** nark is the only widely-known system that combines *no LLM on the write path*, *structured-and-versioned memory*, and *single-binary local deployment*. Today this is rhetoric. The bench turns it into chart-able numbers (write-path token cost, cold-start time, network calls).

Positioning is **B + C + D**, in the framing established during brainstorming:

- **B. Local-first / sovereign alternative** — privacy, offline, no vendor lock-in. Hosted competitors (mem0 cloud, Zep cloud) are not direct rivals; they're a different design point.
- **C. The Claude Code / agent-CLI memory** — built for terminal agents specifically.
- **D. Internal iteration** — numbers exist primarily to drive Sean's own improvements; public scoreboard is secondary.

Mem0/Letta/Zep are *reference points*, not enemies. The benchmark exists to keep nark honest and to give serious evaluators something concrete.

## 2. Survey findings that shape the design

(Full survey: see prior brainstorming transcript. Key load-bearing facts repeated here so this doc is self-contained.)

- **LOCOMO scores are contested.** Mem0, Zep, and Letta have published mutually-inconsistent SOTA claims on the same dataset; an independent reproduction in the Zep-papers repo showed mem0's "68.5%" was closer to ~58% with corrected eval; Letta showed a trivial filesystem-tool agent hits ~74%. LOCOMO is a methodology landmine — we run it as a secondary signal, never the headline.
- **LongMemEval (ICLR 2025, arxiv 2410.10813) is the better target.** 500 curated questions, 5 ability classes (info-extraction, multi-session reasoning, temporal, knowledge-updates, abstention). Commercial systems drop ~30% on it. Cleaner methodology, harder to game. **This becomes the headline benchmark.**
- **nark sits in a quadrant no other tool occupies.** Across the surveyed tools (mem0, Letta, Graphiti/Zep, Cognee, LangMem, OpenAI memory, Claude memory): every competitor does LLM-on-write, none have content-addressed versioning, none ship as a single binary. The honest threats to nark are: zero-effort ingest (all competitors auto-extract memories), temporal reasoning (Graphiti's bi-temporal model is more expressive than nark's versioned-supersedes chain), self-improvement loops (Cognee's Memify, Anthropic's Dreaming), and agent-framework integration (everyone else has MCP servers).

## 3. Repo layout

A new top-level **`bench/`** workspace member, sibling to `src/`. The top-level `Cargo.toml` becomes a workspace root with members `["", "bench"]`. nark itself keeps publishing as `nark` to crates.io; `bench` is `publish = false`.

```
nark/
├── Cargo.toml                    # promoted to workspace root
├── src/                          # unchanged — nark stays as-is
└── bench/
    ├── Cargo.toml                # [[bin]] name = "nark-bench"
    ├── README.md                 # how to run; docker-compose for Letta/Graphiti
    ├── src/
    │   ├── main.rs               # clap dispatch
    │   ├── protocol.rs           # Adapter trait
    │   ├── adapters/
    │   │   ├── nark.rs           # subprocess: cargo run --bin nark
    │   │   ├── fts5.rs           # direct rusqlite — pure-Rust baseline
    │   │   ├── vector.rs         # ndarray + ort — pure-Rust baseline
    │   │   ├── mem0.rs           # subprocess: bench/shims/mem0_shim.py
    │   │   ├── letta.rs          # HTTP: ureq → localhost:8283
    │   │   └── graphiti.rs       # subprocess: bench/shims/graphiti_shim.py
    │   ├── tasks/
    │   │   ├── ir.rs             # Task A — classical IR
    │   │   ├── longmemeval.rs    # Task B primary
    │   │   ├── locomo.rs         # Task B secondary
    │   │   ├── replay.rs         # Task C
    │   │   └── curation.rs       # Task D — stubbed pending Librarian
    │   ├── metrics/
    │   │   ├── ir.rs             # Recall@k, MRR, nDCG
    │   │   ├── perf.rs           # latency, tokens, mem, cold-start, network
    │   │   └── judge.rs          # LLM-as-judge with on-disk cache
    │   ├── judges/
    │   │   ├── claude_cli.rs     # subprocess: claude -p --model claude-opus-4-7
    │   │   ├── codex_cli.rs      # subprocess: codex exec --model gpt-5.5
    │   │   ├── api.rs            # fallback: ureq → Anthropic / OpenAI
    │   │   └── panel.rs          # consensus wrapper over Vec<Box<dyn Judge>>
    │   ├── bin/
    │   │   ├── label.rs          # one-off corpus labelling helper
    │   │   └── badge-json.rs     # shields.io endpoint data
    │   └── report.rs             # JSON → BENCHMARKS.md table generator
    ├── shims/                    # Python adapter shims (one file each, ~50 LOC)
    │   ├── mem0_shim.py
    │   ├── graphiti_shim.py
    │   └── requirements.txt
    ├── judges/                   # judge prompts, committed
    │   ├── longmemeval.md
    │   ├── locomo.md
    │   └── replay.md
    ├── datasets/                 # small fixtures in-tree (~2 MB cap)
    │   ├── ir/
    │   │   ├── nark-self/        # scrubbed real vault notes + queries
    │   │   └── synthetic-domain/
    │   ├── longmemeval/fetch.sh  # download script
    │   ├── locomo/fetch.sh
    │   └── replay/sessions.jsonl # scrubbed real transcripts (committed)
    ├── cache/
    │   └── judge.db              # committed sqlite — keyed cache of verdicts
    ├── results/                  # committed JSON outputs over time
    │   ├── main/                 # PR regression baseline
    │   ├── v0.14.0/...
    │   └── latest/ → v0.14.0/
    ├── ci/
    │   └── pr.yml                # GitHub Actions: Lane 1 (PR check)
    └── scripts/
        └── run.sh                # canonical manual invocations
```

Three notable structural decisions:

1. **`bench/` is a workspace member, not a `benches/` directory.** `cargo bench` is opinionated about microbenchmarks; we want a multi-system harness with subcommands and result aggregation. That wants a `[[bin]]` target. Keeping it as a separate workspace member also isolates heavier dev-only deps (`reqwest`, `parquet`, `csv`) from nark's release binary.
2. **Result files are committed.** `bench/results/` lives in git. Provides `git log` for benchmarks across nark versions.
3. **Python shims exist.** Three files, ~150 LOC total, pinned via `bench/shims/requirements.txt`. They contain no logic — only protocol translation to Python-only competitors (mem0, Graphiti). The Rust adapter is where the real code is. This is the one non-Rust concession in the repo.

## 4. The benchmark protocol

A single Rust trait — `Adapter` — that every system-under-test implements. The harness only sees the trait; it doesn't care whether the implementation is in-process Rust, a Python subprocess speaking JSON-lines, or an HTTP call to a daemon.

```rust
pub struct Document {
    pub id: String,
    pub body: String,
    pub metadata: serde_json::Value,  // frontmatter / tags / timestamps
}

pub struct SearchResult {
    pub doc_id: String,
    pub score: f32,
    pub snippet: Option<String>,
}

pub struct WriteMetrics {
    pub latency_ms: u64,
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}

pub struct SearchMetrics {
    pub latency_ms: u64,
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}

pub trait Adapter {
    fn name(&self) -> &str;
    fn version(&self) -> Result<String>;
    fn setup(&mut self, workdir: &Path) -> Result<()>;
    fn write(&mut self, doc: &Document) -> Result<WriteMetrics>;
    fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)>;
    fn teardown(&mut self) -> Result<()>;
}
```

### Three implementation shapes

1. **Native Rust** — `adapters/fts5.rs`, `adapters/vector.rs`. Direct `rusqlite` / `ndarray` / `ort` calls. Fast lane on every PR.
2. **Subprocess + JSON-lines** — `adapters/mem0.rs`, `adapters/graphiti.rs`, `adapters/nark.rs`. The Rust adapter spawns a long-running child process and pipes JSON lines (`{"op":"setup"}`, `{"op":"write","doc":{...}}`, `{"op":"search","query":"...","k":10}`, `{"op":"teardown"}`). The child responds with one JSON line per request. Setup cost (model loading, DB connect) paid once per run.
3. **HTTP** — `adapters/letta.rs`. Each method is a `ureq::post()` to a Letta server brought up by `docker compose`. Harness fails-soft with `LettaUnavailable` if the port isn't listening.

### Metadata pass-through

Nark uses `frontmatter` (domain / kind / intent / trust / status / tags) heavily; mem0 mostly ignores metadata; Letta uses `user_id`. The harness passes `Document.metadata` through as `serde_json::Value` and each adapter decides what to use.

**Nark is run in two configurations side-by-side**: `nark-full` (uses metadata) and `nark-body-only` (body string only). Reporting both pre-empts the "you cheated by using your taxonomy" critique and makes metadata's contribution itself a measurable axis.

### Long-running subprocess, not one-shot

Every adapter holds setup state for the duration of the run. Spinning up the embedding model or opening a DB connection per query would dominate latency measurements and make them meaningless. The protocol is explicitly chatty within a single process lifetime.

### Token metrics flow through the protocol

The protocol carries `llm_tokens_in` / `llm_tokens_out` per `write()` and `search()`. nark's adapter returns `0` by construction; mem0's shim wraps the LiteLLM client and counts. This makes "nark has zero LLM cost on the write path" a measured, audited number rather than a footnote.

## 5. The four eval tasks

### Task A — Classical IR (the regression harness)

**Dataset.** Hand-curated, in-tree at `bench/datasets/ir/`:

- **`nark-self`** — ~150 notes scrubbed from Sean's real vault. Queries (~50) written by Sean reflecting real research scenarios. Each query has 1–5 hand-labeled relevant note IDs. The corpus that actually reflects how nark is used.
- **`synthetic-domain`** — ~200 generated notes across topic clusters with planted overlaps and distractors. ~100 queries with per-query-class labels (single-hop, multi-hop, synonym, freshness-biased).

**Scoring.** Recall@1, Recall@5, Recall@10, MRR, nDCG@10. All deterministic. Per-query-class breakdowns where the corpus has them.

**Runtime.** Full run <30 seconds on a single laptop. **Runs on every PR.** Regression-gated at >2% drop from `bench/results/main/ir.json`.

### Task B — LongMemEval (primary) + LOCOMO (secondary)

**Datasets.**

- **LongMemEval** (arxiv 2410.10813). Download into `bench/datasets/longmemeval/`. 500 questions × 5 ability classes. **This is the headline number.**
- **LOCOMO** (arxiv 2402.17753). Same download pattern. Run for comparability to mem0/Zep/Letta papers, but always reported with the "trivial-filesystem-agent baseline" column visible and a methodology disclaimer linking Zep's and Letta's critiques.

**Scoring.** Answer accuracy via LLM-as-judge. Default judge: `claude -p --model claude-opus-4-7`. Release-sweep mode: panel of `[claude-opus-4-7, gpt-5.5]` via `codex exec`, agreement-required for `confident` verdicts.

**Caching.** Judgments cached in `bench/cache/judge.db` keyed on `sha256(question + answer + gold + judge_model_id + judge_prompt_version)`. Cache committed to repo. First run pays subscription limits / API tokens; subsequent runs are free.

**Per-ability breakdown.** Five separate accuracy numbers per system. This is where structural choices pay off — nark's typed edges should help multi-session reasoning, CAS versioning should help knowledge-updates.

**Runtime.** First run: 30–60 minutes wall-clock; usage-limited (or ~$5–15 if using API tokens). Cached re-runs: minutes.

### Task C — Agent replay (the realism check)

**Dataset.** Anonymized real session transcripts from Sean's Claude Code / agent use (the 40+ sessions referenced in his agent-power-user reviews). For each transcript: extract retrieval moments, label the "ideal" notes that should have come back. Stored at `bench/datasets/replay/sessions.jsonl`.

**Scoring.** Recall@k on the labeled gold set + token-efficiency (`tokens_returned / tokens_useful` — how much fluff did the system pull in alongside the right answer).

**Runtime.** Fast (deterministic given gold labels). Bottleneck is the one-time labelling pass, expected ~1.5 weekends of Sean's time, optionally accelerated via `bench/src/bin/label.rs` running `codex exec` in the background.

### Task D — Librarian curation (gated)

**Status.** Scaffolded as a typed stub. Real implementation waits on the Librarian agent existing per the current nark direction.

**Planned metrics.** Retirement precision/recall, merge precision (heavily weighted against false positives), summary fidelity vs source via LLM-judge, surfacing relevance.

For now: `bench/src/tasks/curation.rs` is a stub with TODOs; no metric runs.

## 6. Metrics

### Bucket 1 — Classical IR (Task A, parts of C)

Recall@1, Recall@5, Recall@10, MRR, nDCG@10. Standard formulas. Implemented once in `bench/src/metrics/ir.rs` with unit tests against canonical examples. Per-query-class breakdowns reported when the corpus has class labels.

### Bucket 2 — Answer quality (Task B, optionally C)

- **Answer accuracy** — `judge(question, predicted, gold) ∈ {correct, partial, incorrect}` → `1.0 / 0.5 / 0.0`, averaged.
- **Per-ability accuracy** — LongMemEval's 5 classes scored separately.
- **Abstention precision** — for abstention-class questions: did the system correctly say "I don't know"?
- **Judge agreement rate** — in panel mode, fraction of `(question, answer)` pairs where Claude and Codex agree. Reporting this publicly is honest in a way mem0/Zep/Letta papers aren't; if it drops below ~0.9 the judge prompt itself is the bug, not the system under test.

### Bucket 3 — Performance & cost (every task, every system)

The unique-positioning bucket. nark wins these by construction.

| Metric | Definition |
|---|---|
| `write_latency_p50_ms` / `p99_ms` | Wall-clock per `write()` |
| `write_tokens_in_per_doc` / `out_per_doc` | LLM tokens per ingested doc (0 for nark/FTS5/vector) |
| `write_cost_usd_per_1k_docs` | Derived from tokens × current Anthropic/OpenAI pricing |
| `search_latency_p50_ms` / `p99_ms` | Wall-clock per `search()` |
| `search_tokens_in_per_query` / `out_per_query` | LLM tokens per query |
| `peak_resident_mb` | Max RSS during the run, via `/proc/self/status` or `proc_pid_rusage` |
| `cold_start_ms` | Time from `setup()` to first successful `search()` |
| `disk_mb_per_1k_docs` | Storage footprint after ingesting 1000 docs |
| `network_calls_total` | Count of outbound network requests, measured runtime via Linux netns. macOS dev runs declare statically and flag `unverified`; canonical number comes from a one-off Linux run. |

### Result file format

One JSON per `(task, system, config)`, schema-versioned:

```json
{
  "schema_version": "1",
  "task": "longmemeval",
  "system": "nark",
  "config": "full",
  "system_version": "0.13.0",
  "bench_version": "0.1.0",
  "judge_models": ["claude-opus-4-7", "gpt-5.5"],
  "run_started_at": "2026-05-13T18:42:00Z",
  "host": { "os": "darwin", "cpu": "...", "ram_gb": 32 },
  "ir": { "recall_at_5": 0.74, "mrr": 0.61, "ndcg_at_10": 0.68 },
  "ir_per_class": { "single_hop": {...}, "multi_hop": {...} },
  "answer": {
    "accuracy": 0.81,
    "per_ability": {...},
    "abstention_precision": 0.92,
    "judge_panel": {
      "judges": ["claude-opus-4-7", "gpt-5.5"],
      "agreement_rate": 0.94,
      "disagreement_count": 12
    }
  },
  "perf": { "write": {...}, "search": {...}, "peak_resident_mb": 84, "cold_start_ms": 78, "disk_mb_per_1k_docs": 14, "network_calls_total": 0 },
  "errors": []
}
```

### Published `BENCHMARKS.md` layout

Auto-generated from `bench/results/latest/` by `bench/src/report.rs`. Opinionated section ordering:

1. **Headline** — LongMemEval aggregate, one row per system, sorted descending. `nark-full` and `nark-body-only` both shown.
2. **Per-ability table** — same systems × 5 abilities. Cells where nark wins bolded; gaps shown where it loses.
3. **The local-first chart** — bar visualization of `write_cost_usd_per_1k_docs` and `cold_start_ms`. nark at $0 / ~80ms; competitors at their measured values. The visualization makes the positioning unforgeable.
4. **LOCOMO table** — same shape as headline, with one-paragraph methodology disclaimer linking the Zep and Letta critique posts.
5. **Honest threats panel** — verbatim list of where competitors do something nark currently doesn't: zero-effort ingest, bi-temporal reasoning, self-improvement loops, agent-framework integration. Updated as features ship.

### README badges

Three badges sourced from `bench/results/latest/`:

- LongMemEval aggregate
- `cold_start` (ms)
- `write_cost` ($/1k)

The cold-start and write-cost badges are the ones that *describe nark to a stranger* in a way no other memory tool can match. They link to `BENCHMARKS.md`.

## 7. CI shape

Two lanes — one automated, one manual.

### Lane 1 — PR check (GitHub-hosted)

`bench/ci/pr.yml`, `runs-on: ubuntu-latest`, timeout 5 min.

```yaml
jobs:
  bench-regression:
    steps:
      - checkout
      - cache cargo
      - cargo build --release -p nark -p nark-bench
      - cargo run -p nark-bench --release -- \
          --task ir \
          --systems nark-full,nark-body-only,fts5,vector \
          --output bench/results/pr/
      - bench/ci/regression-check.sh
        # fails if any IR metric drops >2% vs bench/results/main/ir.json
```

Properties:
- Deterministic, no LLM, no network, no Docker
- <2 min wall-clock
- Only pure-Rust adapters
- Fails the PR if Recall@5/MRR/nDCG drops by >2% (configurable)

**This is the only automated lane.** It enforces the "did your change make search worse" discipline.

### Lane 2 — Manual benchmark run

A wrapper shell script `bench/scripts/run.sh` documenting canonical invocations:

```bash
# Quick local check (no LLM)
cargo run -p nark-bench --release -- \
  --task ir --systems nark-full,nark-body-only,fts5

# Full local sweep (uses your Claude Code session)
cargo run -p nark-bench --release -- \
  --task ir,longmemeval,locomo,replay \
  --systems nark-full,nark-body-only,fts5,vector,mem0,letta,graphiti \
  --judge claude-cli

# Release sweep — panel of judges, 3 runs for variance
cargo run -p nark-bench --release -- \
  --task all --systems all \
  --judges claude-cli:claude-opus-4-7,codex-cli:gpt-5.5 \
  --panel-mode strict --runs 3 \
  --output bench/results/v0.14.0/
```

Runs on Sean's own machine, uses logged-in Claude Code + Codex subscriptions for the judges. Judgments cached after the first run.

### Why no cron / scheduled nightly

Subscription-bound judging requires a logged-in user; hosted CI runners cannot use OAuth-bound subscriptions and would have to fall back to paid API tokens. Manual runs preserve the free path and make every results commit an intentional act.

### Optional fallback (deferred): hosted CI with API tokens

If at some point publishing numbers on a regular cadence matters, `--judge-backend api` switches to `ureq` against the Anthropic / OpenAI HTTP APIs using `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` repo secrets. Cost: first run pays for judgments at current per-token pricing; cache makes re-runs free. Not built in v1.

## 8. Error handling, determinism, testing, anti-cheat

### Error handling

- **Adapter failures are non-fatal.** `Result` at every call boundary; failures recorded as `errors: [...]` in the result file, system shows N/A in published tables.
- **Partial results are explicit.** "Accuracy = 0.78 over 480 successful judgments (20 errored, 0 timed out)" — never silently average over fewer samples without saying so.
- **Subprocess crashes are recoverable.** State for "what's done" lives in an append-only JSONL during the run; on restart, we resume from the last completed query.
- **Reproducibility floor**: benchmark refuses to run if any adapter reports `version() = "unknown"`.

### Determinism

- **Seed = `0xBADA55`** by default for any adapter with a random component. Recorded in result JSON.
- **Judge cache keyed on `sha256(question + answer + gold + judge_model_id + judge_prompt_version)`.** Prompt changes invalidate exactly the affected judgments. Cache committed.
- **Wall-clock measurements report `mean ± stddev`** in release-sweep mode (3 runs). Manual / nightly are single-shot and note the fact.

### Testing the harness itself

Unit tests in `bench/src/metrics/ir.rs` against canonical examples (Manning IR-book P@k / nDCG examples). One integration test `full_harness_smoke` that runs the full benchmark against a tiny in-tree fixture corpus (~5 notes, 3 queries, pure-Rust adapters only) — keeps Lane 1 honest about "the harness compiles and runs end-to-end."

Competitor adapters (mem0/Letta/Graphiti) are not exercised under `cargo test`. Their correctness is verified by running them against the tiny fixture during a manual sweep; failures show as N/A and are diagnosable from the `errors: [...]` field.

### Anti-cheat

Three rules:

1. **No competitor forking.** Every adapter calls only the competitor's public API at a pinned version.
2. **Same input, same query.** Identical `Document` and query string to every system. Differences come from the systems, not the harness.
3. **Judge prompts are public.** Committed in `bench/judges/*.md`; iteration history preserved.

These are the credibility floor for any third party reproducing the bench.

## 9. Phasing

Rough order of work. Each phase delivers something useful on its own.

1. **Workspace + protocol + pure-Rust adapters + Task A.** Workspace layout, `Adapter` trait, `nark` / `fts5` / `vector` adapters, classical IR metrics, the `nark-self` and `synthetic-domain` corpora, Lane 1 PR check wired up. End state: every PR is regression-gated on IR metrics.
2. **LongMemEval + LOCOMO + judges.** Dataset fetchers, `claude-cli` + `codex-cli` + `api` judge backends, panel + cache. End state: headline numbers exist; `BENCHMARKS.md` v1 publishes.
3. **Python-subprocess competitor adapters.** mem0 and Graphiti shims, performance instrumentation (tokens, latency, RSS, network). End state: comparison rows populated; the unique-positioning chart is real.
4. **Letta HTTP adapter.** Docker compose for Letta + Postgres. End state: full competitor set covered.
5. **Task C (agent replay).** Labelling pass on real transcripts; replay scoring. End state: realism check exists.
6. **Task D (Librarian curation).** Implementation gated on the Librarian agent shipping. Currently a typed stub.

Phase 1 alone gives the regression-gating discipline that pays back immediately on every commit to `src/registry/`. Phases 2–4 are the public-credibility deliverables. Phases 5–6 are roadmap.

## 10. Open items

- **`nark-self` corpus scrub** — Sean to do a one-time anonymization pass on ~150 vault notes, decide what's safe to commit publicly.
- **Self-hosted Linux runner for netns measurement** — declared static on macOS dev; the canonical `network_calls_total` number requires a one-off Linux run, frequency to be decided once results stabilize.
- **Judge prompt versioning convention** — store the prompt version as a header line in `bench/judges/*.md` (e.g., `<!-- version: 1 -->`) so the cache-key invalidation logic has something concrete to read. Implementation detail for Phase 2.
