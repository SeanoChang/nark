# nark Bench — Phase 2 (LongMemEval + LOCOMO + LLM Backend) Design

**Date:** 2026-05-17
**Status:** Design — awaiting user review before plan
**Author:** Sean (with Claude as collaborator)
**Parent design:** `docs/superpowers/specs/2026-05-13-nark-benchmarking-design.md`
**Predecessors (all merged to main):**
- Phase 1a: bench harness foundation
- Phase 1b: vector adapter + embeddings-enabled nark
- Phase 1c: nark-self scaffold + baseline naming fix

---

## 1. Goal & framing

Phase 2 ships **Task B** — answer-accuracy benchmarks against two public datasets, scored by LLM-as-judge. After Phase 2 lands, the bench produces numbers directly comparable to what mem0, Letta, and Zep publish in their papers, against the same datasets they use.

Three concrete deliverables:

1. **LLM call infrastructure** — `LlmBackend` trait abstracting `claude -p`, `codex exec`, and HTTP-API subprocess/call patterns. SQLite cache so reruns are free. Prompt-versioning convention so cache invalidates correctly when prompts change.
2. **LongMemEval integration** — Task B primary. Use upstream's dataset, gold answers, question-type labels (xiaowu0162/LongMemEval pinned commit). Our infrastructure runs gen + judge via `LlmBackend`. Numbers reportable as "nark X% on LongMemEval" comparable to published baselines.
3. **LOCOMO integration** — Task B secondary. Same shape as LongMemEval; methodology is contested (Zep/mem0/Letta papers disagree on numbers) so the result JSON carries an inline methodology disclaimer.

Phase 2 explicitly **does NOT** include:
- A panel-of-judges feature (single judge matches literature; was earlier proposed and dropped during brainstorming).
- Validation that our methodology reproduces published mem0/Zep numbers — requires Phase 3's mem0 adapter to exist. Until then, Phase 2 numbers are "internal directional," not "publishable comparable."
- CI inclusion of Task B (too expensive per-PR; remains manual-trigger only).
- Mem0/Letta/Graphiti competitor adapters (Phase 3).
- LongMemEval-M or LongMemEval-L variants (start with `-S`, 500 short questions).

## 2. Survey findings that shape the design

Recap from the original benchmarking design's survey (verified once more):

- **LongMemEval** is the cleaner headline. ICLR 2025, 500 curated questions, 5 ability classes (info-extraction, multi-session reasoning, temporal, knowledge-updates, abstention). Commercial systems drop ~30% on it. The upstream repo (xiaowu0162/LongMemEval) ships eval scripts + prompts; using their dataset and gold answers (Option C from brainstorming) makes our numbers directly comparable.
- **LOCOMO** is methodologically contested. Mem0 reports 91.6%; Zep's reproduction shows ~58% with corrected eval; Letta showed a trivial filesystem-tool agent hits ~74%. We run it as a *secondary* signal, never the headline, and ship the baseline JSON with a `properties.methodology_disclaimer` field linking to the Zep + Letta critiques.
- **Standard evaluation methodology** for both datasets: retrieve top-k context, generate an answer with an LLM, judge the answer against gold with another LLM call. We follow this exactly.
- **Single judge** matches published methodology. Panel-of-judges was an earlier proposal that would make our numbers stricter (and harder to compare) than what papers report. Dropped during brainstorming.

## 3. Why these design choices

**One `LlmBackend` trait, two roles (gen + judge) as free functions.** Subprocess plumbing for `claude -p`/`codex exec` is identical; the only difference between gen and judge is the prompt and the parsing of the response. Making `Judge` and `Generator` separate traits would duplicate ~150 lines of subprocess+cache+retry logic for no benefit.

**Cache committed to git.** A full Task B sweep is ~500 questions × 3 systems × 2 LLM calls = ~3000 LLM invocations. With cache hits, subsequent runs finish in seconds. Committing the cache (~few MB sqlite after first sweep) means anyone can `git pull && cargo run` and reproduce results without burning their own subscription/credits.

**Prompt versioning via `<!-- prompt-version: N -->` header.** Cache keys include `prompt_version` so changing a prompt invalidates exactly the affected entries. Manual bumping discipline — when you change a prompt's intent (not just whitespace), bump N. Auditable in git diff.

**Per-question adapter isolation (setup → ingest → search → teardown) per question.** Matches upstream's eval model (each question evaluates against its own haystack with no carryover). Implementation overhead is real (500 `nark init` calls + 500 model-cache stagings) but bounded and predictable.

**Use upstream LongMemEval data + labels + gold answers; reimplement gen+judge in our infrastructure.** "Their data, our LLM-call machinery." Keeps results comparable to published numbers (same questions, same gold) while letting us cache, version prompts, and integrate with existing bench infrastructure.

**Schema bump to "3" with backward-compatible `answer: Option<AnswerMetrics>`.** Old Task A baselines remain valid; new Task B results add the new block. Schema version is symbolic — JSON parsers won't break either way thanks to serde's `Option` handling.

## 4. Core abstraction: `LlmBackend` + cache

### 4.1 The trait

```rust
// bench/src/llm/mod.rs (NEW directory)
pub trait LlmBackend: Send {
    fn name(&self) -> &str;        // e.g. "claude-cli", "codex-cli", "api"
    fn model_id(&self) -> &str;    // e.g. "claude-opus-4-7", "gpt-5.5"
    fn call(&mut self, prompt: &str) -> Result<LlmResponse>;
}

pub struct LlmResponse {
    pub text: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_micros: u64,  // u64 dollars × 1e6 — avoid float drift in cost aggregation
}
```

### 4.2 Three implementations

| Backend | File | Subprocess / call | Auth |
|---|---|---|---|
| `ClaudeCliBackend` | `bench/src/llm/claude_cli.rs` | `claude -p --model <model> --output-format json <prompt>` | OAuth (Claude Pro/Max session on disk) |
| `CodexCliBackend` | `bench/src/llm/codex_cli.rs` | `codex exec --model <model> --json <prompt>` | OAuth (Codex Plus session on disk) |
| `ApiBackend` | `bench/src/llm/api.rs` | HTTP POST to Anthropic API via `ureq` | `ANTHROPIC_API_KEY` env var |

**Concurrency**: each backend instance holds a synchronous semaphore (`std::sync::Mutex<usize>` or `tokio::sync::Semaphore` depending on whether we adopt async) with capacity 4 by default. Bench loops issue calls in parallel up to that cap. First-run cost goes from ~15 min serial to ~3-4 min concurrent. CLI flag `--llm-concurrency N` overrides.

**Retries**: transient failures (network, rate-limit, subprocess crash) retried once with exponential backoff. Persistent failures bubble up as `Err`. The caller (gen or judge) decides whether to map to a verdict like `JudgeError` or propagate.

**Token + cost reporting**: where the backend returns these (codex JSON output, claude `--output-format json` includes `cost_usd`), we read directly. Where unavailable (api backend), we compute via a tokenizer estimate. Cost stored as `cost_usd_micros: u64` (dollars × 1e6) to avoid float drift.

### 4.3 Cache

`bench/cache/llm.db` — sqlite. Schema:

```sql
CREATE TABLE responses (
  cache_key TEXT PRIMARY KEY,
  backend_name TEXT NOT NULL,
  model_id TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  call_kind TEXT NOT NULL,           -- 'judge' | 'generate'; descriptive only
  request TEXT NOT NULL,              -- full prompt
  response TEXT NOT NULL,             -- full text
  tokens_in INTEGER NOT NULL,
  tokens_out INTEGER NOT NULL,
  cost_usd_micros INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_responses_backend ON responses(backend_name, model_id);
```

**Cache key**: `sha256(prompt + "\n" + model_id + "\n" + prompt_version)`. Same prompt + same model + same prompt version = cache hit.

**Commit policy**: cache file *is* committed to git. Will grow to a few MB after the first full sweep; manageable. Re-runs are free. Anyone cloning the repo can reproduce numbers without LLM credits.

**Invalidation**: bump `<!-- prompt-version: N -->` in the prompt file. Cache key changes; old entries become unreachable. Periodic `VACUUM` (manual, not automated) shrinks the db if needed.

### 4.4 API on top of the trait

Two free functions in `bench/src/llm/eval.rs`:

```rust
pub fn generate_answer(
    backend: &mut dyn LlmBackend,
    cache: &mut LlmCache,
    prompt_template: &str,
    prompt_version: &str,
    question: &str,
    context_snippets: &[String],
) -> Result<GenerationResult>;

pub fn judge_answer(
    backend: &mut dyn LlmBackend,
    cache: &mut LlmCache,
    prompt_template: &str,
    prompt_version: &str,
    question: &str,
    gold: &str,
    candidate: &str,
) -> Result<JudgmentResult>;

pub struct GenerationResult { pub candidate: String, pub from_cache: bool, pub tokens_in: u64, pub tokens_out: u64, pub cost_usd_micros: u64 }
pub struct JudgmentResult { pub verdict: Verdict, pub reason: String, pub from_cache: bool, pub tokens_in: u64, pub tokens_out: u64, pub cost_usd_micros: u64 }
```

Both functions:
1. Substitute `{{var}}` placeholders in the prompt template.
2. Compute cache key from substituted prompt + model_id + prompt_version.
3. Check cache — return cached response if hit.
4. Otherwise call `backend.call(prompt)`, parse the response (for judge: extract JSON verdict; for gen: trim text), store in cache, return.
5. On parse failure for judge: retry the backend call once; if still bad, return `Verdict::JudgeError`.

## 5. Prompts + verdict semantics

### 5.1 Prompt file layout

```
bench/llm/                                   # NEW directory
├── README.md                                 # how prompts work, versioning convention
└── prompts/
    ├── longmemeval-generate.md               # Task B / LongMemEval generation
    ├── longmemeval-judge.md                  # Task B / LongMemEval judgment
    ├── locomo-generate.md                    # Task B / LOCOMO generation
    └── locomo-judge.md                       # Task B / LOCOMO judgment
```

Each prompt file's first line is `<!-- prompt-version: N -->`. The loader (`bench/src/llm/prompt.rs`) extracts N at load time and includes it in the cache key.

### 5.2 LongMemEval generation prompt (v1)

```markdown
<!-- prompt-version: 1 -->

You are answering a question based on a memory of past conversations.

Retrieved context (may or may not be relevant):
{{context}}

Question: {{question}}

Answer the question concisely. If the context doesn't contain enough information to answer, respond with exactly: "I don't know"
```

`{{context}}` = top-k retrieved snippets joined with `\n---\n`. `{{question}}` = LongMemEval question text.

### 5.3 LongMemEval judge prompt (v1)

```markdown
<!-- prompt-version: 1 -->

You are evaluating whether a candidate answer matches a gold answer for a memory benchmark.

Question: {{question}}
Gold answer: {{gold}}
Candidate answer: {{candidate}}

Respond with ONLY a JSON object on a single line:
{"verdict": "correct" | "partial" | "incorrect", "reason": "<one-sentence rationale>"}

Rules:
- "correct" if the candidate conveys the same factual content as gold, even if worded differently
- "partial" if it captures part of gold's content but omits important specifics
- "incorrect" if the candidate states something contradictory or unrelated to gold
- If gold is "I don't know" or similar abstention, "correct" iff candidate also abstains
```

LOCOMO prompts (v1) follow the same structure; only the rules in the judge prompt vary slightly to match LOCOMO's question types.

### 5.4 Verdict + parsing

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Correct,
    Partial,
    Incorrect,
    JudgeError,      // JSON parse failure after retry — recorded distinctly
}

impl Verdict {
    pub fn score(self) -> f64 {
        match self {
            Verdict::Correct => 1.0,
            Verdict::Partial => 0.5,
            Verdict::Incorrect | Verdict::JudgeError => 0.0,
        }
    }
}
```

The aggregator reports `judge_error_rate` separately from `incorrect_rate` so a flaky judge doesn't look like a bad system.

### 5.5 Prompt iteration discipline

1. Write `<!-- prompt-version: 1 -->` initial prompt.
2. Run a subset (10-20 questions) and eyeball verdicts.
3. If unsatisfactory (judge too harsh on paraphrases, etc.), edit prompt and bump to version 2.
4. Cache invalidates affected entries. Re-run.
5. Once stable, run full sweep. Commit final prompt to git so methodology is auditable.

## 6. Task B runner + dataset pipeline

### 6.1 Dataset acquisition

```
bench/datasets/longmemeval/
├── fetch.sh                  # clone xiaowu0162/LongMemEval @ pinned commit; checksum verify
├── upstream/                  # .gitignored; cloned files live here
└── README.md                  # methodology notes, pinned commit SHA

bench/datasets/locomo/
├── fetch.sh                  # clone snap-research/locomo @ pinned commit
├── upstream/                  # .gitignored
└── README.md
```

`fetch.sh` for LongMemEval (sketch):

```bash
#!/usr/bin/env bash
set -euo pipefail
UPSTREAM_URL="https://github.com/xiaowu0162/LongMemEval"
PINNED_SHA="<exact-commit-sha-to-pin-during-implementation>"

TARGET="$(dirname "$0")/upstream"
if [[ -d "$TARGET" ]]; then
  echo "already cloned at $TARGET"
  exit 0
fi
mkdir -p "$TARGET"
git clone --depth 1 "$UPSTREAM_URL" "$TARGET/repo"
cd "$TARGET/repo"
git fetch origin "$PINNED_SHA"
git checkout "$PINNED_SHA"
echo "LongMemEval ready at $PINNED_SHA"
```

LOCOMO's script follows the same pattern.

Sizes: LongMemEval-S is small (~few MB of question files); LOCOMO is moderate. Both fit in `upstream/` without git-LFS concerns.

### 6.2 Loader modules

`bench/src/tasks/longmemeval.rs` exports:

```rust
pub fn run_longmemeval_task(
    adapter: &mut dyn Adapter,
    dataset_path: &Path,         // bench/datasets/longmemeval/upstream/...
    gen_backend: &mut dyn LlmBackend,
    judge_backend: &mut dyn LlmBackend,
    cache: &mut LlmCache,
    model_cache: &Path,          // for adapter setup (Phase 1b's bench model cache)
    config_label: &str,
) -> Result<BenchResult>;
```

Internally:

1. **Parse dataset**: load each question entry from upstream JSON. Each entry has `{question_id, question, answer, question_type, haystack_sessions}` where `haystack_sessions` is a list of multi-turn conversations.
2. **Per-question loop** (per upstream's eval model — each question evaluates against its own haystack with no carryover):
   - Create tempdir workdir; `adapter.setup(workdir)`.
   - For each session in `haystack_sessions`, for each turn, build a `Document` (`id = format!("s{}_t{}", session_idx, turn_idx)`, body = turn text). `adapter.write(doc)`.
   - `(hits, _) = adapter.search(question, k=10)` — k matches upstream's default; document in README if it differs.
   - `gen_result = generate_answer(gen_backend, cache, gen_prompt, "1", question, &hits.iter().map(|h| h.snippet.unwrap_or_default()).collect::<Vec<_>>())`.
   - `judge_result = judge_answer(judge_backend, cache, judge_prompt, "1", question, gold, &gen_result.candidate)`.
   - Record `(question_id, question_type, verdict)` into per-question results.
   - `adapter.teardown()`.
3. **Aggregate**:
   - Overall accuracy = `mean(verdict.score())` across all 500 questions.
   - Per-ability accuracy = same averaged by `question_type` (5 LongMemEval classes).
   - Abstention precision = correctness rate on questions where `gold == "I don't know"` or normalized equivalent.
   - `judge_error_rate` = fraction with `Verdict::JudgeError`.
   - Token + cost aggregates: sum of `gen_result.tokens_in/out` and `judge_result.tokens_in/out` across all questions (split into the new `perf.generation` and `perf.judging` blocks).
4. **Emit `BenchResult`** with `answer: Some(AnswerMetrics)` populated, `ir: None`, schema_version `"3"`.

### 6.3 Per-question isolation overhead

500 setup/teardown cycles per system. For nark: 500 `nark init` calls (~10-50ms each = 5-25s total) + 500 model-cache stagings (cheap hard-links). For vector: 500 OnnxProvider loads (~100ms each = 50s total). Acceptable. Optimization (keep provider alive across questions) deferred — not a correctness issue, just throughput.

### 6.4 LOCOMO equivalent

`bench/src/tasks/locomo.rs` mirrors LongMemEval with format adjustments. Different upstream JSON structure (LOCOMO has conversations + multiple question types per conversation), but the gen+judge+aggregate flow is identical. Both tasks call the same `LlmCache` instance during a sweep — cache hits cross datasets where prompts coincide.

LOCOMO's `BenchResult.properties` block carries a methodology disclaimer JSON object inline:

```json
"properties": {
  "methodology_disclaimer": {
    "note": "LOCOMO scores are methodologically contested across published papers. Compare with caution.",
    "references": [
      "https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/",
      "https://www.letta.com/blog/benchmarking-ai-agent-memory"
    ]
  }
}
```

## 7. CLI, schema, baseline filenames

### 7.1 CLI additions

```bash
# Task A (Phase 1, unchanged)
cargo run -p nark-bench --release -- run --task ir \
  --systems fts5,nark,vector --corpus synthetic-tiny \
  --output bench/results/main

# Task B — LongMemEval (new)
cargo run -p nark-bench --release -- run --task longmemeval \
  --systems fts5,nark,vector \
  --gen-backend codex-cli --gen-model gpt-5.5 \
  --judge-backend codex-cli --judge-model gpt-5.5 \
  --output bench/results/main

# Task B — LOCOMO (same shape; different --task value)
cargo run -p nark-bench --release -- run --task locomo \
  --systems fts5,nark,vector \
  --gen-backend codex-cli --gen-model gpt-5.5 \
  --judge-backend codex-cli --judge-model gpt-5.5 \
  --output bench/results/main
```

New flags:
- `--gen-backend {claude-cli|codex-cli|api}` — which subprocess/HTTP path
- `--gen-model <id>` — pinned model identifier
- `--judge-backend {claude-cli|codex-cli|api}` — same
- `--judge-model <id>` — same
- `--llm-concurrency <N>` — parallel cap, default 4
- `--yes` — skip the pre-run confirmation prompt

Required when `--task` is `longmemeval` or `locomo`. Ignored when `--task` is `ir`.

### 7.2 Pre-run confirmation prompt

When a Task B run is about to issue >100 cache-miss LLM calls, print a summary to stderr and require `y` to proceed (unless `--yes` is set):

```
About to make ~3000 LLM calls (500 questions × 3 systems × 1 gen + 1 judge).
Estimated cost: subscription limits hit or ~$X.XX in API spend.
First run; subsequent runs use cache and finish in seconds.

Continue? [y/N]
```

Estimate based on cache miss count × average historical tokens-per-call from the cache (or a rough constant if cache is empty). Avoids "I just burned my Codex limits on a typo."

### 7.3 Schema bump to v3

`BenchResult.schema_version` becomes `"3"`. Three additions:

- **`answer: Option<AnswerMetrics>`** — Task B answer-accuracy block
- **`properties: serde_json::Value`** (default `{}`) — general-purpose extension point at the result level; used by LOCOMO to embed the methodology disclaimer (§6.4) and available for future per-run metadata (run-id, git SHA, env tags, etc.) without requiring further schema bumps
- **`perf.generation` / `perf.judging`** — new optional LLM-cost blocks

The new `AnswerMetrics` block:

```rust
pub struct AnswerMetrics {
    pub accuracy: f64,
    pub per_ability: HashMap<String, AbilityMetrics>,
    pub abstention_precision: f64,
    pub judge_error_rate: f64,
    pub questions: usize,
}

pub struct AbilityMetrics {
    pub accuracy: f64,
    pub questions: usize,
}
```

`ir: Option<IrMetrics>` (Phase 1) stays; both Options coexist. Task A fills `ir`, Task B fills `answer`, neither fills both.

`perf` gains two new optional blocks:

```rust
pub struct PerfMetrics {
    pub write: PhaseMetrics,
    pub search: PhaseMetrics,
    pub generation: Option<LlmPhaseMetrics>,   // populated for Task B
    pub judging: Option<LlmPhaseMetrics>,       // populated for Task B
}

pub struct LlmPhaseMetrics {
    pub calls: usize,
    pub cache_hits: usize,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_micros: u64,
}
```

**Backward compatibility**: serde `Option` deserialization means schema v2 baseline JSONs still parse with serde — `answer` and the new perf blocks become `None`. No forced regeneration of Task A baselines required, though the regression-check script's schema warning will fire once until v2 baselines are re-bootstrapped.

### 7.4 Baseline filenames

Phase 1c established `<task>-<system>-<corpus>-<config>.json`. For Task B the corpus slot holds the dataset name:

- `bench/results/main/longmemeval-fts5-longmemeval-default.json`
- `bench/results/main/longmemeval-nark-longmemeval-default.json`
- `bench/results/main/longmemeval-vector-longmemeval-default.json`
- `bench/results/main/locomo-{fts5,nark,vector}-locomo-default.json`

Redundant (task and corpus name match for Task B), but consistent with existing naming. **Decision deferred**: a `<corpus_version>` (e.g. upstream commit short-hash) component could be added later when multiple dataset variants need to coexist (LongMemEval-S vs -M).

### 7.5 Validation step — NOT in Phase 2

The plan should *call out* that until Phase 3 ships a mem0 adapter and we can run mem0's published methodology through our infrastructure and reproduce their ~91.6% on LongMemEval (±2%), our Phase 2 numbers are "internal directional," not "publishable comparable." Document this clearly in Phase 2's release notes / commit message so no one publishes premature numbers.

## 8. Repo changes summary

```
bench/
├── cache/
│   └── llm.db                                  # NEW (committed; sqlite cache)
├── datasets/
│   ├── longmemeval/                            # NEW
│   │   ├── fetch.sh
│   │   ├── README.md
│   │   └── upstream/                           # .gitignored
│   └── locomo/                                 # NEW
│       ├── fetch.sh
│       ├── README.md
│       └── upstream/                           # .gitignored
├── llm/                                        # NEW (prompts)
│   ├── README.md
│   └── prompts/
│       ├── longmemeval-generate.md
│       ├── longmemeval-judge.md
│       ├── locomo-generate.md
│       └── locomo-judge.md
├── src/
│   ├── llm/                                    # NEW
│   │   ├── mod.rs
│   │   ├── claude_cli.rs
│   │   ├── codex_cli.rs
│   │   ├── api.rs
│   │   ├── cache.rs
│   │   ├── prompt.rs
│   │   └── eval.rs                             # generate_answer, judge_answer
│   ├── tasks/
│   │   ├── ir.rs                               # Phase 1 (unchanged)
│   │   ├── longmemeval.rs                      # NEW
│   │   └── locomo.rs                           # NEW
│   ├── main.rs                                 # MODIFIED — task dispatch + new flags
│   ├── result.rs                               # MODIFIED — schema v3, AnswerMetrics
│   └── protocol.rs                             # unchanged (Adapter trait stable)
└── results/main/
    ├── (Phase 1 baselines, schema v2)
    ├── longmemeval-fts5-longmemeval-default.json    # NEW after first run
    ├── longmemeval-nark-longmemeval-default.json    # NEW
    ├── longmemeval-vector-longmemeval-default.json  # NEW
    └── locomo-{fts5,nark,vector}-locomo-default.json # NEW
```

`bench/Cargo.toml` gains:
- `tokio = { version = "1", features = ["sync", "rt"] }` if we adopt async semaphores (else `std::sync` works)
- No other new deps; `ureq`, `rusqlite`, `serde_json` already present

## 9. Phasing for the implementation plan

15 tasks roughly in dependency order:

1. **`LlmBackend` trait + `ClaudeCliBackend`** with subprocess + retry + concurrency-cap plumbing. Unit-tested with a fake `EchoBackend` (returns canned responses).
2. **`CodexCliBackend`** mirroring claude implementation; verified by hand against a small prompt.
3. **`ApiBackend`** (HTTP via ureq to Anthropic). Gated on `ANTHROPIC_API_KEY` env; CI skip default.
4. **`LlmCache` (sqlite)** at `bench/cache/llm.db`. Schema, get/put, cache-key derivation. Unit tests with in-memory sqlite.
5. **Prompt loader** — `<!-- prompt-version: N -->` header extraction, `{{var}}` substitution. Unit-tested.
6. **`Verdict` + `judge_answer` + `generate_answer` free functions**. Unit-tested with `EchoBackend`.
7. **Schema v3** — add `answer: Option<AnswerMetrics>`, generation/judging perf blocks; bump `BenchResult::new` version string.
8. **LongMemEval `fetch.sh` + loader module** — parses upstream JSON into structured questions.
9. **`run_longmemeval_task` runner** — per-question loop; gen + judge; aggregation.
10. **CLI flag additions** — `--gen-backend`, `--gen-model`, `--judge-backend`, `--judge-model`, `--llm-concurrency`, `--yes`; pre-run confirmation prompt.
11. **First LongMemEval run + commit baseline** — manual; expect 1-3h on first sweep depending on subscription throughput.
12. **LOCOMO `fetch.sh` + loader module**.
13. **`run_locomo_task` runner** + methodology disclaimer in result `properties`.
14. **First LOCOMO run + commit baseline**.
15. **Integration smoke test for Task B** — uses tiny fixture dataset (5-10 questions, 1-2 systems, `EchoBackend` for LLM mocking) to verify the pipeline end-to-end without real LLM calls. Add to `cargo test -p nark-bench --test smoke` alongside the Task A smoke.

**CI workflow does NOT change.** Lane 1 stays IR-only. Task B is too expensive per PR. Manual `cargo run` is the run pattern for Task B.

## 10. Open items + deferred scope

### To verify during implementation

- **LongMemEval upstream methodology details**: what `k` does their eval use for retrieval (we assume k=10), what gen model do their published baselines use (we assume GPT-4-class via `codex-cli:gpt-5.5`), what judge model. The plan must include a verification task that reads `upstream/eval.py` (or equivalent) and either confirms our defaults match or documents the substitution.
- **`codex exec` output format flag** — spec assumes `--json` modeled on `claude -p --output-format json`. Needs verification against the actual codex CLI when implementer first hooks up `CodexCliBackend`.
- **Tokenizer-aware token counting** — CLI tools may not return token counts. Fallback strategy: tiktoken-rs crate for OpenAI models, anthropic-tokenizer for Claude. Final choice deferred to Task 1's implementer.

### Explicitly deferred to later phases

- **Mem0 / Letta / Graphiti competitor adapters** — Phase 3. Lets us run cross-system comparisons (and crucially, validates that our LongMemEval implementation reproduces published numbers).
- **CI inclusion of Task B** — Phase 3 or later; needs API tokens in repo secrets, budget discussion.
- **LongMemEval-M / LongMemEval-L variants** — Phase 4 or later; just longer haystacks, same code path.
- **Panel-of-judges feature** — explicitly dropped during brainstorming. Could be revisited if a "validation lane" is needed in Phase 4+.
- **Generation prompt iteration beyond v1** — bench ships a faithful reproduction of upstream's gen prompt; if numbers look wrong, prompt-engineering is a separate cycle.
- **Per-question telemetry in result JSON** — aggregate totals only in `perf.generation/judging`; per-question cost/latency is a future telemetry feature.
- **A `--validate-methodology` CLI flag** — once Phase 3's mem0 adapter exists, a special run mode that compares mem0's output to mem0's published 91.6% baseline. Phase 3 or later.

## 11. What success looks like

After Phase 2 lands and the first sweep runs:

- `cargo run -p nark-bench -- run --task longmemeval --systems fts5,nark,vector --gen-backend codex-cli --gen-model gpt-5.5 --judge-backend codex-cli --judge-model gpt-5.5 --output bench/results/main` produces three result JSONs with `answer` blocks populated.
- nark's `accuracy` is reportable; per-ability breakdown shows which LongMemEval classes (info-extraction, multi-session, etc.) nark's hybrid pipeline handles well or poorly.
- `bench/cache/llm.db` is committed (~MB), so future contributors can `git pull && cargo run` and reproduce the same numbers without burning their own LLM credits.
- The integration smoke test `cargo test -p nark-bench --test smoke` continues to pass on every PR (Task A still gated; Task B verified via fixture + EchoBackend).
- A note in the commit message and in `bench/llm/README.md` records: "Phase 2 numbers are internal directional. Reproducibility-vs-published-baselines validation requires Phase 3 mem0 adapter and is not yet done."
