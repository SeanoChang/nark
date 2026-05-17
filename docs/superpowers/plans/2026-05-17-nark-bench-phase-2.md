# nark Bench — Phase 2 (LongMemEval + LOCOMO + LLM Backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Task B (answer-accuracy benchmarks against LongMemEval + LOCOMO with LLM-as-judge) to the bench. Ship the supporting `LlmBackend` infrastructure (subprocess + cache + prompt versioning), the per-question retrieve→generate→judge runner, and the first set of locked-in baselines.

**Architecture:** A new `bench/src/llm/` module hosts a single `LlmBackend` trait with three implementations (`ClaudeCliBackend`, `CodexCliBackend`, `ApiBackend`) and an `EchoBackend` fixture for testing. A SQLite cache at `bench/cache/llm.db` is committed to git so re-runs are free. Free functions `generate_answer` and `judge_answer` compose the trait with prompt templates. Two new task runners (`run_longmemeval_task`, `run_locomo_task`) iterate per-question, isolating each question's haystack. Result schema bumps to `"3"` with backward-compatible `answer: Option<AnswerMetrics>` and `properties: serde_json::Value` fields.

**Tech Stack:** Rust 2024; existing `rusqlite` (Phase 1); `ureq` (Phase 1) for HTTP-API backend; `sha2` (Phase 1, used by nark) for cache key hashing; `subprocess` via `std::process::Command`. No new heavy dependencies — possible additions if implementer needs them: `tiktoken-rs` for token counting (Task 1 decides).

**Spec reference:** `docs/superpowers/specs/2026-05-17-phase-2-design.md`

**Branch:** `feat/bench-phase-2` (already on this branch).

---

## File structure

**Files to create (new):**

| Path | Responsibility |
|---|---|
| `bench/src/llm/mod.rs` | `LlmBackend` trait, `LlmResponse` struct, module declarations |
| `bench/src/llm/echo.rs` | `EchoBackend` — test fixture returning canned responses |
| `bench/src/llm/claude_cli.rs` | `ClaudeCliBackend` — subprocess to `claude -p` |
| `bench/src/llm/codex_cli.rs` | `CodexCliBackend` — subprocess to `codex exec` |
| `bench/src/llm/api.rs` | `ApiBackend` — HTTP POST to Anthropic API |
| `bench/src/llm/cache.rs` | `LlmCache` — SQLite cache for LLM responses |
| `bench/src/llm/prompt.rs` | Prompt loader (version-header extraction + `{{var}}` substitution) |
| `bench/src/llm/eval.rs` | `Verdict`, `judge_answer`, `generate_answer` free functions |
| `bench/llm/README.md` | How prompts work, versioning convention |
| `bench/llm/prompts/longmemeval-generate.md` | LongMemEval generation prompt (v1) |
| `bench/llm/prompts/longmemeval-judge.md` | LongMemEval judge prompt (v1) |
| `bench/llm/prompts/locomo-generate.md` | LOCOMO generation prompt (v1) |
| `bench/llm/prompts/locomo-judge.md` | LOCOMO judge prompt (v1) |
| `bench/datasets/longmemeval/fetch.sh` | Clone xiaowu0162/LongMemEval at pinned SHA |
| `bench/datasets/longmemeval/README.md` | Methodology notes + pinned commit SHA |
| `bench/datasets/locomo/fetch.sh` | Clone snap-research/locomo at pinned SHA |
| `bench/datasets/locomo/README.md` | Same shape as LongMemEval |
| `bench/src/tasks/longmemeval.rs` | `run_longmemeval_task` — per-question loop, gen+judge, aggregate |
| `bench/src/tasks/locomo.rs` | `run_locomo_task` — same shape, LOCOMO format |
| `bench/tests/smoke_b.rs` | Integration smoke for Task B (uses `EchoBackend`) |
| `bench/cache/llm.db` | SQLite cache, committed to git after first run |
| `bench/results/main/longmemeval-{fts5,nark,vector}-longmemeval-default.json` | Baselines after first LongMemEval run |
| `bench/results/main/locomo-{fts5,nark,vector}-locomo-default.json` | Baselines after first LOCOMO run |

**Files to modify:**

| Path | Change |
|---|---|
| `bench/Cargo.toml` | Add `sha2` (already a workspace dep via nark) as direct dep; possibly `tiktoken-rs` if Task 1 chooses |
| `bench/src/main.rs` | `mod llm;`; add `--task longmemeval` / `--task locomo` dispatch; add `--gen-backend` / `--gen-model` / `--judge-backend` / `--judge-model` / `--llm-concurrency` / `--yes` flags |
| `bench/src/result.rs` | Bump `schema_version` to `"3"`; add `answer: Option<AnswerMetrics>`, `properties: serde_json::Value`, `perf.generation`, `perf.judging` |
| `bench/src/tasks/mod.rs` | `pub mod longmemeval; pub mod locomo;` |
| `.gitignore` | Add `bench/datasets/*/upstream/` entries |

**No changes to:**
- `nark/` (the main crate; Phase 2 is bench-only)
- `bench/src/adapters/*` (existing Adapter trait stable; Task B uses it unchanged)
- `bench/tests/smoke.rs` (Task A smoke untouched; Task B has its own smoke at `smoke_b.rs`)
- `.github/workflows/bench-pr.yml` (CI stays IR-only; Task B runs are manual)

---

## Task 1: `LlmBackend` trait + `EchoBackend` fixture + `ClaudeCliBackend`

**Files:**
- Create: `bench/src/llm/mod.rs`
- Create: `bench/src/llm/echo.rs`
- Create: `bench/src/llm/claude_cli.rs`
- Modify: `bench/src/main.rs` (add `mod llm;`)

The trait + the test fixture + the first real backend, all in one task because they're tightly coupled: writing the trait without an `EchoBackend` makes it untestable, and writing `ClaudeCliBackend` without the trait is meaningless. `CodexCliBackend` and `ApiBackend` come in Tasks 2 and 3.

- [ ] **Step 1.1: Create `bench/src/llm/mod.rs`**

```rust
//! LLM call abstraction for Task B (generation + judging).
//!
//! Single trait `LlmBackend` abstracts over `claude -p`, `codex exec`, and the
//! Anthropic HTTP API. Subprocess plumbing for the CLI backends is identical;
//! only the binary name + JSON parsing differs. The trait is shared between
//! generation and judging — the difference between those two roles is the
//! prompt template + the response parsing (see `eval.rs`).

use anyhow::Result;

pub mod cache;
pub mod claude_cli;
pub mod echo;
// pub mod codex_cli;   // wired in Task 2
// pub mod api;         // wired in Task 3
// pub mod prompt;      // wired in Task 5
// pub mod eval;        // wired in Task 6

/// A backend capable of taking a prompt string and returning an LLM response.
///
/// Implementations:
/// - [`echo::EchoBackend`] — test fixture; returns canned responses.
/// - [`claude_cli::ClaudeCliBackend`] — `claude -p` subprocess (OAuth from disk).
/// - [`codex_cli::CodexCliBackend`] — `codex exec` subprocess (OAuth from disk).
/// - [`api::ApiBackend`] — HTTP POST to Anthropic API (`ANTHROPIC_API_KEY`).
///
/// Backends are not `Sync` by default — they hold internal mutable state
/// (concurrency semaphores, subprocess handles). For parallel use, wrap in
/// `Mutex` at the call site.
pub trait LlmBackend: Send {
    /// Human-readable backend name, e.g. `"claude-cli"`, `"echo"`. Used in
    /// result JSON and cache keys.
    fn name(&self) -> &str;

    /// Model identifier, e.g. `"claude-opus-4-7"`, `"gpt-5.5"`. Used in
    /// cache keys so the same prompt against different models is cached
    /// separately.
    fn model_id(&self) -> &str;

    /// Send `prompt` to the LLM and return the response. Implementations
    /// SHOULD retry transient failures once with exponential backoff before
    /// returning Err.
    fn call(&mut self, prompt: &str) -> Result<LlmResponse>;
}

/// One LLM response.
///
/// `cost_usd_micros` is dollars × 1e6 (`u64` to avoid float drift in
/// aggregate cost reporting). Token counts are as reported by the backend
/// where possible; estimated via tokenizer-aware count for backends that
/// don't return them (currently none — both CLI backends and the HTTP API
/// report token counts).
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_micros: u64,
}
```

- [ ] **Step 1.2: Create `bench/src/llm/echo.rs` (test fixture)**

```rust
//! Echo backend — test fixture for `LlmBackend`. Returns canned responses
//! by prompt prefix. Used by all `LlmBackend` consumers in unit + integration
//! tests so real LLM calls never happen during `cargo test`.

use anyhow::{anyhow, Result};
use std::collections::HashMap;

use super::{LlmBackend, LlmResponse};

pub struct EchoBackend {
    /// Map from prompt prefix to canned response text. The longest matching
    /// prefix wins.
    responses: HashMap<String, String>,
    /// Default response if no prefix matches. If None, `call` returns Err
    /// for unmatched prompts (useful for tests that should fail loudly on
    /// unexpected calls).
    default: Option<String>,
}

impl EchoBackend {
    pub fn new() -> Self {
        Self { responses: HashMap::new(), default: None }
    }

    /// Register a canned response for prompts starting with `prefix`.
    pub fn with(mut self, prefix: &str, response: &str) -> Self {
        self.responses.insert(prefix.to_string(), response.to_string());
        self
    }

    /// Set a default response for unmatched prompts.
    pub fn with_default(mut self, response: &str) -> Self {
        self.default = Some(response.to_string());
        self
    }
}

impl Default for EchoBackend {
    fn default() -> Self { Self::new() }
}

impl LlmBackend for EchoBackend {
    fn name(&self) -> &str { "echo" }
    fn model_id(&self) -> &str { "echo-v1" }

    fn call(&mut self, prompt: &str) -> Result<LlmResponse> {
        // Find the longest matching prefix.
        let best = self.responses.iter()
            .filter(|(prefix, _)| prompt.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, response)| response.clone())
            .or_else(|| self.default.clone());

        match best {
            Some(text) => Ok(LlmResponse {
                tokens_in: prompt.len() as u64 / 4,      // rough estimate
                tokens_out: text.len() as u64 / 4,
                cost_usd_micros: 0,
                text,
            }),
            None => Err(anyhow!("EchoBackend: no matching prefix for prompt: {}", &prompt[..prompt.len().min(80)])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_returns_canned_response() {
        let mut b = EchoBackend::new().with("hello", "world");
        let resp = b.call("hello there").unwrap();
        assert_eq!(resp.text, "world");
    }

    #[test]
    fn echo_returns_longest_matching_prefix() {
        let mut b = EchoBackend::new()
            .with("h", "short")
            .with("hello", "long");
        let resp = b.call("hello world").unwrap();
        assert_eq!(resp.text, "long");
    }

    #[test]
    fn echo_uses_default_when_no_match() {
        let mut b = EchoBackend::new().with_default("fallback");
        let resp = b.call("anything").unwrap();
        assert_eq!(resp.text, "fallback");
    }

    #[test]
    fn echo_errors_when_no_match_and_no_default() {
        let mut b = EchoBackend::new().with("foo", "bar");
        let err = b.call("unknown prompt").unwrap_err();
        assert!(err.to_string().contains("no matching prefix"));
    }
}
```

- [ ] **Step 1.3: Create `bench/src/llm/claude_cli.rs`**

```rust
//! `claude -p` subprocess backend.
//!
//! Spawns `claude -p --model <model> --output-format json <prompt>`. The
//! `--output-format json` flag wraps Claude's reply in a JSON envelope that
//! includes `result` (the text), `cost_usd`, and (where available) token
//! counts. We parse that envelope, return `LlmResponse`.
//!
//! Auth: uses the OAuth session on disk (`~/.claude/credentials`). No
//! API key required if the user is logged in to Claude Code.
//!
//! Concurrency: each backend instance holds an internal mutex-guarded
//! counter capping in-flight calls. Default 4. Override via `with_concurrency`.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::process::Command;
use std::sync::{Arc, Mutex};

use super::{LlmBackend, LlmResponse};

const DEFAULT_CONCURRENCY: usize = 4;

pub struct ClaudeCliBackend {
    model: String,
    binary: String,          // typically "claude"; overrideable for tests
    in_flight: Arc<Mutex<usize>>,
    max_concurrency: usize,
}

impl ClaudeCliBackend {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            binary: "claude".to_string(),
            in_flight: Arc::new(Mutex::new(0)),
            max_concurrency: DEFAULT_CONCURRENCY,
        }
    }

    pub fn with_binary(mut self, binary: impl Into<String>) -> Self {
        self.binary = binary.into();
        self
    }

    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }
}

#[derive(Debug, Deserialize)]
struct ClaudeOutput {
    /// The text response.
    result: String,
    /// Cost in USD (e.g. 0.0123). May be absent for cached responses.
    #[serde(default)]
    cost_usd: Option<f64>,
    /// Token usage. Field name varies by version; check both shapes.
    #[serde(default, alias = "tokens")]
    usage: Option<ClaudeUsage>,
}

#[derive(Debug, Deserialize)]
struct ClaudeUsage {
    #[serde(default, alias = "input_tokens", alias = "in")]
    input: u64,
    #[serde(default, alias = "output_tokens", alias = "out")]
    output: u64,
}

impl LlmBackend for ClaudeCliBackend {
    fn name(&self) -> &str { "claude-cli" }
    fn model_id(&self) -> &str { &self.model }

    fn call(&mut self, prompt: &str) -> Result<LlmResponse> {
        // Concurrency gate: block until in_flight < max.
        loop {
            let mut count = self.in_flight.lock().unwrap();
            if *count < self.max_concurrency {
                *count += 1;
                break;
            }
            drop(count);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // Defer count decrement until function exit via guard.
        struct Guard<'a> { counter: &'a Mutex<usize> }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                let mut count = self.counter.lock().unwrap();
                *count = count.saturating_sub(1);
            }
        }
        let _guard = Guard { counter: &self.in_flight };

        // Subprocess invocation. Retry once on transient failure.
        let mut attempts = 0;
        loop {
            attempts += 1;
            let output = Command::new(&self.binary)
                .arg("-p")
                .arg("--model").arg(&self.model)
                .arg("--output-format").arg("json")
                .arg(prompt)
                .output()
                .with_context(|| format!("failed to spawn `{} -p`", self.binary))?;

            if output.status.success() {
                let parsed: ClaudeOutput = serde_json::from_slice(&output.stdout)
                    .with_context(|| format!(
                        "failed to parse claude --output-format json: {}",
                        String::from_utf8_lossy(&output.stdout)
                    ))?;

                let (tokens_in, tokens_out) = parsed.usage
                    .map(|u| (u.input, u.output))
                    .unwrap_or((0, 0));
                let cost_usd_micros = parsed.cost_usd
                    .map(|c| (c * 1_000_000.0).round() as u64)
                    .unwrap_or(0);

                return Ok(LlmResponse {
                    text: parsed.result,
                    tokens_in,
                    tokens_out,
                    cost_usd_micros,
                });
            }

            // Failure path: retry once with brief backoff.
            if attempts >= 2 {
                return Err(anyhow!(
                    "claude -p failed after {} attempts: {}",
                    attempts,
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_claude_output_with_usage() {
        let json = r#"{
            "result": "Hello, world",
            "cost_usd": 0.0042,
            "usage": {"input_tokens": 10, "output_tokens": 3}
        }"#;
        let parsed: ClaudeOutput = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.result, "Hello, world");
        assert!(parsed.cost_usd.is_some());
        let usage = parsed.usage.unwrap();
        assert_eq!(usage.input, 10);
        assert_eq!(usage.output, 3);
    }

    #[test]
    fn parses_claude_output_without_usage() {
        let json = r#"{"result": "ok"}"#;
        let parsed: ClaudeOutput = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.result, "ok");
        assert!(parsed.cost_usd.is_none());
        assert!(parsed.usage.is_none());
    }

    // No live `claude -p` invocation in unit tests — that's covered by
    // the integration smoke at bench/tests/smoke_b.rs (which gates on
    // a CLAUDE_CLI_AVAILABLE env var so CI skips it).
}
```

- [ ] **Step 1.4: Wire `mod llm;` into `bench/src/main.rs`**

Read `bench/src/main.rs`. After the existing `mod tasks;` declaration, add:

```rust
mod llm;
```

- [ ] **Step 1.5: Verify build**

Run: `cargo build -p nark-bench`
Expected: completes; only expected dead_code warnings (LlmBackend trait not yet wired into a runner). The cache module declaration in `mod.rs` is commented out — Task 4 will uncomment.

- [ ] **Step 1.6: Run unit tests**

Run: `cargo test -p nark-bench llm::`
Expected: 4 echo tests pass + 2 claude_cli parse tests pass = 6 tests passing.

- [ ] **Step 1.7: Commit**

```bash
git add bench/src/llm/ bench/src/main.rs
git commit -m "$(cat <<'EOF'
feat(bench/llm): LlmBackend trait + EchoBackend + ClaudeCliBackend

LlmBackend is the single abstraction for all LLM calls in Task B
(generation and judging both go through it). Three implementations
ship across this and the next two tasks: claude-cli (this task),
codex-cli (next), and api (HTTP, third).

EchoBackend is the test fixture used by every downstream consumer
in unit and integration tests — no real LLM calls during cargo test.

ClaudeCliBackend spawns `claude -p --model X --output-format json
<prompt>` and parses the JSON envelope for text + cost + token
usage. Internal mutex-guarded concurrency cap (default 4) prevents
runaway parallel subprocess spawns.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `CodexCliBackend`

**Files:**
- Create: `bench/src/llm/codex_cli.rs`
- Modify: `bench/src/llm/mod.rs` (uncomment `pub mod codex_cli;`)

Codex CLI's JSON output flag must be verified by the implementer at task start. Spec assumes `--json` (modeled on Claude's `--output-format json`); reality may differ.

- [ ] **Step 2.1: Verify the codex CLI JSON output flag**

Run: `codex exec --help 2>&1 | head -40`

Look for a JSON output option. Likely candidates: `--json`, `--output-format json`, `--format json`. Record the actual flag.

If `codex exec --help` is not available or codex is not installed: install via `npm install -g @openai/codex` or whatever the current install path is, then re-run. If installation isn't possible at this time, STOP and report BLOCKED — `CodexCliBackend` cannot be tested without it.

- [ ] **Step 2.2: Sanity-check the actual JSON output shape**

Run: `codex exec --model gpt-5.5 <whatever-flag-from-2.1> "Say hello in one word"`

Inspect the JSON. Expected fields likely include: the response text (probably `result` or `text`), token usage (probably `usage.input_tokens` / `output_tokens`), cost (probably `cost_usd`). Record the actual shape.

- [ ] **Step 2.3: Create `bench/src/llm/codex_cli.rs`**

Use the actual flag from Step 2.1 and the actual JSON shape from Step 2.2. The file's shape mirrors `claude_cli.rs` from Task 1. Substitute fields and command flags to match what `codex exec` produces.

If the actual JSON output matches Claude's shape closely (e.g. also has `result` / `cost_usd` / `usage.input_tokens`), the implementation is nearly a clone of `ClaudeCliBackend` with `binary = "codex"`, the discovered JSON output flag, and possibly different `name()` / `model_id()` values.

Sample (adjust to actual codex output):

```rust
//! `codex exec` subprocess backend.
//!
//! Spawns `codex exec --model <model> <JSON-FLAG> <prompt>` and parses the
//! JSON output. JSON-FLAG and parsing details discovered at Task 2 start
//! by running `codex exec --help` and inspecting actual output.
//!
//! Auth: uses the OAuth session on disk (`~/.codex/...`). No API key
//! required if the user is logged in to Codex Plus.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::process::Command;
use std::sync::{Arc, Mutex};

use super::{LlmBackend, LlmResponse};

const DEFAULT_CONCURRENCY: usize = 4;

pub struct CodexCliBackend {
    model: String,
    binary: String,
    in_flight: Arc<Mutex<usize>>,
    max_concurrency: usize,
}

impl CodexCliBackend {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            binary: "codex".to_string(),
            in_flight: Arc::new(Mutex::new(0)),
            max_concurrency: DEFAULT_CONCURRENCY,
        }
    }

    pub fn with_binary(mut self, binary: impl Into<String>) -> Self {
        self.binary = binary.into();
        self
    }

    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }
}

// Replace with actual codex JSON shape from Step 2.2.
#[derive(Debug, Deserialize)]
struct CodexOutput {
    // Example shape — adjust to match reality:
    result: String,
    #[serde(default)]
    cost_usd: Option<f64>,
    #[serde(default)]
    usage: Option<CodexUsage>,
}

#[derive(Debug, Deserialize)]
struct CodexUsage {
    #[serde(default, alias = "input_tokens", alias = "in")]
    input: u64,
    #[serde(default, alias = "output_tokens", alias = "out")]
    output: u64,
}

impl LlmBackend for CodexCliBackend {
    fn name(&self) -> &str { "codex-cli" }
    fn model_id(&self) -> &str { &self.model }

    fn call(&mut self, prompt: &str) -> Result<LlmResponse> {
        // Concurrency gate (same shape as ClaudeCliBackend)
        loop {
            let mut count = self.in_flight.lock().unwrap();
            if *count < self.max_concurrency {
                *count += 1;
                break;
            }
            drop(count);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        struct Guard<'a> { counter: &'a Mutex<usize> }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                let mut count = self.counter.lock().unwrap();
                *count = count.saturating_sub(1);
            }
        }
        let _guard = Guard { counter: &self.in_flight };

        let mut attempts = 0;
        loop {
            attempts += 1;
            // REPLACE the JSON-FLAG below with the actual codex flag.
            let output = Command::new(&self.binary)
                .arg("exec")
                .arg("--model").arg(&self.model)
                .arg("--json")     // <-- VERIFY against `codex exec --help`
                .arg(prompt)
                .output()
                .with_context(|| format!("failed to spawn `{} exec`", self.binary))?;

            if output.status.success() {
                let parsed: CodexOutput = serde_json::from_slice(&output.stdout)
                    .with_context(|| format!(
                        "failed to parse codex output JSON: {}",
                        String::from_utf8_lossy(&output.stdout)
                    ))?;

                let (tokens_in, tokens_out) = parsed.usage
                    .map(|u| (u.input, u.output))
                    .unwrap_or((0, 0));
                let cost_usd_micros = parsed.cost_usd
                    .map(|c| (c * 1_000_000.0).round() as u64)
                    .unwrap_or(0);

                return Ok(LlmResponse {
                    text: parsed.result,
                    tokens_in,
                    tokens_out,
                    cost_usd_micros,
                });
            }

            if attempts >= 2 {
                return Err(anyhow!(
                    "codex exec failed after {} attempts: {}",
                    attempts,
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_codex_output_with_usage() {
        // Adjust JSON to match actual codex output discovered in Step 2.2.
        let json = r#"{
            "result": "hello",
            "cost_usd": 0.0042,
            "usage": {"input_tokens": 10, "output_tokens": 3}
        }"#;
        let parsed: CodexOutput = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.result, "hello");
        assert!(parsed.cost_usd.is_some());
    }

    #[test]
    fn parses_codex_output_without_usage() {
        let json = r#"{"result": "ok"}"#;
        let parsed: CodexOutput = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.result, "ok");
    }
}
```

- [ ] **Step 2.4: Uncomment `pub mod codex_cli;` in `bench/src/llm/mod.rs`**

Edit `mod.rs`. Change the line:

```rust
// pub mod codex_cli;   // wired in Task 2
```

to:

```rust
pub mod codex_cli;
```

- [ ] **Step 2.5: Verify build + tests**

Run: `cargo build -p nark-bench && cargo test -p nark-bench llm::codex_cli`
Expected: build clean; 2 parse tests pass.

- [ ] **Step 2.6: Manual smoke (optional)**

If you have a Codex Plus subscription logged in:

```bash
cargo run -p nark-bench --release --bin nark-bench -- /dev/null  # not the real entrypoint; will fail
```

A proper end-to-end smoke against the real codex binary is gated on Tasks 6+ existing. For now, the parse-test coverage + the manual run from Step 2.2 are sufficient.

- [ ] **Step 2.7: Commit**

```bash
git add bench/src/llm/codex_cli.rs bench/src/llm/mod.rs
git commit -m "$(cat <<'EOF'
feat(bench/llm): CodexCliBackend

Mirror of ClaudeCliBackend with `codex exec` as the subprocess.
JSON output flag and response shape verified against the actual
codex CLI during this task (see Step 2.1-2.2 of the plan).

Default concurrency 4; configurable via with_concurrency.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `ApiBackend` (Anthropic HTTP)

**Files:**
- Create: `bench/src/llm/api.rs`
- Modify: `bench/src/llm/mod.rs` (uncomment `pub mod api;`)

Fallback backend for environments without OAuth-bound CLI subscriptions (CI, contributors without Claude Code). Uses `ANTHROPIC_API_KEY` env var.

- [ ] **Step 3.1: Create `bench/src/llm/api.rs`**

```rust
//! HTTP backend talking directly to the Anthropic API.
//!
//! Auth: `ANTHROPIC_API_KEY` env var. Returns an error if unset.
//! Use case: CI environments, contributors without Claude Code installed,
//! or scenarios where the OAuth CLIs aren't viable.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use super::{LlmBackend, LlmResponse};

const DEFAULT_CONCURRENCY: usize = 4;
const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

pub struct ApiBackend {
    model: String,
    api_key: String,
    in_flight: Arc<Mutex<usize>>,
    max_concurrency: usize,
}

impl ApiBackend {
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| anyhow!("ANTHROPIC_API_KEY env var not set"))?;
        if api_key.is_empty() {
            return Err(anyhow!("ANTHROPIC_API_KEY is empty"));
        }
        Ok(Self {
            model: model.into(),
            api_key,
            in_flight: Arc::new(Mutex::new(0)),
            max_concurrency: DEFAULT_CONCURRENCY,
        })
    }

    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }
}

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<MessageItem<'a>>,
}

#[derive(Serialize)]
struct MessageItem<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
}

impl LlmBackend for ApiBackend {
    fn name(&self) -> &str { "api" }
    fn model_id(&self) -> &str { &self.model }

    fn call(&mut self, prompt: &str) -> Result<LlmResponse> {
        // Concurrency gate
        loop {
            let mut count = self.in_flight.lock().unwrap();
            if *count < self.max_concurrency {
                *count += 1;
                break;
            }
            drop(count);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        struct Guard<'a> { counter: &'a Mutex<usize> }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                let mut count = self.counter.lock().unwrap();
                *count = count.saturating_sub(1);
            }
        }
        let _guard = Guard { counter: &self.in_flight };

        let req = MessagesRequest {
            model: &self.model,
            max_tokens: 2048,
            messages: vec![MessageItem { role: "user", content: prompt }],
        };

        let mut attempts = 0;
        loop {
            attempts += 1;
            let resp = ureq::post(API_URL)
                .set("x-api-key", &self.api_key)
                .set("anthropic-version", API_VERSION)
                .set("content-type", "application/json")
                .send_json(serde_json::to_value(&req).unwrap());

            match resp {
                Ok(r) => {
                    let body: MessagesResponse = r.into_json()
                        .context("failed to parse Anthropic API response")?;
                    let text = body.content.into_iter()
                        .filter(|b| b.block_type == "text")
                        .filter_map(|b| b.text)
                        .collect::<Vec<_>>()
                        .join("");
                    // Cost estimate: rough per-token pricing for claude-opus-4-7
                    // (~$15/Mtok in, ~$75/Mtok out). Update when pricing changes.
                    let cost_usd_micros = body.usage.input_tokens * 15
                        + body.usage.output_tokens * 75;
                    return Ok(LlmResponse {
                        text,
                        tokens_in: body.usage.input_tokens,
                        tokens_out: body.usage.output_tokens,
                        cost_usd_micros,
                    });
                }
                Err(e) => {
                    if attempts >= 2 {
                        return Err(anyhow!("Anthropic API failed after {} attempts: {}", attempts, e));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_errors_when_key_missing() {
        // SAFETY: single-threaded test, no other test reads this key here.
        unsafe { std::env::remove_var("ANTHROPIC_API_KEY"); }
        let result = ApiBackend::from_env("claude-opus-4-7");
        assert!(result.is_err());
    }

    #[test]
    fn parses_messages_response() {
        let json = r#"{
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 1}
        }"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.usage.input_tokens, 5);
        assert_eq!(parsed.content[0].text.as_deref(), Some("Hello"));
    }
}
```

- [ ] **Step 3.2: Uncomment `pub mod api;` in `bench/src/llm/mod.rs`**

Edit `mod.rs`. Change:

```rust
// pub mod api;         // wired in Task 3
```

to:

```rust
pub mod api;
```

- [ ] **Step 3.3: Verify build + tests**

Run: `cargo build -p nark-bench && cargo test -p nark-bench llm::api`
Expected: build clean; 2 tests pass.

- [ ] **Step 3.4: Commit**

```bash
git add bench/src/llm/api.rs bench/src/llm/mod.rs
git commit -m "$(cat <<'EOF'
feat(bench/llm): ApiBackend — HTTP fallback to Anthropic API

For CI environments + contributors without Claude Code's OAuth on
disk. Requires ANTHROPIC_API_KEY env var.

POSTs to /v1/messages with the standard messages payload. Cost
computed from token usage + a per-token rate constant (refresh when
Anthropic pricing changes). Concurrency cap same as CLI backends.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `LlmCache` (SQLite)

**Files:**
- Create: `bench/src/llm/cache.rs`
- Modify: `bench/src/llm/mod.rs` (uncomment `pub mod cache;`)

SQLite cache for LLM responses. Cache key derivation, get/put, in-memory unit tests.

- [ ] **Step 4.1: Create `bench/src/llm/cache.rs`**

```rust
//! SQLite-backed cache for LLM responses.
//!
//! Cache key: sha256(prompt + "\n" + model_id + "\n" + prompt_version).
//! Same prompt + same model + same prompt version = cache hit.
//!
//! Persisted to disk at the path provided to `LlmCache::open`. The bench
//! defaults to `bench/cache/llm.db` and commits this file so re-runs are
//! free across contributors.
//!
//! Schema is created idempotently on first open.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};
use std::path::Path;

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS responses (
  cache_key TEXT PRIMARY KEY,
  backend_name TEXT NOT NULL,
  model_id TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  call_kind TEXT NOT NULL,
  request TEXT NOT NULL,
  response TEXT NOT NULL,
  tokens_in INTEGER NOT NULL,
  tokens_out INTEGER NOT NULL,
  cost_usd_micros INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_backend ON responses(backend_name, model_id);
";

pub struct LlmCache {
    conn: Connection,
}

/// Entry stored in the cache. `request` is the full prompt; `response` is the
/// full text returned by the LLM.
#[derive(Debug, Clone)]
pub struct CachedEntry {
    pub backend_name: String,
    pub model_id: String,
    pub prompt_version: String,
    pub call_kind: String,
    pub request: String,
    pub response: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_micros: u64,
}

impl LlmCache {
    /// Open (or create) a cache at the given path. Use `":memory:"` for tests.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open llm cache at {:?}", path))?;
        conn.execute_batch(SCHEMA_SQL)
            .context("failed to initialize llm cache schema")?;
        Ok(Self { conn })
    }

    /// Compute the cache key from prompt + model_id + prompt_version.
    pub fn key(prompt: &str, model_id: &str, prompt_version: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        hasher.update(b"\n");
        hasher.update(model_id.as_bytes());
        hasher.update(b"\n");
        hasher.update(prompt_version.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn get(&self, key: &str) -> Result<Option<CachedEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT backend_name, model_id, prompt_version, call_kind, request, response,
                    tokens_in, tokens_out, cost_usd_micros
             FROM responses WHERE cache_key = ?1"
        )?;
        let mut rows = stmt.query(params![key])?;
        if let Some(row) = rows.next()? {
            Ok(Some(CachedEntry {
                backend_name: row.get(0)?,
                model_id: row.get(1)?,
                prompt_version: row.get(2)?,
                call_kind: row.get(3)?,
                request: row.get(4)?,
                response: row.get(5)?,
                tokens_in: row.get::<_, i64>(6)? as u64,
                tokens_out: row.get::<_, i64>(7)? as u64,
                cost_usd_micros: row.get::<_, i64>(8)? as u64,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn put(&self, key: &str, entry: &CachedEntry) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT OR REPLACE INTO responses
             (cache_key, backend_name, model_id, prompt_version, call_kind,
              request, response, tokens_in, tokens_out, cost_usd_micros, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                key,
                entry.backend_name,
                entry.model_id,
                entry.prompt_version,
                entry.call_kind,
                entry.request,
                entry.response,
                entry.tokens_in as i64,
                entry.tokens_out as i64,
                entry.cost_usd_micros as i64,
                now,
            ],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn mem() -> LlmCache {
        LlmCache::open(&PathBuf::from(":memory:")).unwrap()
    }

    #[test]
    fn key_is_deterministic() {
        let k1 = LlmCache::key("hello", "claude", "1");
        let k2 = LlmCache::key("hello", "claude", "1");
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_changes_with_prompt() {
        let k1 = LlmCache::key("hello", "claude", "1");
        let k2 = LlmCache::key("hello!", "claude", "1");
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_with_model() {
        let k1 = LlmCache::key("hello", "claude", "1");
        let k2 = LlmCache::key("hello", "gpt", "1");
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_with_prompt_version() {
        let k1 = LlmCache::key("hello", "claude", "1");
        let k2 = LlmCache::key("hello", "claude", "2");
        assert_ne!(k1, k2);
    }

    #[test]
    fn put_then_get_roundtrips() {
        let cache = mem();
        let key = "abc";
        let entry = CachedEntry {
            backend_name: "echo".into(),
            model_id: "echo-v1".into(),
            prompt_version: "1".into(),
            call_kind: "judge".into(),
            request: "Q".into(),
            response: "A".into(),
            tokens_in: 1,
            tokens_out: 2,
            cost_usd_micros: 100,
        };
        cache.put(key, &entry).unwrap();
        let got = cache.get(key).unwrap().unwrap();
        assert_eq!(got.response, "A");
        assert_eq!(got.tokens_in, 1);
    }

    #[test]
    fn get_returns_none_for_unknown_key() {
        let cache = mem();
        let got = cache.get("nonexistent").unwrap();
        assert!(got.is_none());
    }
}
```

- [ ] **Step 4.2: Add `chrono` import to `bench/Cargo.toml` if not already present**

Check `bench/Cargo.toml` — `chrono` should already be there from Phase 1a (used in `result.rs`). If not, add `chrono = { version = "0.4", features = ["serde"] }` to `[dependencies]`.

Also: `sha2` needs to be a direct dep. Check `bench/Cargo.toml`; if not present, add:

```toml
sha2 = "0.10"
```

(nark already uses sha2 transitively; bench may or may not have it as a direct dep yet.)

- [ ] **Step 4.3: Uncomment `pub mod cache;` in `bench/src/llm/mod.rs`**

- [ ] **Step 4.4: Verify build + run cache tests**

Run: `cargo build -p nark-bench && cargo test -p nark-bench llm::cache`
Expected: build clean; 6 cache tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add bench/src/llm/cache.rs bench/src/llm/mod.rs bench/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(bench/llm): LlmCache — SQLite-backed cache for LLM responses

Cache key: sha256(prompt + model_id + prompt_version). Designed to
be persisted at bench/cache/llm.db and committed to git, so reruns
of Task B are free across contributors.

Six unit tests with in-memory SQLite cover key determinism + each
of the three key components + put/get roundtrip + None-on-miss.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Prompt loader

**Files:**
- Create: `bench/src/llm/prompt.rs`
- Create: `bench/llm/README.md`
- Modify: `bench/src/llm/mod.rs` (uncomment `pub mod prompt;`)

Parses `<!-- prompt-version: N -->` headers; substitutes `{{var}}` placeholders.

- [ ] **Step 5.1: Create `bench/src/llm/prompt.rs`**

```rust
//! Prompt template loader.
//!
//! Convention: each prompt file begins with `<!-- prompt-version: N -->`
//! on the first line. The loader extracts N and uses it as the
//! `prompt_version` in cache keys. When the prompt's intent changes,
//! bumping N invalidates cache entries that used the old prompt.
//!
//! Templates use mustache-style `{{var}}` placeholders. The loader
//! substitutes from a key-value map at render time.

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct PromptTemplate {
    pub version: String,
    pub template: String,
}

impl PromptTemplate {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read prompt file {:?}", path))?;
        Self::parse(&raw)
    }

    pub fn parse(raw: &str) -> Result<Self> {
        // First line should be `<!-- prompt-version: N -->`.
        let first_line = raw.lines().next()
            .ok_or_else(|| anyhow!("prompt is empty"))?;
        let version = first_line
            .strip_prefix("<!-- prompt-version: ")
            .and_then(|s| s.strip_suffix(" -->"))
            .ok_or_else(|| anyhow!(
                "prompt missing or malformed first-line `<!-- prompt-version: N -->` header; got: {}",
                first_line
            ))?
            .trim()
            .to_string();

        if version.is_empty() {
            return Err(anyhow!("prompt-version is empty"));
        }

        // Template is everything after the first line + its trailing newline.
        let template = raw.strip_prefix(first_line)
            .map(|rest| rest.strip_prefix('\n').unwrap_or(rest))
            .unwrap_or(raw)
            .to_string();

        Ok(PromptTemplate { version, template })
    }

    /// Substitute `{{var}}` placeholders using the given map. Missing keys
    /// are left in place (with the `{{var}}` syntax intact) — this lets the
    /// caller stage substitutions in multiple passes if needed.
    pub fn render(&self, vars: &HashMap<&str, &str>) -> String {
        let mut output = self.template.clone();
        for (key, value) in vars {
            let placeholder = format!("{{{{{}}}}}", key);
            output = output.replace(&placeholder, value);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_prompt() {
        let raw = "<!-- prompt-version: 1 -->\n\nHello {{name}}!";
        let p = PromptTemplate::parse(raw).unwrap();
        assert_eq!(p.version, "1");
        assert_eq!(p.template, "\nHello {{name}}!");
    }

    #[test]
    fn errors_on_missing_header() {
        let raw = "Hello world";
        let err = PromptTemplate::parse(raw).unwrap_err();
        assert!(err.to_string().contains("prompt-version"));
    }

    #[test]
    fn errors_on_empty_prompt() {
        let raw = "";
        let err = PromptTemplate::parse(raw).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn substitutes_placeholders() {
        let p = PromptTemplate { version: "1".into(), template: "Hi {{name}}, age {{age}}.".into() };
        let mut vars = HashMap::new();
        vars.insert("name", "Alice");
        vars.insert("age", "30");
        assert_eq!(p.render(&vars), "Hi Alice, age 30.");
    }

    #[test]
    fn leaves_unmatched_placeholders_intact() {
        let p = PromptTemplate { version: "1".into(), template: "Hi {{name}}, age {{age}}.".into() };
        let mut vars = HashMap::new();
        vars.insert("name", "Alice");
        assert_eq!(p.render(&vars), "Hi Alice, age {{age}}.");
    }

    #[test]
    fn parses_version_with_whitespace() {
        let raw = "<!-- prompt-version:   42   -->\nbody";
        let p = PromptTemplate::parse(raw).unwrap();
        assert_eq!(p.version, "42");
    }
}
```

- [ ] **Step 5.2: Create `bench/llm/README.md`**

```markdown
# nark-bench LLM Prompts

This directory holds the prompt templates used by Task B's generation
and judging steps. Each task (LongMemEval, LOCOMO) has two prompts:

- `prompts/<task>-generate.md` — sent to the generation LLM after retrieval
- `prompts/<task>-judge.md` — sent to the judging LLM to score the generated answer

## Versioning convention

Every prompt file's first line MUST be `<!-- prompt-version: N -->`.

The prompt loader extracts N and uses it as part of the cache key:
`sha256(prompt + model_id + prompt_version)`. When you change a prompt's
intent (not just whitespace or wording polish), bump N. The cache will
re-query the LLM for affected entries; old cached responses become
unreachable.

Whitespace-only or comment-only changes do not require a version bump —
the rendered prompt is identical, so the cache key is identical.

When in doubt, bump. Cost of an unnecessary bump: re-querying ~500
questions × 1-2 backends = a one-time subscription-limits hit (then
cached). Cost of NOT bumping when you should: silently mixed-methodology
results in the same baseline JSON, hard to detect after the fact.

## Placeholder syntax

Templates use `{{var}}` placeholders. The loader substitutes from a
`HashMap<&str, &str>` at render time. Missing keys are left in place —
this lets multi-pass substitution work cleanly.

Common placeholders:

- `{{question}}` — the question text from the dataset
- `{{gold}}` — the gold answer (judge prompts only)
- `{{candidate}}` — the generated answer being judged (judge prompts only)
- `{{context}}` — retrieved context snippets joined with `\n---\n` (generation prompts only)

## Iteration discipline

1. Write `<!-- prompt-version: 1 -->` initial prompt.
2. Run a subset of questions (10-20) with `--task longmemeval --systems vector` and inspect verdicts manually.
3. If verdicts look wrong (judge too harsh, too lenient, misclassifying), edit prompt and bump version.
4. Cache invalidates affected entries on next run. Re-run subset.
5. Once stable, run full sweep. Commit the locked-in prompt.

Prompt history lives in git — `git log -p bench/llm/prompts/<file>` shows
methodology evolution, which is the credibility argument when publishing
numbers.
```

- [ ] **Step 5.3: Uncomment `pub mod prompt;` in `bench/src/llm/mod.rs`**

- [ ] **Step 5.4: Verify build + tests**

Run: `cargo build -p nark-bench && cargo test -p nark-bench llm::prompt`
Expected: 6 prompt tests pass.

- [ ] **Step 5.5: Commit**

```bash
git add bench/src/llm/prompt.rs bench/llm/README.md bench/src/llm/mod.rs
git commit -m "$(cat <<'EOF'
feat(bench/llm): prompt template loader + versioning convention

PromptTemplate::load reads a file; ::parse parses the
<!-- prompt-version: N --> header and extracts the template body.
render() substitutes {{var}} placeholders from a HashMap.

bench/llm/README.md documents the convention: prompts live in
bench/llm/prompts/, versioning bumps when intent changes (cache
invalidates), templates use {{var}} substitution.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `Verdict` + `judge_answer` + `generate_answer`

**Files:**
- Create: `bench/src/llm/eval.rs`
- Modify: `bench/src/llm/mod.rs` (uncomment `pub mod eval;`; re-export `Verdict` for convenience)

The free functions that compose `LlmBackend` + `PromptTemplate` + `LlmCache` into generation and judging operations.

- [ ] **Step 6.1: Create `bench/src/llm/eval.rs`**

```rust
//! Free functions that compose LlmBackend + PromptTemplate + LlmCache
//! into the two roles Task B needs: generation and judging.
//!
//! `generate_answer` substitutes question + context into the gen template,
//! checks cache, calls backend, caches the result. Returns the candidate
//! answer text.
//!
//! `judge_answer` substitutes question + gold + candidate into the judge
//! template, checks cache, calls backend, parses the JSON verdict from the
//! response. Retries once on JSON parse failure. Returns Verdict + reason.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::cache::{CachedEntry, LlmCache};
use super::prompt::PromptTemplate;
use super::LlmBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Correct,
    Partial,
    Incorrect,
    JudgeError,
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

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub candidate: String,
    pub from_cache: bool,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_micros: u64,
}

#[derive(Debug, Clone)]
pub struct JudgmentResult {
    pub verdict: Verdict,
    pub reason: String,
    pub from_cache: bool,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_micros: u64,
}

/// What the judge LLM returns in its JSON envelope. The actual prompt
/// instructs the LLM to emit ONLY this JSON on a single line.
#[derive(Debug, Deserialize)]
struct JudgeOutput {
    verdict: String,
    #[serde(default)]
    reason: String,
}

pub fn generate_answer(
    backend: &mut dyn LlmBackend,
    cache: &LlmCache,
    template: &PromptTemplate,
    question: &str,
    context_snippets: &[String],
) -> Result<GenerationResult> {
    let context_joined = context_snippets.join("\n---\n");
    let mut vars: HashMap<&str, &str> = HashMap::new();
    vars.insert("question", question);
    vars.insert("context", &context_joined);
    let prompt = template.render(&vars);

    let key = LlmCache::key(&prompt, backend.model_id(), &template.version);

    if let Some(entry) = cache.get(&key)? {
        return Ok(GenerationResult {
            candidate: entry.response.trim().to_string(),
            from_cache: true,
            tokens_in: 0,
            tokens_out: 0,
            cost_usd_micros: 0,
        });
    }

    let resp = backend.call(&prompt)?;
    let candidate = resp.text.trim().to_string();
    let entry = CachedEntry {
        backend_name: backend.name().to_string(),
        model_id: backend.model_id().to_string(),
        prompt_version: template.version.clone(),
        call_kind: "generate".to_string(),
        request: prompt,
        response: candidate.clone(),
        tokens_in: resp.tokens_in,
        tokens_out: resp.tokens_out,
        cost_usd_micros: resp.cost_usd_micros,
    };
    cache.put(&key, &entry)?;

    Ok(GenerationResult {
        candidate,
        from_cache: false,
        tokens_in: resp.tokens_in,
        tokens_out: resp.tokens_out,
        cost_usd_micros: resp.cost_usd_micros,
    })
}

pub fn judge_answer(
    backend: &mut dyn LlmBackend,
    cache: &LlmCache,
    template: &PromptTemplate,
    question: &str,
    gold: &str,
    candidate: &str,
) -> Result<JudgmentResult> {
    let mut vars: HashMap<&str, &str> = HashMap::new();
    vars.insert("question", question);
    vars.insert("gold", gold);
    vars.insert("candidate", candidate);
    let prompt = template.render(&vars);

    let key = LlmCache::key(&prompt, backend.model_id(), &template.version);

    if let Some(entry) = cache.get(&key)? {
        let (verdict, reason) = parse_verdict(&entry.response);
        return Ok(JudgmentResult {
            verdict,
            reason,
            from_cache: true,
            tokens_in: 0,
            tokens_out: 0,
            cost_usd_micros: 0,
        });
    }

    // Call backend; retry once on parse failure.
    let mut attempts = 0;
    let resp = loop {
        attempts += 1;
        let r = backend.call(&prompt)?;
        let trimmed = r.text.trim();
        let parsed: Result<JudgeOutput, _> = serde_json::from_str(trimmed);
        if parsed.is_ok() || attempts >= 2 {
            break r;
        }
        // Retry once.
    };

    let trimmed = resp.text.trim();
    let (verdict, reason) = parse_verdict(trimmed);
    let entry = CachedEntry {
        backend_name: backend.name().to_string(),
        model_id: backend.model_id().to_string(),
        prompt_version: template.version.clone(),
        call_kind: "judge".to_string(),
        request: prompt,
        response: trimmed.to_string(),
        tokens_in: resp.tokens_in,
        tokens_out: resp.tokens_out,
        cost_usd_micros: resp.cost_usd_micros,
    };
    cache.put(&key, &entry)?;

    Ok(JudgmentResult {
        verdict,
        reason,
        from_cache: false,
        tokens_in: resp.tokens_in,
        tokens_out: resp.tokens_out,
        cost_usd_micros: resp.cost_usd_micros,
    })
}

fn parse_verdict(text: &str) -> (Verdict, String) {
    match serde_json::from_str::<JudgeOutput>(text) {
        Ok(j) => {
            let v = match j.verdict.to_lowercase().as_str() {
                "correct" => Verdict::Correct,
                "partial" => Verdict::Partial,
                "incorrect" => Verdict::Incorrect,
                _ => Verdict::JudgeError,
            };
            (v, j.reason)
        }
        Err(e) => (Verdict::JudgeError, format!("parse error: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::echo::EchoBackend;
    use std::path::PathBuf;

    fn cache() -> LlmCache {
        LlmCache::open(&PathBuf::from(":memory:")).unwrap()
    }

    fn gen_template() -> PromptTemplate {
        PromptTemplate::parse(
            "<!-- prompt-version: 1 -->\n\nContext: {{context}}\nQ: {{question}}\nA:"
        ).unwrap()
    }

    fn judge_template() -> PromptTemplate {
        PromptTemplate::parse(
            "<!-- prompt-version: 1 -->\n\nQ: {{question}}\nGold: {{gold}}\nCandidate: {{candidate}}"
        ).unwrap()
    }

    #[test]
    fn generate_returns_answer_and_caches() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("Context", "Generated answer.");
        let tmpl = gen_template();
        let snippets = vec!["snippet-1".to_string(), "snippet-2".to_string()];

        let r1 = generate_answer(&mut backend, &cache, &tmpl, "What?", &snippets).unwrap();
        assert_eq!(r1.candidate, "Generated answer.");
        assert!(!r1.from_cache);

        let r2 = generate_answer(&mut backend, &cache, &tmpl, "What?", &snippets).unwrap();
        assert!(r2.from_cache);
        assert_eq!(r2.candidate, "Generated answer.");
    }

    #[test]
    fn judge_parses_correct_verdict() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("Q:",
            r#"{"verdict": "correct", "reason": "Exact match"}"#);
        let tmpl = judge_template();

        let r = judge_answer(&mut backend, &cache, &tmpl, "Q", "gold", "candidate").unwrap();
        assert_eq!(r.verdict, Verdict::Correct);
        assert_eq!(r.reason, "Exact match");
    }

    #[test]
    fn judge_parses_partial_verdict() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("Q:",
            r#"{"verdict": "partial", "reason": "Missing detail"}"#);
        let r = judge_answer(&mut backend, &cache, &judge_template(), "Q", "g", "c").unwrap();
        assert_eq!(r.verdict, Verdict::Partial);
    }

    #[test]
    fn judge_returns_judge_error_on_malformed_json() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("Q:", "not json");
        let r = judge_answer(&mut backend, &cache, &judge_template(), "Q", "g", "c").unwrap();
        assert_eq!(r.verdict, Verdict::JudgeError);
    }

    #[test]
    fn judge_unknown_verdict_string_maps_to_judge_error() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("Q:",
            r#"{"verdict": "maybe", "reason": ""}"#);
        let r = judge_answer(&mut backend, &cache, &judge_template(), "Q", "g", "c").unwrap();
        assert_eq!(r.verdict, Verdict::JudgeError);
    }

    #[test]
    fn judge_caches_responses() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("Q:",
            r#"{"verdict": "correct", "reason": "ok"}"#);
        let _ = judge_answer(&mut backend, &cache, &judge_template(), "Q", "g", "c").unwrap();
        let r2 = judge_answer(&mut backend, &cache, &judge_template(), "Q", "g", "c").unwrap();
        assert!(r2.from_cache);
    }

    #[test]
    fn verdict_scores() {
        assert_eq!(Verdict::Correct.score(), 1.0);
        assert_eq!(Verdict::Partial.score(), 0.5);
        assert_eq!(Verdict::Incorrect.score(), 0.0);
        assert_eq!(Verdict::JudgeError.score(), 0.0);
    }
}
```

- [ ] **Step 6.2: Uncomment `pub mod eval;` in `bench/src/llm/mod.rs`; re-export `Verdict`**

In `bench/src/llm/mod.rs`, after uncommenting `pub mod eval;`, add a re-export:

```rust
pub use eval::Verdict;
```

- [ ] **Step 6.3: Verify build + tests**

Run: `cargo build -p nark-bench && cargo test -p nark-bench llm::eval`
Expected: 7 eval tests pass.

- [ ] **Step 6.4: Commit**

```bash
git add bench/src/llm/eval.rs bench/src/llm/mod.rs
git commit -m "$(cat <<'EOF'
feat(bench/llm): Verdict + judge_answer + generate_answer

Free functions composing LlmBackend + PromptTemplate + LlmCache into
the two roles Task B needs. judge_answer parses JSON verdict from
LLM response; retries once on parse failure; returns Verdict::JudgeError
on persistent failure. generate_answer returns the LLM text verbatim
(trimmed).

Both functions check cache before calling LLM, store on miss. From-cache
results report zero new tokens (the original cost was charged on first
write).

Seven unit tests cover the gen + judge + cache + parse paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Schema v3 — AnswerMetrics + perf.generation/judging + properties

**Files:**
- Modify: `bench/src/result.rs`
- Modify: `bench/src/tasks/ir.rs` (no semantic change; verify still compiles)

Backward-compatible additions. Existing IR baselines deserialize fine (new fields are Option/default).

- [ ] **Step 7.1: Update `bench/src/result.rs`**

Read the current `bench/src/result.rs`. Add three new structs and modify `BenchResult` + `PerfMetrics`.

Insert these structs after the existing `PhaseMetrics` definition:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmPhaseMetrics {
    pub calls: usize,
    pub cache_hits: usize,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_micros: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnswerMetrics {
    pub accuracy: f64,
    pub per_ability: std::collections::HashMap<String, AbilityMetrics>,
    pub abstention_precision: f64,
    pub judge_error_rate: f64,
    pub questions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AbilityMetrics {
    pub accuracy: f64,
    pub questions: usize,
}
```

Modify `PerfMetrics` to add the two new optional fields:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerfMetrics {
    pub write: PhaseMetrics,
    pub search: PhaseMetrics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<LlmPhaseMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judging: Option<LlmPhaseMetrics>,
}
```

Modify `BenchResult` to add `answer` and `properties`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub schema_version: String,
    pub task: String,
    pub system: String,
    pub config: String,
    pub system_version: String,
    pub bench_version: String,
    pub run_started_at: String,
    pub corpus: String,
    pub ir: Option<IrMetrics>,
    pub ir_per_class: std::collections::HashMap<String, IrMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<AnswerMetrics>,
    pub perf: PerfMetrics,
    #[serde(default)]
    pub properties: serde_json::Value,
    pub errors: Vec<BenchError>,
}
```

Update `BenchResult::new` — bump schema_version to "3", initialize new fields:

```rust
impl BenchResult {
    pub fn new(task: &str, system: &str, config: &str, system_version: &str, corpus: &str) -> Self {
        Self {
            schema_version: "3".to_string(),     // was "2"
            task: task.to_string(),
            system: system.to_string(),
            config: config.to_string(),
            system_version: system_version.to_string(),
            bench_version: env!("CARGO_PKG_VERSION").to_string(),
            run_started_at: Utc::now().to_rfc3339(),
            corpus: corpus.to_string(),
            ir: None,
            ir_per_class: std::collections::HashMap::new(),
            answer: None,                         // NEW
            perf: PerfMetrics::default(),
            properties: serde_json::Value::Object(serde_json::Map::new()),  // NEW: default to {}
            errors: vec![],
        }
    }

    // write_to_disk stays unchanged from Phase 1c
}
```

- [ ] **Step 7.2: Verify build + run all tests**

Run: `cargo build -p nark-bench`
Expected: completes; existing IR task and adapters compile unchanged.

Run: `cargo test -p nark-bench --lib`
Expected: all unit tests pass (~30 by now, depending on prior tasks).

- [ ] **Step 7.3: Verify smoke test still passes**

Run: `cargo build -p nark --release && cargo test -p nark-bench --release --test smoke`
Expected: 1 smoke test passes.

Note: the smoke test reads existing baselines under schema v2. The regression-check script will now emit a "schema version mismatch (new=3, baseline=2) — re-bootstrap baseline if intentional" warning when run against the old baselines. That's expected and harmless until Task A baselines are regenerated for any other reason; we don't force-regenerate here.

- [ ] **Step 7.4: Commit**

```bash
git add bench/src/result.rs
git commit -m "$(cat <<'EOF'
feat(bench/result): schema v3 — AnswerMetrics, properties, LLM perf

Three backward-compatible additions:
- answer: Option<AnswerMetrics> — populated for Task B; None for Task A
- properties: serde_json::Value (default {}) — extension point at the
  result level (LOCOMO uses it for methodology disclaimer)
- perf.generation / perf.judging: Option<LlmPhaseMetrics>

schema_version bumps "2" -> "3". Phase 1 baselines deserialize fine
because new fields are Option/default. Regression-check.sh will warn
on schema mismatch until Task A baselines are regenerated for any
other reason — no force-regen here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: LongMemEval fetch script + loader

**Files:**
- Create: `bench/datasets/longmemeval/fetch.sh`
- Create: `bench/datasets/longmemeval/README.md`
- Create: `bench/src/tasks/longmemeval_loader.rs` (loader module separate from runner for testability)
- Modify: `bench/src/tasks/mod.rs` — `pub mod longmemeval_loader;`
- Modify: `.gitignore` — add `bench/datasets/*/upstream/`

The fetch script + loader. Task 9 has the runner that wires loader + LLM eval into a full Task B pipeline.

- [ ] **Step 8.1: Identify the pinned upstream SHA**

```bash
git ls-remote https://github.com/xiaowu0162/LongMemEval HEAD | head -1
```

Record the SHA (40 hex chars).

If the repo URL is wrong or unreachable, search GitHub for "LongMemEval" and use the official ICLR 2025 paper repo. The exact URL is critical; verify before pinning.

- [ ] **Step 8.2: Inspect the upstream dataset format**

Clone the repo to a temp directory and look at the JSON file structure:

```bash
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/xiaowu0162/LongMemEval "$TMP/lme"
ls "$TMP/lme"
# Look for files like data/longmemeval_s.json, data/longmemeval_m.json, etc.
# Inspect the JSON shape:
jq '.[0] | keys' "$TMP/lme/data/longmemeval_s.json" 2>/dev/null | head -20
jq '.[0]' "$TMP/lme/data/longmemeval_s.json" 2>/dev/null | head -50
```

Record:
- The relative path to the `-S` (short) dataset file
- The JSON structure of a question entry (field names)
- What `question_type` values appear (these become the per-ability classes)
- Whether `haystack_sessions` is a list of lists of turns, or some other shape

If the upstream format is different from the spec's assumption (`{question_id, question, answer, question_type, haystack_sessions}`), STOP and report DONE_WITH_CONCERNS. The loader code below assumes the spec's structure; the implementer must adapt the loader to actual format before proceeding.

Also record what `k` (retrieval cutoff) the upstream eval scripts use, and what gen + judge models they invoke. These inform defaults in Task 10.

- [ ] **Step 8.3: Create `bench/datasets/longmemeval/fetch.sh`**

```bash
#!/usr/bin/env bash
# Clones xiaowu0162/LongMemEval at a pinned commit SHA into upstream/.
# Idempotent: if upstream/ exists, prints "already cloned" and exits.

set -euo pipefail

UPSTREAM_URL="https://github.com/xiaowu0162/LongMemEval"
# REPLACE with the SHA recorded in Step 8.1:
PINNED_SHA="REPLACE_WITH_ACTUAL_SHA_FROM_STEP_8_1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/upstream"

if [[ -d "$TARGET/repo/.git" ]]; then
  echo "LongMemEval already cloned at $TARGET/repo"
  exit 0
fi

mkdir -p "$TARGET"
git clone "$UPSTREAM_URL" "$TARGET/repo"
cd "$TARGET/repo"
git fetch origin "$PINNED_SHA"
git checkout "$PINNED_SHA"

echo "LongMemEval ready at $TARGET/repo (pinned $PINNED_SHA)"
```

- [ ] **Step 8.4: Create `bench/datasets/longmemeval/README.md`**

```markdown
# LongMemEval Dataset

Public benchmark from ICLR 2025 — 500 curated questions across 5 ability
classes designed to test long-term conversational memory in LLM agents.

Upstream: https://github.com/xiaowu0162/LongMemEval
Pinned commit SHA: (recorded in `fetch.sh`)

## How to fetch

```bash
bash bench/datasets/longmemeval/fetch.sh
```

Clones the upstream repo at the pinned SHA into `upstream/repo/`.
`upstream/` is gitignored (the dataset files are MB-scale and live
upstream's git, not ours).

## Dataset structure

(Filled in by the implementer in Step 8.2 of Phase 2 plan after
inspecting the actual upstream JSON. Sketch:)

- `data/longmemeval_s.json` — short variant (the one we use in Phase 2)
- Each question entry has fields TBD-from-step-8.2 (e.g. question_id,
  question, answer, question_type, haystack_sessions)
- `question_type` values become per-ability breakdown keys in the
  result JSON

## Five ability classes

(From the LongMemEval paper, ICLR 2025:)

1. Information extraction
2. Multi-session reasoning
3. Temporal reasoning
4. Knowledge updates
5. Abstention (correctly saying "I don't know")

## Methodology

This bench uses the upstream dataset and gold answers but runs the
gen + judge pipeline through our `LlmBackend` infrastructure. See
`bench/llm/README.md` for the prompt-versioning convention; see
`bench/src/tasks/longmemeval.rs` for the runner.

## Phase 2 status

Phase 2 ships baseline numbers but does NOT include validation that
our methodology reproduces published mem0/Zep numbers — that requires
Phase 3's mem0 adapter. Until then, our LongMemEval numbers are
"internal directional" not "publishable comparable."
```

- [ ] **Step 8.5: Create `bench/src/tasks/longmemeval_loader.rs`**

```rust
//! Loader for the LongMemEval dataset.
//!
//! Parses the upstream JSON files (cloned via bench/datasets/longmemeval/fetch.sh)
//! into typed Rust structs the runner can consume.
//!
//! Adjust field names + shape based on what the implementer found in
//! Step 8.2 of Phase 2 plan. The structs below are the assumed format
//! per the spec.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct LongMemEvalQuestion {
    pub question_id: String,
    pub question: String,
    /// Gold answer.
    pub answer: String,
    /// Ability class — drives per-ability breakdown in the result JSON.
    pub question_type: String,
    /// Haystack: nested list (list of sessions, each session is a list of turns).
    /// Each turn is a string containing the speaker and content. The exact
    /// shape may need adjustment per Step 8.2; if so, the runner needs
    /// matching changes.
    pub haystack_sessions: Vec<Vec<String>>,
}

/// Load all questions from a LongMemEval JSON file (typically
/// `bench/datasets/longmemeval/upstream/repo/data/longmemeval_s.json`).
pub fn load_questions(path: &Path) -> Result<Vec<LongMemEvalQuestion>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read LongMemEval data file {:?}", path))?;
    let questions: Vec<LongMemEvalQuestion> = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse LongMemEval JSON at {:?}", path))?;
    Ok(questions)
}

/// Flatten a question's haystack into a list of (document_id, text) tuples,
/// one per turn. Document IDs are stable: `s{session_idx}_t{turn_idx}`.
pub fn haystack_to_documents(q: &LongMemEvalQuestion) -> Vec<(String, String)> {
    q.haystack_sessions.iter().enumerate()
        .flat_map(|(session_idx, session)| {
            session.iter().enumerate().map(move |(turn_idx, turn_text)| {
                (format!("s{}_t{}", session_idx, turn_idx), turn_text.clone())
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_question() {
        let json = r#"[
            {
                "question_id": "q1",
                "question": "What did Alice say?",
                "answer": "She said hello.",
                "question_type": "info_extraction",
                "haystack_sessions": [
                    ["Alice: Hello", "Bob: Hi"],
                    ["Alice: How are you", "Bob: Fine"]
                ]
            }
        ]"#;
        let qs: Vec<LongMemEvalQuestion> = serde_json::from_str(json).unwrap();
        assert_eq!(qs.len(), 1);
        assert_eq!(qs[0].question_id, "q1");
        assert_eq!(qs[0].haystack_sessions.len(), 2);
    }

    #[test]
    fn flattens_haystack_to_documents() {
        let q = LongMemEvalQuestion {
            question_id: "q1".into(),
            question: "Q".into(),
            answer: "A".into(),
            question_type: "info_extraction".into(),
            haystack_sessions: vec![
                vec!["t0".into(), "t1".into()],
                vec!["t2".into()],
            ],
        };
        let docs = haystack_to_documents(&q);
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].0, "s0_t0");
        assert_eq!(docs[1].0, "s0_t1");
        assert_eq!(docs[2].0, "s1_t0");
        assert_eq!(docs[0].1, "t0");
    }
}
```

- [ ] **Step 8.6: Update `bench/src/tasks/mod.rs`**

Add `pub mod longmemeval_loader;` to the module declarations.

- [ ] **Step 8.7: Update `.gitignore`**

Read `.gitignore` at the repo root. Add:

```
# Bench dataset upstream clones — large; live in upstream's git.
bench/datasets/*/upstream/
```

- [ ] **Step 8.8: Verify build + tests**

Run: `cargo build -p nark-bench && cargo test -p nark-bench tasks::longmemeval_loader`
Expected: 2 loader tests pass.

- [ ] **Step 8.9: Make fetch.sh executable + smoke-run it**

```bash
chmod +x bench/datasets/longmemeval/fetch.sh
bash bench/datasets/longmemeval/fetch.sh
ls bench/datasets/longmemeval/upstream/repo/data/
```

Expected: clones to upstream/repo/, lists at least `longmemeval_s.json` (or whatever the actual file is per Step 8.2).

- [ ] **Step 8.10: Commit**

```bash
git add bench/datasets/longmemeval/ bench/src/tasks/longmemeval_loader.rs bench/src/tasks/mod.rs .gitignore
git commit -m "$(cat <<'EOF'
feat(bench/longmemeval): fetch script + dataset loader

bench/datasets/longmemeval/fetch.sh clones xiaowu0162/LongMemEval at
a pinned SHA into upstream/repo. .gitignore excludes the upstream
clone (~MB of data lives in their git, not ours).

bench/src/tasks/longmemeval_loader.rs parses the upstream JSON into
typed LongMemEvalQuestion structs. haystack_to_documents flattens
the nested haystack format into stable (doc_id, text) pairs.

README documents the methodology + the "Phase 2 numbers are internal
directional until Phase 3 validation" caveat.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `run_longmemeval_task` runner

**Files:**
- Create: `bench/src/tasks/longmemeval.rs`
- Create: `bench/llm/prompts/longmemeval-generate.md`
- Create: `bench/llm/prompts/longmemeval-judge.md`
- Modify: `bench/src/tasks/mod.rs` — `pub mod longmemeval;`

The runner that wires everything together: loads questions, iterates per-question, ingests haystack into adapter, retrieves, generates, judges, aggregates.

- [ ] **Step 9.1: Create `bench/llm/prompts/longmemeval-generate.md`**

```markdown
<!-- prompt-version: 1 -->

You are answering a question based on a memory of past conversations.

Retrieved context (may or may not be relevant):
{{context}}

Question: {{question}}

Answer the question concisely. If the context doesn't contain enough information to answer, respond with exactly: "I don't know"
```

- [ ] **Step 9.2: Create `bench/llm/prompts/longmemeval-judge.md`**

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

- [ ] **Step 9.3: Create `bench/src/tasks/longmemeval.rs`**

```rust
//! Task B — LongMemEval runner.
//!
//! Per-question:
//!   adapter.setup(workdir)
//!   for turn in haystack: adapter.write(doc)
//!   hits = adapter.search(question, k=10)
//!   candidate = generate_answer(gen_backend, gen_prompt, question, hits.snippets)
//!   verdict = judge_answer(judge_backend, judge_prompt, question, gold, candidate)
//!   adapter.teardown()
//!
//! Aggregates: overall accuracy, per_ability accuracy (keyed by question_type),
//! abstention_precision, judge_error_rate, perf.generation + perf.judging.

use anyhow::{Context, Result};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

use crate::llm::cache::LlmCache;
use crate::llm::eval::{generate_answer, judge_answer, Verdict};
use crate::llm::prompt::PromptTemplate;
use crate::llm::LlmBackend;
use crate::protocol::{Adapter, Document};
use crate::result::{AbilityMetrics, AnswerMetrics, BenchError, BenchResult, LlmPhaseMetrics};
use crate::tasks::longmemeval_loader::{haystack_to_documents, load_questions};

const K: usize = 10;

pub fn run_longmemeval_task(
    adapter_factory: &mut dyn FnMut() -> Result<Box<dyn Adapter>>,
    dataset_path: &Path,
    gen_backend: &mut dyn LlmBackend,
    gen_template_path: &Path,
    judge_backend: &mut dyn LlmBackend,
    judge_template_path: &Path,
    cache: &LlmCache,
    config_label: &str,
) -> Result<BenchResult> {
    // Load template files (read once, used per-question).
    let gen_template = PromptTemplate::load(gen_template_path)
        .with_context(|| format!("failed to load gen template at {:?}", gen_template_path))?;
    let judge_template = PromptTemplate::load(judge_template_path)
        .with_context(|| format!("failed to load judge template at {:?}", judge_template_path))?;

    // Load dataset.
    let questions = load_questions(dataset_path)?;

    // Construct a tentative adapter to record name + version for the result header.
    let mut probe = adapter_factory()?;
    let system_name = probe.name().to_string();
    let system_version = probe.version().unwrap_or_else(|_| "unknown".to_string());
    drop(probe);

    if system_version == "unknown" {
        anyhow::bail!(
            "adapter '{}' reports unknown version — refusing to run for reproducibility",
            system_name
        );
    }

    let corpus_name = "longmemeval".to_string();
    let mut result = BenchResult::new("longmemeval", &system_name, config_label, &system_version, &corpus_name);

    // Per-question aggregation buckets.
    let mut all_verdicts: Vec<(String, Verdict)> = Vec::new(); // (question_type, verdict)
    let mut abstention_correct = 0;
    let mut abstention_total = 0;
    let mut judge_errors = 0;

    let mut gen_calls = 0usize;
    let mut gen_cache_hits = 0usize;
    let mut gen_tokens_in = 0u64;
    let mut gen_tokens_out = 0u64;
    let mut gen_cost = 0u64;

    let mut judge_calls = 0usize;
    let mut judge_cache_hits = 0usize;
    let mut judge_tokens_in = 0u64;
    let mut judge_tokens_out = 0u64;
    let mut judge_cost = 0u64;

    for (i, q) in questions.iter().enumerate() {
        let workdir = tempfile::tempdir()?;
        let mut adapter = adapter_factory()?;

        if let Err(e) = adapter.setup(workdir.path()) {
            result.errors.push(BenchError {
                phase: format!("setup:{}", q.question_id),
                message: e.to_string(),
            });
            continue;
        }

        // Ingest haystack.
        let docs = haystack_to_documents(q);
        let mut ingest_failed = false;
        for (doc_id, body) in docs {
            let doc = Document { id: doc_id, body, metadata: json!({}) };
            if let Err(e) = adapter.write(&doc) {
                result.errors.push(BenchError {
                    phase: format!("write:{}:doc", q.question_id),
                    message: e.to_string(),
                });
                ingest_failed = true;
                break;
            }
        }
        if ingest_failed {
            let _ = adapter.teardown();
            continue;
        }

        // Retrieve.
        let (hits, _search_metrics) = match adapter.search(&q.question, K) {
            Ok(x) => x,
            Err(e) => {
                result.errors.push(BenchError {
                    phase: format!("search:{}", q.question_id),
                    message: e.to_string(),
                });
                let _ = adapter.teardown();
                continue;
            }
        };

        // Build snippets from hits — fall back to empty string if missing.
        let snippets: Vec<String> = hits.into_iter()
            .map(|h| h.snippet.unwrap_or_default())
            .collect();

        // Generate.
        let gen_result = match generate_answer(gen_backend, cache, &gen_template, &q.question, &snippets) {
            Ok(g) => g,
            Err(e) => {
                result.errors.push(BenchError {
                    phase: format!("generate:{}", q.question_id),
                    message: e.to_string(),
                });
                let _ = adapter.teardown();
                continue;
            }
        };
        gen_calls += 1;
        if gen_result.from_cache { gen_cache_hits += 1; }
        gen_tokens_in += gen_result.tokens_in;
        gen_tokens_out += gen_result.tokens_out;
        gen_cost += gen_result.cost_usd_micros;

        // Judge.
        let judgment = match judge_answer(judge_backend, cache, &judge_template, &q.question, &q.answer, &gen_result.candidate) {
            Ok(j) => j,
            Err(e) => {
                result.errors.push(BenchError {
                    phase: format!("judge:{}", q.question_id),
                    message: e.to_string(),
                });
                let _ = adapter.teardown();
                continue;
            }
        };
        judge_calls += 1;
        if judgment.from_cache { judge_cache_hits += 1; }
        judge_tokens_in += judgment.tokens_in;
        judge_tokens_out += judgment.tokens_out;
        judge_cost += judgment.cost_usd_micros;

        if matches!(judgment.verdict, Verdict::JudgeError) {
            judge_errors += 1;
        }

        // Abstention precision: questions where the gold answer is an abstention.
        let gold_abstains = q.answer.trim().eq_ignore_ascii_case("I don't know")
            || q.answer.trim().eq_ignore_ascii_case("idk");
        if gold_abstains {
            abstention_total += 1;
            if matches!(judgment.verdict, Verdict::Correct) {
                abstention_correct += 1;
            }
        }

        all_verdicts.push((q.question_type.clone(), judgment.verdict));

        let _ = adapter.teardown();

        // Progress every 50 questions.
        if (i + 1) % 50 == 0 {
            eprintln!("  longmemeval: {}/{} questions ({})", i + 1, questions.len(), system_name);
        }
    }

    // Aggregate.
    let n = all_verdicts.len() as f64;
    let overall = if n > 0.0 {
        all_verdicts.iter().map(|(_, v)| v.score()).sum::<f64>() / n
    } else { 0.0 };

    let mut per_ability_buckets: HashMap<String, Vec<Verdict>> = HashMap::new();
    for (qt, v) in &all_verdicts {
        per_ability_buckets.entry(qt.clone()).or_default().push(*v);
    }
    let per_ability: HashMap<String, AbilityMetrics> = per_ability_buckets.into_iter()
        .map(|(k, vs)| {
            let acc = if vs.is_empty() { 0.0 }
                else { vs.iter().map(|v| v.score()).sum::<f64>() / vs.len() as f64 };
            (k, AbilityMetrics { accuracy: acc, questions: vs.len() })
        })
        .collect();

    let abstention_precision = if abstention_total > 0 {
        abstention_correct as f64 / abstention_total as f64
    } else { 1.0 };

    let judge_error_rate = if !all_verdicts.is_empty() {
        judge_errors as f64 / all_verdicts.len() as f64
    } else { 0.0 };

    result.answer = Some(AnswerMetrics {
        accuracy: overall,
        per_ability,
        abstention_precision,
        judge_error_rate,
        questions: all_verdicts.len(),
    });

    result.perf.generation = Some(LlmPhaseMetrics {
        calls: gen_calls,
        cache_hits: gen_cache_hits,
        tokens_in: gen_tokens_in,
        tokens_out: gen_tokens_out,
        cost_usd_micros: gen_cost,
    });
    result.perf.judging = Some(LlmPhaseMetrics {
        calls: judge_calls,
        cache_hits: judge_cache_hits,
        tokens_in: judge_tokens_in,
        tokens_out: judge_tokens_out,
        cost_usd_micros: judge_cost,
    });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::echo::EchoBackend;
    use crate::adapters::fts5::Fts5Adapter;
    use std::path::PathBuf;

    fn cache() -> LlmCache {
        LlmCache::open(&PathBuf::from(":memory:")).unwrap()
    }

    #[test]
    fn smoke_runs_longmemeval_with_echo_backends() {
        // Build a tiny synthetic dataset on the fly.
        let dataset = serde_json::json!([
            {
                "question_id": "q1",
                "question": "What did Alice say?",
                "answer": "Hello",
                "question_type": "info_extraction",
                "haystack_sessions": [["Alice said hello", "Bob said hi"]]
            }
        ]);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), serde_json::to_string(&dataset).unwrap()).unwrap();

        let gen_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("llm/prompts/longmemeval-generate.md");
        let judge_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("llm/prompts/longmemeval-judge.md");

        let mut gen_backend = EchoBackend::new()
            .with_default("Alice said hello");
        let mut judge_backend = EchoBackend::new()
            .with_default(r#"{"verdict": "correct", "reason": "match"}"#);
        let cache = cache();

        let mut factory = || -> anyhow::Result<Box<dyn Adapter>> {
            Ok(Box::new(Fts5Adapter::new()))
        };

        let result = run_longmemeval_task(
            &mut factory,
            tmp.path(),
            &mut gen_backend,
            &gen_path,
            &mut judge_backend,
            &judge_path,
            &cache,
            "smoke",
        ).unwrap();

        assert_eq!(result.task, "longmemeval");
        assert_eq!(result.schema_version, "3");
        assert!(result.answer.is_some());
        let answer = result.answer.unwrap();
        assert_eq!(answer.questions, 1);
        assert_eq!(answer.accuracy, 1.0);  // judge said correct
    }
}
```

- [ ] **Step 9.4: Update `bench/src/tasks/mod.rs`**

Add `pub mod longmemeval;` to module declarations.

- [ ] **Step 9.5: Verify build + tests**

```bash
cargo build -p nark-bench
cargo test -p nark-bench tasks::longmemeval
```

Expected: build clean; 1 smoke test passes (uses EchoBackend so no real LLM calls).

- [ ] **Step 9.6: Commit**

```bash
git add bench/src/tasks/longmemeval.rs bench/llm/prompts/longmemeval-*.md bench/src/tasks/mod.rs
git commit -m "$(cat <<'EOF'
feat(bench/tasks): run_longmemeval_task + generate/judge prompts (v1)

Per-question runner: setup adapter, ingest haystack, retrieve top-10,
generate answer via gen LlmBackend, judge via judge LlmBackend, aggregate
to AnswerMetrics. Adapter is constructed via factory closure so the
runner can create a fresh isolated workdir per question (matching
upstream LongMemEval's eval model).

Generation + judge prompts shipped as v1 in bench/llm/prompts/. Both
follow the {{var}} substitution convention; both have the canonical
<!-- prompt-version: 1 --> header for cache versioning.

Integration smoke test in the runner module uses EchoBackend for both
gen and judge so cargo test never hits a real LLM.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: CLI flags + pre-run confirmation prompt

**Files:**
- Modify: `bench/src/main.rs`

Adds `--gen-backend`, `--gen-model`, `--judge-backend`, `--judge-model`, `--llm-concurrency`, `--yes` flags. Dispatches `--task longmemeval` to the Task 9 runner.

- [ ] **Step 10.1: Read current `bench/src/main.rs`**

Read the file to understand the current `Commands::Run` shape (Phase 1c version with `--task ir` dispatch only).

- [ ] **Step 10.2: Replace the `Commands::Run` arm + add helper functions**

Replace `bench/src/main.rs` with:

```rust
use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod protocol;
mod metrics;
mod adapters;
mod result;
mod tasks;
mod model_cache;
mod llm;

use crate::llm::api::ApiBackend;
use crate::llm::claude_cli::ClaudeCliBackend;
use crate::llm::codex_cli::CodexCliBackend;
use crate::llm::cache::LlmCache;
use crate::llm::LlmBackend;

#[derive(Parser)]
#[command(name = "nark-bench", about = "Benchmark harness for nark")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a benchmark task
    Run {
        /// Task name: "ir" | "longmemeval" | "locomo"
        #[arg(long)]
        task: String,
        /// Comma-separated systems (e.g. "fts5,nark,vector")
        #[arg(long)]
        systems: String,
        /// For --task ir: corpus name (e.g. "synthetic-tiny"). Ignored for Task B.
        #[arg(long, default_value = "synthetic-tiny")]
        corpus: String,
        /// Output directory for result JSON files
        #[arg(long, default_value = "bench/results/local")]
        output: PathBuf,

        // Task B (longmemeval / locomo) flags
        /// Generation backend: claude-cli | codex-cli | api
        #[arg(long)]
        gen_backend: Option<String>,
        /// Generation model id (e.g. "gpt-5.5", "claude-opus-4-7")
        #[arg(long)]
        gen_model: Option<String>,
        /// Judge backend: claude-cli | codex-cli | api
        #[arg(long)]
        judge_backend: Option<String>,
        /// Judge model id
        #[arg(long)]
        judge_model: Option<String>,
        /// Concurrency cap for parallel LLM calls (per backend)
        #[arg(long, default_value = "4")]
        llm_concurrency: usize,
        /// Skip the pre-run confirmation prompt
        #[arg(long)]
        yes: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { task, systems, corpus, output, gen_backend, gen_model, judge_backend, judge_model, llm_concurrency, yes } => {
            match task.as_str() {
                "ir" => run_ir(systems, corpus, output),
                "longmemeval" => run_task_b("longmemeval", systems, output, gen_backend, gen_model, judge_backend, judge_model, llm_concurrency, yes),
                "locomo" => run_task_b("locomo", systems, output, gen_backend, gen_model, judge_backend, judge_model, llm_concurrency, yes),
                other => anyhow::bail!("unknown --task value: {}", other),
            }
        }
    }
}

fn run_ir(systems: String, corpus: String, output: PathBuf) -> Result<()> {
    let corpus_root = PathBuf::from("bench/datasets/ir").join(&corpus);
    if !corpus_root.exists() {
        anyhow::bail!("corpus not found at {:?}", corpus_root);
    }
    let cache = model_cache::cache_root()?;
    let needs_model = systems.split(',').any(|s| matches!(s.trim(), "nark" | "vector"));
    if needs_model {
        model_cache::ensure_ready(&cache)?;
    }
    for system in systems.split(',') {
        let system = system.trim();
        if system.is_empty() { continue; }
        let mut adapter = adapters::make_adapter(system, Some(&cache))?;
        let result = tasks::ir::run_ir_task(adapter.as_mut(), &corpus_root, "default")?;
        let path = result.write_to_disk(&output)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn build_backend(name: Option<&str>, model: Option<&str>, concurrency: usize, role: &str) -> Result<Box<dyn LlmBackend>> {
    let name = name.ok_or_else(|| anyhow!("--{}-backend is required for Task B", role))?;
    let model = model.ok_or_else(|| anyhow!("--{}-model is required for Task B", role))?;
    match name {
        "claude-cli" => Ok(Box::new(ClaudeCliBackend::new(model).with_concurrency(concurrency))),
        "codex-cli" => Ok(Box::new(CodexCliBackend::new(model).with_concurrency(concurrency))),
        "api" => Ok(Box::new(ApiBackend::from_env(model)?.with_concurrency(concurrency))),
        other => anyhow::bail!("unknown {}-backend: {}", role, other),
    }
}

fn run_task_b(
    task_name: &str,
    systems: String,
    output: PathBuf,
    gen_backend_name: Option<String>,
    gen_model: Option<String>,
    judge_backend_name: Option<String>,
    judge_model: Option<String>,
    concurrency: usize,
    yes: bool,
) -> Result<()> {
    // For Task B we always need the model cache (nark and vector both need it).
    let model_cache_path = model_cache::cache_root()?;
    let needs_model = systems.split(',').any(|s| matches!(s.trim(), "nark" | "vector"));
    if needs_model {
        model_cache::ensure_ready(&model_cache_path)?;
    }

    let mut gen_backend = build_backend(gen_backend_name.as_deref(), gen_model.as_deref(), concurrency, "gen")?;
    let mut judge_backend = build_backend(judge_backend_name.as_deref(), judge_model.as_deref(), concurrency, "judge")?;
    let llm_cache_path = PathBuf::from("bench/cache/llm.db");
    std::fs::create_dir_all(llm_cache_path.parent().unwrap())?;
    let llm_cache = LlmCache::open(&llm_cache_path)?;

    let dataset_path = match task_name {
        "longmemeval" => PathBuf::from("bench/datasets/longmemeval/upstream/repo/data/longmemeval_s.json"),
        "locomo" => PathBuf::from("bench/datasets/locomo/upstream/repo/data/locomo.json"),
        _ => unreachable!(),
    };
    if !dataset_path.exists() {
        anyhow::bail!(
            "dataset not found at {:?}. Run bench/datasets/{}/fetch.sh first.",
            dataset_path, task_name
        );
    }

    let gen_template_path = PathBuf::from(format!("bench/llm/prompts/{}-generate.md", task_name));
    let judge_template_path = PathBuf::from(format!("bench/llm/prompts/{}-judge.md", task_name));

    // Pre-run confirmation prompt.
    let system_count = systems.split(',').filter(|s| !s.trim().is_empty()).count();
    let est_calls = 500usize * system_count * 2;
    if !yes {
        eprintln!(
            "About to run {} on {} systems with up to ~{} LLM calls (gen + judge per question).\n\
             First-run cost: subscription limits or API spend.\n\
             Subsequent runs use cache (bench/cache/llm.db) and finish in seconds.\n\
             \n\
             Continue? [y/N]: ",
            task_name, system_count, est_calls
        );
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).ok();
        if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
            eprintln!("aborted.");
            std::process::exit(1);
        }
    }

    for system in systems.split(',') {
        let system = system.trim();
        if system.is_empty() { continue; }

        let cache_path = model_cache_path.clone();
        let sys_owned = system.to_string();
        let mut factory = move || -> Result<Box<dyn crate::protocol::Adapter>> {
            adapters::make_adapter(&sys_owned, Some(&cache_path))
        };

        let result = match task_name {
            "longmemeval" => tasks::longmemeval::run_longmemeval_task(
                &mut factory,
                &dataset_path,
                gen_backend.as_mut(),
                &gen_template_path,
                judge_backend.as_mut(),
                &judge_template_path,
                &llm_cache,
                "default",
            )?,
            "locomo" => tasks::locomo::run_locomo_task(
                &mut factory,
                &dataset_path,
                gen_backend.as_mut(),
                &gen_template_path,
                judge_backend.as_mut(),
                &judge_template_path,
                &llm_cache,
                "default",
            )?,
            _ => unreachable!(),
        };
        let path = result.write_to_disk(&output)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
```

Note: this references `tasks::locomo` which lands in Task 13. To keep this commit buildable, **add a stub `bench/src/tasks/locomo.rs`** with just the function signature returning an error:

- [ ] **Step 10.3: Create stub `bench/src/tasks/locomo.rs`**

```rust
//! Task B — LOCOMO runner. Full implementation in Task 13 of Phase 2 plan.
//! Stub allows main.rs to compile while LongMemEval lands first.

use anyhow::{anyhow, Result};
use std::path::Path;

use crate::llm::cache::LlmCache;
use crate::llm::LlmBackend;
use crate::protocol::Adapter;
use crate::result::BenchResult;

pub fn run_locomo_task(
    _adapter_factory: &mut dyn FnMut() -> Result<Box<dyn Adapter>>,
    _dataset_path: &Path,
    _gen_backend: &mut dyn LlmBackend,
    _gen_template_path: &Path,
    _judge_backend: &mut dyn LlmBackend,
    _judge_template_path: &Path,
    _cache: &LlmCache,
    _config_label: &str,
) -> Result<BenchResult> {
    Err(anyhow!("LOCOMO runner not yet implemented — see Task 13 of Phase 2 plan"))
}
```

Add `pub mod locomo;` to `bench/src/tasks/mod.rs`.

- [ ] **Step 10.4: Verify build**

Run: `cargo build -p nark-bench`
Expected: completes; warnings about unused `tasks::locomo::run_locomo_task` arguments are OK (stub).

- [ ] **Step 10.5: Smoke `--help`**

Run: `cargo run -p nark-bench --release -- run --help`
Expected: help output lists all the new flags (`--gen-backend`, `--gen-model`, etc.).

- [ ] **Step 10.6: Smoke `--task ir` still works**

Run: `cargo run -p nark-bench --release -- run --task ir --systems fts5 --corpus synthetic-tiny --output /tmp/p2-task10`
Expected: completes; produces `/tmp/p2-task10/ir-fts5-synthetic-tiny-default.json`.

Clean up: `rm -rf /tmp/p2-task10`.

- [ ] **Step 10.7: Commit**

```bash
git add bench/src/main.rs bench/src/tasks/locomo.rs bench/src/tasks/mod.rs
git commit -m "$(cat <<'EOF'
feat(bench/cli): --task longmemeval/locomo dispatch + LLM backend flags

main.rs gains six new flags (gen/judge backend + model + concurrency + yes)
and a pre-run confirmation prompt that estimates total LLM calls.

--task ir continues to work unchanged (calls into Phase 1's run_ir_task).
--task longmemeval routes to run_longmemeval_task (Task 9).
--task locomo currently routes to a stub (full impl in Task 13).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: First LongMemEval run + commit baseline

**Files:**
- Create: `bench/cache/llm.db` (committed after first run)
- Create: `bench/results/main/longmemeval-{fts5,nark,vector}-longmemeval-default.json`

**Manual task.** No code changes. Expect 1-3 hour runtime on first sweep depending on subscription throughput.

- [ ] **Step 11.1: Fetch the dataset**

```bash
bash bench/datasets/longmemeval/fetch.sh
ls bench/datasets/longmemeval/upstream/repo/data/
```

Confirm `longmemeval_s.json` exists.

- [ ] **Step 11.2: Verify backends are available**

```bash
which claude && claude -p --help 2>&1 | head -3
which codex && codex exec --help 2>&1 | head -3
```

Expected: both binaries present. If either is missing, install before continuing.

- [ ] **Step 11.3: Build nark and nark-bench in release**

```bash
cargo build -p nark --release
cargo build -p nark-bench --release
```

- [ ] **Step 11.4: Make `bench/cache/` directory**

```bash
mkdir -p bench/cache
```

- [ ] **Step 11.5: Run LongMemEval against all three systems**

This is the multi-hour step. Expected to consume Codex Plus subscription limits.

```bash
cargo run -p nark-bench --release -- run \
  --task longmemeval \
  --systems fts5,nark,vector \
  --gen-backend codex-cli --gen-model gpt-5.5 \
  --judge-backend codex-cli --judge-model gpt-5.5 \
  --llm-concurrency 4 \
  --output bench/results/main
```

When prompted, type `y` to confirm. Watch progress: stderr prints `longmemeval: 50/500 questions (fts5)` style updates every 50 questions.

If subscription limits hit mid-run: stop, wait for the limit window to reset, re-run. Cache hits make resumption fast (already-processed questions skip the LLM call).

- [ ] **Step 11.6: Inspect baseline results**

```bash
for sys in fts5 nark vector; do
  echo "=== $sys ==="
  jq '.schema_version, .corpus, .answer.accuracy, .answer.questions, (.answer.per_ability | to_entries | map({(.key): .value.accuracy}))' \
    bench/results/main/longmemeval-${sys}-longmemeval-default.json
done
```

Sanity check:
- All three: `schema_version == "3"`, `corpus == "longmemeval"`.
- All three: `answer.questions == 500` (or whatever the upstream `-s` dataset size is).
- FTS5: accuracy somewhere in the 0.20-0.40 range (pure BM25 baseline).
- nark: accuracy higher than FTS5 (hybrid pipeline).
- vector: accuracy comparable to or above nark (synonym/semantic-heavy questions favor pure embedding).
- Per-ability breakdowns show varying scores across the 5 classes.
- `answer.judge_error_rate < 0.05` (judge prompt mostly producing parseable JSON).

If any system shows wildly anomalous numbers (e.g. 0% accuracy, suggesting an infrastructure bug; or 100% accuracy, suggesting the judge is rubber-stamping), STOP and investigate. Don't commit garbage baselines.

- [ ] **Step 11.7: Commit baselines + cache**

```bash
git add bench/results/main/longmemeval-*.json bench/cache/llm.db
git commit -m "$(cat <<'EOF'
bench: lock in first LongMemEval baselines + commit LLM cache

Three baselines under schema v3, --systems fts5,nark,vector --task
longmemeval. Generation + judge both via codex-cli:gpt-5.5.

NOTE: These numbers are "internal directional" not "publishable
comparable" until Phase 3's mem0 adapter exists and we validate our
methodology reproduces mem0's published 91.6% within ±2%. Do NOT
publish these numbers as "nark vs mem0" comparisons yet.

LLM cache committed at bench/cache/llm.db. Re-runs of the same prompt
against the same model + same prompt-version are free.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: LOCOMO fetch script + loader

**Files:**
- Create: `bench/datasets/locomo/fetch.sh`
- Create: `bench/datasets/locomo/README.md`
- Create: `bench/src/tasks/locomo_loader.rs`
- Modify: `bench/src/tasks/mod.rs` — `pub mod locomo_loader;`

Same shape as Task 8 (LongMemEval), adapted for LOCOMO's dataset format.

- [ ] **Step 12.1: Identify LOCOMO upstream + inspect format**

```bash
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/snap-research/locomo "$TMP/locomo"
ls "$TMP/locomo"
# Inspect the data file shape:
find "$TMP/locomo" -name '*.json' | head -3
jq 'keys' "$TMP/locomo/<data-file>" 2>/dev/null | head
```

Record:
- Pinned commit SHA from `git ls-remote https://github.com/snap-research/locomo HEAD | head -1`
- Path to the actual data file
- JSON structure (LOCOMO is structured differently from LongMemEval: each "sample" is a multi-turn conversation with multiple QA pairs attached)
- Question type / category labels if any

If the structure is materially different from what's shown below, adjust the loader accordingly.

- [ ] **Step 12.2: Create `bench/datasets/locomo/fetch.sh`**

Mirror of LongMemEval's fetch.sh:

```bash
#!/usr/bin/env bash
# Clones snap-research/locomo at a pinned commit SHA into upstream/.

set -euo pipefail

UPSTREAM_URL="https://github.com/snap-research/locomo"
# REPLACE with SHA from Step 12.1:
PINNED_SHA="REPLACE_WITH_ACTUAL_SHA_FROM_STEP_12_1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/upstream"

if [[ -d "$TARGET/repo/.git" ]]; then
  echo "LOCOMO already cloned at $TARGET/repo"
  exit 0
fi

mkdir -p "$TARGET"
git clone "$UPSTREAM_URL" "$TARGET/repo"
cd "$TARGET/repo"
git fetch origin "$PINNED_SHA"
git checkout "$PINNED_SHA"

echo "LOCOMO ready at $TARGET/repo (pinned $PINNED_SHA)"
```

Make executable: `chmod +x bench/datasets/locomo/fetch.sh`.

- [ ] **Step 12.3: Create `bench/datasets/locomo/README.md`**

```markdown
# LOCOMO Dataset

Long-Conversation Memory benchmark from Snap Research.

Upstream: https://github.com/snap-research/locomo
Pinned commit SHA: (recorded in `fetch.sh`)

## Methodology disclaimer

LOCOMO scores reported in the literature are CONTESTED across papers:
- mem0 reports 91.6% (paper)
- Zep's independent reproduction shows ~58% with corrected eval
  (https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- Letta showed a trivial filesystem-tool agent hits ~74%
  (https://www.letta.com/blog/benchmarking-ai-agent-memory)

For this reason, our bench treats LOCOMO as a SECONDARY signal, not
a headline. The baseline JSON's `properties.methodology_disclaimer`
field embeds these references inline so anyone reading the results
sees the caveat.

## How to fetch

```bash
bash bench/datasets/locomo/fetch.sh
```

## Dataset structure

(Filled in by the implementer in Step 12.1 of Phase 2 plan after
inspecting the actual upstream JSON. Sketch:)

- TBD-from-step-12.1

## Running

```bash
cargo run -p nark-bench --release -- run \
  --task locomo \
  --systems fts5,nark,vector \
  --gen-backend codex-cli --gen-model gpt-5.5 \
  --judge-backend codex-cli --judge-model gpt-5.5 \
  --output bench/results/main
```
```

- [ ] **Step 12.4: Create `bench/src/tasks/locomo_loader.rs`**

Adjust struct fields per actual format discovered in Step 12.1. Skeleton:

```rust
//! Loader for the LOCOMO dataset.
//!
//! Adjust struct shape based on what the implementer found in Step 12.1
//! of Phase 2 plan. The structs below assume LOCOMO's format is roughly
//! "list of samples, each with a conversation + a list of question-answer
//! pairs attached." Actual format may differ.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct LocomoSample {
    pub sample_id: String,
    /// Conversation turns. Adjust field name to match upstream.
    pub conversation: Vec<String>,
    /// QA pairs attached to this sample.
    pub qa: Vec<LocomoQA>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LocomoQA {
    pub question_id: String,
    pub question: String,
    pub answer: String,
    /// Question category if present in upstream — drives per-ability breakdown.
    #[serde(default)]
    pub category: Option<String>,
}

pub fn load_samples(path: &Path) -> Result<Vec<LocomoSample>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read LOCOMO data file {:?}", path))?;
    let samples: Vec<LocomoSample> = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse LOCOMO JSON at {:?}", path))?;
    Ok(samples)
}

/// Flatten a sample's conversation into stable (doc_id, text) tuples.
/// Document IDs: `<sample_id>_t<turn_idx>`.
pub fn conversation_to_documents(s: &LocomoSample) -> Vec<(String, String)> {
    s.conversation.iter().enumerate()
        .map(|(turn_idx, turn_text)| {
            (format!("{}_t{}", s.sample_id, turn_idx), turn_text.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_sample() {
        let json = r#"[
            {
                "sample_id": "s1",
                "conversation": ["Hello", "Hi back"],
                "qa": [
                    {"question_id": "q1", "question": "Q?", "answer": "A.", "category": "factual"}
                ]
            }
        ]"#;
        let ss: Vec<LocomoSample> = serde_json::from_str(json).unwrap();
        assert_eq!(ss.len(), 1);
        assert_eq!(ss[0].qa.len(), 1);
    }
}
```

- [ ] **Step 12.5: Update `bench/src/tasks/mod.rs`**

Add `pub mod locomo_loader;` to module declarations.

- [ ] **Step 12.6: Verify build + tests + fetch**

```bash
cargo build -p nark-bench
cargo test -p nark-bench tasks::locomo_loader
bash bench/datasets/locomo/fetch.sh
ls bench/datasets/locomo/upstream/repo/
```

Expected: build clean; 1 loader test passes; fetch succeeds.

- [ ] **Step 12.7: Commit**

```bash
git add bench/datasets/locomo/ bench/src/tasks/locomo_loader.rs bench/src/tasks/mod.rs
git commit -m "$(cat <<'EOF'
feat(bench/locomo): fetch script + dataset loader

Mirror of Task 8's LongMemEval scaffolding. Pinned upstream commit SHA
in fetch.sh; .gitignore already excludes bench/datasets/*/upstream/
from Phase 1.

README documents the methodology disclaimer (mem0/Zep/Letta reproductions
disagree on numbers) — embedded into result properties in Task 13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `run_locomo_task` runner + methodology disclaimer

**Files:**
- Modify: `bench/src/tasks/locomo.rs` — replace stub with real implementation
- Create: `bench/llm/prompts/locomo-generate.md`
- Create: `bench/llm/prompts/locomo-judge.md`

- [ ] **Step 13.1: Create `bench/llm/prompts/locomo-generate.md`**

Same as LongMemEval's gen prompt (the gen task is the same: given retrieved context, answer the question):

```markdown
<!-- prompt-version: 1 -->

You are answering a question based on a memory of past conversations.

Retrieved context (may or may not be relevant):
{{context}}

Question: {{question}}

Answer the question concisely. If the context doesn't contain enough information to answer, respond with exactly: "I don't know"
```

- [ ] **Step 13.2: Create `bench/llm/prompts/locomo-judge.md`**

```markdown
<!-- prompt-version: 1 -->

You are evaluating whether a candidate answer matches a gold answer for a long-conversation memory benchmark.

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

- [ ] **Step 13.3: Replace stub `bench/src/tasks/locomo.rs` with real implementation**

```rust
//! Task B — LOCOMO runner. Mirrors LongMemEval's structure with LOCOMO's
//! sample-with-many-QAs format.

use anyhow::{Context, Result};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

use crate::llm::cache::LlmCache;
use crate::llm::eval::{generate_answer, judge_answer, Verdict};
use crate::llm::prompt::PromptTemplate;
use crate::llm::LlmBackend;
use crate::protocol::{Adapter, Document};
use crate::result::{AbilityMetrics, AnswerMetrics, BenchError, BenchResult, LlmPhaseMetrics};
use crate::tasks::locomo_loader::{conversation_to_documents, load_samples};

const K: usize = 10;

pub fn run_locomo_task(
    adapter_factory: &mut dyn FnMut() -> Result<Box<dyn Adapter>>,
    dataset_path: &Path,
    gen_backend: &mut dyn LlmBackend,
    gen_template_path: &Path,
    judge_backend: &mut dyn LlmBackend,
    judge_template_path: &Path,
    cache: &LlmCache,
    config_label: &str,
) -> Result<BenchResult> {
    let gen_template = PromptTemplate::load(gen_template_path)
        .with_context(|| format!("failed to load gen template at {:?}", gen_template_path))?;
    let judge_template = PromptTemplate::load(judge_template_path)
        .with_context(|| format!("failed to load judge template at {:?}", judge_template_path))?;

    let samples = load_samples(dataset_path)?;

    let mut probe = adapter_factory()?;
    let system_name = probe.name().to_string();
    let system_version = probe.version().unwrap_or_else(|_| "unknown".to_string());
    drop(probe);

    if system_version == "unknown" {
        anyhow::bail!(
            "adapter '{}' reports unknown version — refusing to run for reproducibility",
            system_name
        );
    }

    let mut result = BenchResult::new("locomo", &system_name, config_label, &system_version, "locomo");

    // Methodology disclaimer in properties — this is what makes LOCOMO results
    // honest about the contested-numbers situation.
    result.properties = json!({
        "methodology_disclaimer": {
            "note": "LOCOMO scores are methodologically contested across published papers. Compare with caution.",
            "references": [
                "https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/",
                "https://www.letta.com/blog/benchmarking-ai-agent-memory"
            ]
        }
    });

    let mut all_verdicts: Vec<(String, Verdict)> = Vec::new();
    let mut abstention_correct = 0;
    let mut abstention_total = 0;
    let mut judge_errors = 0;

    let mut gen_calls = 0usize;
    let mut gen_cache_hits = 0usize;
    let mut gen_tokens_in = 0u64;
    let mut gen_tokens_out = 0u64;
    let mut gen_cost = 0u64;

    let mut judge_calls = 0usize;
    let mut judge_cache_hits = 0usize;
    let mut judge_tokens_in = 0u64;
    let mut judge_tokens_out = 0u64;
    let mut judge_cost = 0u64;

    let total_qa: usize = samples.iter().map(|s| s.qa.len()).sum();
    let mut processed = 0;

    for sample in &samples {
        // Ingest this sample's conversation ONCE per adapter setup
        // (multiple QAs share the same haystack).
        let workdir = tempfile::tempdir()?;
        let mut adapter = adapter_factory()?;
        if let Err(e) = adapter.setup(workdir.path()) {
            result.errors.push(BenchError {
                phase: format!("setup:{}", sample.sample_id),
                message: e.to_string(),
            });
            continue;
        }

        let docs = conversation_to_documents(sample);
        let mut ingest_failed = false;
        for (doc_id, body) in docs {
            let doc = Document { id: doc_id, body, metadata: json!({}) };
            if let Err(e) = adapter.write(&doc) {
                result.errors.push(BenchError {
                    phase: format!("write:{}", sample.sample_id),
                    message: e.to_string(),
                });
                ingest_failed = true;
                break;
            }
        }
        if ingest_failed {
            let _ = adapter.teardown();
            continue;
        }

        for qa in &sample.qa {
            let (hits, _) = match adapter.search(&qa.question, K) {
                Ok(x) => x,
                Err(e) => {
                    result.errors.push(BenchError {
                        phase: format!("search:{}", qa.question_id),
                        message: e.to_string(),
                    });
                    continue;
                }
            };

            let snippets: Vec<String> = hits.into_iter()
                .map(|h| h.snippet.unwrap_or_default())
                .collect();

            let gen_result = match generate_answer(gen_backend, cache, &gen_template, &qa.question, &snippets) {
                Ok(g) => g,
                Err(e) => {
                    result.errors.push(BenchError {
                        phase: format!("generate:{}", qa.question_id),
                        message: e.to_string(),
                    });
                    continue;
                }
            };
            gen_calls += 1;
            if gen_result.from_cache { gen_cache_hits += 1; }
            gen_tokens_in += gen_result.tokens_in;
            gen_tokens_out += gen_result.tokens_out;
            gen_cost += gen_result.cost_usd_micros;

            let judgment = match judge_answer(judge_backend, cache, &judge_template, &qa.question, &qa.answer, &gen_result.candidate) {
                Ok(j) => j,
                Err(e) => {
                    result.errors.push(BenchError {
                        phase: format!("judge:{}", qa.question_id),
                        message: e.to_string(),
                    });
                    continue;
                }
            };
            judge_calls += 1;
            if judgment.from_cache { judge_cache_hits += 1; }
            judge_tokens_in += judgment.tokens_in;
            judge_tokens_out += judgment.tokens_out;
            judge_cost += judgment.cost_usd_micros;

            if matches!(judgment.verdict, Verdict::JudgeError) {
                judge_errors += 1;
            }

            let category = qa.category.clone().unwrap_or_else(|| "unknown".to_string());

            let gold_abstains = qa.answer.trim().eq_ignore_ascii_case("I don't know")
                || qa.answer.trim().eq_ignore_ascii_case("idk");
            if gold_abstains {
                abstention_total += 1;
                if matches!(judgment.verdict, Verdict::Correct) {
                    abstention_correct += 1;
                }
            }

            all_verdicts.push((category, judgment.verdict));
            processed += 1;
        }

        let _ = adapter.teardown();

        if processed % 50 == 0 {
            eprintln!("  locomo: {}/{} QAs ({})", processed, total_qa, system_name);
        }
    }

    let n = all_verdicts.len() as f64;
    let overall = if n > 0.0 {
        all_verdicts.iter().map(|(_, v)| v.score()).sum::<f64>() / n
    } else { 0.0 };

    let mut per_category_buckets: HashMap<String, Vec<Verdict>> = HashMap::new();
    for (cat, v) in &all_verdicts {
        per_category_buckets.entry(cat.clone()).or_default().push(*v);
    }
    let per_ability: HashMap<String, AbilityMetrics> = per_category_buckets.into_iter()
        .map(|(k, vs)| {
            let acc = if vs.is_empty() { 0.0 }
                else { vs.iter().map(|v| v.score()).sum::<f64>() / vs.len() as f64 };
            (k, AbilityMetrics { accuracy: acc, questions: vs.len() })
        })
        .collect();

    let abstention_precision = if abstention_total > 0 {
        abstention_correct as f64 / abstention_total as f64
    } else { 1.0 };

    let judge_error_rate = if !all_verdicts.is_empty() {
        judge_errors as f64 / all_verdicts.len() as f64
    } else { 0.0 };

    result.answer = Some(AnswerMetrics {
        accuracy: overall,
        per_ability,
        abstention_precision,
        judge_error_rate,
        questions: all_verdicts.len(),
    });

    result.perf.generation = Some(LlmPhaseMetrics {
        calls: gen_calls,
        cache_hits: gen_cache_hits,
        tokens_in: gen_tokens_in,
        tokens_out: gen_tokens_out,
        cost_usd_micros: gen_cost,
    });
    result.perf.judging = Some(LlmPhaseMetrics {
        calls: judge_calls,
        cache_hits: judge_cache_hits,
        tokens_in: judge_tokens_in,
        tokens_out: judge_tokens_out,
        cost_usd_micros: judge_cost,
    });

    Ok(result)
}
```

- [ ] **Step 13.4: Verify build**

Run: `cargo build -p nark-bench`
Expected: completes; existing stub warnings disappear.

- [ ] **Step 13.5: Commit**

```bash
git add bench/src/tasks/locomo.rs bench/llm/prompts/locomo-*.md
git commit -m "$(cat <<'EOF'
feat(bench/tasks): run_locomo_task — Task B / LOCOMO runner

Mirrors run_longmemeval_task with LOCOMO's sample-with-many-QAs format.
Ingests each sample's conversation once, runs all attached QAs against
that ingested haystack.

Crucially: the result.properties block carries an inline methodology
disclaimer linking the Zep and Letta critiques of LOCOMO. This is
visible in every committed baseline so future readers cannot
inadvertently treat the number as canonical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: First LOCOMO run + commit baseline

**Files:**
- Modify: `bench/cache/llm.db` (gains LOCOMO entries)
- Create: `bench/results/main/locomo-{fts5,nark,vector}-locomo-default.json`

Manual task. Similar shape to Task 11.

- [ ] **Step 14.1: Confirm dataset is fetched**

```bash
ls bench/datasets/locomo/upstream/repo/ 2>/dev/null || bash bench/datasets/locomo/fetch.sh
```

- [ ] **Step 14.2: Run LOCOMO against all three systems**

```bash
cargo run -p nark-bench --release -- run \
  --task locomo \
  --systems fts5,nark,vector \
  --gen-backend codex-cli --gen-model gpt-5.5 \
  --judge-backend codex-cli --judge-model gpt-5.5 \
  --llm-concurrency 4 \
  --output bench/results/main
```

Confirm `y` at prompt. LOCOMO has fewer total QA pairs than LongMemEval (estimate ~200-300 depending on the specific dataset variant), so this run is shorter.

- [ ] **Step 14.3: Inspect LOCOMO baselines**

```bash
for sys in fts5 nark vector; do
  echo "=== $sys ==="
  jq '.schema_version, .corpus, .answer.accuracy, .answer.questions, .properties.methodology_disclaimer.note' \
    bench/results/main/locomo-${sys}-locomo-default.json
done
```

Sanity-check:
- `schema_version == "3"`, `corpus == "locomo"`.
- All three: `properties.methodology_disclaimer.note` is present.
- Accuracy numbers similar order of magnitude to LongMemEval (typically lower because LOCOMO is harder for short-context retrieval).

- [ ] **Step 14.4: Commit baselines + updated cache**

```bash
git add bench/results/main/locomo-*.json bench/cache/llm.db
git commit -m "$(cat <<'EOF'
bench: lock in first LOCOMO baselines (with methodology disclaimer)

Three LOCOMO baselines under schema v3. Generation + judge via
codex-cli:gpt-5.5 (same configuration as LongMemEval; identical
prompts where possible).

Each baseline's properties.methodology_disclaimer field embeds links
to the Zep and Letta critiques of LOCOMO. This is visible in every
committed result so the number can never be quoted out of context.

LLM cache (bench/cache/llm.db) now contains both LongMemEval and
LOCOMO entries. Re-runs of the same prompt + model + version are free.

Same "internal directional" caveat as LongMemEval applies until Phase 3
validation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Integration smoke test for Task B

**Files:**
- Create: `bench/tests/smoke_b.rs`

Uses `EchoBackend` for both gen and judge so `cargo test` never hits real LLMs. Verifies the runner end-to-end against a synthetic fixture dataset.

- [ ] **Step 15.1: Create `bench/tests/smoke_b.rs`**

```rust
//! Integration smoke test for Task B (LongMemEval + LOCOMO runners).
//!
//! Uses EchoBackend for both gen and judge so cargo test never hits
//! a real LLM. Verifies the pipeline end-to-end against a tiny
//! synthetic dataset.

use anyhow::Result;
use nark_bench::adapters::fts5::Fts5Adapter;
use nark_bench::llm::cache::LlmCache;
use nark_bench::llm::echo::EchoBackend;
use nark_bench::protocol::Adapter;
use nark_bench::tasks::longmemeval::run_longmemeval_task;
use serde_json::json;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

#[test]
fn smoke_b_longmemeval_with_echo_backends() {
    // Build a tiny synthetic dataset on the fly (2 questions).
    let dataset = json!([
        {
            "question_id": "q1",
            "question": "What did Alice say?",
            "answer": "Hello",
            "question_type": "info_extraction",
            "haystack_sessions": [["Alice said hello", "Bob said hi"]]
        },
        {
            "question_id": "q2",
            "question": "What is the answer?",
            "answer": "I don't know",
            "question_type": "abstention",
            "haystack_sessions": [["irrelevant text"]]
        }
    ]);
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), serde_json::to_string(&dataset).unwrap()).unwrap();

    let gen_path = workspace_root().join("bench/llm/prompts/longmemeval-generate.md");
    let judge_path = workspace_root().join("bench/llm/prompts/longmemeval-judge.md");

    // Gen always answers "Hello"; judge always says correct.
    let mut gen_backend = EchoBackend::new().with_default("Hello");
    let mut judge_backend = EchoBackend::new()
        .with_default(r#"{"verdict": "correct", "reason": "ok"}"#);

    let cache = LlmCache::open(&PathBuf::from(":memory:")).unwrap();

    let mut factory = || -> Result<Box<dyn Adapter>> {
        Ok(Box::new(Fts5Adapter::new()))
    };

    let result = run_longmemeval_task(
        &mut factory,
        tmp.path(),
        &mut gen_backend,
        &gen_path,
        &mut judge_backend,
        &judge_path,
        &cache,
        "smoke",
    ).unwrap();

    assert_eq!(result.task, "longmemeval");
    assert_eq!(result.schema_version, "3");
    assert!(result.answer.is_some());
    let answer = result.answer.unwrap();
    assert_eq!(answer.questions, 2);
    assert_eq!(answer.accuracy, 1.0); // judge said correct for both
    assert_eq!(answer.judge_error_rate, 0.0);

    // Both abilities present.
    assert!(answer.per_ability.contains_key("info_extraction"));
    assert!(answer.per_ability.contains_key("abstention"));

    // Perf blocks populated.
    assert!(result.perf.generation.is_some());
    assert!(result.perf.judging.is_some());
    let gen_perf = result.perf.generation.unwrap();
    let judge_perf = result.perf.judging.unwrap();
    assert_eq!(gen_perf.calls, 2);
    assert_eq!(judge_perf.calls, 2);
}
```

Note: this test imports `nark_bench::...` paths, which means the `bench/Cargo.toml` needs `[lib]` exposure for the `nark_bench` crate so tests can import its modules. Phase 1b's lib hybrid was for `nark`; bench is currently a binary-only crate. Two options:

(a) Add a `[lib]` target to `bench/Cargo.toml` similarly:

```toml
[lib]
name = "nark_bench"
path = "src/main.rs"   # not ideal — main.rs has a fn main()
```

(b) Cleaner: refactor `bench/src/main.rs` to split into `bench/src/lib.rs` + a thin `bench/src/main.rs` that just calls into the lib. This is structurally correct but is a bigger change.

Pragmatic for Phase 2: put the smoke test inline as a unit test inside `bench/src/tasks/longmemeval.rs` (already done in Task 9!) rather than an integration test that needs lib exposure. Mark this whole task as "the smoke test from Task 9 IS the integration coverage; no separate integration file needed."

**Revised plan for Task 15:** Verify the Task 9 unit test (which is exactly the same shape as the integration smoke proposed here) passes after Tasks 10-14 have shipped, and that's the integration coverage for Task B. The CI Lane 1 still runs Task A's smoke test from Phase 1; Task B coverage lives in the unit test inside `bench/src/tasks/longmemeval.rs`.

Skip creating `bench/tests/smoke_b.rs`. Instead:

- [ ] **Step 15.1 (revised): Verify the in-module smoke test from Task 9 passes after all subsequent commits**

```bash
cargo test -p nark-bench tasks::longmemeval::tests::smoke_runs_longmemeval_with_echo_backends
```

Expected: passes.

- [ ] **Step 15.2: Add a similar smoke test inside `bench/src/tasks/locomo.rs` for LOCOMO coverage**

Read `bench/src/tasks/locomo.rs`. After the `pub fn run_locomo_task` definition (i.e. at the end of the file), append a tests module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::fts5::Fts5Adapter;
    use crate::llm::echo::EchoBackend;
    use std::path::PathBuf;

    fn cache() -> LlmCache {
        LlmCache::open(&PathBuf::from(":memory:")).unwrap()
    }

    #[test]
    fn smoke_runs_locomo_with_echo_backends() {
        let dataset = json!([
            {
                "sample_id": "s1",
                "conversation": ["Alice said hello", "Bob said hi"],
                "qa": [
                    {"question_id": "q1", "question": "Q?", "answer": "Hello", "category": "factual"}
                ]
            }
        ]);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), serde_json::to_string(&dataset).unwrap()).unwrap();

        let gen_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("llm/prompts/locomo-generate.md");
        let judge_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("llm/prompts/locomo-judge.md");

        let mut gen_backend = EchoBackend::new().with_default("Hello");
        let mut judge_backend = EchoBackend::new()
            .with_default(r#"{"verdict": "correct", "reason": "ok"}"#);
        let cache = cache();

        let mut factory = || -> anyhow::Result<Box<dyn Adapter>> {
            Ok(Box::new(Fts5Adapter::new()))
        };

        let result = run_locomo_task(
            &mut factory,
            tmp.path(),
            &mut gen_backend,
            &gen_path,
            &mut judge_backend,
            &judge_path,
            &cache,
            "smoke",
        ).unwrap();

        assert_eq!(result.task, "locomo");
        assert_eq!(result.schema_version, "3");
        assert!(result.properties.get("methodology_disclaimer").is_some());
        assert!(result.answer.is_some());
        let answer = result.answer.unwrap();
        assert_eq!(answer.questions, 1);
        assert_eq!(answer.accuracy, 1.0);
    }
}
```

- [ ] **Step 15.3: Verify all tests pass**

```bash
cargo test -p nark-bench
```

Expected: all unit tests pass (Phase 1's set + Phase 2's set ~50+ tests total), 1 ignored. Task A integration smoke (`cargo test -p nark-bench --test smoke`) still passes (Phase 1c unchanged).

- [ ] **Step 15.4: Commit**

```bash
git add bench/src/tasks/locomo.rs
git commit -m "$(cat <<'EOF'
test(bench/tasks): in-module smoke test for run_locomo_task

Mirrors Task 9's LongMemEval smoke test inside the locomo module.
Uses EchoBackend for both gen and judge so cargo test never hits a
real LLM. Verifies the runner end-to-end against a tiny synthetic
sample with one QA pair, asserts the methodology disclaimer is
present in result.properties.

This is the integration coverage for Task B. CI Lane 1 still runs
Task A's smoke test from Phase 1; Task B unit tests cover the runner
shape end-to-end without LLM calls.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Wrap-up

After all 15 tasks land, Phase 2 is complete. Concretely:

- `bench/src/llm/` exists with the `LlmBackend` trait + 4 implementations (`EchoBackend` for tests; `ClaudeCliBackend`, `CodexCliBackend`, `ApiBackend` for production).
- `bench/cache/llm.db` is committed (a few MB after first sweeps); re-runs are free.
- `bench/llm/prompts/` holds the 4 prompt templates (gen + judge × LongMemEval + LOCOMO), version-tracked via `<!-- prompt-version: N -->` headers.
- `bench/src/tasks/longmemeval.rs` + `bench/src/tasks/locomo.rs` host the Task B runners.
- `bench/datasets/longmemeval/` + `bench/datasets/locomo/` have fetch scripts + READMEs; upstream clones are gitignored.
- `bench/results/main/` contains 6 new baseline JSONs: `{longmemeval,locomo}-{fts5,nark,vector}-{longmemeval,locomo}-default.json`.
- Result schema is at "3"; old Phase 1 baselines deserialize without breakage (new fields are Option/default).
- `bench/src/main.rs` dispatches `--task longmemeval` and `--task locomo`; new flags `--gen-backend`/`--gen-model`/`--judge-backend`/`--judge-model`/`--llm-concurrency`/`--yes`.
- 50+ unit tests passing; 2 integration smokes (Phase 1's `smoke.rs` for Task A + the in-module Task B smokes); CI Lane 1 still gates on Task A only.
- Phase 2 numbers are "internal directional" pending Phase 3's mem0 adapter for methodology validation. Commit messages + README make this explicit.

## Follow-up plans (not part of Phase 2)

- **Validation against published mem0/Zep numbers** — requires Phase 3's mem0 adapter to exist. Re-running our LongMemEval pipeline with mem0 as a system, comparing our accuracy number to mem0's published 91.6% (±2%), and tuning prompts if necessary. Until then: numbers are internal directional.
- **Phase 3 — Mem0 / Letta / Graphiti competitor adapters.** Adds new systems to the bench so `--systems mem0,letta,graphiti,nark,vector,fts5` produces a comparison row across all systems on Task A + Task B.
- **Phase 4 — Report generator → BENCHMARKS.md + README badges.** Once Phase 3 baselines exist, auto-generate a public-facing comparison page.
- **Phase 5 — LongMemEval-M / LongMemEval-L variants.** Longer haystacks; same code path.
- **CI inclusion of Task B** — possibly nightly cron on a self-hosted runner with API tokens. Depends on budget discussion.
- **Per-question telemetry export** — currently aggregate-only in result JSON; per-question latency/cost would be useful for diagnostics.
