//! LLM call abstraction for Task B (generation + judging).
//!
//! Single trait `LlmBackend` abstracts over `claude -p` and the Anthropic
//! HTTP API. The trait is shared between generation and judging — the
//! difference between those two roles is the prompt template + the
//! response parsing (see `eval.rs`).
//!
//! NOTE: Codex CLI is intentionally NOT supported in Phase 2. Codex's
//! non-interactive mode requires `--dangerously-bypass-approvals-and-sandbox`
//! which gives the model unconfined access to the local filesystem and
//! credentials — unsafe for benchmark workloads where prompt-injected
//! haystack content could induce tool use. Re-introduce only when codex
//! ships a safe non-agentic mode, or run inside a sandboxed container.

use anyhow::Result;

pub mod cache;
pub mod claude_cli;
pub mod echo;
pub mod api;
pub mod prompt;
pub mod eval;

pub use eval::Verdict;

/// A backend capable of taking a prompt string and returning an LLM response.
///
/// Implementations:
/// - [`echo::EchoBackend`] — test fixture; returns canned responses.
/// - [`claude_cli::ClaudeCliBackend`] — `claude -p` subprocess (OAuth from disk).
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
