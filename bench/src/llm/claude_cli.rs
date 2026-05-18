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
