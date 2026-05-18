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
