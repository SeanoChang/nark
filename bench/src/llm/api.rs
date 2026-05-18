//! HTTP backend talking directly to the Anthropic API.
//!
//! Auth: `ANTHROPIC_API_KEY` env var. Returns an error if unset.
//! Use case: CI environments, contributors without Claude Code installed,
//! or scenarios where the OAuth CLI isn't viable.

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
                    // Per-model pricing — see `pricing_for_model` below.
                    // Anthropic CLI returns cost directly; we have to estimate
                    // it for the HTTP path.
                    let (in_per_tok, out_per_tok) = pricing_for_model(&self.model);
                    let cost_usd_micros = body.usage.input_tokens * in_per_tok
                        + body.usage.output_tokens * out_per_tok;
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

/// Returns (input_micros_per_token, output_micros_per_token) for the given
/// Anthropic model. Defaults to Opus pricing for unknown models so we
/// never underestimate cost. Update when Anthropic pricing changes.
///
/// Approximate prices (May 2026):
/// - Opus:   $15.00 / Mtok in,  $75.00 / Mtok out  → 15 / 75 micros per token
/// - Sonnet:  $3.00 / Mtok in,  $15.00 / Mtok out  →  3 / 15 micros per token
/// - Haiku:   $1.00 / Mtok in,   $5.00 / Mtok out  →  1 /  5 micros per token
fn pricing_for_model(model: &str) -> (u64, u64) {
    let m = model.to_lowercase();
    if m.contains("haiku") {
        (1, 5)
    } else if m.contains("sonnet") {
        (3, 15)
    } else {
        (15, 75)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pricing_for_known_models() {
        assert_eq!(pricing_for_model("claude-opus-4-7"), (15, 75));
        assert_eq!(pricing_for_model("claude-sonnet-4-6"), (3, 15));
        assert_eq!(pricing_for_model("claude-haiku-4-5"), (1, 5));
        // Unknown defaults to Opus (overestimate, safer).
        assert_eq!(pricing_for_model("claude-future-7-0"), (15, 75));
    }

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
