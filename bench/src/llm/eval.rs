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
        let mut backend = EchoBackend::new().with("\nContext", "Generated answer.");
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
        let mut backend = EchoBackend::new().with("\nQ:",
            r#"{"verdict": "correct", "reason": "Exact match"}"#);
        let tmpl = judge_template();

        let r = judge_answer(&mut backend, &cache, &tmpl, "Q", "gold", "candidate").unwrap();
        assert_eq!(r.verdict, Verdict::Correct);
        assert_eq!(r.reason, "Exact match");
    }

    #[test]
    fn judge_parses_partial_verdict() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("\nQ:",
            r#"{"verdict": "partial", "reason": "Missing detail"}"#);
        let r = judge_answer(&mut backend, &cache, &judge_template(), "Q", "g", "c").unwrap();
        assert_eq!(r.verdict, Verdict::Partial);
    }

    #[test]
    fn judge_returns_judge_error_on_malformed_json() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("\nQ:", "not json");
        let r = judge_answer(&mut backend, &cache, &judge_template(), "Q", "g", "c").unwrap();
        assert_eq!(r.verdict, Verdict::JudgeError);
    }

    #[test]
    fn judge_unknown_verdict_string_maps_to_judge_error() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("\nQ:",
            r#"{"verdict": "maybe", "reason": ""}"#);
        let r = judge_answer(&mut backend, &cache, &judge_template(), "Q", "g", "c").unwrap();
        assert_eq!(r.verdict, Verdict::JudgeError);
    }

    #[test]
    fn judge_caches_responses() {
        let cache = cache();
        let mut backend = EchoBackend::new().with("\nQ:",
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
