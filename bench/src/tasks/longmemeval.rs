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
//! Aggregates: overall accuracy, per_ability accuracy (keyed by paper-aligned
//! ability class), abstention_precision, judge_error_rate, perf.generation + perf.judging.

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
use crate::tasks::longmemeval_loader::{ability_class, haystack_to_documents, load_questions};

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
    let gen_template = PromptTemplate::load(gen_template_path)
        .with_context(|| format!("failed to load gen template at {:?}", gen_template_path))?;
    let judge_template = PromptTemplate::load(judge_template_path)
        .with_context(|| format!("failed to load judge template at {:?}", judge_template_path))?;

    let questions = load_questions(dataset_path)?;

    let probe = adapter_factory()?;
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

    let mut all_verdicts: Vec<(String, Verdict)> = Vec::new(); // (ability_class, verdict)
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
            let _ = adapter.teardown();
            continue;
        }

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

        let snippets: Vec<String> = hits.into_iter()
            .map(|h| h.snippet.unwrap_or_default())
            .collect();

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
        // `calls` counts LIVE LLM invocations; `cache_hits` counts skipped ones.
        // tokens/cost are only populated on live calls (cache returns zeros).
        // Keeping these counters consistent avoids cost-per-call drift on re-runs.
        if gen_result.from_cache {
            gen_cache_hits += 1;
        } else {
            gen_calls += 1;
            gen_tokens_in += gen_result.tokens_in;
            gen_tokens_out += gen_result.tokens_out;
            gen_cost += gen_result.cost_usd_micros;
        }

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
        if judgment.from_cache {
            judge_cache_hits += 1;
        } else {
            judge_calls += 1;
            judge_tokens_in += judgment.tokens_in;
            judge_tokens_out += judgment.tokens_out;
            judge_cost += judgment.cost_usd_micros;
        }

        if matches!(judgment.verdict, Verdict::JudgeError) {
            judge_errors += 1;
        }

        let gold_abstains = q.answer.trim().eq_ignore_ascii_case("I don't know")
            || q.answer.trim().eq_ignore_ascii_case("idk");
        if gold_abstains {
            abstention_total += 1;
            if matches!(judgment.verdict, Verdict::Correct) {
                abstention_correct += 1;
            }
        }

        // Use paper-aligned 5-class ability (free function from loader, NOT a method).
        // The loader exposes `ability_class(question_type: &str)` mapping 6 raw
        // question_type values to 5 paper-aligned ability classes.
        all_verdicts.push((ability_class(&q.question_type).to_string(), judgment.verdict));

        let _ = adapter.teardown();

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
    use crate::adapters::fts5::Fts5Adapter;
    use crate::llm::echo::EchoBackend;
    use std::path::PathBuf;

    fn cache() -> LlmCache {
        LlmCache::open(&PathBuf::from(":memory:")).unwrap()
    }

    #[test]
    fn smoke_runs_longmemeval_with_echo_backends() {
        // Build a tiny synthetic dataset matching Task 8's discovered format:
        // haystack_sessions is Vec<Vec<{role, content}>>, NOT Vec<Vec<String>>.
        let dataset = serde_json::json!([
            {
                "question_id": "q1",
                "question": "What did Alice say?",
                "answer": "Hello",
                "question_type": "single-session-user",
                "question_date": "2024-01-01",
                "answer_session_ids": ["s0"],
                "haystack_session_ids": ["s0"],
                "haystack_dates": ["2024-01-01"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "Alice said hello"},
                        {"role": "assistant", "content": "Bob said hi"}
                    ]
                ]
            }
        ]);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), serde_json::to_string(&dataset).unwrap()).unwrap();

        let gen_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("llm/prompts/longmemeval-generate.md");
        let judge_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("llm/prompts/longmemeval-judge.md");

        let mut gen_backend = EchoBackend::new().with_default("Alice said hello");
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
