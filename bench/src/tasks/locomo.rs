//! Task B — LOCOMO runner. Mirrors run_longmemeval_task with LOCOMO's
//! sample-with-many-QAs format.
//!
//! Per sample: ingest the conversation ONCE into the adapter, then run all
//! attached QAs against that ingested haystack (much more efficient than
//! re-ingesting per QA — LOCOMO has up to 260 QAs per sample).

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
use crate::tasks::locomo_loader::{category_label, conversation_to_documents, load_samples};

const K: usize = 10;

#[allow(clippy::too_many_arguments)]
pub fn run_locomo_task(
    adapter_factory: &mut dyn FnMut() -> Result<Box<dyn Adapter>>,
    dataset_path: &Path,
    gen_backend: &mut dyn LlmBackend,
    gen_template_path: &Path,
    judge_backend: &mut dyn LlmBackend,
    judge_template_path: &Path,
    cache: &LlmCache,
    config_label: &str,
    limit: Option<usize>,
) -> Result<BenchResult> {
    let gen_template = PromptTemplate::load(gen_template_path)
        .with_context(|| format!("failed to load gen template at {:?}", gen_template_path))?;
    let judge_template = PromptTemplate::load(judge_template_path)
        .with_context(|| format!("failed to load judge template at {:?}", judge_template_path))?;

    let samples = load_samples(dataset_path)?;

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

    let mut result =
        BenchResult::new("locomo", &system_name, config_label, &system_version, "locomo");

    // Methodology disclaimer inline in properties.
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
    let mut stop = false;

    for sample in &samples {
        // Ingest this sample's conversation ONCE per adapter setup
        // (multiple QAs share the same haystack — that's LOCOMO's structure).
        let workdir = tempfile::tempdir()?;
        let mut adapter = adapter_factory()?;
        if let Err(e) = adapter.setup(workdir.path()) {
            result.errors.push(BenchError {
                phase: format!("setup:{}", sample.sample_id),
                message: e.to_string(),
            });
            let _ = adapter.teardown();
            continue;
        }

        let docs: Vec<Document> = conversation_to_documents(sample)
            .into_iter()
            .map(|(doc_id, body)| Document { id: doc_id, body, metadata: json!({}) })
            .collect();
        if let Err(e) = adapter.write_batch(&docs) {
            result.errors.push(BenchError {
                phase: format!("write_batch:{}", sample.sample_id),
                message: e.to_string(),
            });
            let _ = adapter.teardown();
            continue;
        }

        for qa in sample.qa.iter() {
            // qa.question_id is already synthesized by the loader as
            // `<sample_id>_q<idx>` — no need to re-synthesize.
            let question_id = &qa.question_id;

            // qa.answer is serde_json::Value; convert to string for the judge.
            let gold = match &qa.answer {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Null => "null".to_string(),
                other => other.to_string(),
            };

            let (hits, _) = match adapter.search(&qa.question, K) {
                Ok(x) => x,
                Err(e) => {
                    result.errors.push(BenchError {
                        phase: format!("search:{}", question_id),
                        message: e.to_string(),
                    });
                    continue;
                }
            };

            let snippets: Vec<String> = hits
                .into_iter()
                .map(|h| h.snippet.unwrap_or_default())
                .collect();

            let gen_result =
                match generate_answer(gen_backend, cache, &gen_template, &qa.question, &snippets) {
                    Ok(g) => g,
                    Err(e) => {
                        result.errors.push(BenchError {
                            phase: format!("generate:{}", question_id),
                            message: e.to_string(),
                        });
                        continue;
                    }
                };
            // `calls` counts LIVE LLM invocations; cache_hits counts skipped ones.
            // Keeps cost-per-call meaningful on re-runs.
            if gen_result.from_cache {
                gen_cache_hits += 1;
            } else {
                gen_calls += 1;
                gen_tokens_in += gen_result.tokens_in;
                gen_tokens_out += gen_result.tokens_out;
                gen_cost += gen_result.cost_usd_micros;
            }

            let judgment = match judge_answer(
                judge_backend,
                cache,
                &judge_template,
                &qa.question,
                &gold,
                &gen_result.candidate,
            ) {
                Ok(j) => j,
                Err(e) => {
                    result.errors.push(BenchError {
                        phase: format!("judge:{}", question_id),
                        message: e.to_string(),
                    });
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

            let category = category_label(qa.category).to_string();

            let gold_abstains = gold.trim().eq_ignore_ascii_case("I don't know")
                || gold.trim().eq_ignore_ascii_case("idk")
                || gold.trim().eq_ignore_ascii_case("null");
            if gold_abstains {
                abstention_total += 1;
                if matches!(judgment.verdict, Verdict::Correct) {
                    abstention_correct += 1;
                }
            }

            all_verdicts.push((category, judgment.verdict));
            processed += 1;

            // --limit early-break: still let adapter.teardown() run so SQLite
            // handles close cleanly; just break out after the current sample.
            if let Some(cap) = limit {
                if processed >= cap {
                    stop = true;
                    break;
                }
            }
        }

        let _ = adapter.teardown();

        if stop {
            break;
        }
        if processed % 50 == 0 {
            eprintln!("  locomo: {}/{} QAs ({})", processed, total_qa, system_name);
        }
    }

    let n = all_verdicts.len() as f64;
    let overall = if n > 0.0 {
        all_verdicts.iter().map(|(_, v)| v.score()).sum::<f64>() / n
    } else {
        0.0
    };

    let mut per_category_buckets: HashMap<String, Vec<Verdict>> = HashMap::new();
    for (cat, v) in &all_verdicts {
        per_category_buckets.entry(cat.clone()).or_default().push(*v);
    }
    let per_ability: HashMap<String, AbilityMetrics> = per_category_buckets
        .into_iter()
        .map(|(k, vs)| {
            let acc = if vs.is_empty() {
                0.0
            } else {
                vs.iter().map(|v| v.score()).sum::<f64>() / vs.len() as f64
            };
            (k, AbilityMetrics { accuracy: acc, questions: vs.len() })
        })
        .collect();

    let abstention_precision = if abstention_total > 0 {
        abstention_correct as f64 / abstention_total as f64
    } else {
        1.0
    };

    let judge_error_rate = if !all_verdicts.is_empty() {
        judge_errors as f64 / all_verdicts.len() as f64
    } else {
        0.0
    };

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
    fn smoke_runs_locomo_with_echo_backends() {
        // Build a tiny synthetic LOCOMO dataset matching the loader format.
        // Conversation has speaker_a, speaker_b, and dynamic session_<N> keys.
        // The loader also expects observation, session_summary, and event_summary fields.
        let dataset = serde_json::json!([
            {
                "sample_id": "s1",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "Alice said hello"},
                        {"speaker": "Bob",   "dia_id": "D1:2", "text": "Bob said hi"}
                    ]
                },
                "qa": [
                    {
                        "question": "What did Alice say?",
                        "answer": "Hello",
                        "category": 2,
                        "evidence": ["D1:1"]
                    }
                ],
                "observation": {},
                "session_summary": {},
                "event_summary": {}
            }
        ]);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), serde_json::to_string(&dataset).unwrap()).unwrap();

        let gen_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("llm/prompts/locomo-generate.md");
        let judge_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
            None,
        ).unwrap();

        assert_eq!(result.task, "locomo");
        assert_eq!(result.schema_version, "3");
        assert!(result.properties.get("methodology_disclaimer").is_some(),
            "methodology_disclaimer must be present in result.properties");
        assert!(result.answer.is_some());
        let answer = result.answer.unwrap();
        assert_eq!(answer.questions, 1);
        assert_eq!(answer.accuracy, 1.0);
    }
}
