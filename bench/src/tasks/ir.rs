//! Task A: classical IR. Loads a query→relevant-ids fixture, runs every
//! query through every adapter, computes Recall@k / MRR / nDCG, emits one
//! BenchResult per (system, config).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::metrics::ir::{ndcg_at_k, recall_at_k, reciprocal_rank};
use crate::protocol::{Adapter, Document};
use crate::result::{percentile, BenchError, BenchResult, IrMetrics};

#[derive(Debug, Deserialize)]
struct QueryRow {
    query_id: String,
    query: String,
    relevant: Vec<String>,
    #[serde(default)]
    class: Option<String>,
}

pub fn run_ir_task(
    adapter: &mut dyn Adapter,
    corpus_root: &Path,
    config_label: &str,
) -> Result<BenchResult> {
    let system_name = adapter.name().to_string();
    let system_version = adapter.version().unwrap_or_else(|_| "unknown".to_string());
    let corpus_name = corpus_root.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    if system_version == "unknown" {
        anyhow::bail!("adapter '{}' refused to report a version — refusing to run for reproducibility", system_name);
    }

    let mut result = BenchResult::new("ir", &system_name, config_label, &system_version, &corpus_name);

    let workdir = tempfile::tempdir()?;
    adapter.setup(workdir.path())?;

    // Ingest corpus
    let corpus_dir = corpus_root.join("corpus");
    let mut write_latencies = Vec::new();
    let mut tokens_in_total = 0u64;
    let mut tokens_out_total = 0u64;

    // Sort corpus entries by file name so ingest order is deterministic across
    // platforms (read_dir order is filesystem-dependent on macOS vs Linux).
    let mut entries: Vec<_> = std::fs::read_dir(&corpus_dir)
        .with_context(|| format!("failed to read corpus dir {:?}", corpus_dir))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let body = std::fs::read_to_string(&path)?;
        let id = path.file_stem().unwrap().to_string_lossy().to_string();
        let doc = Document { id: id.clone(), body, metadata: serde_json::json!({}) };
        match adapter.write(&doc) {
            Ok(m) => {
                write_latencies.push(m.latency_ms);
                tokens_in_total += m.llm_tokens_in;
                tokens_out_total += m.llm_tokens_out;
            }
            Err(e) => {
                result.errors.push(BenchError {
                    phase: format!("write:{}", id),
                    message: e.to_string(),
                });
            }
        }
    }

    // Load queries
    let queries_path = corpus_root.join("queries.jsonl");
    let queries_str = std::fs::read_to_string(&queries_path)
        .with_context(|| format!("failed to read queries file {:?}", queries_path))?;
    let queries: Vec<QueryRow> = queries_str
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<QueryRow>(l)
            .with_context(|| format!("bad query JSON: {}", l)))
        .collect::<Result<_>>()?;

    // Run queries, score each
    let mut search_latencies = Vec::new();
    let mut s_tokens_in_total = 0u64;
    let mut s_tokens_out_total = 0u64;
    let mut per_class_scores: HashMap<String, Vec<(Vec<String>, HashSet<String>)>> = HashMap::new();
    let mut all_scores: Vec<(Vec<String>, HashSet<String>)> = Vec::new();

    for q in &queries {
        let relevant: HashSet<String> = q.relevant.iter().cloned().collect();
        match adapter.search(&q.query, 10) {
            Ok((hits, m)) => {
                search_latencies.push(m.latency_ms);
                s_tokens_in_total += m.llm_tokens_in;
                s_tokens_out_total += m.llm_tokens_out;
                let ranked: Vec<String> = hits.into_iter().map(|h| h.doc_id).collect();
                all_scores.push((ranked.clone(), relevant.clone()));
                if let Some(cls) = &q.class {
                    per_class_scores.entry(cls.clone()).or_default().push((ranked, relevant));
                }
            }
            Err(e) => {
                result.errors.push(BenchError {
                    phase: format!("search:{}", q.query_id),
                    message: e.to_string(),
                });
            }
        }
    }

    adapter.teardown()?;

    // Compute aggregate metrics
    result.ir = Some(aggregate(&all_scores));
    for (cls, rows) in per_class_scores {
        result.ir_per_class.insert(cls, aggregate(&rows));
    }

    // Performance
    write_latencies.sort_unstable();
    search_latencies.sort_unstable();
    result.perf.write.latency_p50_ms = percentile(&write_latencies, 0.5);
    result.perf.write.latency_p99_ms = percentile(&write_latencies, 0.99);
    result.perf.write.llm_tokens_in_total = tokens_in_total;
    result.perf.write.llm_tokens_out_total = tokens_out_total;
    result.perf.search.latency_p50_ms = percentile(&search_latencies, 0.5);
    result.perf.search.latency_p99_ms = percentile(&search_latencies, 0.99);
    result.perf.search.llm_tokens_in_total = s_tokens_in_total;
    result.perf.search.llm_tokens_out_total = s_tokens_out_total;

    Ok(result)
}

fn aggregate(rows: &[(Vec<String>, HashSet<String>)]) -> IrMetrics {
    if rows.is_empty() {
        return IrMetrics::default();
    }
    let n = rows.len() as f64;
    let r1: f64 = rows.iter().map(|(r, g)| recall_at_k(r, g, 1)).sum::<f64>() / n;
    let r5: f64 = rows.iter().map(|(r, g)| recall_at_k(r, g, 5)).sum::<f64>() / n;
    let r10: f64 = rows.iter().map(|(r, g)| recall_at_k(r, g, 10)).sum::<f64>() / n;
    let mrr: f64 = rows.iter().map(|(r, g)| reciprocal_rank(r, g)).sum::<f64>() / n;
    let nd: f64 = rows.iter().map(|(r, g)| ndcg_at_k(r, g, 10)).sum::<f64>() / n;
    IrMetrics {
        recall_at_1: r1,
        recall_at_5: r5,
        recall_at_10: r10,
        mrr,
        ndcg_at_10: nd,
        queries: rows.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_empty() {
        let m = aggregate(&[]);
        assert_eq!(m.queries, 0);
        assert_eq!(m.mrr, 0.0);
    }

    #[test]
    fn aggregate_one_perfect_query() {
        let ranked = vec!["a".to_string(), "b".to_string()];
        let relevant: HashSet<String> = ["a".to_string()].iter().cloned().collect();
        let m = aggregate(&[(ranked, relevant)]);
        assert_eq!(m.queries, 1);
        assert!((m.recall_at_1 - 1.0).abs() < 1e-9);
        assert!((m.mrr - 1.0).abs() < 1e-9);
    }
}
