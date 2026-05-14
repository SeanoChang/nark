use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

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
    pub ir_per_class: HashMap<String, IrMetrics>,
    pub perf: PerfMetrics,
    pub errors: Vec<BenchError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IrMetrics {
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub ndcg_at_10: f64,
    /// Number of queries this metric was averaged over.
    pub queries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerfMetrics {
    pub write: PhaseMetrics,
    pub search: PhaseMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhaseMetrics {
    pub latency_p50_ms: u64,
    pub latency_p99_ms: u64,
    pub llm_tokens_in_total: u64,
    pub llm_tokens_out_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchError {
    pub phase: String,
    pub message: String,
}

impl BenchResult {
    pub fn new(task: &str, system: &str, config: &str, system_version: &str, corpus: &str) -> Self {
        Self {
            schema_version: "1".to_string(),
            task: task.to_string(),
            system: system.to_string(),
            config: config.to_string(),
            system_version: system_version.to_string(),
            bench_version: env!("CARGO_PKG_VERSION").to_string(),
            run_started_at: Utc::now().to_rfc3339(),
            corpus: corpus.to_string(),
            ir: None,
            ir_per_class: HashMap::new(),
            perf: PerfMetrics::default(),
            errors: vec![],
        }
    }

    pub fn write_to_disk(&self, output_dir: &Path) -> Result<std::path::PathBuf> {
        std::fs::create_dir_all(output_dir)?;
        let filename = format!("{}-{}-{}.json", self.task, self.system, self.config);
        let path = output_dir.join(filename);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(path)
    }
}

/// Helper: compute p50/p99 from a sorted vector of latencies.
pub fn percentile(sorted_ms: &[u64], p: f64) -> u64 {
    if sorted_ms.is_empty() {
        return 0;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx.min(sorted_ms.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_basic() {
        let mut v = vec![10u64, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        v.sort();
        assert_eq!(percentile(&v, 0.5), 60);
        assert_eq!(percentile(&v, 0.99), 100);
        assert_eq!(percentile(&v, 0.0), 10);
    }

    #[test]
    fn percentile_empty() {
        assert_eq!(percentile(&[], 0.5), 0);
    }
}
