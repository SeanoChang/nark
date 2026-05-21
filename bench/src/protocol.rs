use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A document to be ingested by an adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub body: String,
    /// Frontmatter / tags / timestamps. Adapters decide what to use.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// One ranked search hit returned by an adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub doc_id: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WriteMetrics {
    pub latency_us: u64,
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchMetrics {
    pub latency_us: u64,
    pub llm_tokens_in: u64,
    pub llm_tokens_out: u64,
}

/// The single trait every system-under-test implements.
pub trait Adapter {
    /// Human-readable name used in result files and CLI flags (e.g. "fts5", "nark").
    fn name(&self) -> &str;

    /// Version string for the system under test, e.g. "nark 0.13.0" or "sqlite 3.45.0 FTS5".
    /// Recorded in result JSON for reproducibility. Returning "unknown" should be avoided —
    /// the harness refuses to run if any adapter reports "unknown".
    fn version(&self) -> Result<String>;

    /// Called once before any write/search. `workdir` is a clean temp directory
    /// the adapter can use freely.
    fn setup(&mut self, workdir: &Path) -> Result<()>;

    /// Ingest one document. Latency is wall-clock; token counts are 0 for
    /// adapters that don't call LLMs on write.
    fn write(&mut self, doc: &Document) -> Result<WriteMetrics>;

    /// Ingest a batch of documents. The default loops over `write`, but adapters
    /// that pay heavy per-invocation overhead (subprocess startup, model loads)
    /// should override this to amortize across the batch. Returns aggregate
    /// metrics; per-document latency is summed into `latency_us`.
    ///
    /// Required for the LongMemEval/LOCOMO tasks where haystacks are 500+ turns
    /// — subprocess-per-turn ingest is otherwise unworkable.
    fn write_batch(&mut self, docs: &[Document]) -> Result<WriteMetrics> {
        let mut total = WriteMetrics::default();
        for doc in docs {
            let m = self.write(doc)?;
            total.latency_us += m.latency_us;
            total.llm_tokens_in += m.llm_tokens_in;
            total.llm_tokens_out += m.llm_tokens_out;
        }
        Ok(total)
    }

    /// Run a query, return top-k hits ranked by the adapter's own scoring.
    fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)>;

    /// Called once at the end. Should release any subprocess / connection / temp state.
    fn teardown(&mut self) -> Result<()>;
}
