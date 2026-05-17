//! Pure-Rust vector baseline. Embeds every doc on write, brute-force cosine
//! on search. Uses the same OnnxProvider as nark to keep model choice
//! constant (apples-to-apples comparison).

use anyhow::{anyhow, Context, Result};
use nark::embed::{cosine_similarity, init_embedding, EmbeddingProvider, OnnxProvider};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::model_cache;
use crate::protocol::{Adapter, Document, SearchMetrics, SearchResult, WriteMetrics};

pub struct VectorAdapter {
    model_cache: PathBuf,
    provider: Option<OnnxProvider>,
    embeddings: HashMap<String, Vec<f32>>,
    workdir: Option<PathBuf>,
}

impl VectorAdapter {
    pub fn new(model_cache: PathBuf) -> Self {
        Self {
            model_cache,
            provider: None,
            embeddings: HashMap::new(),
            workdir: None,
        }
    }
}

impl Adapter for VectorAdapter {
    fn name(&self) -> &str { "vector" }

    fn version(&self) -> Result<String> {
        // Reports the intended model identifier. Actual init success is
        // signalled by setup() returning Ok(()). The bench harness only
        // calls version() after setup() succeeds, so this is safe.
        Ok(format!("vector {}", nark::embed::MODEL_NAME))
    }

    fn setup(&mut self, workdir: &Path) -> Result<()> {
        model_cache::stage_into(&self.model_cache, workdir)
            .context("failed to stage embedding model into vector adapter workdir")?;
        let mut provider = init_embedding(workdir)
            .ok_or_else(|| anyhow!("ONNX init returned None after staging — model files corrupt?"))?;

        // Defensive normalization probe: confirm provider outputs are L2-normalized.
        // We rely on this assumption when calling nark::embed::cosine_similarity
        // (which is dot-product-only). If a future change breaks the normalization
        // contract, this assertion catches it loudly instead of silent degradation.
        let probe = provider.embed_document("probe")
            .context("normalization probe: failed to embed test string")?;
        let norm: f32 = probe.iter().map(|x| x * x).sum::<f32>().sqrt();
        if (norm - 1.0).abs() > 1e-3 {
            return Err(anyhow!(
                "OnnxProvider output is not L2-normalized (norm = {:.6}); \
                 cosine_similarity will return incorrect results. \
                 Check nark::embed::OnnxProvider::run_inference for an l2_normalize call.",
                norm
            ));
        }

        self.provider = Some(provider);
        self.workdir = Some(workdir.to_path_buf());
        self.embeddings.clear();
        Ok(())
    }

    fn write(&mut self, doc: &Document) -> Result<WriteMetrics> {
        let t0 = Instant::now();
        let provider = self.provider.as_mut().ok_or_else(|| anyhow!("setup not called"))?;
        let embedding = provider.embed_document(&doc.body)
            .with_context(|| format!("failed to embed doc {}", doc.id))?;
        self.embeddings.insert(doc.id.clone(), embedding);
        Ok(WriteMetrics {
            latency_us: t0.elapsed().as_micros() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        })
    }

    fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)> {
        let t0 = Instant::now();
        let provider = self.provider.as_mut().ok_or_else(|| anyhow!("setup not called"))?;
        let query_emb = provider.embed_query(query)?;
        let mut scored: Vec<(String, f32)> = self.embeddings.iter()
            .map(|(id, emb)| (id.clone(), cosine_similarity(&query_emb, emb)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let results: Vec<SearchResult> = scored.into_iter().take(k)
            .map(|(doc_id, score)| SearchResult { doc_id, score, snippet: None })
            .collect();
        Ok((results, SearchMetrics {
            latency_us: t0.elapsed().as_micros() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        }))
    }

    fn teardown(&mut self) -> Result<()> {
        self.provider = None;
        self.embeddings.clear();
        self.workdir = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nark::embed::l2_normalize;

    #[test]
    fn cosine_identical_unit_vectors_returns_one() {
        let v = l2_normalize(&[1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "expected ~1.0, got {}", sim);
    }

    #[test]
    fn cosine_perpendicular_unit_vectors_returns_zero() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "expected ~0.0, got {}", sim);
    }

    #[test]
    fn cosine_opposite_unit_vectors_returns_negative_one() {
        let a = l2_normalize(&[1.0, 2.0, 3.0]);
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "expected ~-1.0, got {}", sim);
    }
}
