pub mod fts5;
pub mod nark;
pub mod vector;

use crate::protocol::Adapter;
use anyhow::{anyhow, bail, Result};
use std::path::Path;

/// Construct an adapter by short name.
///
/// `model_cache`: path to the populated ONNX model cache. Required for
/// the `vector` adapter (errors if None). The `nark` adapter uses it
/// when provided to enable embeddings; if None, nark runs BM25-only.
/// The `fts5` adapter ignores it.
pub fn make_adapter(name: &str, model_cache: Option<&Path>) -> Result<Box<dyn Adapter>> {
    match name {
        "fts5" => Ok(Box::new(fts5::Fts5Adapter::new())),
        "nark" => Ok(Box::new(nark::NarkAdapter::new(model_cache.map(Path::to_path_buf)))),
        "vector" => {
            let cache = model_cache.ok_or_else(|| anyhow!(
                "vector adapter requires a model cache; ensure NARK_BENCH_MODEL_CACHE env var \
                 is set or that the runner has called model_cache::ensure_ready"
            ))?;
            Ok(Box::new(vector::VectorAdapter::new(cache.to_path_buf())))
        }
        other => bail!("unknown adapter: {}", other),
    }
}
