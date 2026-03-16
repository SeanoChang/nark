pub mod download;

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

const MODEL_NAME: &str = "bge-base-en-v1.5";
const EMBED_DIM: usize = 768;
const MAX_TOKENS: usize = 512;

pub struct EmbeddingEngine {
    session: Session,
    tokenizer: Tokenizer,
}

/// Try to load the embedding engine. Returns None if dylib or model files are missing.
pub fn init_embedding(vault_dir: &Path) -> Option<EmbeddingEngine> {
    let dylib = vault_dir.join("lib").join(onnx_dylib_name());
    let model = vault_dir.join("models").join(MODEL_NAME).join("model.onnx");
    let tok_path = vault_dir.join("models").join(MODEL_NAME).join("tokenizer.json");

    if !dylib.exists() || !model.exists() || !tok_path.exists() {
        return None;
    }

    ort::init_from(&dylib).ok()?.commit();

    let session = Session::builder()
        .ok()?
        .with_intra_threads(1)
        .ok()?
        .commit_from_file(&model)
        .ok()?;

    let mut tokenizer = Tokenizer::from_file(&tok_path).ok()?;
    tokenizer.with_truncation(Some(tokenizers::TruncationParams {
        max_length: MAX_TOKENS,
        ..Default::default()
    })).ok()?;
    tokenizer.with_padding(None);

    Some(EmbeddingEngine { session, tokenizer })
}

impl EmbeddingEngine {
    /// Embed a document (note content). No query prefix.
    pub fn embed_document(&mut self, text: &str) -> Result<Vec<f32>> {
        self.run_inference(text)
    }

    /// Embed a search query with BGE query prefix.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let prefixed = format!("Represent this sentence for searching relevant passages: {query}");
        self.run_inference(&prefixed)
    }

    fn run_inference(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let seq_len = input_ids.len();

        let ids = TensorRef::from_array_view(([1usize, seq_len], &*input_ids))
            .context("failed to create input_ids tensor")?;
        let mask = TensorRef::from_array_view(([1usize, seq_len], &*attention_mask))
            .context("failed to create attention_mask tensor")?;
        let type_ids = TensorRef::from_array_view(([1usize, seq_len], &*token_type_ids))
            .context("failed to create token_type_ids tensor")?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids,
            "attention_mask" => mask,
            "token_type_ids" => type_ids
        ]).context("inference failed")?;

        // Output: last_hidden_state [1, seq_len, 768]. Take CLS token (index 0).
        let hidden = outputs[0].try_extract_array::<f32>()
            .context("failed to extract output tensor")?;

        // CLS token embedding: hidden[[0, 0, ..]] → 768-dim
        let cls: Vec<f32> = (0..EMBED_DIM)
            .map(|i| hidden[[0, 0, i]])
            .collect();

        Ok(l2_normalize(&cls))
    }
}

/// Build the text input for document embedding.
/// Prefixes with bracketed taxonomy context to help bge-base distinguish domains.
pub fn build_embed_input(
    title: &str,
    domain: &str,
    kind: &str,
    intent: &str,
    tags: &[String],
    _aliases: &[String],
    body: &str,
) -> String {
    let tags_str = tags.join(", ");
    format!("[{domain} {kind} — {intent}] [{tags_str}] {title}\n\n{body}")
}

/// Cosine similarity between two L2-normalized vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn onnx_dylib_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libonnxruntime.1.24.2.dylib"
    } else {
        "libonnxruntime.so.1.24.2"
    }
}
