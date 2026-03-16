use anyhow::{bail, Context, Result};

use super::{l2_normalize, EmbeddingProvider};

const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_DIMS: usize = 1536;

pub struct ApiProvider {
    model: String,
    api_key: String,
    dimensions: usize,
}

impl ApiProvider {
    /// Create from environment. Returns None if OPENAI_API_KEY is not set.
    pub fn from_env(model_override: Option<&str>) -> Option<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").ok()?;
        if api_key.is_empty() {
            return None;
        }
        let model = model_override.unwrap_or(DEFAULT_MODEL).to_string();
        let dimensions = if model.contains("3-small") {
            DEFAULT_DIMS
        } else if model.contains("3-large") {
            3072
        } else {
            DEFAULT_DIMS
        };
        Some(Self { model, api_key, dimensions })
    }

    fn call_api(&self, input: &str) -> Result<Vec<f32>> {
        let body = serde_json::json!({
            "input": input,
            "model": self.model,
        });

        let resp = ureq::post("https://api.openai.com/v1/embeddings")
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .context("OpenAI API request failed")?;

        let resp_body = resp.into_string()
            .context("failed to read OpenAI response body")?;
        let json: serde_json::Value = serde_json::from_str(&resp_body)
            .context("failed to parse OpenAI response")?;

        let embedding = json["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("unexpected OpenAI response structure"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<f32>>();

        if embedding.is_empty() {
            bail!("OpenAI returned empty embedding");
        }

        Ok(l2_normalize(&embedding))
    }
}

impl EmbeddingProvider for ApiProvider {
    fn embed_document(&mut self, text: &str) -> Result<Vec<f32>> {
        self.call_api(text)
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        // OpenAI doesn't use instruction prefixes
        self.call_api(query)
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
