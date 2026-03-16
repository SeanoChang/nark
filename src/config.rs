use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

pub const BUILTIN_KINDS: &[&str] = &[
    "spec", "decision", "runbook", "report", "reference",
    "incident", "experiment", "dataset", "skill", "memory", "note", "journal",
];

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub search: SearchConfig,
    pub taxonomy: TaxonomyConfig,
    pub embedding: EmbeddingConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub provider: String,           // "local" | "openai"
    pub api_model: Option<String>,  // override default API model
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "local".to_string(),
            api_model: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TaxonomyConfig {
    pub extra_kinds: Vec<String>,
}

impl Default for TaxonomyConfig {
    fn default() -> Self {
        Self {
            extra_kinds: Vec::new(),
        }
    }
}

impl TaxonomyConfig {
    pub fn valid_kinds(&self) -> Vec<&str> {
        let mut kinds: Vec<&str> = BUILTIN_KINDS.to_vec();
        kinds.extend(self.extra_kinds.iter().map(|s| s.as_str()));
        kinds
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub threshold: f64,
    pub top_n: usize,
    pub bm25: Bm25Config,
    pub weights: BlendWeights,
    pub graph: GraphConfig,
    pub engagement: EngagementConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Bm25Config {
    pub top_k: usize,
    pub weight_title: f64,
    pub weight_body: f64,
    pub weight_spine: f64,
    pub weight_aliases: f64,
    pub weight_keywords: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BlendWeights {
    pub cosine: f64,
    pub graph: f64,
    #[serde(alias = "activation")]
    pub engagement: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EngagementConfig {
    pub half_life_hours: f64,
    pub saturation_reads: f64,
    pub weight_recency: f64,
    pub weight_popularity: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GraphConfig {
    pub decay: f64,
    pub max_hops: usize,
    pub respect_domain_filter: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            threshold: 0.10,
            top_n: 20,
            bm25: Bm25Config::default(),
            weights: BlendWeights::default(),
            graph: GraphConfig::default(),
            engagement: EngagementConfig::default(),
        }
    }
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            top_k: 100,
            weight_title: 5.0,
            weight_body: 1.0,
            weight_spine: 2.0,
            weight_aliases: 3.0,
            weight_keywords: 10.0,
        }
    }
}

impl Default for BlendWeights {
    fn default() -> Self {
        Self {
            cosine: 0.50,
            graph: 0.25,
            engagement: 0.25,
        }
    }
}

impl Default for EngagementConfig {
    fn default() -> Self {
        Self {
            half_life_hours: 168.0,
            saturation_reads: 20.0,
            weight_recency: 0.60,
            weight_popularity: 0.40,
        }
    }
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            decay: 0.5,
            max_hops: 1,
            respect_domain_filter: false,
        }
    }
}

impl Bm25Config {
    /// Format BM25 column weights as the argument string for FTS5 bm25() function.
    /// Column order: note_id (0, unindexed), title, body, spine, aliases, keywords.
    pub fn fts5_weights_arg(&self) -> String {
        format!(
            "0.0, {}, {}, {}, {}, {}",
            self.weight_title, self.weight_body, self.weight_spine,
            self.weight_aliases, self.weight_keywords
        )
    }
}

pub fn load(vault_dir: &Path) -> Result<Config> {
    let path = vault_dir.join("config.toml");
    if path.exists() {
        let content = std::fs::read_to_string(&path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    } else {
        Ok(Config::default())
    }
}
