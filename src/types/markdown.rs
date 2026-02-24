use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frontmatter {
    pub title: String,
    pub author: String,
    pub domain: String,
    pub intent: String,
    pub kind: String,
    pub trust: String,
    pub status: String,
    pub tags: Vec<String>,
}
