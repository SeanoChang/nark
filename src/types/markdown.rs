use std::fmt;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Active,
    Deprecated,
    Retracted,
    Draft,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Status::Active => write!(f, "active"),
            Status::Deprecated => write!(f, "deprecated"),
            Status::Retracted => write!(f, "retracted"),
            Status::Draft => write!(f, "draft"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontmatterLink {
    pub target: String,
    pub rel: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frontmatter {
    pub title: String,
    pub author: String,
    pub domain: String,
    pub intent: String,
    pub kind: String,
    pub status: Status,
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<FrontmatterLink>,
}
