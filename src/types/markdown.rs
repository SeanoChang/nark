use std::fmt;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Systems,
    Security,
    Finance,
    AiMl,
    Data,
    Programming,
    Math,
    Writing,
    Product,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Intent {
    Build,
    Debug,
    Operate,
    Design,
    Research,
    Evaluate,
    Decide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Spec,
    Decision,
    Runbook,
    Report,
    Reference,
    Incident,
    Experiment,
    Dataset,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Trust {
    Hypothesis,
    Reviewed,
    Verified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Active,
    Deprecated,
    Retracted,
    Draft,
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Domain::Systems => write!(f, "systems"),
            Domain::Security => write!(f, "security"),
            Domain::Finance => write!(f, "finance"),
            Domain::AiMl => write!(f, "ai_ml"),
            Domain::Data => write!(f, "data"),
            Domain::Programming => write!(f, "programming"),
            Domain::Math => write!(f, "math"),
            Domain::Writing => write!(f, "writing"),
            Domain::Product => write!(f, "product"),
        }
    }
}

impl fmt::Display for Intent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Intent::Build => write!(f, "build"),
            Intent::Debug => write!(f, "debug"),
            Intent::Operate => write!(f, "operate"),
            Intent::Design => write!(f, "design"),
            Intent::Research => write!(f, "research"),
            Intent::Evaluate => write!(f, "evaluate"),
            Intent::Decide => write!(f, "decide"),
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Spec => write!(f, "spec"),
            Kind::Decision => write!(f, "decision"),
            Kind::Runbook => write!(f, "runbook"),
            Kind::Report => write!(f, "report"),
            Kind::Reference => write!(f, "reference"),
            Kind::Incident => write!(f, "incident"),
            Kind::Experiment => write!(f, "experiment"),
            Kind::Dataset => write!(f, "dataset"),
        }
    }
}

impl fmt::Display for Trust {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Trust::Hypothesis => write!(f, "hypothesis"),
            Trust::Reviewed => write!(f, "reviewed"),
            Trust::Verified => write!(f, "verified"),
        }
    }
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
pub struct Frontmatter {
    pub title: String,
    pub author: String,
    pub domain: Domain,
    pub intent: Intent,
    pub kind: Kind,
    pub trust: Trust,
    pub status: Status,
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}
