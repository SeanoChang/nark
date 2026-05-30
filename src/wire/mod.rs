use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Notify,
    Request,
    Reply,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    #[default]
    Normal,
    High,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Body {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Envelope {
    pub id: String,
    pub from: String,
    pub to: String,
    pub kind: Kind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_cap: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ts: Option<String>,
    #[serde(default)]
    pub priority: Priority,
    pub body: Body,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_round_trip() {
        let env = Envelope {
            id: "msg-1".to_string(),
            from: "alice".to_string(),
            to: "neo".to_string(),
            kind: Kind::Notify,
            intent: Some("handoff".to_string()),
            correlation_id: None,
            reply_cap: None,
            ttl: Some("72h".to_string()),
            ts: None,
            priority: Priority::Normal,
            body: Body {
                subject: Some("hi".to_string()),
                text: Some("ping".to_string()),
                attachments: Vec::new(),
            },
        };

        let json = serde_json::to_string(&env).expect("serialize");
        let got: Envelope = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(got.id, "msg-1");
        assert!(matches!(got.kind, Kind::Notify));
    }
}
