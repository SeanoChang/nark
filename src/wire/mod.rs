use serde::{Deserialize, Serialize};

mod rpc;

// Full JSON-RPC surface. The `nark serve` bin uses only `RPCRequest`/
// `RPCResponse` directly (the success/error variants carry `ResultResponse`/
// `ErrorResponse`/`RPCError`), but these are the module's public types and are
// part of the library surface, so re-export them all rather than narrowing the
// API to what one consumer needs today.
#[allow(unused_imports)]
pub use rpc::{ErrorResponse, RPCError, RPCRequest, RPCResponse, ResultResponse};

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

impl Priority {
    fn is_normal(&self) -> bool {
        matches!(self, Priority::Normal)
    }
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
    #[serde(default, skip_serializing_if = "Priority::is_normal")]
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

    fn minimal_envelope() -> Envelope {
        Envelope {
            id: "msg-2".to_string(),
            from: "alice".to_string(),
            to: "neo".to_string(),
            kind: Kind::Notify,
            intent: None,
            correlation_id: None,
            reply_cap: None,
            ttl: None,
            ts: None,
            priority: Priority::Normal,
            body: Body {
                subject: None,
                text: None,
                attachments: Vec::new(),
            },
        }
    }

    // (a) minimal-envelope round-trip with intent/ts/correlation_id = None
    // proving skip_serializing_if keeps optional fields off the wire.
    #[test]
    fn minimal_envelope_round_trip() {
        let env = minimal_envelope();

        let json = serde_json::to_string(&env).expect("serialize");
        assert!(!json.contains("intent"), "json: {json}");
        assert!(!json.contains("ts"), "json: {json}");
        assert!(!json.contains("correlation_id"), "json: {json}");

        let got: Envelope = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(got.id, "msg-2");
        assert!(got.intent.is_none());
        assert!(got.ts.is_none());
        assert!(got.correlation_id.is_none());
    }

    // (b) golden: a default-priority Envelope's serialized JSON does NOT
    // contain "priority" (matches Go's omitempty so Phase 1b fixtures match).
    #[test]
    fn default_priority_omitted_from_wire() {
        let env = minimal_envelope();
        let json = serde_json::to_string(&env).expect("serialize");
        assert!(!json.contains("priority"), "json: {json}");
    }

    // (c) deserialize a Go-style JSON literal with NO priority key and
    // assert it defaults to Normal.
    #[test]
    fn missing_priority_defaults_to_normal() {
        let json = r#"{"id":"msg-3","from":"alice","to":"neo","kind":"notify","body":{}}"#;
        let got: Envelope = serde_json::from_str(json).expect("deserialize");
        assert!(matches!(got.priority, Priority::Normal));
    }

    // High priority still serializes (sanity that skip only drops Normal).
    #[test]
    fn high_priority_present_on_wire() {
        let mut env = minimal_envelope();
        env.priority = Priority::High;
        let json = serde_json::to_string(&env).expect("serialize");
        assert!(json.contains("\"priority\":\"high\""), "json: {json}");
    }
}
