//! JSON-RPC 2.0 request/response wire types for the Ark comm protocol.
//!
//! `nark serve` speaks a one-request-per-connection JSON-RPC dialect: the
//! client writes a single [`RPCRequest`] as one JSON line, the daemon dispatches
//! it to a method, and writes back exactly one [`RPCResponse`] line.
//!
//! These types intentionally model only what the protocol uses today:
//!
//! * [`RPCRequest`] — `id` + `method` + optional `params`.
//! * [`RPCResponse`] — an untagged either/or of a success [`ResultResponse`] or
//!   an [`ErrorResponse`]. A response carries a `result` XOR an `error`, never
//!   both; the untagged enum serializes to exactly one shape so the wire stays
//!   compatible with a standard JSON-RPC peer.
//!
//! The constructors ([`RPCResponse::result`] / [`RPCResponse::error`]) keep the
//! `id` echo and the error `code`/`message`/`data` shape in one place so the
//! method router does not hand-roll JSON.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single JSON-RPC request: `{ "id": ..., "method": ..., "params"?: ... }`.
///
/// `id` echoes back on the response so a client can correlate. `params` is
/// optional and method-defined.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RPCRequest {
    pub id: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// A successful JSON-RPC result: `{ "id": ..., "result": ... }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultResponse {
    pub id: String,
    pub result: Value,
}

/// The error payload of a JSON-RPC error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RPCError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// A JSON-RPC error response: `{ "id": ..., "error": { code, message, data? } }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub id: String,
    pub error: RPCError,
}

/// A JSON-RPC response: either a [`ResultResponse`] or an [`ErrorResponse`].
///
/// `#[serde(untagged)]` means the response serializes to exactly one of the two
/// shapes (a `result` key XOR an `error` key) with no discriminant wrapper, so
/// it is wire-compatible with a standard JSON-RPC peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RPCResponse {
    Result(ResultResponse),
    Error(ErrorResponse),
}

impl RPCResponse {
    /// Build a success response echoing `id` with `result`.
    pub fn result(id: impl Into<String>, result: Value) -> Self {
        RPCResponse::Result(ResultResponse {
            id: id.into(),
            result,
        })
    }

    /// Build an error response echoing `id` with the given `code`/`message`/`data`.
    pub fn error(
        id: impl Into<String>,
        code: i64,
        message: impl Into<String>,
        data: Option<Value>,
    ) -> Self {
        RPCResponse::Error(ErrorResponse {
            id: id.into(),
            error: RPCError {
                code,
                message: message.into(),
                data,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_parses_with_params() {
        let line = r#"{"id":"1","method":"resolve","params":{"id":"abc"}}"#;
        let req: RPCRequest = serde_json::from_str(line).expect("parse request");
        assert_eq!(req.id, "1");
        assert_eq!(req.method, "resolve");
        assert_eq!(req.params, Some(json!({"id":"abc"})));
    }

    #[test]
    fn request_parses_without_params() {
        let line = r#"{"id":"7","method":"ping"}"#;
        let req: RPCRequest = serde_json::from_str(line).expect("parse request");
        assert_eq!(req.id, "7");
        assert_eq!(req.method, "ping");
        assert!(req.params.is_none());
    }

    #[test]
    fn result_response_shape() {
        let resp = RPCResponse::result("9", json!({"pong": true}));
        let v: Value = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(v, json!({"id": "9", "result": {"pong": true}}));
        // No `error` key on a success response.
        assert!(v.get("error").is_none());
    }

    #[test]
    fn error_response_shape() {
        let resp = RPCResponse::error("9", -32601, "method not found", None);
        let v: Value = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(
            v,
            json!({"id": "9", "error": {"code": -32601, "message": "method not found"}})
        );
        // No `result` key on an error response, and `data` omitted when None.
        assert!(v.get("result").is_none());
        assert!(v["error"].get("data").is_none());
    }

    #[test]
    fn error_response_includes_data_when_present() {
        let resp = RPCResponse::error("3", -32000, "boom", Some(json!({"why": "x"})));
        let v: Value = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(v["error"]["data"], json!({"why": "x"}));
    }
}
