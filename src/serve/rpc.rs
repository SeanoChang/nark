//! JSON-RPC method router for the `nark serve` daemon.
//!
//! Phase 3 replaces the Phase-2 line protocol with one JSON-RPC request per
//! connection: the client writes a single [`crate::wire::RPCRequest`] line, the
//! connection handler parses it and calls [`dispatch`], and writes back the one
//! [`crate::wire::RPCResponse`] this returns.
//!
//! This slice (3.2) lands the framing and the router skeleton with a single
//! method:
//!
//! * `ping` -> `{"pong": true}` (a success result),
//! * any other method -> JSON-RPC `-32601 method not found`.
//!
//! The READ methods (`resolve`, `stats`, `search`, ...) hang off [`dispatch`] in
//! later slices; they call the same `registry::*` functions the CLI handlers do.

use serde_json::json;

use crate::wire::{RPCRequest, RPCResponse};

/// JSON-RPC error code: the requested method is not implemented.
const METHOD_NOT_FOUND: i64 = -32601;

/// Dispatch a parsed [`RPCRequest`] to its method and return the response.
///
/// Unknown methods produce a `-32601 method not found` error echoing the
/// request id, so a client always gets exactly one response per request.
pub fn dispatch(req: &RPCRequest) -> RPCResponse {
    match req.method.as_str() {
        "ping" => RPCResponse::result(req.id.clone(), json!({"pong": true})),
        _ => RPCResponse::error(req.id.clone(), METHOD_NOT_FOUND, "method not found", None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::RPCResponse;

    fn request(id: &str, method: &str) -> RPCRequest {
        RPCRequest {
            id: id.to_string(),
            method: method.to_string(),
            params: None,
        }
    }

    #[test]
    fn ping_returns_pong_with_matching_id() {
        let resp = dispatch(&request("42", "ping"));
        match resp {
            RPCResponse::Result(r) => {
                assert_eq!(r.id, "42");
                assert_eq!(r.result, json!({"pong": true}));
            }
            RPCResponse::Error(e) => panic!("ping should succeed, got error {e:?}"),
        }
    }

    #[test]
    fn unknown_method_is_method_not_found() {
        let resp = dispatch(&request("7", "no-such-method"));
        match resp {
            RPCResponse::Error(e) => {
                assert_eq!(e.id, "7");
                assert_eq!(e.error.code, METHOD_NOT_FOUND);
                assert_eq!(e.error.message, "method not found");
                assert!(e.error.data.is_none());
            }
            RPCResponse::Result(r) => panic!("unknown method should error, got result {r:?}"),
        }
    }
}
