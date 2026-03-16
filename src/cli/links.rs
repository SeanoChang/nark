use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::{edges, resolve};

pub fn run(vault_dir: &Path, id: &str) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;

    let meta = resolve::get_meta(&conn, id)
        .map_err(|_| anyhow::anyhow!("note not found: {}", id))?;

    let (outgoing, incoming) = edges::get_edges(&conn, &meta.note_id)?;

    let out_json: Vec<serde_json::Value> = outgoing.iter().map(|e| {
        let mut v = serde_json::json!({
            "note_id": e.note_id,
            "title": e.title,
            "edge_type": e.edge_type,
            "weight": e.weight,
            "source_type": e.source_type,
        });
        if let Some(ctx) = &e.context {
            v["context"] = serde_json::json!(ctx);
        }
        v
    }).collect();

    let in_json: Vec<serde_json::Value> = incoming.iter().map(|e| {
        let mut v = serde_json::json!({
            "note_id": e.note_id,
            "title": e.title,
            "edge_type": e.edge_type,
            "weight": e.weight,
            "source_type": e.source_type,
        });
        if let Some(ctx) = &e.context {
            v["context"] = serde_json::json!(ctx);
        }
        v
    }).collect();

    let out = serde_json::json!({
        "id": meta.note_id,
        "title": meta.title,
        "outgoing": out_json,
        "incoming": in_json,
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
