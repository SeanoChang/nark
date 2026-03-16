use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::resolve;

pub fn run(vault_dir: &Path, id: &str) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let meta = resolve::get_meta(&conn, id)?;

    let out = serde_json::json!({
        "id": meta.note_id,
        "title": meta.title,
        "domain": meta.domain,
        "intent": meta.intent,
        "kind": meta.kind,
        "status": meta.status,
        "tags": meta.tags,
        "updated_at": meta.updated_at,
        "links_in": meta.links_in,
        "links_out": meta.links_out,
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
