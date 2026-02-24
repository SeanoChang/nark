use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::search;

pub fn run(vault_dir: &Path, query: &str, domain: Option<&str>, limit: usize) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let hits = search::search(&conn, query, domain, limit)?;

    let results: Vec<serde_json::Value> = hits.iter().map(|h| {
        serde_json::json!({
            "id": h.note_id,
            "title": h.title,
            "domain": h.domain,
            "kind": h.kind,
            "snippet": h.snippet,
            "rank": h.rank,
        })
    }).collect();

    let out = serde_json::json!({
        "query": query,
        "domain": domain,
        "hits": results.len(),
        "results": results,
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
