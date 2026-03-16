use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::browse::{self, BrowseResult};

pub fn run(vault_dir: &Path, path: Option<&str>, include_tags: bool) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let result = browse::browse(&conn, path, include_tags)?;

    let out = match result {
        BrowseResult::Groups { level, items } => {
            let results: Vec<serde_json::Value> = items.iter().map(|g| {
                serde_json::json!({ "name": g.name, "count": g.count })
            }).collect();
            serde_json::json!({
                "level": level,
                "path": path.unwrap_or(""),
                "results": results,
            })
        }
        BrowseResult::Notes(notes) => {
            let results: Vec<serde_json::Value> = notes.iter().map(|n| {
                let mut obj = serde_json::json!({
                    "id": n.note_id,
                    "title": n.title,
                    "updated_at": n.updated_at,
                });
                if let Some(tags) = &n.tags {
                    obj["tags"] = serde_json::json!(tags);
                }
                obj
            }).collect();
            serde_json::json!({
                "level": "note",
                "path": path.unwrap_or(""),
                "results": results,
            })
        }
    };

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
