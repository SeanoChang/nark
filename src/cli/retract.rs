use anyhow::{bail, Result};
use std::path::Path;

use crate::cli::search::parse_temporal;
use crate::db;
use crate::registry::{bulk::BulkFilter, delete};

pub fn run(
    vault_dir: &Path,
    ids: Vec<String>,
    domain: Option<String>,
    kind: Option<String>,
    tag: Vec<String>,
    since: Option<String>,
    before: Option<String>,
    confirm: bool,
) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;

    let filter = BulkFilter {
        domain,
        kind,
        tags: tag,
        since: since.map(|s| parse_temporal(&s)).transpose()?,
        before: before.map(|b| parse_temporal(&b)).transpose()?,
    };

    // Mode 1: ID-based retract (no --confirm needed)
    if !ids.is_empty() {
        if filter.has_any() {
            bail!("cannot combine note IDs with filter flags");
        }
        let notes = delete::validate_ids(&conn, &ids)?;
        delete::soft_delete(&conn, &notes)?;

        let out = serde_json::json!({
            "retracted": notes.len(),
            "notes": notes.iter().map(|n| {
                serde_json::json!({ "id": n.note_id, "title": n.title })
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    // Mode 2: filter-based bulk retract
    if !filter.has_any() {
        bail!("provide note IDs or at least one filter flag (--domain, --kind, --tag, --since, --before)");
    }

    let matched = crate::registry::bulk::find_matching_notes(&conn, &filter)?;

    if !confirm {
        let notes: Vec<serde_json::Value> = matched.iter().map(|(id, title)| {
            serde_json::json!({ "id": id, "title": title })
        }).collect();
        let out = serde_json::json!({
            "mode": "dry_run",
            "matched": matched.len(),
            "notes": notes,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    if matched.is_empty() {
        let out = serde_json::json!({ "retracted": 0, "notes": [] });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    // Build DeletedNote structs for soft_delete
    let ids: Vec<String> = matched.iter().map(|(id, _)| id.clone()).collect();
    let notes = delete::validate_ids(&conn, &ids)?;
    delete::soft_delete(&conn, &notes)?;

    let out = serde_json::json!({
        "retracted": notes.len(),
        "notes": notes.iter().map(|n| {
            serde_json::json!({ "id": n.note_id, "title": n.title })
        }).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
