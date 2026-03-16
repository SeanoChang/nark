use anyhow::{bail, Result};
use std::path::Path;

use crate::cli::search::parse_temporal;
use crate::db;
use crate::registry::{bulk::BulkFilter, resolve, tags};

pub struct BulkTagOpts {
    pub domain: Option<String>,
    pub kind: Option<String>,
    pub filter_tag: Vec<String>,
    pub since: Option<String>,
    pub before: Option<String>,
    pub confirm: bool,
}

pub fn run(
    vault_dir: &Path,
    args: Vec<String>,
    list: bool,
    find: Vec<String>,
    bulk: BulkTagOpts,
) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;

    // Mode 1: --list
    if list {
        let counts = tags::list_tags(&conn)?;
        let out: Vec<serde_json::Value> = counts.iter().map(|t| {
            serde_json::json!({ "tag": t.name, "count": t.count })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    // Mode 2: --find
    if !find.is_empty() {
        let notes = tags::find_by_tags(&conn, &find)?;
        let out: Vec<serde_json::Value> = notes.iter().map(|n| {
            serde_json::json!({
                "id": n.note_id,
                "title": n.title,
                "domain": n.domain,
                "kind": n.kind,
                "tags": n.tags,
            })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    // Parse args into note IDs and +/-tag modifiers
    let (note_ids, add, remove) = parse_args(&args)?;

    // Mode 5: bulk tag by filter
    let filter = BulkFilter {
        domain: bulk.domain,
        kind: bulk.kind,
        tags: bulk.filter_tag,
        since: bulk.since.map(|s| parse_temporal(&s)).transpose()?,
        before: bulk.before.map(|b| parse_temporal(&b)).transpose()?,
    };

    if filter.has_any() {
        if add.is_empty() && remove.is_empty() {
            bail!("bulk tag mode requires at least one +tag or -tag modifier");
        }
        if !note_ids.is_empty() {
            bail!("cannot combine note IDs with filter flags for bulk operations");
        }

        let matched = crate::registry::bulk::find_matching_notes(&conn, &filter)?;

        if !bulk.confirm {
            let notes: Vec<serde_json::Value> = matched.iter().map(|(id, title)| {
                serde_json::json!({ "id": id, "title": title })
            }).collect();
            let out = serde_json::json!({
                "mode": "dry_run",
                "matched": matched.len(),
                "notes": notes,
                "would_add": add,
                "would_remove": remove,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
            return Ok(());
        }

        let ids: Vec<String> = matched.iter().map(|(id, _)| id.clone()).collect();
        if ids.is_empty() {
            let out = serde_json::json!({ "tagged": 0, "notes": [], "added": add, "removed": remove });
            println!("{}", serde_json::to_string_pretty(&out)?);
            return Ok(());
        }
        tags::mutate_tags(&conn, &ids, &add, &remove)?;

        let out = serde_json::json!({
            "tagged": ids.len(),
            "notes": ids,
            "added": add,
            "removed": remove,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    if note_ids.is_empty() {
        bail!("no note IDs provided");
    }

    // Validate all IDs exist and resolve prefixes
    let note_ids: Vec<String> = note_ids.iter().map(|id| {
        let meta = resolve::get_meta(&conn, id)
            .map_err(|_| anyhow::anyhow!("note not found: {}", id))?;
        Ok(meta.note_id)
    }).collect::<Result<Vec<String>>>()?;

    // Mode 3: read-only (no modifiers)
    if add.is_empty() && remove.is_empty() {
        if note_ids.len() == 1 {
            let t = tags::get_tags(&conn, &note_ids[0])?;
            println!("{}", serde_json::to_string_pretty(&t)?);
        } else {
            let out: Vec<serde_json::Value> = note_ids.iter().map(|id| {
                let t = tags::get_tags(&conn, id).unwrap_or_default();
                serde_json::json!({ "id": id, "tags": t })
            }).collect();
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        return Ok(());
    }

    // Mode 4: mutate
    tags::mutate_tags(&conn, &note_ids, &add, &remove)?;

    let out = serde_json::json!({
        "tagged": note_ids.len(),
        "notes": note_ids,
        "added": add,
        "removed": remove,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn parse_args(args: &[String]) -> Result<(Vec<String>, Vec<String>, Vec<String>)> {
    let mut ids = Vec::new();
    let mut add = Vec::new();
    let mut remove = Vec::new();

    for arg in args {
        if let Some(tag) = arg.strip_prefix('+') {
            let tag = validate_tag(tag)?;
            add.push(tag);
        } else if let Some(tag) = arg.strip_prefix('-') {
            let tag = validate_tag(tag)?;
            remove.push(tag);
        } else {
            ids.push(arg.clone());
        }
    }

    Ok((ids, add, remove))
}

fn validate_tag(tag: &str) -> Result<String> {
    let tag = tag.to_lowercase();
    if tag.is_empty() {
        bail!("tag name cannot be empty");
    }
    if !tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        bail!("invalid tag '{}': only lowercase alphanumeric and hyphens allowed", tag);
    }
    Ok(tag)
}
