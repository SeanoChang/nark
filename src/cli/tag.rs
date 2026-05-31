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
    // Write command: hold the advisory write lock for the whole invocation. A
    // conflicting RW open is refused with the plain write-locked error; the
    // handle derefs to the `Connection` so every mode below is unchanged, and
    // the lock is released when the handle drops at end of function.
    //
    // Note: `--list` / `--find` / read-only tag lookups are technically
    // non-mutating, but `tag` is dispatched as a single write command, so it is
    // guarded as a whole. The write CLIs are a defense-in-depth safety net (not
    // used directly on the deploy box), so gating these rarely-direct read
    // sub-modes too is acceptable and keeps the open path uniform.
    let conn = db::open_registry_guarded(vault_dir)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-tag-lock-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        dir
    }

    fn empty_bulk() -> BulkTagOpts {
        BulkTagOpts {
            domain: None,
            kind: None,
            filter_tag: Vec::new(),
            since: None,
            before: None,
            confirm: false,
        }
    }

    /// Spot-check: `tag` routes through the guarded open, so while the write
    /// lock is held it is refused with the plain write-locked error (before any
    /// mode dispatch).
    #[test]
    fn tag_refuses_when_write_locked() {
        let dir = fresh_vault();
        let held = db::open_registry_guarded(&dir).expect("hold the write lock");

        let err = run(
            &dir,
            vec!["someid".to_string(), "+topic".to_string()],
            false,
            Vec::new(),
            empty_bulk(),
        )
        .expect_err("tag must be refused while the write lock is held");
        assert_eq!(
            err.to_string(),
            "registry is write-locked by another process"
        );

        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
