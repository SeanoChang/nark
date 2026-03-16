use anyhow::{bail, Result};
use similar::{ChangeTag, TextDiff};
use std::path::Path;

use crate::db;
use crate::registry::resolve;
use crate::vault::fs::Vault;

struct VersionRef {
    version_id: String,
    fm_hash: String,
    md_hash: String,
}

fn get_version_ref(conn: &rusqlite::Connection, note_id: &str, version_id: &str) -> Result<VersionRef> {
    let mut stmt = conn.prepare(
        "SELECT version_id, fm_hash, md_hash FROM note_versions
         WHERE note_id = ?1 AND version_id = ?2"
    )?;

    let r = stmt.query_row(rusqlite::params![note_id, version_id], |row| {
        Ok(VersionRef {
            version_id: row.get(0)?,
            fm_hash: row.get(1)?,
            md_hash: row.get(2)?,
        })
    }).map_err(|_| anyhow::anyhow!("version '{}' not found for note '{}'", version_id, note_id))?;

    Ok(r)
}

fn get_head_info(conn: &rusqlite::Connection, note_id: &str) -> Result<(String, Option<String>)> {
    let mut stmt = conn.prepare(
        "SELECT nv.version_id, nv.prev_version_id
         FROM notes n JOIN note_versions nv ON n.head_version_id = nv.version_id
         WHERE n.note_id = ?1"
    )?;

    let (head_id, prev_id) = stmt.query_row([note_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
        ))
    })?;

    Ok((head_id, prev_id))
}

fn read_version_content(vault: &Vault, vref: &VersionRef) -> Result<String> {
    let fm = vault.read_object("objects/fm", &vref.fm_hash, "yaml")?;
    let body = vault.read_object("objects/md", &vref.md_hash, "md")?;
    Ok(format!("---\n{}---\n{}", fm, body))
}

pub fn run(vault_dir: &Path, id: &str, from: Option<&str>, to: Option<&str>) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());

    // Validate note exists and resolve prefix
    let meta = resolve::get_meta(&conn, id)
        .map_err(|_| anyhow::anyhow!("note not found: {}", id))?;

    let (head_id, head_prev_id) = get_head_info(&conn, &meta.note_id)?;

    // Resolve "from" version
    let from_id = match from {
        Some(v) => v.to_string(),
        None => match head_prev_id {
            Some(ref prev) => prev.clone(),
            None => bail!("note has only one version — nothing to diff"),
        },
    };

    // Resolve "to" version
    let to_id = match to {
        Some(v) => v.to_string(),
        None => head_id,
    };

    let from_ref = get_version_ref(&conn, &meta.note_id, &from_id)?;
    let to_ref = get_version_ref(&conn, &meta.note_id, &to_id)?;

    let from_content = read_version_content(&vault, &from_ref)?;
    let to_content = read_version_content(&vault, &to_ref)?;

    // Generate unified diff
    let diff = TextDiff::from_lines(&from_content, &to_content);
    let mut diff_lines: Vec<serde_json::Value> = Vec::new();
    let mut additions = 0usize;
    let mut deletions = 0usize;

    for change in diff.iter_all_changes() {
        let tag = match change.tag() {
            ChangeTag::Delete => {
                deletions += 1;
                "-"
            }
            ChangeTag::Insert => {
                additions += 1;
                "+"
            }
            ChangeTag::Equal => " ",
        };
        diff_lines.push(serde_json::json!({
            "tag": tag,
            "line": change.value().trim_end_matches('\n'),
        }));
    }

    let out = serde_json::json!({
        "note_id": meta.note_id,
        "from_version": from_ref.version_id,
        "to_version": to_ref.version_id,
        "additions": additions,
        "deletions": deletions,
        "changes": additions + deletions,
        "diff": diff_lines,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
