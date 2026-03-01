use anyhow::{bail, Result};
use std::path::Path;

use crate::db;
use crate::registry::{resolve, write::commit_version};
use crate::types::markdown::{Frontmatter, FrontmatterLink};
use crate::vault::fs::Vault;

/// Convert a rel type like "depends-on" to a section heading like "## Depends On"
fn rel_to_heading(rel: &str) -> String {
    let words: Vec<String> = rel
        .split('-')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(first) => {
                    let upper: String = first.to_uppercase().collect();
                    format!("{}{}", upper, c.as_str())
                }
            }
        })
        .collect();
    format!("## {}", words.join(" "))
}

/// Insert a wikilink into the correct rel section of the body.
///
/// Cases:
/// 1. Section exists → append `- [[target]]` after the last `- [[...]]` line in that section
/// 2. Section doesn't exist → append the section + link at the end of the body
/// 3. Link already present in body anywhere → return body unchanged
fn insert_body_link(body: &str, target: &str, rel: &str) -> (String, bool) {
    let wikilink_entry = format!("- [[{}]]", target);
    let wikilink_bare = format!("[[{}]]", target);

    // Already present anywhere in body — skip
    if body.contains(&wikilink_bare) {
        return (body.to_string(), false);
    }

    let heading = rel_to_heading(rel);
    let lines: Vec<&str> = body.lines().collect();

    // Find the section for this rel type
    if let Some(section_idx) = lines.iter().position(|l| l.trim() == heading) {
        // Find the last `- [[...]]` line within this section (before next ## or end)
        let mut insert_after = section_idx;
        for i in (section_idx + 1)..lines.len() {
            let trimmed = lines[i].trim();
            if trimmed.starts_with("## ") {
                break;
            }
            if trimmed.starts_with("- [[") {
                insert_after = i;
            }
        }

        let mut result: Vec<&str> = Vec::with_capacity(lines.len() + 1);
        result.extend_from_slice(&lines[..=insert_after]);
        // We need to push an owned string, so build the full output
        let mut out = result.join("\n");
        out.push('\n');
        out.push_str(&wikilink_entry);
        if insert_after + 1 < lines.len() {
            out.push('\n');
            out.push_str(&lines[insert_after + 1..].join("\n"));
        }
        return (out, true);
    }

    // Section doesn't exist — append at end
    let trimmed_body = body.trim_end();
    let new_body = format!("{}\n\n{}\n{}", trimmed_body, heading, wikilink_entry);
    (new_body, true)
}

pub fn run(vault_dir: &Path, sources: Vec<String>, target: &str, rel: &str) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());

    // Validate target exists
    let target_meta = resolve::get_meta(&conn, target)
        .map_err(|_| anyhow::anyhow!("target note not found: {}", target))?;

    let mut results: Vec<serde_json::Value> = Vec::new();

    for source in &sources {
        // Validate source exists
        let source_meta = resolve::get_meta(&conn, source)
            .map_err(|_| anyhow::anyhow!("source note not found: {}", source))?;

        // Reject self-links
        if source_meta.note_id == target_meta.note_id {
            bail!("cannot link a note to itself: {}", source);
        }

        // Read current source note content
        let refs = resolve::get_ref(&conn, source)?;
        let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
        let body = vault.read_object("objects/md", &refs.md_hash, "md")?;

        let mut fm: Frontmatter = serde_yaml::from_str(&fm_raw)?;

        // Check idempotency — both frontmatter and body already have the link
        let has_fm_link = fm.links.iter().any(|l| l.target == target && l.rel == rel);
        let wikilink_bare = format!("[[{}]]", target);
        let has_body_link = body.contains(&wikilink_bare);

        if has_fm_link && has_body_link {
            results.push(serde_json::json!({
                "source": source,
                "source_title": source_meta.title,
                "status": "no-op",
            }));
            continue;
        }

        // Mutate frontmatter
        if !has_fm_link {
            fm.links.push(FrontmatterLink {
                target: target.to_string(),
                rel: rel.to_string(),
            });
        }

        // Mutate body — insert into the correct rel section
        let (new_body, _) = if has_body_link {
            (body, false)
        } else {
            insert_body_link(&body, target, rel)
        };

        // Reassemble and re-ingest
        let full_note = format!("---\n{}---\n{}", serde_yaml::to_string(&fm)?, new_body);
        let result = vault.ingest(&full_note, Some(source))?;
        commit_version(&conn, &result)?;

        results.push(serde_json::json!({
            "source": source,
            "source_title": source_meta.title,
            "status": "linked",
        }));
    }

    let out = serde_json::json!({
        "target": target,
        "target_title": target_meta.title,
        "rel": rel,
        "linked": results.iter().filter(|r| r["status"] == "linked").count(),
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
