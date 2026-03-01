use anyhow::Result;
use regex::Regex;
use rusqlite::Connection;
use std::collections::HashSet;
use std::sync::LazyLock;

use crate::types::markdown::FrontmatterLink;

static WIKILINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\[([^\]]+)\]\]").unwrap());

struct ParsedEdge {
    target: String,
    edge_type: String,
    source_type: String,
    context: Option<String>,
}

fn edge_weight(edge_type: &str) -> f64 {
    match edge_type {
        "references" => 1.0,
        "depends-on" => 2.0,
        "supersedes" => 1.5,
        "contradicts" => 1.5,
        "extends" => 1.2,
        "informed-by" => 0.8,
        _ => 1.0,
    }
}

fn extract_wikilinks(body: &str) -> Vec<ParsedEdge> {
    let mut seen = HashSet::new();
    let mut edges = Vec::new();

    for line in body.lines() {
        for cap in WIKILINK_RE.captures_iter(line) {
            let target = cap[1].to_string();
            if seen.insert(target.clone()) {
                edges.push(ParsedEdge {
                    target,
                    edge_type: "references".to_string(),
                    source_type: "body".to_string(),
                    context: Some(line.trim().to_string()),
                });
            }
        }
    }

    edges
}

fn map_frontmatter_links(links: &[FrontmatterLink]) -> Vec<ParsedEdge> {
    links
        .iter()
        .map(|link| ParsedEdge {
            target: link.target.clone(),
            edge_type: link.rel.clone(),
            source_type: "frontmatter".to_string(),
            context: None,
        })
        .collect()
}

/// Resolve a target string to a note_id.
/// Tries note_id match first, then title match. Skips retracted notes.
fn resolve_target(tx: &Connection, target: &str) -> Result<Option<String>> {
    // Try by note_id
    let mut stmt = tx.prepare_cached(
        "SELECT note_id FROM current_notes
         WHERE note_id = ?1 AND status != 'retracted'",
    )?;
    if let Some(id) = stmt
        .query_map([target], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .next()
    {
        return Ok(Some(id));
    }

    // Try by title (case-insensitive)
    let mut stmt = tx.prepare_cached(
        "SELECT note_id FROM current_notes
         WHERE title = ?1 COLLATE NOCASE AND status != 'retracted'
         LIMIT 1",
    )?;
    if let Some(id) = stmt
        .query_map([target], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .next()
    {
        return Ok(Some(id));
    }

    Ok(None)
}

fn update_link_count(tx: &Connection, note_id: &str) -> Result<()> {
    tx.execute(
        "UPDATE current_notes SET links_out_count = (
            SELECT COUNT(*) FROM note_edges WHERE src_note_id = ?1
        ) WHERE note_id = ?1",
        [note_id],
    )?;
    tx.execute(
        "UPDATE current_notes SET links_in_count = (
            SELECT COUNT(*) FROM note_edges WHERE dst_note_id = ?1
        ) WHERE note_id = ?1",
        [note_id],
    )?;
    Ok(())
}

/// Synchronize edges for a note within an existing transaction.
pub fn sync_edges(
    tx: &Connection,
    note_id: &str,
    version_id: &str,
    body: &str,
    links: &[FrontmatterLink],
    now: &str,
) -> Result<()> {
    // 1. Collect old destination IDs for link count fixup
    let old_dsts: Vec<String> = {
        let mut stmt = tx.prepare_cached(
            "SELECT DISTINCT dst_note_id FROM note_edges WHERE src_note_id = ?1",
        )?;
        stmt.query_map([note_id], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect()
    };

    // 2. Delete existing edges from this source
    tx.execute(
        "DELETE FROM note_edges WHERE src_note_id = ?1",
        [note_id],
    )?;

    // 3. Extract edges from body + frontmatter
    let mut parsed = extract_wikilinks(body);
    parsed.extend(map_frontmatter_links(links));

    // 4. Resolve targets, skip self-links and dead links, insert
    let mut new_dsts: HashSet<String> = HashSet::new();
    for edge in &parsed {
        let dst_id = match resolve_target(tx, &edge.target)? {
            Some(id) => id,
            None => continue, // dead link — silently skip
        };

        if dst_id == note_id {
            continue; // skip self-links
        }

        new_dsts.insert(dst_id.clone());

        tx.execute(
            "INSERT OR IGNORE INTO note_edges
                (src_note_id, dst_note_id, edge_type, weight, source_type, context, version_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                note_id,
                dst_id,
                edge.edge_type,
                edge_weight(&edge.edge_type),
                edge.source_type,
                edge.context,
                version_id,
                now,
            ],
        )?;
    }

    // 5. Update links_out_count on source
    update_link_count(tx, note_id)?;

    // 6. Update links_in_count on union of old + new destinations
    let all_dsts: HashSet<&str> = old_dsts
        .iter()
        .map(|s| s.as_str())
        .chain(new_dsts.iter().map(|s| s.as_str()))
        .collect();

    for dst in all_dsts {
        update_link_count(tx, dst)?;
    }

    Ok(())
}
