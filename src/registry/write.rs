use rusqlite::Connection;
use anyhow::Result;

use crate::db::DEFAULT_AGENT_ID;
use crate::types::markdown::Frontmatter;
use crate::types::note::IngestResult;

const NAMESPACE: &str = "ark";

pub fn commit_version(conn: &Connection, r: &IngestResult) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    let fm = &r.frontmatter;
    let tx = conn.unchecked_transaction()?;

    upsert_note(&tx, r, &now)?;
    insert_version(&tx, r, &now)?;
    upsert_current_note(&tx, r, fm, &now)?;
    replace_note_text(&tx, r, fm)?;
    sync_tags(&tx, &r.note_id, &fm.tags)?;
    super::edges::sync_edges(&tx, &r.note_id, &r.version_id, &r.body, &fm.links, &now)?;

    tx.commit()?;
    Ok(())
}

fn insert_version(tx: &Connection, r: &IngestResult, now: &str) -> Result<()> {
    tx.execute(
        "INSERT INTO note_versions
            (version_id, note_id, prev_version_id, author_agent_id,
             content_hash, fm_hash, md_hash, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            r.version_id, r.note_id, r.prev_version_id, DEFAULT_AGENT_ID,
            r.content_hash, r.fm_hash, r.md_hash, now,
        ],
    )?;
    Ok(())
}

fn upsert_note(tx: &Connection, r: &IngestResult, now: &str) -> Result<()> {
    tx.execute(
        "INSERT INTO notes (note_id, namespace, head_version_id, author_agent_id, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(note_id) DO UPDATE SET head_version_id = excluded.head_version_id",
        rusqlite::params![r.note_id, NAMESPACE, r.version_id, DEFAULT_AGENT_ID, now],
    )?;
    Ok(())
}

fn upsert_current_note(tx: &Connection, r: &IngestResult, fm: &Frontmatter, now: &str) -> Result<()> {
    tx.execute(
        "INSERT INTO current_notes
            (note_id, namespace, head_version_id, author_agent_id,
             title, domain, intent, kind, trust, status, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
         ON CONFLICT(note_id) DO UPDATE SET
             head_version_id = excluded.head_version_id,
             title = excluded.title,
             domain = excluded.domain,
             intent = excluded.intent,
             kind = excluded.kind,
             trust = excluded.trust,
             status = excluded.status,
             updated_at = excluded.updated_at",
        rusqlite::params![
            r.note_id, NAMESPACE, r.version_id, DEFAULT_AGENT_ID,
            fm.title, fm.domain.to_string(), fm.intent.to_string(), fm.kind.to_string(),
            fm.trust.to_string(), fm.status.to_string(), now,
        ],
    )?;
    Ok(())
}

fn replace_note_text(tx: &Connection, r: &IngestResult, fm: &Frontmatter) -> Result<()> {
    let spine = format!("{}/{}/{}", fm.domain.to_string(), fm.intent.to_string(), fm.kind.to_string());
    let keywords = fm.tags.join(" ");
    let aliases = fm.aliases.join(" ");

    tx.execute("DELETE FROM note_text WHERE note_id = ?1", [&r.note_id])?;
    tx.execute(
        "INSERT INTO note_text
            (note_id, title, body, spine, aliases, keywords)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![r.note_id, fm.title, r.body, spine, aliases, keywords],
    )?;
    Ok(())
}

fn sync_tags(tx: &Connection, note_id: &str, tags: &[String]) -> Result<()> {
    tx.execute("DELETE FROM note_tags WHERE note_id = ?1", [note_id])?;

    for tag in tags {
        tx.execute("INSERT OR IGNORE INTO tags (name) VALUES (?1)", [tag])?;
        tx.execute(
            "INSERT INTO note_tags (note_id, tag_id)
             SELECT ?1, tag_id FROM tags WHERE name = ?2",
            rusqlite::params![note_id, tag],
        )?;
    }
    Ok(())
}
