use anyhow::{bail, Result};
use rusqlite::Connection;

/// Resolve a full or prefix note ID to the full 36-char UUID.
/// - Full UUID (36 chars with hyphens) → returned as-is (no DB query).
/// - Short prefix → matched against `current_notes.note_id`.
///   Exactly 1 match → full ID. 0 → error. 2+ → ambiguous error listing matches.
pub fn resolve_id(conn: &Connection, prefix: &str) -> Result<String> {
    if prefix.is_empty() {
        bail!("note ID prefix cannot be empty");
    }

    if !prefix.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
        bail!("invalid note ID prefix '{prefix}': must contain only hex characters and hyphens");
    }

    // Fast path: already a full UUID
    if prefix.len() == 36 && prefix.chars().filter(|c| *c == '-').count() == 4 {
        return Ok(prefix.to_string());
    }

    let mut stmt = conn.prepare(
        "SELECT note_id FROM current_notes WHERE note_id LIKE ?1 || '%'"
    )?;

    let matches: Vec<String> = stmt
        .query_map([prefix], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    match matches.len() {
        0 => bail!("note not found: {prefix}"),
        1 => Ok(matches.into_iter().next().unwrap()),
        n => {
            let ids = matches.join(", ");
            bail!("ambiguous prefix '{prefix}' matches {n} notes: {ids}");
        }
    }
}

pub struct NoteMeta {
    pub note_id: String,
    pub title: String,
    pub domain: String,
    pub intent: String,
    pub kind: String,
    pub status: String,
    pub tags: Vec<String>,
    pub updated_at: String,
}

pub struct NoteRef {
    pub fm_hash: String,
    pub md_hash: String,
}

pub fn get_meta(conn: &Connection, note_id: &str) -> Result<NoteMeta> {
    let note_id = resolve_id(conn, note_id)?;
    let mut stmt = conn.prepare(
        "SELECT note_id, title, domain, intent, kind, status, updated_at
         FROM current_notes WHERE note_id = ?1"
    )?;

    let meta = stmt.query_row([&note_id], |row| {
        Ok(NoteMeta {
            note_id: row.get(0)?,
            title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            intent: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            kind: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            status: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
            updated_at: row.get::<_, Option<String>>(6)?.unwrap_or_default(),
            tags: Vec::new(),
        })
    })?;

    let mut tag_stmt = conn.prepare(
        "SELECT t.name FROM tags t
         JOIN note_tags nt ON t.tag_id = nt.tag_id
         WHERE nt.note_id = ?1
         ORDER BY t.name"
    )?;

    let tags: Vec<String> = tag_stmt
        .query_map([&note_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(NoteMeta { tags, ..meta })
}

pub fn get_ref(conn: &Connection, note_id: &str) -> Result<NoteRef> {
    let note_id = resolve_id(conn, note_id)?;
    let mut stmt = conn.prepare(
        "SELECT nv.fm_hash, nv.md_hash
         FROM notes n
         JOIN note_versions nv ON n.head_version_id = nv.version_id
         WHERE n.note_id = ?1"
    )?;

    let r = stmt.query_row([&note_id], |row| {
        Ok(NoteRef {
            fm_hash: row.get(0)?,
            md_hash: row.get(1)?,
        })
    })?;

    Ok(r)
}
