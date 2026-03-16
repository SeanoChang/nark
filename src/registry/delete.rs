use anyhow::{bail, Result};
use rusqlite::Connection;

use crate::db::{DEFAULT_AGENT_ID, DEFAULT_CLIENT_ID};
use crate::registry::resolve;

pub struct DeletedNote {
    pub note_id: String,
    pub title: String,
    pub fm_hash: String,
    pub md_hash: String,
}

/// Validate all IDs exist, return metadata needed for deletion.
pub fn validate_ids(conn: &Connection, ids: &[String]) -> Result<Vec<DeletedNote>> {
    let mut notes = Vec::with_capacity(ids.len());
    let mut bad_ids = Vec::new();

    for id in ids {
        match resolve::get_meta(conn, id) {
            Ok(meta) => {
                let refs = resolve::get_ref(conn, &meta.note_id)?;
                notes.push(DeletedNote {
                    note_id: meta.note_id,
                    title: meta.title,
                    fm_hash: refs.fm_hash,
                    md_hash: refs.md_hash,
                });
            }
            Err(_) => bad_ids.push(id.as_str()),
        }
    }

    if !bad_ids.is_empty() {
        bail!("note IDs not found: {}", bad_ids.join(", "));
    }

    Ok(notes)
}

/// Soft delete: set status = 'retracted'.
pub fn soft_delete(conn: &Connection, notes: &[DeletedNote]) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    let tx = conn.unchecked_transaction()?;

    for note in notes {
        tx.execute(
            "UPDATE current_notes SET status = 'retracted', updated_at = ?1 WHERE note_id = ?2",
            rusqlite::params![now, note.note_id],
        )?;

        audit(&tx, "retract", &note.note_id, &now)?;
    }

    tx.commit()?;
    Ok(())
}

/// Hard delete: remove all registry rows.
pub fn hard_delete(conn: &Connection, notes: &[DeletedNote]) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    let tx = conn.unchecked_transaction()?;

    for note in notes {
        // Collect edge neighbors before deletion for link count fixup
        let neighbors: Vec<String> = {
            let mut stmt = tx.prepare(
                "SELECT DISTINCT dst_note_id FROM note_edges WHERE src_note_id = ?1
                 UNION
                 SELECT DISTINCT src_note_id FROM note_edges WHERE dst_note_id = ?1",
            )?;
            stmt.query_map([&note.note_id], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .filter(|id: &String| id != &note.note_id)
                .collect()
        };

        tx.execute("DELETE FROM note_text WHERE note_id = ?1", [&note.note_id])?;
        tx.execute("DELETE FROM note_tags WHERE note_id = ?1", [&note.note_id])?;
        tx.execute(
            "DELETE FROM note_edges WHERE src_note_id = ?1 OR dst_note_id = ?1",
            [&note.note_id],
        )?;
        tx.execute("DELETE FROM note_collaborators WHERE note_id = ?1", [&note.note_id])?;
        tx.execute("DELETE FROM current_notes WHERE note_id = ?1", [&note.note_id])?;
        tx.execute("DELETE FROM note_versions WHERE note_id = ?1", [&note.note_id])?;
        tx.execute("DELETE FROM notes WHERE note_id = ?1", [&note.note_id])?;

        // Recompute link counts on affected neighbors
        for neighbor_id in &neighbors {
            tx.execute(
                "UPDATE current_notes SET links_out_count = (
                    SELECT COUNT(*) FROM note_edges WHERE src_note_id = ?1
                ) WHERE note_id = ?1",
                [neighbor_id],
            )?;
            tx.execute(
                "UPDATE current_notes SET links_in_count = (
                    SELECT COUNT(*) FROM note_edges WHERE dst_note_id = ?1
                ) WHERE note_id = ?1",
                [neighbor_id],
            )?;
        }

        audit(&tx, "hard_delete", &note.note_id, &now)?;
    }

    tx.commit()?;
    Ok(())
}

fn audit(tx: &Connection, action: &str, note_id: &str, now: &str) -> Result<()> {
    let event_id = uuid::Uuid::new_v4().to_string();
    tx.execute(
        "INSERT INTO audit_log (event_id, client_id, agent_id, action, target_type, target_id, created_at)
         VALUES (?1, ?2, ?3, ?4, 'note', ?5, ?6)",
        rusqlite::params![event_id, DEFAULT_CLIENT_ID, DEFAULT_AGENT_ID, action, note_id, now],
    )?;
    Ok(())
}
