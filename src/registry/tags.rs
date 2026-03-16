use anyhow::Result;
use rusqlite::Connection;

use crate::db::{DEFAULT_AGENT_ID, DEFAULT_CLIENT_ID};

pub struct TagCount {
    pub name: String,
    pub count: i64,
}

pub struct TaggedNote {
    pub note_id: String,
    pub title: String,
    pub domain: String,
    pub kind: String,
    pub tags: Vec<String>,
}

/// Get all tags for a note.
pub fn get_tags(conn: &Connection, note_id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT t.name FROM tags t
         JOIN note_tags nt ON t.tag_id = nt.tag_id
         WHERE nt.note_id = ?1
         ORDER BY t.name"
    )?;
    let tags: Vec<String> = stmt
        .query_map([note_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(tags)
}

/// Add and remove tags on notes in a single transaction.
pub fn mutate_tags(
    conn: &Connection,
    note_ids: &[String],
    add: &[String],
    remove: &[String],
) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    let tx = conn.unchecked_transaction()?;

    for note_id in note_ids {
        for tag in add {
            tx.execute("INSERT OR IGNORE INTO tags (name) VALUES (?1)", [tag])?;
            tx.execute(
                "INSERT OR IGNORE INTO note_tags (note_id, tag_id)
                 SELECT ?1, tag_id FROM tags WHERE name = ?2",
                rusqlite::params![note_id, tag],
            )?;
        }

        for tag in remove {
            tx.execute(
                "DELETE FROM note_tags WHERE note_id = ?1 AND tag_id = (
                    SELECT tag_id FROM tags WHERE name = ?2
                )",
                rusqlite::params![note_id, tag],
            )?;
        }

        refresh_keywords(&tx, note_id)?;

        let detail = serde_json::json!({ "added": add, "removed": remove }).to_string();
        let event_id = uuid::Uuid::new_v4().to_string();
        tx.execute(
            "INSERT INTO audit_log (event_id, client_id, agent_id, action, target_type, target_id, detail, created_at)
             VALUES (?1, ?2, ?3, 'tag', 'note', ?4, ?5, ?6)",
            rusqlite::params![event_id, DEFAULT_CLIENT_ID, DEFAULT_AGENT_ID, note_id, detail, now],
        )?;
    }

    tx.commit()?;
    Ok(())
}

/// List all tags with usage counts (excludes retracted notes).
pub fn list_tags(conn: &Connection) -> Result<Vec<TagCount>> {
    let mut stmt = conn.prepare(
        "SELECT t.name, COUNT(nt.note_id) AS count
         FROM tags t
         JOIN note_tags nt ON nt.tag_id = t.tag_id
         JOIN current_notes cn ON cn.note_id = nt.note_id
         WHERE cn.status != 'retracted'
         GROUP BY t.name
         ORDER BY count DESC"
    )?;
    let rows: Vec<TagCount> = stmt
        .query_map([], |row| {
            Ok(TagCount { name: row.get(0)?, count: row.get(1)? })
        })?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

/// Find notes matching ALL given tags (AND logic).
pub fn find_by_tags(conn: &Connection, tags: &[String]) -> Result<Vec<TaggedNote>> {
    if tags.is_empty() {
        return Ok(Vec::new());
    }

    let placeholders: Vec<String> = (1..=tags.len()).map(|i| format!("?{}", i)).collect();
    let tag_count = tags.len();

    let sql = format!(
        "SELECT cn.note_id, cn.title, cn.domain, cn.kind
         FROM current_notes cn
         JOIN note_tags nt ON nt.note_id = cn.note_id
         JOIN tags t ON t.tag_id = nt.tag_id
         WHERE cn.status != 'retracted'
           AND t.name IN ({})
         GROUP BY cn.note_id
         HAVING COUNT(DISTINCT t.name) = ?{}
         ORDER BY cn.updated_at DESC",
        placeholders.join(", "),
        tag_count + 1
    );

    let mut stmt = conn.prepare(&sql)?;

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = tags
        .iter()
        .map(|t| Box::new(t.clone()) as Box<dyn rusqlite::types::ToSql>)
        .collect();
    params.push(Box::new(tag_count as i64));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows: Vec<TaggedNote> = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok(TaggedNote {
                note_id: row.get(0)?,
                title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                kind: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                tags: Vec::new(),
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    // Hydrate tags for each result
    let mut result = Vec::with_capacity(rows.len());
    for mut note in rows {
        note.tags = get_tags(conn, &note.note_id)?;
        result.push(note);
    }

    Ok(result)
}

fn refresh_keywords(tx: &Connection, note_id: &str) -> Result<()> {
    tx.execute(
        "UPDATE note_text SET keywords = (
            SELECT COALESCE(GROUP_CONCAT(t.name, ' '), '')
            FROM note_tags nt JOIN tags t ON t.tag_id = nt.tag_id
            WHERE nt.note_id = ?1
        ) WHERE note_id = ?1",
        [note_id],
    )?;
    Ok(())
}
