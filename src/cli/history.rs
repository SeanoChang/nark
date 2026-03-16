use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::resolve;

pub fn run(vault_dir: &Path, id: &str) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;

    // Validate note exists and resolve prefix
    let meta = resolve::get_meta(&conn, id)
        .map_err(|_| anyhow::anyhow!("note not found: {}", id))?;

    let mut stmt = conn.prepare(
        "WITH RECURSIVE history AS (
            SELECT version_id, prev_version_id, content_hash, created_at
            FROM note_versions
            WHERE note_id = ?1 AND version_id = (SELECT head_version_id FROM notes WHERE note_id = ?1)
            UNION ALL
            SELECT nv.version_id, nv.prev_version_id, nv.content_hash, nv.created_at
            FROM note_versions nv JOIN history h ON nv.version_id = h.prev_version_id
        )
        SELECT version_id, prev_version_id, content_hash, created_at FROM history"
    )?;

    let rows: Vec<Result<serde_json::Value, rusqlite::Error>> = stmt
        .query_map([&meta.note_id], |row| {
            Ok(serde_json::json!({
                "version_id": row.get::<_, String>(0)?,
                "prev_version_id": row.get::<_, Option<String>>(1)?,
                "content_hash": row.get::<_, String>(2)?,
                "created_at": row.get::<_, String>(3)?,
            }))
        })?
        .collect();

    let mut versions = Vec::with_capacity(rows.len());
    for row in rows {
        versions.push(row?);
    }

    let out = serde_json::json!({
        "note_id": meta.note_id,
        "version_count": versions.len(),
        "versions": versions,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::db;
    use rusqlite::Connection;

    fn setup_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON;").unwrap();
        db::MIGRATIONS.to_latest(&mut conn).unwrap();
        db::seed_defaults(&conn).unwrap();
        conn
    }

    fn insert_versioned_note(conn: &Connection, note_id: &str, versions: &[(&str, Option<&str>)]) {
        let now = chrono::Utc::now().to_rfc3339();
        let head = versions.last().unwrap().0;

        conn.execute(
            "INSERT INTO notes (note_id, namespace, head_version_id, author_agent_id, created_at)
             VALUES (?1, 'ark', ?2, ?3, ?4)",
            rusqlite::params![note_id, head, db::DEFAULT_AGENT_ID, now],
        ).unwrap();

        conn.execute(
            "INSERT INTO current_notes (note_id, namespace, head_version_id, author_agent_id, title, domain, kind, status, updated_at)
             VALUES (?1, 'ark', ?2, ?3, 'Test', 'systems', 'spec', 'active', ?4)",
            rusqlite::params![note_id, head, db::DEFAULT_AGENT_ID, now],
        ).unwrap();

        for (vid, prev) in versions {
            conn.execute(
                "INSERT INTO note_versions (version_id, note_id, prev_version_id, author_agent_id, content_hash, fm_hash, md_hash, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![vid, note_id, prev, db::DEFAULT_AGENT_ID, format!("ch-{}", vid), "fh", "mh", now],
            ).unwrap();
        }
    }

    #[test]
    fn test_history_walks_full_chain() {
        let conn = setup_db();
        insert_versioned_note(&conn, "n1", &[
            ("v1", None),
            ("v2", Some("v1")),
            ("v3", Some("v2")),
        ]);

        let mut stmt = conn.prepare(
            "WITH RECURSIVE history AS (
                SELECT version_id, prev_version_id, content_hash, created_at
                FROM note_versions
                WHERE note_id = ?1 AND version_id = (SELECT head_version_id FROM notes WHERE note_id = ?1)
                UNION ALL
                SELECT nv.version_id, nv.prev_version_id, nv.content_hash, nv.created_at
                FROM note_versions nv JOIN history h ON nv.version_id = h.prev_version_id
            )
            SELECT version_id FROM history"
        ).unwrap();

        let versions: Vec<String> = stmt
            .query_map(["n1"], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(versions, vec!["v3", "v2", "v1"]);
    }

    #[test]
    fn test_history_single_version() {
        let conn = setup_db();
        insert_versioned_note(&conn, "n1", &[("v1", None)]);

        let mut stmt = conn.prepare(
            "WITH RECURSIVE history AS (
                SELECT version_id, prev_version_id, content_hash, created_at
                FROM note_versions
                WHERE note_id = ?1 AND version_id = (SELECT head_version_id FROM notes WHERE note_id = ?1)
                UNION ALL
                SELECT nv.version_id, nv.prev_version_id, nv.content_hash, nv.created_at
                FROM note_versions nv JOIN history h ON nv.version_id = h.prev_version_id
            )
            SELECT version_id FROM history"
        ).unwrap();

        let versions: Vec<String> = stmt
            .query_map(["n1"], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(versions, vec!["v1"]);
    }
}
