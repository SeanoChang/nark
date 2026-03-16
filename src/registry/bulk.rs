use anyhow::{bail, Result};
use rusqlite::Connection;

pub struct BulkFilter {
    pub domain: Option<String>,
    pub kind: Option<String>,
    pub tags: Vec<String>,
    pub since: Option<String>,
    pub before: Option<String>,
}

impl BulkFilter {
    pub fn has_any(&self) -> bool {
        self.domain.is_some()
            || self.kind.is_some()
            || !self.tags.is_empty()
            || self.since.is_some()
            || self.before.is_some()
    }
}

/// Returns (note_id, title) pairs matching the filter. Excludes retracted notes.
pub fn find_matching_notes(conn: &Connection, filter: &BulkFilter) -> Result<Vec<(String, String)>> {
    if !filter.has_any() {
        bail!("at least one filter flag is required for bulk operations");
    }

    let mut sql = String::from(
        "SELECT cn.note_id, cn.title FROM current_notes cn WHERE cn.status != 'retracted'"
    );
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    let mut pi = 1usize;

    if let Some(ref d) = filter.domain {
        sql.push_str(&format!(" AND cn.domain = ?{}", pi));
        params.push(Box::new(d.clone()));
        pi += 1;
    }
    if let Some(ref k) = filter.kind {
        sql.push_str(&format!(" AND cn.kind = ?{}", pi));
        params.push(Box::new(k.clone()));
        pi += 1;
    }
    if let Some(ref s) = filter.since {
        sql.push_str(&format!(" AND cn.updated_at >= ?{}", pi));
        params.push(Box::new(s.clone()));
        pi += 1;
    }
    if let Some(ref b) = filter.before {
        sql.push_str(&format!(" AND cn.updated_at <= ?{}", pi));
        params.push(Box::new(b.clone()));
        pi += 1;
    }
    if !filter.tags.is_empty() {
        let (sub, next_pi) = tag_subquery(&filter.tags, pi);
        sql.push_str(&format!(" AND cn.note_id IN ({})", sub));
        for tag in &filter.tags {
            params.push(Box::new(tag.clone()));
        }
        params.push(Box::new(filter.tags.len() as i64));
        pi = next_pi;
    }
    let _ = pi; // suppress unused warning

    sql.push_str(" ORDER BY cn.updated_at DESC");

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows: Vec<(String, String)> = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

fn tag_subquery(tags: &[String], start_pi: usize) -> (String, usize) {
    let mut pi = start_pi;
    let placeholders: Vec<String> = tags
        .iter()
        .map(|_| {
            let p = format!("?{}", pi);
            pi += 1;
            p
        })
        .collect();

    let having_pi = pi;
    pi += 1;

    let sql = format!(
        "SELECT ntg.note_id FROM note_tags ntg \
         JOIN tags t ON t.tag_id = ntg.tag_id \
         WHERE t.name IN ({}) \
         GROUP BY ntg.note_id \
         HAVING COUNT(DISTINCT t.name) = ?{}",
        placeholders.join(", "),
        having_pi
    );

    (sql, pi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use rusqlite::Connection;

    fn setup_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON;").unwrap();
        db::MIGRATIONS.to_latest(&mut conn).unwrap();
        db::seed_defaults(&conn).unwrap();
        conn
    }

    fn insert_note(conn: &Connection, id: &str, title: &str, domain: &str, kind: &str, status: &str) {
        let now = chrono::Utc::now().to_rfc3339();
        let vid = format!("v-{}", id);

        conn.execute(
            "INSERT INTO notes (note_id, namespace, head_version_id, author_agent_id, created_at)
             VALUES (?1, 'ark', ?2, ?3, ?4)",
            rusqlite::params![id, vid, db::DEFAULT_AGENT_ID, now],
        ).unwrap();

        conn.execute(
            "INSERT INTO note_versions (version_id, note_id, author_agent_id, content_hash, fm_hash, md_hash, created_at)
             VALUES (?1, ?2, ?3, 'ch', 'fh', 'mh', ?4)",
            rusqlite::params![vid, id, db::DEFAULT_AGENT_ID, now],
        ).unwrap();

        conn.execute(
            "INSERT INTO current_notes (note_id, namespace, head_version_id, author_agent_id, title, domain, kind, status, updated_at)
             VALUES (?1, 'ark', ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![id, vid, db::DEFAULT_AGENT_ID, title, domain, kind, status, now],
        ).unwrap();

        conn.execute(
            "INSERT INTO note_text (note_id, title, body, spine, aliases, keywords)
             VALUES (?1, ?2, '', '', '', '')",
            rusqlite::params![id, title],
        ).unwrap();
    }

    fn add_tag(conn: &Connection, note_id: &str, tag: &str) {
        conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?1)", [tag]).unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO note_tags (note_id, tag_id) SELECT ?1, tag_id FROM tags WHERE name = ?2",
            rusqlite::params![note_id, tag],
        ).unwrap();
    }

    #[test]
    fn test_find_matching_notes_by_domain() {
        let conn = setup_db();
        insert_note(&conn, "n1", "Finance Report", "finance", "report", "active");
        insert_note(&conn, "n2", "Tech Overview", "tech", "reference", "active");
        insert_note(&conn, "n3", "Finance Summary", "finance", "journal", "active");

        let filter = BulkFilter {
            domain: Some("finance".into()),
            kind: None,
            tags: vec![],
            since: None,
            before: None,
        };
        let results = find_matching_notes(&conn, &filter).unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"n1"));
        assert!(ids.contains(&"n3"));
    }

    #[test]
    fn test_find_matching_notes_by_tag() {
        let conn = setup_db();
        insert_note(&conn, "n1", "Note A", "finance", "report", "active");
        insert_note(&conn, "n2", "Note B", "finance", "report", "active");
        insert_note(&conn, "n3", "Note C", "tech", "reference", "active");
        add_tag(&conn, "n1", "stale");
        add_tag(&conn, "n2", "stale");
        add_tag(&conn, "n2", "important");
        add_tag(&conn, "n3", "important");

        // Single tag
        let filter = BulkFilter {
            domain: None,
            kind: None,
            tags: vec!["stale".into()],
            since: None,
            before: None,
        };
        let results = find_matching_notes(&conn, &filter).unwrap();
        assert_eq!(results.len(), 2);

        // AND logic: stale + important
        let filter = BulkFilter {
            domain: None,
            kind: None,
            tags: vec!["stale".into(), "important".into()],
            since: None,
            before: None,
        };
        let results = find_matching_notes(&conn, &filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "n2");
    }

    #[test]
    fn test_find_matching_notes_excludes_retracted() {
        let conn = setup_db();
        insert_note(&conn, "n1", "Active Note", "finance", "report", "active");
        insert_note(&conn, "n2", "Retracted Note", "finance", "report", "retracted");

        let filter = BulkFilter {
            domain: Some("finance".into()),
            kind: None,
            tags: vec![],
            since: None,
            before: None,
        };
        let results = find_matching_notes(&conn, &filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "n1");
    }

    #[test]
    fn test_find_matching_notes_combined_filters() {
        let conn = setup_db();
        insert_note(&conn, "n1", "Finance Report", "finance", "report", "active");
        insert_note(&conn, "n2", "Finance Journal", "finance", "journal", "active");
        insert_note(&conn, "n3", "Tech Report", "tech", "report", "active");
        add_tag(&conn, "n1", "quarterly");
        add_tag(&conn, "n2", "quarterly");

        let filter = BulkFilter {
            domain: Some("finance".into()),
            kind: Some("report".into()),
            tags: vec!["quarterly".into()],
            since: None,
            before: None,
        };
        let results = find_matching_notes(&conn, &filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "n1");
    }

    #[test]
    fn test_find_matching_notes_requires_filter() {
        let conn = setup_db();
        let filter = BulkFilter {
            domain: None,
            kind: None,
            tags: vec![],
            since: None,
            before: None,
        };
        assert!(find_matching_notes(&conn, &filter).is_err());
    }
}
