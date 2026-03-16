use anyhow::Result;
use rusqlite::Connection;

const FILTER: &str = "namespace = 'ark' AND status != 'retracted'";

pub struct GroupRow {
    pub name: String,
    pub count: i64,
}

pub struct NoteRow {
    pub note_id: String,
    pub title: String,
    pub updated_at: String,
    pub tags: Option<Vec<String>>,
}

pub enum BrowseResult {
    Groups { level: &'static str, items: Vec<GroupRow> },
    Notes(Vec<NoteRow>),
}

pub fn browse(conn: &Connection, path: Option<&str>, include_tags: bool) -> Result<BrowseResult> {
    let segments = parse_path(path);

    match segments.as_slice() {
        [] => list_domains(conn),
        [domain] => list_intents(conn, domain),
        [domain, intent] => list_kinds(conn, domain, intent),
        [domain, intent, kind] => list_notes(conn, domain, intent, kind, include_tags),
        _ => anyhow::bail!("too many path segments (max 3: domain/intent/kind)"),
    }
}

fn parse_path(path: Option<&str>) -> Vec<String> {
    match path {
        None => Vec::new(),
        Some(p) => p.trim_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect(),
    }
}

fn list_domains(conn: &Connection) -> Result<BrowseResult> {
    let sql = format!(
        "SELECT domain, COUNT(*) FROM current_notes WHERE {} GROUP BY domain ORDER BY domain",
        FILTER
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(GroupRow { name: row.get(0)?, count: row.get(1)? })
    })?;
    let items: Vec<GroupRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(BrowseResult::Groups { level: "domain", items })
}

fn list_intents(conn: &Connection, domain: &str) -> Result<BrowseResult> {
    let sql = format!(
        "SELECT intent, COUNT(*) FROM current_notes WHERE {} AND domain = ?1 GROUP BY intent ORDER BY intent",
        FILTER
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([domain], |row| {
        Ok(GroupRow { name: row.get(0)?, count: row.get(1)? })
    })?;
    let items: Vec<GroupRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(BrowseResult::Groups { level: "intent", items })
}

fn list_kinds(conn: &Connection, domain: &str, intent: &str) -> Result<BrowseResult> {
    let sql = format!(
        "SELECT kind, COUNT(*) FROM current_notes WHERE {} AND domain = ?1 AND intent = ?2 GROUP BY kind ORDER BY kind",
        FILTER
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params![domain, intent], |row| {
        Ok(GroupRow { name: row.get(0)?, count: row.get(1)? })
    })?;
    let items: Vec<GroupRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(BrowseResult::Groups { level: "kind", items })
}

fn list_notes(conn: &Connection, domain: &str, intent: &str, kind: &str, include_tags: bool) -> Result<BrowseResult> {
    let sql = if include_tags {
        format!(
            "SELECT cn.note_id, cn.title, cn.updated_at,
                    (SELECT GROUP_CONCAT(t.name, ', ')
                     FROM note_tags nt
                     JOIN tags t ON t.tag_id = nt.tag_id
                     WHERE nt.note_id = cn.note_id
                     ORDER BY t.name
                    ) AS tags
             FROM current_notes cn
             WHERE {} AND cn.domain = ?1 AND cn.intent = ?2 AND cn.kind = ?3
             ORDER BY cn.updated_at DESC
             LIMIT 50",
            FILTER.replace("namespace", "cn.namespace").replace("status", "cn.status")
        )
    } else {
        format!(
            "SELECT note_id, title, updated_at FROM current_notes
             WHERE {} AND domain = ?1 AND intent = ?2 AND kind = ?3
             ORDER BY updated_at DESC
             LIMIT 50",
            FILTER
        )
    };

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params![domain, intent, kind], |row| {
        let tags = if include_tags {
            let tags_str: Option<String> = row.get(3)?;
            tags_str.map(|s| s.split(", ").map(String::from).collect())
        } else {
            None
        };
        Ok(NoteRow {
            note_id: row.get(0)?,
            title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            updated_at: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            tags,
        })
    })?;
    let items: Vec<NoteRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(BrowseResult::Notes(items))
}
