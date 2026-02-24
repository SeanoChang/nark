use anyhow::Result;
use rusqlite::Connection;

pub struct SearchHit {
    pub note_id: String,
    pub title: String,
    pub domain: String,
    pub kind: String,
    pub snippet: String,
    pub rank: f64,
}

pub fn search(
    conn: &Connection,
    query: &str,
    domain: Option<&str>,
    limit: usize,
) -> Result<Vec<SearchHit>> {
    let sql = build_query(domain);

    let mut stmt = conn.prepare(&sql)?;

    let rows = match domain {
        Some(d) => stmt.query_map(
            rusqlite::params![query, query, d, limit as i64],
            map_row,
        )?,
        None => stmt.query_map(
            rusqlite::params![query, query, limit as i64],
            map_row,
        )?,
    };

    let hits: Vec<SearchHit> = rows.filter_map(|r| r.ok()).collect();
    Ok(hits)
}

fn build_query(domain: Option<&str>) -> String {
    // BM25 weights: title=5, body=1, spine=2, aliases=3, keywords=10
    let mut sql = String::from(
        "SELECT
            nt.note_id,
            cn.title,
            cn.domain,
            cn.kind,
            snippet(note_text, 2, '[', ']', '...', 32),
            bm25(note_text, 0.0, 5.0, 1.0, 2.0, 3.0, 10.0)
         FROM note_text nt
         JOIN current_notes cn ON nt.note_id = cn.note_id
         WHERE note_text MATCH ?1
           AND cn.namespace = 'ark'
           AND cn.status != 'deprecated'"
    );

    if domain.is_some() {
        sql.push_str("\n           AND cn.domain = ?3");
    }

    let limit_param = if domain.is_some() { "?4" } else { "?3" };
    sql.push_str(&format!(
        "\n         ORDER BY bm25(note_text, 0.0, 5.0, 1.0, 2.0, 3.0, 10.0)
         LIMIT {}",
        limit_param
    ));

    sql
}

fn map_row(row: &rusqlite::Row) -> rusqlite::Result<SearchHit> {
    Ok(SearchHit {
        note_id: row.get(0)?,
        title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
        domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
        kind: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
        snippet: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
        rank: row.get(5)?,
    })
}
