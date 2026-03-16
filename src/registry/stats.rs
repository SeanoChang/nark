use anyhow::Result;
use rusqlite::Connection;

const FILTER: &str = "namespace = 'ark' AND status != 'retracted'";

pub struct FacetCount {
    pub label: String,
    pub count: i64,
}

pub struct RecentNote {
    pub note_id: String,
    pub title: String,
    pub domain: String,
    pub intent: String,
    pub kind: String,
    pub updated_at: String,
}

pub struct MostAccessed {
    pub title: String,
    pub count: i64,
}

pub struct AccessStats {
    pub total_reads: i64,
    pub most_accessed: Option<MostAccessed>,
    pub never_read: i64,
}

pub struct VaultStats {
    pub total_notes: i64,
    pub total_versions: i64,
    pub by_domain: Vec<FacetCount>,
    pub by_kind: Vec<FacetCount>,
    pub recent: Vec<RecentNote>,
    pub access: AccessStats,
}

pub fn overview(conn: &Connection) -> Result<VaultStats> {
    let total_notes: i64 = conn.query_row(
        &format!("SELECT COUNT(*) FROM current_notes WHERE {}", FILTER),
        [], |r| r.get(0),
    )?;

    let total_versions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM note_versions", [], |r| r.get(0),
    )?;

    let by_domain = facet_query(conn, "domain")?;
    let by_kind = facet_query(conn, "kind")?;
    let recent = recent_query(conn)?;
    let access = access_stats(conn)?;

    Ok(VaultStats {
        total_notes,
        total_versions,
        by_domain,
        by_kind,
        recent,
        access,
    })
}

fn facet_query(conn: &Connection, column: &str) -> Result<Vec<FacetCount>> {
    let sql = format!(
        "SELECT {}, COUNT(*) FROM current_notes WHERE {} GROUP BY {} ORDER BY COUNT(*) DESC",
        column, FILTER, column
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows: Vec<FacetCount> = stmt
        .query_map([], |row| {
            Ok(FacetCount {
                label: row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                count: row.get(1)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

fn recent_query(conn: &Connection) -> Result<Vec<RecentNote>> {
    let sql = format!(
        "SELECT note_id, title, domain, intent, kind, updated_at
         FROM current_notes
         WHERE {}
         ORDER BY updated_at DESC
         LIMIT 5",
        FILTER
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows: Vec<RecentNote> = stmt
        .query_map([], |row| {
            Ok(RecentNote {
                note_id: row.get(0)?,
                title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                intent: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                kind: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                updated_at: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

fn access_stats(conn: &Connection) -> Result<AccessStats> {
    let total_reads: i64 = conn.query_row(
        &format!("SELECT COALESCE(SUM(access_count), 0) FROM current_notes WHERE {}", FILTER),
        [], |r| r.get(0),
    )?;

    let most_accessed = conn.query_row(
        &format!(
            "SELECT title, access_count FROM current_notes \
             WHERE {} AND access_count > 0 \
             ORDER BY access_count DESC LIMIT 1",
            FILTER
        ),
        [],
        |row| {
            Ok(MostAccessed {
                title: row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                count: row.get(1)?,
            })
        },
    ).ok();

    let never_read: i64 = conn.query_row(
        &format!("SELECT COUNT(*) FROM current_notes WHERE {} AND access_count = 0", FILTER),
        [], |r| r.get(0),
    )?;

    Ok(AccessStats {
        total_reads,
        most_accessed,
        never_read,
    })
}

