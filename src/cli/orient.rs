use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;

use crate::cli::search::parse_temporal;
use crate::cli::util::truncate_at_word;
use crate::config;
use crate::db;
use crate::registry::{access, resolve, search, tags};
use crate::vault::fs::Vault;

pub fn run(
    vault_dir: &Path,
    query: Option<&str>,
    domain: Option<&str>,
    kind: Option<&str>,
    tag_filters: &[String],
    limit: usize,
    since: Option<&str>,
    before: Option<&str>,
) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());
    let cfg = config::load(vault_dir)?;

    let since_ts = since.map(parse_temporal).transpose()?;
    let before_ts = before.map(parse_temporal).transpose()?;

    let filters = search::SearchFilters {
        domain,
        kind,
        intent: None,
        tags: tag_filters,
        since: since_ts.as_deref(),
        before: before_ts.as_deref(),
        limit,
    };

    let q = query.unwrap_or("");
    let hits = search::search(&conn, q, &filters, &cfg.search, None, search::SearchMode::Normal)?;

    // Build briefing
    let mut md = String::new();
    let display_query = if q.is_empty() { "vault" } else { q };
    md.push_str(&format!("# Vault Briefing: {}\n\n", display_query));

    // Key notes section
    md.push_str(&format!("## Key Notes ({} most relevant)\n\n", hits.len()));

    let mut all_tags = BTreeSet::new();

    for hit in &hits {
        let refs = resolve::get_ref(&conn, &hit.note_id)?;
        let body = vault.read_object("objects/md", &refs.md_hash, "md")?;
        let preview = truncate_at_word(&body, 300).trim();

        // Get updated_at from current_notes
        let updated_at: String = conn.query_row(
            "SELECT COALESCE(updated_at, '') FROM current_notes WHERE note_id = ?1",
            [&hit.note_id],
            |row| row.get(0),
        )?;
        let date = updated_at.split('T').next().unwrap_or(&updated_at);

        md.push_str(&format!("### {}\n", hit.title));
        md.push_str(&format!("- Domain: {} | Kind: {}\n", hit.domain, hit.kind));
        md.push_str(&format!("- Updated: {}\n", date));
        md.push_str(&format!("> {}\n\n", preview.replace('\n', "\n> ")));

        // Collect tags
        if let Ok(note_tags) = tags::get_tags(&conn, &hit.note_id) {
            for t in note_tags {
                all_tags.insert(t);
            }
        }

        access::bump_access(&conn, &hit.note_id)?;
    }

    // Active tags section
    if !all_tags.is_empty() {
        md.push_str("## Active Tags\n");
        let tag_list: Vec<&str> = all_tags.iter().map(|s| s.as_str()).collect();
        md.push_str(&tag_list.join(", "));
        md.push_str("\n\n");
    }

    // Recent activity section — scoped to the same domain/kind/tag filters
    let seven_days_ago = parse_temporal("7d")?;
    let mut sql = String::from(
        "SELECT COUNT(*) FROM current_notes cn WHERE cn.updated_at >= ?1 AND cn.status != 'retracted'"
    );
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(seven_days_ago)];
    let mut pi = 2usize;
    if let Some(d) = domain {
        sql.push_str(&format!(" AND cn.domain = ?{}", pi));
        params.push(Box::new(d.to_string()));
        pi += 1;
    }
    if let Some(k) = kind {
        sql.push_str(&format!(" AND cn.kind = ?{}", pi));
        params.push(Box::new(k.to_string()));
        pi += 1;
    }
    for t in tag_filters {
        sql.push_str(&format!(
            " AND EXISTS (SELECT 1 FROM note_tags nt JOIN tags tg ON nt.tag_id = tg.tag_id WHERE nt.note_id = cn.note_id AND tg.name = ?{})",
            pi
        ));
        params.push(Box::new(t.clone()));
        pi += 1;
    }
    let _ = pi; // suppress unused warning
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let recent_count: i64 = conn.query_row(&sql, param_refs.as_slice(), |row| row.get(0))?;
    let scope = if domain.is_some() || kind.is_some() || !tag_filters.is_empty() {
        " (matching filters)"
    } else {
        ""
    };
    md.push_str("## Recent Activity\n");
    md.push_str(&format!("{} notes updated in last 7 days{}\n", recent_count, scope));

    print!("{}", md);
    Ok(())
}

