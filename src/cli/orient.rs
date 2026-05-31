use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;

use crate::cli::search::parse_temporal;
use crate::cli::util::truncate_at_word;
use crate::config;
use crate::db;
use crate::registry::{access, resolve, search, tags};
use crate::serve;
use crate::vault::fs::Vault;

/// Build the `nark/orient` JSON-RPC params object, mapping the CLI args onto the
/// exact field names the Phase-3 server's `orient_params` parser expects (see
/// `serve::rpc`): `query` (the server also accepts the `topic` alias, but the CLI
/// only has one query arg, so we send `query`), `domain`, `kind` (strings), `tag`
/// (array), `limit` (number), `since`/`before` (strings). Absent/`None`-default
/// fields are omitted so the server defaults them the same way clap does on the
/// direct path. `limit` is always sent (it has a CLI default); `query` is sent
/// only when present (orient's query is optional).
fn orient_params_json(
    query: Option<&str>,
    domain: Option<&str>,
    kind: Option<&str>,
    tag_filters: &[String],
    limit: usize,
    since: Option<&str>,
    before: Option<&str>,
) -> serde_json::Value {
    let mut params = serde_json::Map::new();
    if let Some(q) = query {
        params.insert("query".into(), serde_json::json!(q));
    }
    if let Some(d) = domain {
        params.insert("domain".into(), serde_json::json!(d));
    }
    if let Some(k) = kind {
        params.insert("kind".into(), serde_json::json!(k));
    }
    if !tag_filters.is_empty() {
        params.insert("tag".into(), serde_json::json!(tag_filters));
    }
    params.insert("limit".into(), serde_json::json!(limit));
    if let Some(s) = since {
        params.insert("since".into(), serde_json::json!(s));
    }
    if let Some(b) = before {
        params.insert("before".into(), serde_json::json!(b));
    }
    serde_json::Value::Object(params)
}

/// Render a socket-HIT `nark/orient` result to the EXACT bytes the direct-open
/// path prints, so dual-mode output is byte-identical socket-vs-direct.
///
/// The serve `orient` method returns the briefing as a JSON **string**
/// ([`serde_json::Value::String`]); the direct path emits that markdown RAW via
/// `print!("{}", md)` (real newlines, unquoted, unescaped, no trailing newline).
/// So when the result is a string we return it verbatim — NOT through
/// `to_string_pretty`, which would quote it, escape every newline to a literal
/// `\n`, collapse it to one line, and (with the old `println!`) append a stray
/// newline. The non-string branch is purely defensive (orient always returns a
/// string today); it pretty-prints with a trailing newline to match the prior
/// `println!` framing.
fn render_orient_output(result: &serde_json::Value) -> Result<String> {
    match result.as_str() {
        Some(md) => Ok(md.to_string()),
        None => Ok(format!("{}\n", serde_json::to_string_pretty(result)?)),
    }
}

// The argument list mirrors the `nark orient` CLI flags 1:1 (clap dispatch in
// `main.rs`); collapsing them into a struct would change that public call site,
// which is out of scope for the dual-mode slice — so the lint is allowed here.
#[allow(clippy::too_many_arguments)]
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
    // Dual-mode: ask a live `nark serve` first (one round-trip via the shared
    // `try_vault_request` seam, which resolves the vault's socket itself). The
    // serve `orient` result is the briefing markdown as a JSON string; on a
    // socket HIT we print it RAW via `render_orient_output` (real newlines,
    // unquoted, unescaped, no trailing newline) so the bytes are IDENTICAL to the
    // direct-open path's `print!("{}", md)` — never `to_string_pretty`, which
    // would quote/escape/one-line the markdown. The socket is an optimization:
    // the seam returns `None` on ANY failure (absent or stale socket, connect
    // timeout, `unauthorized`, error response, malformed JSON, any I/O error),
    // and we then fall through to the always-correct direct-open path below,
    // unchanged (including its per-note access bump, which the read-only serve
    // path intentionally omits).
    let params = orient_params_json(query, domain, kind, tag_filters, limit, since, before);
    if let Some(result) = serve::client::try_vault_request(vault_dir, "nark/orient", params) {
        print!("{}", render_orient_output(&result)?);
        return Ok(());
    }

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
    let hits = search::search(
        &conn,
        q,
        &filters,
        &cfg.search,
        None,
        search::SearchMode::Normal,
    )?;

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
    }

    // Bump access for every hit — agent read these notes' content. Gated by the
    // advisory write lock: non-blocking (the read is never delayed) and skipped
    // for ALL hits if a writer/serve holds the lock (best-effort tracking, no
    // unguarded dual-write). Acquire once, bump all, release.
    let note_ids: Vec<&str> = hits.iter().map(|h| h.note_id.as_str()).collect();
    access::try_bump_access(vault_dir, &conn, &note_ids)?;

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
        "SELECT COUNT(*) FROM current_notes cn WHERE cn.updated_at >= ?1 AND cn.status != 'retracted'",
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
    md.push_str(&format!(
        "{} notes updated in last 7 days{}\n",
        recent_count, scope
    ));

    print!("{}", md);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::client::default_socket;
    use crate::serve::client::test_support::{
        TestServer, current_uid_agent_map, seed_vault, temp_vault_dir,
    };

    /// Rebuild the briefing markdown the direct-open path produces for a
    /// domain-only orient over a seeded vault, WITHOUT the per-note access bump
    /// (the bump is a side effect, not part of the markdown). The serve `orient`
    /// method returns this exact markdown as a JSON string, so this is the parity
    /// baseline for a socket HIT.
    fn direct_orient_markdown(vault_dir: &Path, domain: &str) -> String {
        let conn = db::open_registry(vault_dir).expect("open registry");
        let vault = Vault::new(vault_dir.to_path_buf());
        let cfg = config::load(vault_dir).expect("load config");

        let filters = search::SearchFilters {
            domain: Some(domain),
            kind: None,
            intent: None,
            tags: &[],
            since: None,
            before: None,
            limit: 10,
        };
        let hits = search::search(
            &conn,
            "",
            &filters,
            &cfg.search,
            None,
            search::SearchMode::Normal,
        )
        .expect("registry search");

        let mut md = String::new();
        md.push_str("# Vault Briefing: vault\n\n");
        md.push_str(&format!("## Key Notes ({} most relevant)\n\n", hits.len()));

        let mut all_tags = BTreeSet::new();
        for hit in &hits {
            let refs = resolve::get_ref(&conn, &hit.note_id).expect("get ref");
            let body = vault
                .read_object("objects/md", &refs.md_hash, "md")
                .expect("read body");
            let preview = truncate_at_word(&body, 300);
            let preview = preview.trim();
            let updated_at: String = conn
                .query_row(
                    "SELECT COALESCE(updated_at, '') FROM current_notes WHERE note_id = ?1",
                    [&hit.note_id],
                    |row| row.get(0),
                )
                .expect("updated_at");
            let date = updated_at.split('T').next().unwrap_or(&updated_at);
            md.push_str(&format!("### {}\n", hit.title));
            md.push_str(&format!("- Domain: {} | Kind: {}\n", hit.domain, hit.kind));
            md.push_str(&format!("- Updated: {}\n", date));
            md.push_str(&format!("> {}\n\n", preview.replace('\n', "\n> ")));
            if let Ok(note_tags) = tags::get_tags(&conn, &hit.note_id) {
                for t in note_tags {
                    all_tags.insert(t);
                }
            }
        }

        if !all_tags.is_empty() {
            md.push_str("## Active Tags\n");
            let tag_list: Vec<&str> = all_tags.iter().map(|s| s.as_str()).collect();
            md.push_str(&tag_list.join(", "));
            md.push_str("\n\n");
        }

        let seven_days_ago = parse_temporal("7d").expect("7d");
        let recent_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM current_notes cn WHERE cn.updated_at >= ?1 AND cn.status != 'retracted' AND cn.domain = ?2",
                rusqlite::params![seven_days_ago, domain],
                |row| row.get(0),
            )
            .expect("recent count");
        md.push_str("## Recent Activity\n");
        md.push_str(&format!(
            "{} notes updated in last 7 days (matching filters)\n",
            recent_count
        ));
        md
    }

    /// Rebuild the briefing markdown the direct-open path produces for a
    /// domain + tag-filtered orient over a seeded vault, WITHOUT the per-note
    /// access bump. Mirrors `direct_orient_markdown` but applies a `tag` filter to
    /// BOTH the search pre-filter AND the recent-activity COUNT (the same `EXISTS`
    /// clause `run` appends per tag, alongside the domain clause), and uses the
    /// "(matching filters)" scope suffix since filters are present. This is the
    /// parity baseline for a socket HIT that forwards `domain` + `tag` filters.
    fn direct_orient_markdown_with_tag(vault_dir: &Path, domain: &str, tags: &[String]) -> String {
        let conn = db::open_registry(vault_dir).expect("open registry");
        let vault = Vault::new(vault_dir.to_path_buf());
        let cfg = config::load(vault_dir).expect("load config");

        let filters = search::SearchFilters {
            domain: Some(domain),
            kind: None,
            intent: None,
            tags,
            since: None,
            before: None,
            limit: 10,
        };
        let hits = search::search(
            &conn,
            "",
            &filters,
            &cfg.search,
            None,
            search::SearchMode::Normal,
        )
        .expect("registry search");

        let mut md = String::new();
        md.push_str("# Vault Briefing: vault\n\n");
        md.push_str(&format!("## Key Notes ({} most relevant)\n\n", hits.len()));

        let mut all_tags = BTreeSet::new();
        for hit in &hits {
            let refs = resolve::get_ref(&conn, &hit.note_id).expect("get ref");
            let body = vault
                .read_object("objects/md", &refs.md_hash, "md")
                .expect("read body");
            let preview = truncate_at_word(&body, 300);
            let preview = preview.trim();
            let updated_at: String = conn
                .query_row(
                    "SELECT COALESCE(updated_at, '') FROM current_notes WHERE note_id = ?1",
                    [&hit.note_id],
                    |row| row.get(0),
                )
                .expect("updated_at");
            let date = updated_at.split('T').next().unwrap_or(&updated_at);
            md.push_str(&format!("### {}\n", hit.title));
            md.push_str(&format!("- Domain: {} | Kind: {}\n", hit.domain, hit.kind));
            md.push_str(&format!("- Updated: {}\n", date));
            md.push_str(&format!("> {}\n\n", preview.replace('\n', "\n> ")));
            if let Ok(note_tags) = tags::get_tags(&conn, &hit.note_id) {
                for t in note_tags {
                    all_tags.insert(t);
                }
            }
        }

        if !all_tags.is_empty() {
            md.push_str("## Active Tags\n");
            let tag_list: Vec<&str> = all_tags.iter().map(|s| s.as_str()).collect();
            md.push_str(&tag_list.join(", "));
            md.push_str("\n\n");
        }

        // Recent-activity COUNT scoped to domain + each tag, exactly as `run`
        // assembles it (domain clause at ?2, then one EXISTS clause per tag).
        let seven_days_ago = parse_temporal("7d").expect("7d");
        let mut sql = String::from(
            "SELECT COUNT(*) FROM current_notes cn WHERE cn.updated_at >= ?1 AND cn.status != 'retracted'",
        );
        let mut bound: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(seven_days_ago)];
        let mut pi = 2usize;
        sql.push_str(&format!(" AND cn.domain = ?{}", pi));
        bound.push(Box::new(domain.to_string()));
        pi += 1;
        for t in tags {
            sql.push_str(&format!(
                " AND EXISTS (SELECT 1 FROM note_tags nt JOIN tags tg ON nt.tag_id = tg.tag_id WHERE nt.note_id = cn.note_id AND tg.name = ?{})",
                pi
            ));
            bound.push(Box::new(t.clone()));
            pi += 1;
        }
        let _ = pi;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            bound.iter().map(|p| p.as_ref()).collect();
        let recent_count: i64 = conn
            .query_row(&sql, param_refs.as_slice(), |row| row.get(0))
            .expect("recent count");
        md.push_str("## Recent Activity\n");
        md.push_str(&format!(
            "{} notes updated in last 7 days (matching filters)\n",
            recent_count
        ));
        md
    }

    /// (a) With a live serve + a mapped uid, the dual-mode orient path returns the
    /// SERVER's result for a domain query. The server returns the briefing as a
    /// JSON string equal (value-for-value) to the markdown the direct path builds.
    #[cfg(target_os = "macos")]
    #[test]
    fn orient_socket_hit_matches_direct_path() {
        let server = TestServer::start(current_uid_agent_map());

        // The seeded note's domain is "engineering"; a domain filter makes the
        // query optional, so the briefing falls back to the "vault" label.
        let socket = default_socket(server.dir());
        let params = orient_params_json(None, Some("engineering"), None, &[], 10, None, None);
        let from_socket = serve::client::try_request(&socket, "nark/orient", params)
            .expect("authenticated nark/orient should return Some(result)");

        let served_md = from_socket
            .as_str()
            .expect("orient result is the briefing markdown as a JSON string");
        let direct_md = direct_orient_markdown(server.dir(), "engineering");
        assert_eq!(
            served_md, direct_md,
            "socket-hit orient briefing must match the direct-open path's markdown"
        );
        assert!(
            served_md.contains("### Client Note"),
            "the seeded note should appear as a key note, got: {served_md}"
        );

        // The PRINTED BYTES must be byte-identical between the two paths, not
        // just value-equal at the JSON layer. The direct path prints RAW
        // markdown (`print!("{}", md)`): real newlines, no surrounding quotes,
        // no escaping, no trailing newline. `render_orient_output` is exactly
        // what `run` writes via `print!`, so asserting it equals the direct
        // markdown catches the quoted/escaped/single-line JSON-string regression
        // a naive `to_string_pretty` would emit on a socket HIT.
        let printed = render_orient_output(&from_socket).expect("render socket orient output");
        assert_eq!(
            printed, direct_md,
            "socket-hit orient must print the SAME raw markdown bytes as the direct path"
        );
        assert!(
            !printed.starts_with('"') && printed.contains('\n'),
            "orient output must be raw multi-line markdown, not a quoted/escaped JSON string, got: {printed:?}"
        );

        run(
            server.dir(),
            None,
            Some("engineering"),
            None,
            &[],
            10,
            None,
            None,
        )
        .expect("dual-mode orient over live serve");
    }

    /// (a') FILTERED parity: with a live serve + a mapped uid, the dual-mode orient
    /// path forwards a `domain` + `tag` filter over the socket and returns the
    /// SERVER's briefing markdown byte-identical to the direct-open path applying
    /// the SAME filters. Closes the review gap: a domain-only parity test would not
    /// catch a future rename of the `tag` filter param silently returning wrong
    /// results on a socket HIT. The seeded note carries `tag=delta` in
    /// `domain=engineering`, so the `### Client Note` assertion makes a dropped
    /// filter (which would empty the briefing) unable to pass parity vacuously.
    #[cfg(target_os = "macos")]
    #[test]
    fn orient_socket_hit_matches_direct_path_with_tag_filter() {
        let server = TestServer::start(current_uid_agent_map());

        let tags = vec!["delta".to_string()];
        let socket = default_socket(server.dir());
        let params = orient_params_json(None, Some("engineering"), None, &tags, 10, None, None);
        let from_socket = serve::client::try_request(&socket, "nark/orient", params)
            .expect("authenticated filtered nark/orient should return Some(result)");

        let served_md = from_socket
            .as_str()
            .expect("orient result is the briefing markdown as a JSON string");
        let direct_md = direct_orient_markdown_with_tag(server.dir(), "engineering", &tags);
        assert_eq!(
            served_md, direct_md,
            "socket-hit filtered orient briefing must match the direct-open path's markdown"
        );
        assert!(
            served_md.contains("### Client Note"),
            "the seeded note (domain=engineering, tag=delta) should survive the tag filter, got: {served_md}"
        );

        // The PRINTED BYTES must be byte-identical between the two paths.
        let printed = render_orient_output(&from_socket).expect("render socket orient output");
        assert_eq!(
            printed, direct_md,
            "socket-hit filtered orient must print the SAME raw markdown bytes as the direct path"
        );

        run(
            server.dir(),
            None,
            Some("engineering"),
            None,
            &tags,
            10,
            None,
            None,
        )
        .expect("dual-mode filtered orient over live serve");
    }

    /// (b) With NO serve (socket absent), the direct path is taken — the seam
    /// returns `None` — and the handler succeeds with the seeded vault's data.
    #[test]
    fn orient_no_serve_takes_direct_path() {
        let dir = temp_vault_dir();
        let _id = seed_vault(&dir);

        let socket = default_socket(&dir);
        assert!(!socket.exists(), "precondition: no serve socket");
        let params = orient_params_json(None, Some("engineering"), None, &[], 10, None, None);
        assert!(
            serve::client::try_vault_request(&dir, "nark/orient", params).is_none(),
            "with no serve, the seam must return None so the direct path is taken"
        );

        run(&dir, None, Some("engineering"), None, &[], 10, None, None)
            .expect("direct-open orient with no serve");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// PARITY SWEEP (orient): the BYTES `run` would print on a socket HIT must be
    /// byte-identical to the direct path's printed bytes over the same seeded
    /// vault. Orient prints RAW markdown via `print!("{}", md)` (no trailing
    /// newline, no quoting), so we render the socket result through
    /// `render_orient_output` — exactly what `run` writes — and compare it to the
    /// direct path's reconstructed markdown.
    #[cfg(target_os = "macos")]
    #[test]
    fn orient_socket_vs_direct_output_is_byte_identical() {
        let server = TestServer::start(current_uid_agent_map());

        let params = orient_params_json(None, Some("engineering"), None, &[], 10, None, None);
        let socket_value = serve::client::try_vault_request(server.dir(), "nark/orient", params)
            .expect("socket hit");

        let socket_bytes = render_orient_output(&socket_value).expect("render socket");
        let direct_bytes = direct_orient_markdown(server.dir(), "engineering");
        assert_eq!(
            socket_bytes, direct_bytes,
            "orient output must be byte-identical socket-present vs socket-absent"
        );
    }

    /// FALLBACK HARDENING (orient): a STALE socket — bound but never accepting —
    /// must NOT make the read fail. The seam times out and `run` falls back to
    /// the correct direct path with no hang.
    #[test]
    fn orient_stale_socket_falls_back_to_direct() {
        use std::os::unix::net::UnixListener;

        let dir = temp_vault_dir();
        let _id = seed_vault(&dir);
        let socket = default_socket(&dir);
        std::fs::create_dir_all(socket.parent().unwrap()).expect("create run dir");
        let _listener = UnixListener::bind(&socket).expect("bind stale listener");

        let start = std::time::Instant::now();
        run(&dir, None, Some("engineering"), None, &[], 10, None, None)
            .expect("orient must fall back to direct over a stale socket");
        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "a stale socket must not hang the orient read"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
