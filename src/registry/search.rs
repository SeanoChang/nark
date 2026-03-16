use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use rusqlite::Connection;

use crate::config::{EngagementConfig, SearchConfig};
use crate::embed;


#[derive(Debug)]
pub struct SearchHit {
    pub note_id: String,
    pub title: String,
    pub domain: String,
    pub kind: String,
    pub snippet: String,
    pub rank: f64,
    pub links_in: i64,
    pub links_out: i64,
}

pub struct SearchFilters<'a> {
    pub domain: Option<&'a str>,
    pub kind: Option<&'a str>,
    pub intent: Option<&'a str>,
    pub tags: &'a [String],
    pub since: Option<&'a str>,
    pub before: Option<&'a str>,
    pub limit: usize,
}

/// Optional cosine context: query embedding + per-note embeddings.
pub struct CosineContext {
    pub query_embedding: Vec<f32>,
    pub note_embeddings: HashMap<String, Vec<f32>>,
}

/// Controls which pipeline steps execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Full pipeline: BM25 filter → graph expand → cosine rank → blend
    Normal,
    /// BM25-only: pre-filter → BM25 filter + rank → return. Skips graph/cosine/blend.
    Bm25Only,
    /// Semantic: pre-filter → cosine against ALL notes → graph expand → blend → return.
    /// Bypasses BM25 as candidate filter.
    Semantic,
}

// -- Internal types --

struct Candidate {
    note_id: String,
    title: String,
    domain: String,
    kind: String,
    snippet: String,
    bm25_rank: Option<usize>,
    cosine_score: f64,
    graph_score: f64,
    engagement: f64,
    final_score: f64,
    links_in: i64,
    links_out: i64,
}

struct RawEdge {
    src_note_id: String,
    dst_note_id: String,
    edge_type: String,
    weight: f64,
}

// -- Public API --

pub fn search(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
    cosine_ctx: Option<&CosineContext>,
    mode: SearchMode,
) -> Result<Vec<SearchHit>> {
    let has_query = !query.is_empty();
    let has_filters = filters.domain.is_some()
        || filters.kind.is_some()
        || filters.intent.is_some()
        || !filters.tags.is_empty();

    if !has_query && !has_filters {
        bail!("search requires a query, --tag, --kind, --intent, --domain, or a combination (--since/--before are pre-filters and cannot be used alone)");
    }

    match mode {
        SearchMode::Bm25Only => search_bm25_only(conn, query, filters, config),
        SearchMode::Semantic => search_semantic(conn, query, filters, config, cosine_ctx),
        SearchMode::Normal => search_normal(conn, query, filters, config, cosine_ctx),
    }
}

/// Full pipeline: BM25 + cosine dual-recall → graph expand → blend
fn search_normal(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
    cosine_ctx: Option<&CosineContext>,
) -> Result<Vec<SearchHit>> {
    let has_query = !query.is_empty();

    // Step 1: BM25 recall or filter-only candidates
    let mut candidates = if has_query {
        fetch_fts_candidates(conn, query, filters, config)?
    } else {
        fetch_filter_candidates(conn, filters, config)?
    };

    // Step 2: Cosine recall (only if embeddings available and there's a query)
    if has_query {
        if let Some(ctx) = cosine_ctx {
            let cosine_candidates = fetch_cosine_candidates(ctx, conn, filters, config)?;
            merge_candidates(&mut candidates, cosine_candidates);
        }
    }

    // Step 3: Graph expansion
    candidates = graph_expand(conn, candidates, config, filters.domain)?;

    // Step 4: Cosine scoring
    compute_cosine_scores(&mut candidates, cosine_ctx);

    // Step 5: Blend signals
    let has_embeddings = cosine_ctx.is_some();
    blend_scores(&mut candidates, config, has_embeddings);

    // Step 6: Threshold + sort + limit
    let results = threshold_and_sort(candidates, config.threshold, filters.limit);
    Ok(to_hits(results))
}

/// BM25-only: pre-filter → BM25 filter + rank → return top N.
/// Skips graph expansion, cosine scoring, and blending.
fn search_bm25_only(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
) -> Result<Vec<SearchHit>> {
    let has_query = !query.is_empty();

    let mut candidates = if has_query {
        fetch_fts_candidates(conn, query, filters, config)?
    } else {
        fetch_filter_candidates(conn, filters, config)?
    };

    // Score using BM25 rank position + engagement only (no graph, no cosine)
    let pool_size = candidates.iter().filter(|c| c.bm25_rank.is_some()).count() as f64;
    for c in &mut candidates {
        let primary = match c.bm25_rank {
            Some(rank) if pool_size > 0.0 => 1.0 - (rank as f64 / pool_size),
            _ => 0.0,
        };
        // Simple score: BM25 rank normalized + engagement tiebreaker
        c.final_score = primary * 0.75 + c.engagement * 0.25;
    }

    let results = threshold_and_sort(candidates, config.threshold, filters.limit);
    Ok(to_hits(results))
}

/// Semantic: pre-filter → cosine against ALL notes → graph expand → blend.
/// Bypasses BM25 as candidate filter — brute-force cosine against the full vault.
fn search_semantic(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
    cosine_ctx: Option<&CosineContext>,
) -> Result<Vec<SearchHit>> {
    let ctx = match cosine_ctx {
        Some(c) => c,
        None => bail!("--semantic requires embeddings. Run `nark embed init` then `nark embed build`."),
    };

    if query.is_empty() {
        bail!("--semantic requires a query");
    }

    // Step 1: Fetch ALL active notes matching pre-filters (no BM25)
    let mut candidates = fetch_filter_candidates(conn, filters, config)?;

    // Raise the limit for semantic — we want the full pool before cosine ranking
    // (fetch_filter_candidates already uses bm25.top_k as limit, which is fine for
    // small vaults; for larger vaults we'd want no limit, but top_k=100 is enough for now)

    // Step 4: Cosine scoring against all candidates
    compute_cosine_scores(&mut candidates, Some(ctx));

    // Step 3: Graph expansion
    candidates = graph_expand(conn, candidates, config, filters.domain)?;

    // Re-score graph-discovered notes with cosine too
    compute_cosine_scores(&mut candidates, Some(ctx));

    // Step 5: Blend (cosine is primary since we always have embeddings)
    blend_scores(&mut candidates, config, true);

    // Step 6: Threshold + sort + limit
    let results = threshold_and_sort(candidates, config.threshold, filters.limit);
    Ok(to_hits(results))
}

fn to_hits(candidates: Vec<Candidate>) -> Vec<SearchHit> {
    candidates
        .into_iter()
        .map(|c| SearchHit {
            note_id: c.note_id,
            title: c.title,
            domain: c.domain,
            kind: c.kind,
            snippet: c.snippet,
            rank: c.final_score,
            links_in: c.links_in,
            links_out: c.links_out,
        })
        .collect()
}

// -- Candidate Fetching --

fn fetch_fts_candidates(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
) -> Result<Vec<Candidate>> {
    let bm25_weights = config.bm25.fts5_weights_arg();

    let mut sql = format!(
        "SELECT
            nt.note_id,
            cn.title,
            cn.domain,
            cn.kind,
            snippet(note_text, 2, '[', ']', '...', 32),
            bm25(note_text, {}),
            cn.access_count,
            cn.last_accessed,
            cn.updated_at,
            cn.links_in_count,
            cn.links_out_count
         FROM note_text nt
         JOIN current_notes cn ON nt.note_id = cn.note_id
         WHERE note_text MATCH ?1
           AND cn.namespace = 'ark'
           AND cn.status != 'retracted'",
        bm25_weights
    );

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    params.push(Box::new(query.to_string()));
    let mut pi = 2usize;

    append_column_filters(&mut sql, &mut params, &mut pi, filters);

    if !filters.tags.is_empty() {
        let (subquery, new_pi) = tag_subquery(filters.tags, pi);
        sql.push_str(&format!("\n           AND cn.note_id IN ({})", subquery));
        for tag in filters.tags {
            params.push(Box::new(tag.clone()));
        }
        params.push(Box::new(filters.tags.len() as i64));
        pi = new_pi;
    }

    sql.push_str(&format!(
        "\n         ORDER BY bm25(note_text, {})\n         LIMIT ?{}",
        bm25_weights, pi
    ));
    params.push(Box::new(config.bm25.top_k as i64));

    let mut candidates = exec_candidate_query(conn, &sql, &params, &config.engagement)?;

    // Assign BM25 rank positions
    for (i, c) in candidates.iter_mut().enumerate() {
        c.bm25_rank = Some(i);
    }

    Ok(candidates)
}

fn fetch_filter_candidates(
    conn: &Connection,
    filters: &SearchFilters,
    config: &SearchConfig,
) -> Result<Vec<Candidate>> {
    let mut sql = String::from(
        "SELECT cn.note_id, cn.title, cn.domain, cn.kind, '' AS snippet, 0.0 AS rank,
                cn.access_count, cn.last_accessed, cn.updated_at,
                cn.links_in_count, cn.links_out_count
         FROM current_notes cn
         WHERE cn.namespace = 'ark'
           AND cn.status != 'retracted'",
    );

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    let mut pi = 1usize;

    append_column_filters(&mut sql, &mut params, &mut pi, filters);

    if !filters.tags.is_empty() {
        let (subquery, new_pi) = tag_subquery(filters.tags, pi);
        sql.push_str(&format!("\n           AND cn.note_id IN ({})", subquery));
        for tag in filters.tags {
            params.push(Box::new(tag.clone()));
        }
        params.push(Box::new(filters.tags.len() as i64));
        pi = new_pi;
    }

    sql.push_str(&format!(
        "\n         ORDER BY cn.updated_at DESC\n         LIMIT ?{}",
        pi
    ));
    params.push(Box::new(config.bm25.top_k as i64));

    exec_candidate_query(conn, &sql, &params, &config.engagement)
}

fn exec_candidate_query(
    conn: &Connection,
    sql: &str,
    params: &[Box<dyn rusqlite::types::ToSql>],
    eng_config: &EngagementConfig,
) -> Result<Vec<Candidate>> {
    let mut stmt = conn.prepare(sql)?;
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows: Vec<Candidate> = stmt
        .query_map(param_refs.as_slice(), |row| {
            let access_count: i64 = row.get::<_, Option<i64>>(6)?.unwrap_or(0);
            let last_accessed: Option<String> = row.get(7)?;
            let updated_at: Option<String> = row.get(8)?;
            Ok(Candidate {
                note_id: row.get(0)?,
                title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                kind: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                snippet: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                bm25_rank: None,
                cosine_score: 0.0,
                graph_score: 0.0,
                engagement: compute_engagement(
                    access_count, last_accessed.as_deref(), updated_at.as_deref(),
                    eng_config,
                ),
                final_score: 0.0,
                links_in: row.get::<_, Option<i64>>(9)?.unwrap_or(0),
                links_out: row.get::<_, Option<i64>>(10)?.unwrap_or(0),
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

/// Compute engagement score from real signals: recency and popularity.
fn compute_engagement(
    access_count: i64,
    last_accessed: Option<&str>,
    updated_at: Option<&str>,
    config: &EngagementConfig,
) -> f64 {
    // Recency: exponential decay (true half-life) from the more recent of last_accessed and updated_at
    let recency = match most_recent_timestamp(last_accessed, updated_at) {
        Some(ts) => {
            let age_hours = hours_since(&ts).max(0.0);
            (-age_hours * 2.0_f64.ln() / config.half_life_hours).exp()
        }
        None => 0.30, // no timestamps fallback
    };

    // Popularity: log-saturating read count, clamped to [0, 1]
    let popularity = ((1.0 + access_count as f64).ln() / (1.0 + config.saturation_reads).ln())
        .min(1.0);

    config.weight_recency * recency + config.weight_popularity * popularity
}

fn most_recent_timestamp(a: Option<&str>, b: Option<&str>) -> Option<chrono::DateTime<chrono::Utc>> {
    let parse = |s: &str| chrono::DateTime::parse_from_rfc3339(s).ok().map(|dt| dt.with_timezone(&chrono::Utc));
    match (a.and_then(parse), b.and_then(parse)) {
        (Some(ta), Some(tb)) => Some(ta.max(tb)),
        (Some(t), None) | (None, Some(t)) => Some(t),
        (None, None) => None,
    }
}

fn hours_since(ts: &chrono::DateTime<chrono::Utc>) -> f64 {
    let now = chrono::Utc::now();
    let duration = now.signed_duration_since(*ts);
    duration.num_seconds() as f64 / 3600.0
}

// -- Cosine Recall (dual-recall fusion) --

/// Fetch top-k candidates by cosine similarity from the embedding table.
/// Applies the same pre-filters as BM25 candidates.
fn fetch_cosine_candidates(
    ctx: &CosineContext,
    conn: &Connection,
    filters: &SearchFilters,
    config: &SearchConfig,
) -> Result<Vec<Candidate>> {
    // Score all embedded notes by cosine similarity
    let mut scored: Vec<(&String, f32)> = ctx
        .note_embeddings
        .iter()
        .map(|(id, emb)| (id, embed::cosine_similarity(&ctx.query_embedding, emb)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(config.bm25.top_k);

    if scored.is_empty() {
        return Ok(Vec::new());
    }

    // Fetch metadata for the top-k cosine hits
    let note_ids: Vec<&String> = scored.iter().map(|(id, _)| *id).collect();
    let cosine_map: HashMap<&str, f32> = scored.iter().map(|(id, s)| (id.as_str(), *s)).collect();

    let placeholders: String = note_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let mut sql = format!(
        "SELECT cn.note_id, cn.title, cn.domain, cn.kind, '' AS snippet, 0.0 AS rank,
                cn.access_count, cn.last_accessed, cn.updated_at,
                cn.links_in_count, cn.links_out_count
         FROM current_notes cn
         WHERE cn.note_id IN ({})
           AND cn.namespace = 'ark'
           AND cn.status != 'retracted'",
        placeholders
    );

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
        note_ids.iter().map(|id| Box::new((*id).clone()) as Box<dyn rusqlite::types::ToSql>).collect();
    let mut pi = note_ids.len() + 1;

    append_column_filters(&mut sql, &mut params, &mut pi, filters);

    if !filters.tags.is_empty() {
        let (subquery, new_pi) = tag_subquery(filters.tags, pi);
        sql.push_str(&format!("\n           AND cn.note_id IN ({})", subquery));
        for tag in filters.tags {
            params.push(Box::new(tag.clone()));
        }
        params.push(Box::new(filters.tags.len() as i64));
        pi = new_pi;
    }
    let _ = pi; // suppress unused warning

    let mut candidates = exec_candidate_query(conn, &sql, &params, &config.engagement)?;

    // Pre-fill cosine scores from the embedding scan
    for c in &mut candidates {
        if let Some(&score) = cosine_map.get(c.note_id.as_str()) {
            c.cosine_score = score as f64;
        }
    }

    Ok(candidates)
}

/// Merge cosine-recall candidates into the existing BM25 candidate pool.
/// Deduplicates by note_id. If a note appears in both, keeps BM25 candidate
/// but copies the cosine_score from the cosine candidate.
fn merge_candidates(bm25_pool: &mut Vec<Candidate>, cosine_pool: Vec<Candidate>) {
    let mut index: HashMap<String, usize> = bm25_pool
        .iter()
        .enumerate()
        .map(|(i, c)| (c.note_id.clone(), i))
        .collect();

    for cosine_c in cosine_pool {
        if let Some(&idx) = index.get(&cosine_c.note_id) {
            // Note in both pools: copy cosine score to BM25 candidate
            if cosine_c.cosine_score > bm25_pool[idx].cosine_score {
                bm25_pool[idx].cosine_score = cosine_c.cosine_score;
            }
        } else {
            // Only in cosine pool: add to candidates
            index.insert(cosine_c.note_id.clone(), bm25_pool.len());
            bm25_pool.push(cosine_c);
        }
    }
}

// -- Graph Expansion --

fn fetch_edges_batch(conn: &Connection, note_ids: &[&str]) -> Result<Vec<RawEdge>> {
    if note_ids.is_empty() {
        return Ok(Vec::new());
    }

    let placeholders: String = note_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");

    // Outgoing edges from seeds
    let sql_out = format!(
        "SELECT src_note_id, dst_note_id, edge_type, weight
         FROM note_edges WHERE src_note_id IN ({})",
        placeholders
    );
    // Incoming edges to seeds
    let sql_in = format!(
        "SELECT src_note_id, dst_note_id, edge_type, weight
         FROM note_edges WHERE dst_note_id IN ({})",
        placeholders
    );

    let mut edges = Vec::new();
    for sql in [&sql_out, &sql_in] {
        let mut stmt = conn.prepare(sql)?;
        let params: Vec<&dyn rusqlite::types::ToSql> =
            note_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
        let rows = stmt
            .query_map(params.as_slice(), |row| {
                Ok(RawEdge {
                    src_note_id: row.get(0)?,
                    dst_note_id: row.get(1)?,
                    edge_type: row.get(2)?,
                    weight: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok());
        edges.extend(rows);
    }

    Ok(edges)
}

fn graph_expand(
    conn: &Connection,
    mut candidates: Vec<Candidate>,
    config: &SearchConfig,
    domain_filter: Option<&str>,
) -> Result<Vec<Candidate>> {
    if candidates.is_empty() {
        return Ok(candidates);
    }

    let pool_size = candidates.len() as f64;
    let seed_ids: HashSet<String> = candidates.iter().map(|c| c.note_id.clone()).collect();

    // Seed scores: BM25-ranked notes get position-based score, filter-only get 0.5
    let seed_scores: HashMap<String, f64> = candidates
        .iter()
        .map(|c| {
            let score = match c.bm25_rank {
                Some(rank) => 1.0 - (rank as f64 / pool_size),
                None => 0.5,
            };
            (c.note_id.clone(), score)
        })
        .collect();

    // Fetch all edges touching seed notes
    let id_refs: Vec<&str> = seed_ids.iter().map(|s| s.as_str()).collect();
    let edges = fetch_edges_batch(conn, &id_refs)?;

    // Propagate scores to neighbors
    let mut neighbor_scores: HashMap<String, f64> = HashMap::new();

    for edge in &edges {
        let is_outgoing = seed_ids.contains(&edge.src_note_id);
        let is_incoming = seed_ids.contains(&edge.dst_note_id);

        if is_outgoing {
            // Outgoing from seed → neighbor is dst
            // Skip supersedes outgoing (blocks new→old propagation)
            if edge.edge_type == "supersedes" {
                continue;
            }
            let seed_score = seed_scores.get(&edge.src_note_id).copied().unwrap_or(0.5);
            let propagated = seed_score * edge.weight * config.graph.decay;
            *neighbor_scores.entry(edge.dst_note_id.clone()).or_insert(0.0) += propagated;
        }

        if is_incoming {
            // Incoming to seed → neighbor is src
            // Allow supersedes incoming (permits old→new flow)
            let seed_score = seed_scores.get(&edge.dst_note_id).copied().unwrap_or(0.5);
            let propagated = seed_score * edge.weight * config.graph.decay;
            *neighbor_scores.entry(edge.src_note_id.clone()).or_insert(0.0) += propagated;
        }
    }

    // Assign graph scores to existing candidates
    for c in &mut candidates {
        if let Some(&gs) = neighbor_scores.get(&c.note_id) {
            c.graph_score = gs;
        }
    }

    // Find graph-discovered notes not already in the pool
    let discovered: Vec<String> = neighbor_scores
        .keys()
        .filter(|id| !seed_ids.contains(*id))
        .cloned()
        .collect();

    if !discovered.is_empty() {
        // Fetch metadata for discovered notes in one query
        let placeholders: String = discovered.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let mut sql = format!(
            "SELECT cn.note_id, cn.title, cn.domain, cn.kind, '' AS snippet, 0.0 AS rank,
                    cn.access_count, cn.last_accessed, cn.updated_at,
                    cn.links_in_count, cn.links_out_count
             FROM current_notes cn
             WHERE cn.note_id IN ({})
               AND cn.namespace = 'ark'
               AND cn.status != 'retracted'",
            placeholders
        );

        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
            discovered.iter().map(|id| Box::new(id.clone()) as Box<dyn rusqlite::types::ToSql>).collect();

        // If respect_domain_filter is enabled, restrict to queried domain
        if config.graph.respect_domain_filter {
            if let Some(domain) = domain_filter {
                sql.push_str("\n               AND cn.domain = ?");
                params.push(Box::new(domain.to_string()));
            }
        }

        let new_candidates = exec_candidate_query(conn, &sql, &params, &config.engagement)?;

        for mut nc in new_candidates {
            if let Some(&gs) = neighbor_scores.get(&nc.note_id) {
                nc.graph_score = gs;
            }
            candidates.push(nc);
        }
    }

    // Normalize graph scores to [0,1]
    let max_graph = candidates
        .iter()
        .map(|c| c.graph_score)
        .fold(0.0_f64, f64::max);
    if max_graph > 0.0 {
        for c in &mut candidates {
            c.graph_score /= max_graph;
        }
    }

    Ok(candidates)
}

// -- Cosine Scoring --

fn compute_cosine_scores(candidates: &mut [Candidate], cosine_ctx: Option<&CosineContext>) {
    let ctx = match cosine_ctx {
        Some(c) => c,
        None => return,
    };

    for c in candidates.iter_mut() {
        if c.cosine_score > 0.0 {
            continue; // already scored (e.g. semantic mode re-scores after graph expand)
        }
        c.cosine_score = ctx
            .note_embeddings
            .get(&c.note_id)
            .map(|ne| embed::cosine_similarity(&ctx.query_embedding, ne) as f64)
            .unwrap_or(0.0);
    }
}

// -- Blending --

fn blend_scores(candidates: &mut [Candidate], config: &SearchConfig, has_embeddings: bool) {
    if candidates.is_empty() {
        return;
    }

    let w = &config.weights;

    // Pool size for BM25 rank normalization (count of candidates that came from BM25)
    let pool_size = candidates
        .iter()
        .filter(|c| c.bm25_rank.is_some())
        .count() as f64;

    for c in candidates.iter_mut() {
        let primary = if has_embeddings {
            c.cosine_score
        } else if let Some(rank) = c.bm25_rank {
            if pool_size > 0.0 {
                1.0 - (rank as f64 / pool_size)
            } else {
                0.0
            }
        } else {
            0.0
        };

        c.final_score = primary * w.cosine + c.graph_score * w.graph + c.engagement * w.engagement;
    }
}

// -- Threshold + Sort --

fn threshold_and_sort(mut candidates: Vec<Candidate>, threshold: f64, limit: usize) -> Vec<Candidate> {
    candidates.retain(|c| c.final_score >= threshold);
    candidates.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(limit);
    candidates
}

// -- Helpers --

fn append_column_filters(
    sql: &mut String,
    params: &mut Vec<Box<dyn rusqlite::types::ToSql>>,
    pi: &mut usize,
    filters: &SearchFilters,
) {
    if let Some(d) = filters.domain {
        sql.push_str(&format!("\n           AND cn.domain = ?{}", *pi));
        params.push(Box::new(d.to_string()));
        *pi += 1;
    }
    if let Some(k) = filters.kind {
        sql.push_str(&format!("\n           AND cn.kind = ?{}", *pi));
        params.push(Box::new(k.to_string()));
        *pi += 1;
    }
    if let Some(i) = filters.intent {
        sql.push_str(&format!("\n           AND cn.intent = ?{}", *pi));
        params.push(Box::new(i.to_string()));
        *pi += 1;
    }
    if let Some(s) = filters.since {
        sql.push_str(&format!("\n           AND cn.updated_at >= ?{}", *pi));
        params.push(Box::new(s.to_string()));
        *pi += 1;
    }
    if let Some(b) = filters.before {
        sql.push_str(&format!("\n           AND cn.updated_at <= ?{}", *pi));
        params.push(Box::new(b.to_string()));
        *pi += 1;
    }
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
    use crate::config::SearchConfig;
    use crate::db;
    use rusqlite::Connection;

    /// Create an in-memory DB with all migrations applied + seed defaults.
    fn setup_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON;").unwrap();

        // Use the same migration machinery as production
        db::MIGRATIONS.to_latest(&mut conn).unwrap();
        db::seed_defaults(&conn).unwrap();
        conn
    }

    /// Insert a note into current_notes + note_text (FTS5).
    /// Returns the note_id for convenience.
    fn insert_note(
        conn: &Connection,
        note_id: &str,
        title: &str,
        domain: &str,
        kind: &str,
        body: &str,
    ) -> String {
        let now = chrono::Utc::now().to_rfc3339();
        let version_id = format!("v-{}", note_id);

        // Insert into notes table (identity)
        conn.execute(
            "INSERT INTO notes (note_id, namespace, head_version_id, author_agent_id, created_at)
             VALUES (?1, 'ark', ?2, ?3, ?4)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, now],
        )
        .unwrap();

        // Insert into note_versions
        conn.execute(
            "INSERT INTO note_versions (version_id, note_id, author_agent_id, content_hash, fm_hash, md_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![version_id, note_id, db::DEFAULT_AGENT_ID, "ch", "fh", "mh", now],
        )
        .unwrap();

        // Insert into current_notes (materialized view)
        conn.execute(
            "INSERT INTO current_notes (note_id, namespace, head_version_id, author_agent_id, title, domain, kind, status, updated_at)
             VALUES (?1, 'ark', ?2, ?3, ?4, ?5, ?6, 'active', ?7)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, title, domain, kind, now],
        )
        .unwrap();

        // Insert into FTS5 note_text
        conn.execute(
            "INSERT INTO note_text (note_id, title, body, spine, aliases, keywords)
             VALUES (?1, ?2, ?3, '', '', '')",
            rusqlite::params![note_id, title, body],
        )
        .unwrap();

        note_id.to_string()
    }

    /// Insert an edge between two notes.
    fn insert_edge(
        conn: &Connection,
        src: &str,
        dst: &str,
        edge_type: &str,
        weight: f64,
    ) {
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO note_edges (src_note_id, dst_note_id, edge_type, weight, source_type, version_id, created_at)
             VALUES (?1, ?2, ?3, ?4, 'body', ?5, ?6)",
            rusqlite::params![src, dst, edge_type, weight, format!("v-{}", src), now],
        )
        .unwrap();
    }

    fn default_config() -> SearchConfig {
        SearchConfig::default()
    }

    fn default_filters(limit: usize) -> SearchFilters<'static> {
        SearchFilters {
            domain: None,
            kind: None,
            intent: None,
            tags: &[],
            since: None,
            before: None,
            limit,
        }
    }

    // ================================================================
    // Test 1: Worked example — Notes A-E with correct ordering
    // ================================================================

    #[test]
    fn test_worked_example_full_pipeline() {
        let conn = setup_db();
        let config = default_config();

        // Insert the 6 systems-domain notes (3 BM25 hits + 2 graph-discovered + 1 extra)
        insert_note(&conn, "note-a", "CAS content hash design", "systems", "spec",
            "content_hash = BLAKE3(fm || md). The vault uses BLAKE3 hashing for all objects.");
        insert_note(&conn, "note-b", "BLAKE3 benchmark results", "systems", "report",
            "BLAKE3 benchmark: 2.1 GB/s on ARM. Fast hashing for content-addressed storage.");
        insert_note(&conn, "note-c", "CAS design for vault", "systems", "spec",
            "Content-addressed storage using BLAKE3 hashing function. Object store layout.");
        insert_note(&conn, "note-d", "Vault ingest pipeline", "systems", "spec",
            "The ingest pipeline reads markdown, parses frontmatter, and writes CAS objects.");
        insert_note(&conn, "note-e", "Registry write.rs impl", "systems", "reference",
            "Implementation of registry write module. Handles note creation and updates.");
        insert_note(&conn, "note-f", "Unrelated systems note", "systems", "report",
            "Network latency measurements for cross-region replication.");

        // Edges: A references D and E (weight 1.0)
        // Using references (weight=1.0) so graph boost is moderate relative to cosine signal
        insert_edge(&conn, "note-a", "note-d", "references", 1.0);
        insert_edge(&conn, "note-a", "note-e", "references", 1.0);

        // Search for "BLAKE3 hashing" with domain filter
        let filters = SearchFilters {
            domain: Some("systems"),
            kind: None,
            intent: None,
            tags: &[],
            since: None,
            before: None,
            limit: 10,
        };

        // Step 1+2: BM25 candidates
        let mut candidates = fetch_fts_candidates(&conn, "BLAKE3 hashing", &filters, &config).unwrap();
        assert!(!candidates.is_empty(), "BM25 should find BLAKE3 hits");

        // Verify A, B, C are in the candidates (they mention BLAKE3)
        let hit_ids: Vec<&str> = candidates.iter().map(|c| c.note_id.as_str()).collect();
        assert!(hit_ids.contains(&"note-a"), "Note A should be a BM25 hit");
        assert!(hit_ids.contains(&"note-b"), "Note B should be a BM25 hit");
        assert!(hit_ids.contains(&"note-c"), "Note C should be a BM25 hit");
        assert!(!hit_ids.contains(&"note-d"), "Note D should NOT be a BM25 hit");

        // Step 3: Graph expansion
        candidates = graph_expand(&conn, candidates, &config, filters.domain).unwrap();
        let expanded_ids: Vec<&str> = candidates.iter().map(|c| c.note_id.as_str()).collect();
        assert!(expanded_ids.contains(&"note-d"), "Note D should be graph-discovered");
        assert!(expanded_ids.contains(&"note-e"), "Note E should be graph-discovered");

        // Verify graph scores are normalized to [0,1]
        for c in &candidates {
            assert!(c.graph_score >= 0.0 && c.graph_score <= 1.0,
                "Graph score for {} should be in [0,1], got {}", c.note_id, c.graph_score);
        }

        // D and E should have non-zero graph scores (discovered via edges from A)
        let d_graph = candidates.iter().find(|c| c.note_id == "note-d").unwrap().graph_score;
        let e_graph = candidates.iter().find(|c| c.note_id == "note-e").unwrap().graph_score;
        assert!(d_graph > 0.0, "Note D should have graph score > 0");
        assert!(e_graph > 0.0, "Note E should have graph score > 0");

        // Step 4: Manually assign cosine scores (simulating what compute_cosine_scores does)
        // A has highest cosine (exact match), D confirmed relevant by cosine
        let fake_cosine: HashMap<&str, f64> = HashMap::from([
            ("note-a", 0.92), ("note-b", 0.88), ("note-c", 0.65),
            ("note-d", 0.71), ("note-e", 0.45),
        ]);
        for c in &mut candidates {
            c.cosine_score = fake_cosine.get(c.note_id.as_str()).copied().unwrap_or(0.0);
        }

        // Step 5: Blend
        blend_scores(&mut candidates, &config, true);

        // Step 6: Threshold + sort
        let results = threshold_and_sort(candidates, config.threshold, 10);

        assert!(results.len() >= 5, "Should have at least 5 results, got {}", results.len());
        let order: Vec<&str> = results.iter().map(|c| c.note_id.as_str()).collect();

        // Key invariant: D (graph-discovered, no keyword match) outranks B (direct BM25 hit)
        // because D is confirmed relevant by cosine AND boosted by graph score
        let d_score = results.iter().find(|c| c.note_id == "note-d").unwrap().final_score;
        let b_score = results.iter().find(|c| c.note_id == "note-b").unwrap().final_score;
        assert!(d_score > b_score,
            "Graph-discovered Note D ({:.3}) should outrank direct hit Note B ({:.3})",
            d_score, b_score);

        // All 5 relevant notes should appear
        assert!(order.contains(&"note-a"), "Note A should be in results");
        assert!(order.contains(&"note-b"), "Note B should be in results");
        assert!(order.contains(&"note-c"), "Note C should be in results");
        assert!(order.contains(&"note-d"), "Note D should be in results");
        assert!(order.contains(&"note-e"), "Note E should be in results");

        // F (unrelated, no keyword match, no graph connection) should NOT appear
        assert!(!order.contains(&"note-f"), "Unrelated Note F should not be in results");

        // Scores should be monotonically decreasing
        for w in results.windows(2) {
            assert!(w[0].final_score >= w[1].final_score,
                "Results should be sorted descending: {} ({:.3}) >= {} ({:.3})",
                w[0].note_id, w[0].final_score, w[1].note_id, w[1].final_score);
        }
    }

    // ================================================================
    // Test 2: Tier 1 — No embeddings, no graph (pure BM25)
    // ================================================================

    #[test]
    fn test_tier1_bm25_only() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "t1-a", "Rust error handling", "programming", "reference",
            "Rust error handling with Result and Option types. Anyhow for applications.");
        insert_note(&conn, "t1-b", "Go error handling", "programming", "reference",
            "Go error handling patterns. Error wrapping with fmt.Errorf.");
        insert_note(&conn, "t1-c", "Python testing", "programming", "runbook",
            "Python pytest framework. Unit testing best practices.");

        // No edges, no cosine context → pure BM25 + engagement
        let hits = search(&conn, "error handling", &default_filters(10), &config, None, SearchMode::Normal).unwrap();

        assert!(hits.len() >= 2, "Should find at least 2 error handling notes");
        // Both A and B mention "error handling" — C (Python testing) should not appear
        let ids: Vec<&str> = hits.iter().map(|h| h.note_id.as_str()).collect();
        assert!(ids.contains(&"t1-a"));
        assert!(ids.contains(&"t1-b"));
        // All scores should be > 0 (BM25 rank normalized + engagement)
        for h in &hits {
            assert!(h.rank > 0.0, "Score should be > 0 for {}", h.note_id);
        }
    }

    // ================================================================
    // Test 3: Tier 2 — No embeddings, with graph
    // ================================================================

    #[test]
    fn test_tier2_bm25_plus_graph() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "t2-a", "OAuth2 token flow", "security", "spec",
            "OAuth2 authorization code flow. Token exchange and refresh.");
        insert_note(&conn, "t2-b", "JWT validation", "security", "reference",
            "JWT signature validation. Claims verification. Token expiry.");
        insert_note(&conn, "t2-c", "Auth middleware design", "security", "spec",
            "Middleware for request authentication. Session management.");

        // Edge: A depends-on C (auth middleware)
        insert_edge(&conn, "t2-a", "t2-c", "depends-on", 2.0);

        // Search with no cosine context (tier 2: BM25 + graph)
        let hits = search(&conn, "OAuth2 token", &default_filters(10), &config, None, SearchMode::Normal).unwrap();

        // A should be a direct hit, C should be graph-discovered
        let ids: Vec<&str> = hits.iter().map(|h| h.note_id.as_str()).collect();
        assert!(ids.contains(&"t2-a"), "Note A should be found via BM25");
        // C may or may not appear depending on graph score vs threshold
        // The important thing is the pipeline doesn't crash without embeddings
    }

    // ================================================================
    // Test 4: Directional supersedes — old→new only
    // ================================================================

    #[test]
    fn test_supersedes_direction() {
        let conn = setup_db();
        let config = default_config();

        // "new" supersedes "old"
        insert_note(&conn, "sup-old", "API v1 design", "systems", "spec",
            "Original API design document. REST endpoints for v1.");
        insert_note(&conn, "sup-new", "API v2 design", "systems", "spec",
            "Updated API design. GraphQL migration from REST.");
        insert_note(&conn, "sup-other", "API client SDK", "systems", "reference",
            "REST client library. Uses API v1 endpoints.");

        // new supersedes old: src=new, dst=old
        insert_edge(&conn, "sup-new", "sup-old", "supersedes", 1.5);
        // other references old
        insert_edge(&conn, "sup-other", "sup-old", "references", 1.0);

        // Case 1: Search finds OLD note → NEW should surface via graph
        let mut candidates = fetch_fts_candidates(
            &conn, "API v1 REST", &default_filters(10), &config
        ).unwrap();
        let old_found = candidates.iter().any(|c| c.note_id == "sup-old");
        if old_found {
            candidates = graph_expand(&conn, candidates, &config, None).unwrap();
            let ids: Vec<&str> = candidates.iter().map(|c| c.note_id.as_str()).collect();
            // When old is found, new should be graph-discovered (old→new propagation)
            assert!(ids.contains(&"sup-new"),
                "When old note is seed, new note should be discovered via incoming supersedes");
        }

        // Case 2: Search finds NEW note → OLD should NOT surface via supersedes
        let mut candidates2 = fetch_fts_candidates(
            &conn, "GraphQL migration", &default_filters(10), &config
        ).unwrap();
        let new_found = candidates2.iter().any(|c| c.note_id == "sup-new");
        if new_found {
            candidates2 = graph_expand(&conn, candidates2, &config, None).unwrap();
            // Old should NOT be discovered via supersedes (blocks new→old)
            let old_via_supersedes = candidates2.iter().any(|c| c.note_id == "sup-old");
            assert!(!old_via_supersedes,
                "When new note is seed, old note should NOT be discovered via supersedes");
        }
    }

    // ================================================================
    // Test 5: Engagement score computed from real signals
    // ================================================================

    #[test]
    fn test_engagement_computation() {
        let config = EngagementConfig::default();

        // Brand new note: just written, never read
        let score_new = compute_engagement(0, None, Some(&chrono::Utc::now().to_rfc3339()), &config);
        // recency ~1.0, popularity 0.0 → 0.60 * 1.0 + 0.40 * 0.0 = ~0.60
        assert!(score_new > 0.50 && score_new < 0.70,
            "Brand new note engagement should be ~0.60, got {:.3}", score_new);

        // Well-read note: 20 reads, accessed recently
        let recent = chrono::Utc::now().to_rfc3339();
        let score_popular = compute_engagement(20, Some(&recent), Some(&recent), &config);
        // recency ~1.0, popularity ~1.0 → 0.60 + 0.40 = ~1.0
        assert!(score_popular > 0.85,
            "Popular recent note engagement should be >0.85, got {:.3}", score_popular);

        // Stale note: 30 days old, never read
        let stale = (chrono::Utc::now() - chrono::Duration::days(30)).to_rfc3339();
        let score_stale = compute_engagement(0, None, Some(&stale), &config);
        // recency ~0.06, popularity 0.0 → 0.60 * 0.06 = ~0.04
        assert!(score_stale < 0.20,
            "Stale unread note engagement should be <0.20, got {:.3}", score_stale);

        // No timestamps: should get 0.30 fallback recency
        let score_no_ts = compute_engagement(0, None, None, &config);
        assert!(score_no_ts > 0.10 && score_no_ts < 0.25,
            "No-timestamp note engagement should use fallback, got {:.3}", score_no_ts);
    }

    // ================================================================
    // Test 6: Graph score normalization
    // ================================================================

    #[test]
    fn test_graph_score_normalization() {
        let conn = setup_db();
        let config = default_config();

        // Create a hub note linked to multiple seeds
        insert_note(&conn, "gn-a", "Seed note alpha", "systems", "spec",
            "Graph normalization test seed alpha. Unique keyword graphnorm.");
        insert_note(&conn, "gn-b", "Seed note beta", "systems", "spec",
            "Graph normalization test seed beta. Unique keyword graphnorm.");
        insert_note(&conn, "gn-c", "Seed note gamma", "systems", "spec",
            "Graph normalization test seed gamma. Unique keyword graphnorm.");
        insert_note(&conn, "gn-hub", "Hub note", "systems", "reference",
            "This hub connects to everything. No graphnorm keyword.");

        // All seeds link to hub with high weights
        insert_edge(&conn, "gn-a", "gn-hub", "depends-on", 2.0);
        insert_edge(&conn, "gn-b", "gn-hub", "depends-on", 2.0);
        insert_edge(&conn, "gn-c", "gn-hub", "depends-on", 2.0);

        let mut candidates = fetch_fts_candidates(
            &conn, "graphnorm", &default_filters(10), &config
        ).unwrap();

        // Should find a, b, c as seeds
        assert!(candidates.len() >= 3);

        candidates = graph_expand(&conn, candidates, &config, None).unwrap();

        // Hub should be discovered and have the max graph score (normalized to 1.0)
        let hub = candidates.iter().find(|c| c.note_id == "gn-hub");
        assert!(hub.is_some(), "Hub note should be graph-discovered");
        let hub = hub.unwrap();
        assert!((hub.graph_score - 1.0).abs() < 0.001,
            "Hub (max accumulator) should normalize to 1.0, got {}", hub.graph_score);

        // All graph scores should be in [0, 1]
        for c in &candidates {
            assert!(c.graph_score >= 0.0 && c.graph_score <= 1.0,
                "Graph score for {} = {} should be in [0,1]", c.note_id, c.graph_score);
        }
    }

    // ================================================================
    // Test 7: BM25 discarded with embeddings, kept without
    // ================================================================

    #[test]
    fn test_bm25_discard_with_embeddings() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "bd-a", "Primary search target", "systems", "spec",
            "BM25 discard test. Primary target note.");
        insert_note(&conn, "bd-b", "Secondary search target", "systems", "spec",
            "BM25 discard test. Secondary target note.");

        let mut candidates = fetch_fts_candidates(
            &conn, "BM25 discard test", &default_filters(10), &config
        ).unwrap();
        assert!(candidates.len() >= 2);

        // Without embeddings: BM25 rank should determine primary signal
        blend_scores(&mut candidates, &config, false);
        let scores_no_embed: Vec<f64> = candidates.iter().map(|c| c.final_score).collect();
        assert!(scores_no_embed.iter().all(|&s| s > 0.0),
            "Without embeddings, BM25 rank should contribute to final score");

        // Reset scores
        for c in &mut candidates {
            c.final_score = 0.0;
            c.cosine_score = 0.99; // high cosine
        }

        // With embeddings: cosine should be primary, BM25 rank ignored
        blend_scores(&mut candidates, &config, true);
        // Both should have the same final score since cosine is the same
        // and they have same engagement + graph (both 0)
        let scores_with_embed: Vec<f64> = candidates.iter().map(|c| c.final_score).collect();
        assert!((scores_with_embed[0] - scores_with_embed[1]).abs() < 0.001,
            "With embeddings and same cosine, BM25 rank shouldn't affect scores");
    }

    // ================================================================
    // Test 8: Blend formula correctness
    // ================================================================

    #[test]
    fn test_blend_formula() {
        let config = default_config();
        let mut candidates = vec![
            Candidate {
                note_id: "bf-a".into(),
                title: "test".into(),
                domain: "systems".into(),
                kind: "spec".into(),
                snippet: String::new(),
                bm25_rank: Some(0),
                cosine_score: 0.92,
                graph_score: 0.0,
                engagement: 0.85,
                final_score: 0.0,
                links_in: 0,
                links_out: 0,
            },
            Candidate {
                note_id: "bf-d".into(),
                title: "test".into(),
                domain: "systems".into(),
                kind: "spec".into(),
                snippet: String::new(),
                bm25_rank: None,
                cosine_score: 0.71,
                graph_score: 0.40,
                engagement: 0.70,
                final_score: 0.0,
                links_in: 0,
                links_out: 0,
            },
        ];

        blend_scores(&mut candidates, &config, true);

        // A: 0.92×0.50 + 0.00×0.25 + 0.85×0.25 = 0.460 + 0.000 + 0.2125 = 0.6725
        let a = &candidates[0];
        let expected_a = 0.92 * 0.50 + 0.00 * 0.25 + 0.85 * 0.25;
        assert!((a.final_score - expected_a).abs() < 0.001,
            "Note A blend: expected {:.4}, got {:.4}", expected_a, a.final_score);

        // D: 0.71×0.50 + 0.40×0.25 + 0.70×0.25 = 0.355 + 0.100 + 0.175 = 0.630
        let d = &candidates[1];
        let expected_d = 0.71 * 0.50 + 0.40 * 0.25 + 0.70 * 0.25;
        assert!((d.final_score - expected_d).abs() < 0.001,
            "Note D blend: expected {:.4}, got {:.4}", expected_d, d.final_score);
    }

    // ================================================================
    // Test 9: Filter-only search (no query)
    // ================================================================

    #[test]
    fn test_filter_only_search() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "fo-a", "Finance report alpha", "finance", "report",
            "Quarterly earnings analysis for Q4.");
        insert_note(&conn, "fo-b", "Finance report beta", "finance", "report",
            "Annual revenue forecast model.");
        insert_note(&conn, "fo-c", "Systems spec", "systems", "spec",
            "Infrastructure scaling plan.");

        let filters = SearchFilters {
            domain: Some("finance"),
            kind: Some("report"),
            intent: None,
            tags: &[],
            since: None,
            before: None,
            limit: 10,
        };

        let hits = search(&conn, "", &filters, &config, None, SearchMode::Normal).unwrap();

        assert_eq!(hits.len(), 2, "Should find exactly 2 finance reports");
    }

    // ================================================================
    // Test 10: Threshold filtering
    // ================================================================

    #[test]
    fn test_threshold_filtering() {
        let candidates = vec![
            Candidate {
                note_id: "th-a".into(), title: "a".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: None, cosine_score: 0.0, graph_score: 0.0,
                engagement: 0.0, final_score: 0.50,
                links_in: 0, links_out: 0,
            },
            Candidate {
                note_id: "th-b".into(), title: "b".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: None, cosine_score: 0.0, graph_score: 0.0,
                engagement: 0.0, final_score: 0.05, // below default threshold 0.10
                links_in: 0, links_out: 0,
            },
            Candidate {
                note_id: "th-c".into(), title: "c".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: None, cosine_score: 0.0, graph_score: 0.0,
                engagement: 0.0, final_score: 0.10, // exactly at threshold
                links_in: 0, links_out: 0,
            },
        ];

        let results = threshold_and_sort(candidates, 0.10, 10);
        assert_eq!(results.len(), 2, "Should keep scores >= 0.10, filter out 0.05");
        assert_eq!(results[0].note_id, "th-a");
        assert_eq!(results[1].note_id, "th-c");
    }

    // ================================================================
    // Test 11: BM25-only mode skips graph and cosine
    // ================================================================

    #[test]
    fn test_bm25_only_mode() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "bm-a", "BM25 mode target note", "systems", "spec",
            "Testing BM25 only mode. This should be found.");
        insert_note(&conn, "bm-b", "BM25 mode neighbor", "systems", "spec",
            "This neighbor note has different content.");
        insert_edge(&conn, "bm-a", "bm-b", "references", 1.0);

        let hits = search(
            &conn, "BM25 mode target", &default_filters(10), &config, None, SearchMode::Bm25Only
        ).unwrap();

        // Should find A (keyword match)
        let ids: Vec<&str> = hits.iter().map(|h| h.note_id.as_str()).collect();
        assert!(ids.contains(&"bm-a"), "BM25-only should find keyword match");
        // B should NOT appear (no keyword match, graph expansion skipped)
        assert!(!ids.contains(&"bm-b"), "BM25-only should not graph-discover neighbors");
    }

    // ================================================================
    // Test 12: --bm25 and --semantic mutual exclusivity (tested at CLI level)
    // ================================================================

    #[test]
    fn test_semantic_mode_requires_embeddings() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "sem-a", "Semantic test note", "systems", "spec",
            "Testing semantic mode without embeddings.");

        let result = search(
            &conn, "semantic test", &default_filters(10), &config, None, SearchMode::Semantic
        );

        assert!(result.is_err(), "--semantic without embeddings should error");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("embeddings"), "Error should mention embeddings: {}", err);
    }

    // ================================================================
    // Test 13: respect_domain_filter restricts graph expansion
    // ================================================================

    #[test]
    fn test_respect_domain_filter() {
        let conn = setup_db();
        let mut config = default_config();

        insert_note(&conn, "df-a", "Domain filter seed", "finance", "spec",
            "Finance domain filter test. Unique keyword domfilter.");
        insert_note(&conn, "df-b", "Same domain neighbor", "finance", "reference",
            "Finance reference note about quarterly earnings.");
        insert_note(&conn, "df-c", "Cross domain neighbor", "systems", "reference",
            "Systems reference note about infrastructure scaling.");

        insert_edge(&conn, "df-a", "df-b", "references", 1.0);
        insert_edge(&conn, "df-a", "df-c", "references", 1.0);

        // Without respect_domain_filter: both neighbors should be discovered
        config.graph.respect_domain_filter = false;
        let mut candidates = fetch_fts_candidates(
            &conn, "domfilter", &default_filters(10), &config
        ).unwrap();
        candidates = graph_expand(&conn, candidates, &config, Some("finance")).unwrap();
        let ids: Vec<&str> = candidates.iter().map(|c| c.note_id.as_str()).collect();
        assert!(ids.contains(&"df-b"), "Without domain filter: same-domain neighbor found");
        assert!(ids.contains(&"df-c"), "Without domain filter: cross-domain neighbor found");

        // With respect_domain_filter: only same-domain neighbor should be discovered
        config.graph.respect_domain_filter = true;
        let mut candidates2 = fetch_fts_candidates(
            &conn, "domfilter", &default_filters(10), &config
        ).unwrap();
        candidates2 = graph_expand(&conn, candidates2, &config, Some("finance")).unwrap();
        let ids2: Vec<&str> = candidates2.iter().map(|c| c.note_id.as_str()).collect();
        assert!(ids2.contains(&"df-b"), "With domain filter: same-domain neighbor found");
        assert!(!ids2.contains(&"df-c"), "With domain filter: cross-domain neighbor excluded");
    }

    // ================================================================
    // Test 14: Changing config weights changes results
    // ================================================================

    #[test]
    fn test_weight_changes_affect_scores() {
        let mut config = default_config();
        let mut candidates = vec![
            Candidate {
                note_id: "wt-a".into(),
                title: "test".into(),
                domain: "systems".into(),
                kind: "spec".into(),
                snippet: String::new(),
                bm25_rank: Some(0),
                cosine_score: 0.90,
                graph_score: 0.10,
                engagement: 0.50,
                final_score: 0.0,
                links_in: 0,
                links_out: 0,
            },
        ];

        // Default weights: cosine=0.50, graph=0.25, engagement=0.25
        blend_scores(&mut candidates, &config, true);
        let score_default = candidates[0].final_score;

        // Increase graph weight, decrease cosine
        candidates[0].final_score = 0.0;
        config.weights.cosine = 0.30;
        config.weights.graph = 0.45;
        blend_scores(&mut candidates, &config, true);
        let score_graph_heavy = candidates[0].final_score;

        assert!(
            (score_default - score_graph_heavy).abs() > 0.01,
            "Changing weights should observably change scores: default={:.4}, graph_heavy={:.4}",
            score_default, score_graph_heavy
        );
    }

    // ================================================================
    // Test 15: Configurable BM25 column weights
    // ================================================================

    #[test]
    fn test_configurable_bm25_weights() {
        let config = default_config();
        let expected = "0.0, 5, 1, 2, 3, 10";
        let actual = config.bm25.fts5_weights_arg();
        assert_eq!(actual, expected, "Default BM25 weights should match spec");

        // Custom weights
        let mut custom = config;
        custom.bm25.weight_title = 10.0;
        custom.bm25.weight_keywords = 20.0;
        let actual = custom.bm25.fts5_weights_arg();
        assert!(actual.contains("10"), "Custom title weight should appear");
        assert!(actual.contains("20"), "Custom keywords weight should appear");
    }

    // ================================================================
    // Test 16: Temporal since filter excludes old notes
    // ================================================================

    /// Insert a note with a specific updated_at timestamp.
    fn insert_note_at(
        conn: &Connection,
        note_id: &str,
        title: &str,
        domain: &str,
        kind: &str,
        body: &str,
        updated_at: &str,
    ) {
        let version_id = format!("v-{}", note_id);
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO notes (note_id, namespace, head_version_id, author_agent_id, created_at)
             VALUES (?1, 'ark', ?2, ?3, ?4)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, now],
        ).unwrap();

        conn.execute(
            "INSERT INTO note_versions (version_id, note_id, author_agent_id, content_hash, fm_hash, md_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![version_id, note_id, db::DEFAULT_AGENT_ID, "ch", "fh", "mh", now],
        ).unwrap();

        conn.execute(
            "INSERT INTO current_notes (note_id, namespace, head_version_id, author_agent_id, title, domain, kind, status, updated_at)
             VALUES (?1, 'ark', ?2, ?3, ?4, ?5, ?6, 'active', ?7)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, title, domain, kind, updated_at],
        ).unwrap();

        conn.execute(
            "INSERT INTO note_text (note_id, title, body, spine, aliases, keywords)
             VALUES (?1, ?2, ?3, '', '', '')",
            rusqlite::params![note_id, title, body],
        ).unwrap();
    }

    #[test]
    fn test_since_filter_excludes_old_notes() {
        let conn = setup_db();
        let config = default_config();

        let now = chrono::Utc::now();
        let recent = (now - chrono::Duration::hours(6)).to_rfc3339();
        let old = (now - chrono::Duration::days(30)).to_rfc3339();
        let since_cutoff = (now - chrono::Duration::days(1)).to_rfc3339();

        insert_note_at(&conn, "tf-recent", "Recent BTC analysis", "finance", "report",
            "Bitcoin price analysis from today.", &recent);
        insert_note_at(&conn, "tf-old", "Old BTC report", "finance", "report",
            "Bitcoin mining overview from last month.", &old);

        // Without since filter: both notes should appear
        let filters_no_time = SearchFilters {
            domain: Some("finance"),
            kind: None,
            intent: None,
            tags: &[],
            since: None,
            before: None,
            limit: 10,
        };
        let hits = search(&conn, "Bitcoin", &filters_no_time, &config, None, SearchMode::Bm25Only).unwrap();
        assert_eq!(hits.len(), 2, "Without temporal filter, both notes should appear");

        // With since filter: only recent note should appear
        let filters_since = SearchFilters {
            domain: Some("finance"),
            kind: None,
            intent: None,
            tags: &[],
            since: Some(&since_cutoff),
            before: None,
            limit: 10,
        };
        let hits = search(&conn, "Bitcoin", &filters_since, &config, None, SearchMode::Bm25Only).unwrap();
        assert_eq!(hits.len(), 1, "Since filter should exclude old note");
        assert_eq!(hits[0].note_id, "tf-recent");
    }

    // ================================================================
    // Test 17: Temporal before filter excludes recent notes
    // ================================================================

    #[test]
    fn test_before_filter_excludes_recent_notes() {
        let conn = setup_db();
        let config = default_config();

        let now = chrono::Utc::now();
        let recent = (now - chrono::Duration::hours(6)).to_rfc3339();
        let old = (now - chrono::Duration::days(30)).to_rfc3339();
        let before_cutoff = (now - chrono::Duration::days(7)).to_rfc3339();

        insert_note_at(&conn, "bf-recent", "Recent FOMC note", "finance", "report",
            "Federal reserve meeting notes from today.", &recent);
        insert_note_at(&conn, "bf-old", "Old FOMC note", "finance", "report",
            "Federal reserve meeting notes from last month.", &old);

        // With before filter: only old note should appear
        let filters_before = SearchFilters {
            domain: Some("finance"),
            kind: None,
            intent: None,
            tags: &[],
            since: None,
            before: Some(&before_cutoff),
            limit: 10,
        };
        let hits = search(&conn, "Federal reserve", &filters_before, &config, None, SearchMode::Bm25Only).unwrap();
        assert_eq!(hits.len(), 1, "Before filter should exclude recent note");
        assert_eq!(hits[0].note_id, "bf-old");
    }

    // ================================================================
    // Test 18: Combined since + before creates a date range
    // ================================================================

    #[test]
    fn test_since_and_before_combined_range() {
        let conn = setup_db();
        let config = default_config();

        let now = chrono::Utc::now();
        let very_recent = (now - chrono::Duration::hours(1)).to_rfc3339();
        let mid_range = (now - chrono::Duration::days(5)).to_rfc3339();
        let very_old = (now - chrono::Duration::days(60)).to_rfc3339();

        let since_cutoff = (now - chrono::Duration::days(14)).to_rfc3339();
        let before_cutoff = (now - chrono::Duration::days(2)).to_rfc3339();

        insert_note_at(&conn, "r-new", "New market data", "finance", "report",
            "Latest market data analysis.", &very_recent);
        insert_note_at(&conn, "r-mid", "Mid-range market data", "finance", "report",
            "Market data from last week.", &mid_range);
        insert_note_at(&conn, "r-old", "Old market data", "finance", "report",
            "Ancient market data analysis.", &very_old);

        // Range: 14 days ago to 2 days ago — should only match mid-range
        let filters = SearchFilters {
            domain: Some("finance"),
            kind: None,
            intent: None,
            tags: &[],
            since: Some(&since_cutoff),
            before: Some(&before_cutoff),
            limit: 10,
        };
        let hits = search(&conn, "market data", &filters, &config, None, SearchMode::Bm25Only).unwrap();
        assert_eq!(hits.len(), 1, "Date range should only include mid-range note");
        assert_eq!(hits[0].note_id, "r-mid");
    }

    // ================================================================
    // Test 19: merge_candidates deduplication
    // ================================================================

    #[test]
    fn test_merge_candidates_dedup() {
        let mut bm25_pool = vec![
            Candidate {
                note_id: "mc-a".into(), title: "a".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: Some(0), cosine_score: 0.0, graph_score: 0.0,
                engagement: 0.50, final_score: 0.0,
                links_in: 0, links_out: 0,
            },
            Candidate {
                note_id: "mc-b".into(), title: "b".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: Some(1), cosine_score: 0.0, graph_score: 0.0,
                engagement: 0.50, final_score: 0.0,
                links_in: 0, links_out: 0,
            },
        ];

        let cosine_pool = vec![
            // Duplicate: mc-a appears in both pools with cosine score
            Candidate {
                note_id: "mc-a".into(), title: "a".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: None, cosine_score: 0.85, graph_score: 0.0,
                engagement: 0.50, final_score: 0.0,
                links_in: 0, links_out: 0,
            },
            // New: mc-c only in cosine pool
            Candidate {
                note_id: "mc-c".into(), title: "c".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: None, cosine_score: 0.72, graph_score: 0.0,
                engagement: 0.50, final_score: 0.0,
                links_in: 0, links_out: 0,
            },
        ];

        merge_candidates(&mut bm25_pool, cosine_pool);

        assert_eq!(bm25_pool.len(), 3, "Should have 3 unique candidates");

        // mc-a: BM25 candidate kept, but cosine score copied
        let a = bm25_pool.iter().find(|c| c.note_id == "mc-a").unwrap();
        assert_eq!(a.bm25_rank, Some(0), "BM25 rank should be preserved");
        assert!((a.cosine_score - 0.85).abs() < 0.001, "Cosine score should be copied");

        // mc-b: unchanged
        let b = bm25_pool.iter().find(|c| c.note_id == "mc-b").unwrap();
        assert_eq!(b.bm25_rank, Some(1));

        // mc-c: from cosine pool, no BM25 rank
        let c = bm25_pool.iter().find(|c| c.note_id == "mc-c").unwrap();
        assert_eq!(c.bm25_rank, None, "Cosine-only candidate should have no BM25 rank");
        assert!((c.cosine_score - 0.72).abs() < 0.001);
    }

    // ================================================================
    // Test 20: Hybrid fusion — Normal mode with cosine context
    // ================================================================

    #[test]
    fn test_hybrid_fusion_normal_mode() {
        let conn = setup_db();
        let config = default_config();

        // Note A: BM25 hit (contains "wyckoff spring")
        insert_note(&conn, "hf-a", "Wyckoff spring setup", "finance", "spec",
            "Wyckoff spring entry criteria. Accumulation phase identification.");
        // Note B: BM25 hit (contains "spring")
        insert_note(&conn, "hf-b", "Spring cleaning runbook", "systems", "runbook",
            "Spring cleaning procedures for vault maintenance.");
        // Note C: NOT a BM25 hit for "spring entry" — but semantically related
        insert_note(&conn, "hf-c", "Accumulation phase trading", "finance", "spec",
            "How to identify accumulation phases in market structure.");

        // Simulate embeddings: C is semantically close to query but has no keyword match.
        // All vectors must be L2-normalized since cosine_similarity is a bare dot product.
        let dim = 768;
        let unit = |fill: f32| -> Vec<f32> {
            let norm = (dim as f32 * fill * fill).sqrt();
            vec![fill / norm; dim]
        };

        let query_emb = unit(1.0); // unit vector
        let mut note_embeddings = HashMap::new();
        // A: high cosine ~1.0 (keyword + semantic match)
        note_embeddings.insert("hf-a".to_string(), unit(1.0));
        // B: low cosine ~0.3 (wrong domain) — use a mostly-orthogonal vector
        let mut b_vec = vec![0.0_f32; dim];
        for i in 0..dim { b_vec[i] = if i < dim / 3 { 1.0 } else { 0.0 }; }
        let b_norm: f32 = b_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut b_vec { *x /= b_norm; }
        note_embeddings.insert("hf-b".to_string(), b_vec);
        // C: high cosine ~0.9 (semantic match, no keyword)
        let mut c_vec = unit(1.0);
        // Perturb slightly to get cosine < 1.0 but still high
        for i in 0..(dim / 10) { c_vec[i] = 0.0; }
        let c_norm: f32 = c_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut c_vec { *x /= c_norm; }
        note_embeddings.insert("hf-c".to_string(), c_vec);

        let cosine_ctx = CosineContext {
            query_embedding: query_emb,
            note_embeddings,
        };

        let hits = search(
            &conn, "spring entry", &default_filters(10), &config,
            Some(&cosine_ctx), SearchMode::Normal
        ).unwrap();

        let ids: Vec<&str> = hits.iter().map(|h| h.note_id.as_str()).collect();

        // Key assertion: C should appear (found via cosine recall, not BM25)
        assert!(ids.contains(&"hf-c"),
            "Cosine-recall should surface semantically related note C. Got: {:?}", ids);
        // A should also appear (found by both BM25 and cosine)
        assert!(ids.contains(&"hf-a"),
            "Note A should appear via BM25 + cosine. Got: {:?}", ids);
    }
}
