use anyhow::Result;
use rusqlite::Connection;

use crate::embed;
use crate::registry::edges;

pub struct SimilarNote {
    pub note_id: String,
    pub title: String,
    pub similarity: f32,
}

/// Result of a similarity computation, ready to be attached to output JSON.
pub struct SimilarityResult {
    pub similar: Vec<SimilarNote>,
    pub auto_linked: Option<usize>,
}

/// Find notes similar to the given note by cosine similarity.
/// Skips self, filters by threshold, returns sorted descending.
pub fn find_similar_notes(
    conn: &Connection,
    note_id: &str,
    all_embeddings: &[(String, Vec<f32>)],
    note_embedding: &[f32],
    threshold: f32,
    max: usize,
) -> Vec<SimilarNote> {
    let mut scored: Vec<(&str, f32)> = all_embeddings
        .iter()
        .filter(|(id, _)| id != note_id)
        .map(|(id, emb)| (id.as_str(), embed::cosine_similarity(note_embedding, emb)))
        .filter(|(_, score)| *score >= threshold)
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(max);

    if scored.is_empty() {
        return Vec::new();
    }

    // Batch-fetch titles
    let ids: Vec<&str> = scored.iter().map(|(id, _)| *id).collect();
    let titles = fetch_titles(conn, &ids);

    scored
        .into_iter()
        .map(|(id, score)| SimilarNote {
            note_id: id.to_string(),
            title: titles.get(id).cloned().unwrap_or_default(),
            similarity: score,
        })
        .collect()
}

/// Create auto-edges from src to similar notes that meet the auto-link threshold.
/// Uses INSERT OR IGNORE so existing manual edges are preserved.
/// Uses the source note's head version_id for the edge record.
/// Returns count of newly created edges.
pub fn create_auto_edges(
    conn: &Connection,
    src_note_id: &str,
    similar: &[SimilarNote],
    threshold: f32,
) -> Result<usize> {
    let now = chrono::Utc::now().to_rfc3339();

    // Fetch head version_id for the source note
    let head_version_id: String = conn.query_row(
        "SELECT head_version_id FROM current_notes WHERE note_id = ?1",
        [src_note_id],
        |row| row.get(0),
    )?;

    let mut created = 0usize;

    for sim in similar {
        if sim.similarity < threshold {
            continue;
        }

        let changed = conn.execute(
            "INSERT OR IGNORE INTO note_edges
                (src_note_id, dst_note_id, edge_type, weight, source_type, context, version_id, created_at)
             VALUES (?1, ?2, 'references', 1.0, 'auto', NULL, ?3, ?4)",
            rusqlite::params![src_note_id, sim.note_id, head_version_id, now],
        )?;
        created += changed;
    }

    if created > 0 {
        edges::update_link_count(conn, src_note_id)?;
        for sim in similar {
            if sim.similarity >= threshold {
                edges::update_link_count(conn, &sim.note_id)?;
            }
        }
    }

    Ok(created)
}

/// Compute similarity suggestions and optionally auto-link.
/// Returns None if embeddings are unavailable or dimensions mismatch.
/// Consolidates the dimension check + find + auto-link pattern used by write/jot/edit.
pub fn compute_suggestions(
    conn: &Connection,
    note_id: &str,
    embedding: &[f32],
    all_embeddings: &[(String, Vec<f32>)],
    similarity_threshold: f32,
    auto_link_threshold: f32,
    max_suggestions: usize,
    auto_link: bool,
) -> Option<SimilarityResult> {
    if all_embeddings.is_empty() {
        return None;
    }

    // Dimension check
    let dims_match = all_embeddings.first()
        .map(|(_, v)| v.len() == embedding.len())
        .unwrap_or(false);
    if !dims_match {
        return None;
    }

    let similar = find_similar_notes(
        conn, note_id, all_embeddings, embedding,
        similarity_threshold, max_suggestions,
    );

    if similar.is_empty() {
        return None;
    }

    let auto_linked = if auto_link {
        Some(create_auto_edges(conn, note_id, &similar, auto_link_threshold).unwrap_or(0))
    } else {
        None
    };

    Some(SimilarityResult { similar, auto_linked })
}

/// Attach similarity results to a JSON output value.
pub fn append_to_json(result: &SimilarityResult, output: &mut serde_json::Value) {
    let sim_json: Vec<serde_json::Value> = result.similar.iter().map(|s| {
        serde_json::json!({
            "id": s.note_id,
            "title": s.title,
            "similarity": (s.similarity * 1000.0).round() / 1000.0,
        })
    }).collect();
    output["similar"] = serde_json::json!(sim_json);

    if let Some(linked) = result.auto_linked {
        output["auto_linked"] = serde_json::json!(linked);
    }
}

fn fetch_titles<'a>(conn: &Connection, note_ids: &'a [&'a str]) -> std::collections::HashMap<&'a str, String> {
    let mut map = std::collections::HashMap::new();
    if note_ids.is_empty() {
        return map;
    }

    let placeholders: String = note_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT note_id, title FROM current_notes WHERE note_id IN ({})",
        placeholders
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return map,
    };

    let params: Vec<&dyn rusqlite::types::ToSql> =
        note_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();

    if let Ok(rows) = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?.unwrap_or_default()))
    }) {
        for row in rows.flatten() {
            for &input_id in note_ids {
                if input_id == row.0 {
                    map.insert(input_id, row.1.clone());
                    break;
                }
            }
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    fn setup_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON;").unwrap();
        db::MIGRATIONS.to_latest(&mut conn).unwrap();
        db::seed_defaults(&conn).unwrap();
        conn
    }

    fn insert_note(conn: &Connection, note_id: &str, title: &str) {
        let now = chrono::Utc::now().to_rfc3339();
        let version_id = format!("v-{}", note_id);
        conn.execute(
            "INSERT INTO notes (note_id, namespace, head_version_id, author_agent_id, created_at)
             VALUES (?1, 'ark', ?2, ?3, ?4)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, now],
        ).unwrap();
        conn.execute(
            "INSERT INTO note_versions (version_id, note_id, author_agent_id, content_hash, fm_hash, md_hash, created_at)
             VALUES (?1, ?2, ?3, 'ch', 'fh', 'mh', ?4)",
            rusqlite::params![version_id, note_id, db::DEFAULT_AGENT_ID, now],
        ).unwrap();
        conn.execute(
            "INSERT INTO current_notes (note_id, namespace, head_version_id, author_agent_id, title, domain, kind, status, updated_at)
             VALUES (?1, 'ark', ?2, ?3, ?4, 'test', 'reference', 'active', ?5)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, title, now],
        ).unwrap();
    }

    fn make_embedding(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| seed + i as f32).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_find_similar_excludes_self() {
        let conn = setup_db();
        insert_note(&conn, "note-a", "Note A");
        insert_note(&conn, "note-b", "Note B");

        let emb_a = make_embedding(1.0, 10);
        let emb_b = make_embedding(1.1, 10);
        let all = vec![
            ("note-a".to_string(), emb_a.clone()),
            ("note-b".to_string(), emb_b),
        ];

        let results = find_similar_notes(&conn, "note-a", &all, &emb_a, 0.0, 10);
        assert!(!results.iter().any(|r| r.note_id == "note-a"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].note_id, "note-b");
    }

    #[test]
    fn test_find_similar_respects_threshold() {
        let conn = setup_db();
        insert_note(&conn, "note-a", "Note A");
        insert_note(&conn, "note-b", "Note B");
        insert_note(&conn, "note-c", "Note C");

        let emb_a = make_embedding(1.0, 10);
        let emb_b = make_embedding(1.01, 10);
        let emb_c = make_embedding(100.0, 10);
        let all = vec![
            ("note-a".to_string(), emb_a.clone()),
            ("note-b".to_string(), emb_b),
            ("note-c".to_string(), emb_c),
        ];

        let results = find_similar_notes(&conn, "note-a", &all, &emb_a, 0.99, 10);
        assert!(results.iter().all(|r| r.note_id != "note-c"));
    }

    #[test]
    fn test_find_similar_sorted_descending() {
        let conn = setup_db();
        insert_note(&conn, "note-a", "Note A");
        insert_note(&conn, "note-b", "Note B");
        insert_note(&conn, "note-c", "Note C");

        let emb_a = make_embedding(1.0, 10);
        let emb_b = make_embedding(1.01, 10);
        let emb_c = make_embedding(1.5, 10);
        let all = vec![
            ("note-a".to_string(), emb_a.clone()),
            ("note-b".to_string(), emb_b),
            ("note-c".to_string(), emb_c),
        ];

        let results = find_similar_notes(&conn, "note-a", &all, &emb_a, 0.0, 10);
        assert!(results.len() >= 2);
        for w in results.windows(2) {
            assert!(w[0].similarity >= w[1].similarity);
        }
    }

    #[test]
    fn test_find_similar_respects_max_limit() {
        let conn = setup_db();
        insert_note(&conn, "note-a", "Note A");
        insert_note(&conn, "note-b", "Note B");
        insert_note(&conn, "note-c", "Note C");
        insert_note(&conn, "note-d", "Note D");

        let emb_a = make_embedding(1.0, 10);
        let emb_b = make_embedding(1.01, 10);
        let emb_c = make_embedding(1.02, 10);
        let emb_d = make_embedding(1.03, 10);
        let all = vec![
            ("note-a".to_string(), emb_a.clone()),
            ("note-b".to_string(), emb_b),
            ("note-c".to_string(), emb_c),
            ("note-d".to_string(), emb_d),
        ];

        // All 3 should pass threshold, but max=2 should truncate
        let results = find_similar_notes(&conn, "note-a", &all, &emb_a, 0.0, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_create_auto_edges_inserts_with_correct_source_type() {
        let conn = setup_db();
        insert_note(&conn, "note-a", "Note A");
        insert_note(&conn, "note-b", "Note B");

        let similar = vec![SimilarNote {
            note_id: "note-b".to_string(),
            title: "Note B".to_string(),
            similarity: 0.9,
        }];

        let created = create_auto_edges(&conn, "note-a", &similar, 0.8).unwrap();
        assert_eq!(created, 1);

        // Verify source_type is 'auto' and version_id is the head version
        let (source_type, version_id): (String, String) = conn
            .query_row(
                "SELECT source_type, version_id FROM note_edges WHERE src_note_id = 'note-a' AND dst_note_id = 'note-b'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(source_type, "auto");
        assert_eq!(version_id, "v-note-a");
    }

    #[test]
    fn test_create_auto_edges_skips_existing_manual_edge() {
        let conn = setup_db();
        insert_note(&conn, "note-a", "Note A");
        insert_note(&conn, "note-b", "Note B");

        conn.execute(
            "INSERT INTO note_edges (src_note_id, dst_note_id, edge_type, weight, source_type, version_id, created_at)
             VALUES ('note-a', 'note-b', 'references', 1.0, 'body', 'v-note-a', '2026-01-01T00:00:00Z')",
            [],
        ).unwrap();

        let similar = vec![SimilarNote {
            note_id: "note-b".to_string(),
            title: "Note B".to_string(),
            similarity: 0.9,
        }];

        let created = create_auto_edges(&conn, "note-a", &similar, 0.8).unwrap();
        assert_eq!(created, 0);

        let source_type: String = conn
            .query_row(
                "SELECT source_type FROM note_edges WHERE src_note_id = 'note-a' AND dst_note_id = 'note-b'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(source_type, "body");
    }

    #[test]
    fn test_sync_edges_preserves_auto_edges() {
        let conn = setup_db();
        insert_note(&conn, "note-a", "Note A");
        insert_note(&conn, "note-b", "Note B");
        insert_note(&conn, "note-c", "Note C");

        let similar = vec![SimilarNote {
            note_id: "note-b".to_string(),
            title: "Note B".to_string(),
            similarity: 0.9,
        }];
        let created = create_auto_edges(&conn, "note-a", &similar, 0.8).unwrap();
        assert_eq!(created, 1);

        use crate::types::markdown::FrontmatterLink;
        edges::sync_edges(
            &conn,
            "note-a",
            "v-note-a",
            "body with [[note-c]]",
            &[] as &[FrontmatterLink],
            &chrono::Utc::now().to_rfc3339(),
        )
        .unwrap();

        let auto_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM note_edges WHERE src_note_id = 'note-a' AND dst_note_id = 'note-b' AND source_type = 'auto'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(auto_count, 1, "auto-edge should survive sync_edges");

        let body_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM note_edges WHERE src_note_id = 'note-a' AND dst_note_id = 'note-c' AND source_type = 'body'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(body_count, 1, "body edge should be created by sync_edges");
    }
}
