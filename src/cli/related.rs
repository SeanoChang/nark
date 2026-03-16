use anyhow::{bail, Result};
use std::path::Path;

use crate::config;
use crate::db;
use crate::registry::{embeddings, resolve, similarity};

pub fn run(vault_dir: &Path, id: &str, limit: usize, link: bool) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let cfg = config::load(vault_dir)?;

    let meta = resolve::get_meta(&conn, id)?;

    // Get this note's embedding
    let note_embedding = match embeddings::get_embedding(&conn, &meta.note_id)? {
        Some(emb) => emb,
        None => bail!("no embedding for note {}. Run `nark embed build`.", meta.note_id),
    };

    // Load all embeddings
    let all = embeddings::get_all_embeddings(&conn)?;
    if all.is_empty() {
        bail!("no embeddings found. Run `nark embed init` then `nark embed build`.");
    }

    // Dimension check
    if let Some((_, first_vec)) = all.first() {
        if first_vec.len() != note_embedding.len() {
            bail!(
                "embedding dimension mismatch (note={}, stored={}). Run `nark embed build` to re-embed.",
                note_embedding.len(), first_vec.len()
            );
        }
    }

    let similar = similarity::find_similar_notes(
        &conn,
        &meta.note_id,
        &all,
        &note_embedding,
        cfg.embedding.similarity_threshold as f32,
        limit,
    );

    let linked = if link && !similar.is_empty() {
        similarity::create_auto_edges(
            &conn,
            &meta.note_id,
            &similar,
            cfg.embedding.auto_link_threshold as f32,
        )
        .unwrap_or(0)
    } else {
        0
    };

    let sim_json: Vec<serde_json::Value> = similar
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.note_id,
                "title": s.title,
                "similarity": (s.similarity * 1000.0).round() / 1000.0,
            })
        })
        .collect();

    let mut out = serde_json::json!({
        "id": meta.note_id,
        "title": meta.title,
        "similar": sim_json,
    });

    if link {
        out["auto_linked"] = serde_json::json!(linked);
    }

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
