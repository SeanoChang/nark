use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::embed::{self, build_embed_input};
use crate::registry::{embeddings, resolve, write::commit_version};
use crate::vault::fs::Vault;

pub fn run(vault_dir: &Path, id: &str, version_id: &str) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());

    // Validate note exists and resolve prefix
    let meta = resolve::get_meta(&conn, id)
        .map_err(|_| anyhow::anyhow!("note not found: {}", id))?;

    // Validate version belongs to this note
    let mut stmt = conn.prepare(
        "SELECT fm_hash, md_hash FROM note_versions WHERE note_id = ?1 AND version_id = ?2"
    )?;

    let (fm_hash, md_hash): (String, String) = stmt
        .query_row(rusqlite::params![&meta.note_id, version_id], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .map_err(|_| anyhow::anyhow!("version '{}' not found for note '{}'", version_id, &meta.note_id))?;

    // Read the old version's content
    let fm_raw = vault.read_object("objects/fm", &fm_hash, "yaml")?;
    let body = vault.read_object("objects/md", &md_hash, "md")?;
    let full_doc = format!("---\n{}---\n{}", fm_raw, body);

    // Re-ingest as new version (append-only — not destructive)
    let result = vault.ingest(&full_doc, Some(&meta.note_id))?;
    commit_version(&conn, &result)?;

    // Auto-embed if engine available
    let mut engine = embed::init_embedding(vault_dir);
    if let Some(ref mut eng) = engine {
        let fm = &result.frontmatter;
        let input = build_embed_input(
            &fm.title, &fm.domain.to_string(), &fm.kind.to_string(),
            &fm.intent.to_string(), &fm.tags, &fm.aliases, &result.body,
        );
        if let Ok(embedding) = eng.embed_document(&input) {
            let _ = embeddings::upsert_embedding(
                &conn, &result.note_id, &embedding, "bge-base-en-v1.5",
            );
        }
    }

    let out = serde_json::json!({
        "note_id": result.note_id,
        "version_id": result.version_id,
        "prev_version_id": result.prev_version_id,
        "restored_from": version_id,
        "title": result.frontmatter.title,
        "operation": "rollback",
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
