use anyhow::Result;
use std::io::Read;
use std::path::Path;

use crate::config;
use crate::db;
use crate::embed::{self, build_embed_input};
use crate::registry::{embeddings, resolve, similarity, write::commit_version};
use crate::vault::fs::Vault;

pub fn run(vault_dir: &Path, id: &str, body: Option<String>, auto_link: bool) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());

    let meta = resolve::get_meta(&conn, id)
        .map_err(|_| anyhow::anyhow!("note not found: {}", id))?;

    let content = match body {
        Some(b) => b,
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    if content.trim().is_empty() {
        anyhow::bail!("nothing to append (empty content)");
    }

    // Read current content
    let refs = resolve::get_ref(&conn, &meta.note_id)?;
    let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
    let md_body = vault.read_object("objects/md", &refs.md_hash, "md")?;

    // Append to body, preserving frontmatter
    let new_body = format!("{}\n{}", md_body.trim_end(), content);
    let full_doc = format!("---\n{}---\n{}", fm_raw, new_body);

    // Re-ingest as new version
    let result = vault.ingest(&full_doc, Some(&meta.note_id))?;
    commit_version(&conn, &result)?;

    // Auto-embed
    let cfg = config::load(vault_dir)?;
    let mut provider = embed::init_provider(vault_dir, &cfg.embedding);
    let last_embedding = if let Some(ref mut prov) = provider {
        let fm = &result.frontmatter;
        let input = build_embed_input(
            &fm.title, &fm.domain, &fm.kind,
            &fm.intent, &fm.tags, &fm.aliases, &result.body,
        );
        match prov.embed_document(&input) {
            Ok(embedding) => {
                let _ = embeddings::upsert_embedding(
                    &conn, &result.note_id, &embedding, prov.model_name(),
                );
                Some(embedding)
            }
            Err(_) => None,
        }
    } else {
        None
    };

    let mut out = serde_json::json!({
        "note_id": result.note_id,
        "version_id": result.version_id,
        "prev_version_id": result.prev_version_id,
        "title": result.frontmatter.title,
        "operation": "append",
    });

    if let Some(ref embedding) = last_embedding {
        if embeddings::has_embeddings(&conn) {
            let all = embeddings::get_all_embeddings(&conn).unwrap_or_default();
            if let Some(sim_result) = similarity::compute_suggestions(
                &conn, &result.note_id, embedding, &all,
                cfg.embedding.similarity_threshold as f32,
                cfg.embedding.auto_link_threshold as f32,
                cfg.embedding.max_suggestions, auto_link,
            ) {
                similarity::append_to_json(&sim_result, &mut out);
            }
        }
    }

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
