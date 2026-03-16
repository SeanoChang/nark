use anyhow::Result;
use std::io::Read;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::config;
use crate::db;
use crate::embed::{self, build_embed_input};
use crate::registry::{embeddings, similarity, write::commit_version};
use crate::vault::fs::Vault;

pub fn run(vault_dir: &Path, paths: Vec<String>, depth: Option<u64>, auto_link: bool) -> Result<()> {
    let vault = Vault::new(vault_dir.to_path_buf());
    let conn = db::open_registry(vault_dir)?;

    let files = resolve_paths(&paths, depth)?;

    let cfg = config::load(vault_dir)?;
    let mut provider = embed::init_provider(vault_dir, &cfg.embedding);

    // Pre-load all embeddings once for similarity (only if provider available).
    // Trade-off: in batch writes, note N+1 won't see note N as a similar candidate
    // because we don't refresh this vector mid-loop. Acceptable for typical batch sizes.
    let has_embeds = provider.is_some() && embeddings::has_embeddings(&conn);
    let all_embeddings = if has_embeds {
        embeddings::get_all_embeddings(&conn).unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut wrote = 0u64;
    let mut notes: Vec<serde_json::Value> = Vec::new();

    for file in &files {
        let content = std::fs::read_to_string(file)?;
        let result = vault.ingest(&content, None)?;
        commit_version(&conn, &result)?;

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

        let mut note_json = serde_json::json!({
            "id": result.note_id,
            "title": result.frontmatter.title,
            "file": file.display().to_string(),
        });

        if let Some(ref embedding) = last_embedding {
            if let Some(result) = similarity::compute_suggestions(
                &conn, &result.note_id, embedding, &all_embeddings,
                cfg.embedding.similarity_threshold as f32,
                cfg.embedding.auto_link_threshold as f32,
                cfg.embedding.max_suggestions, auto_link,
            ) {
                similarity::append_to_json(&result, &mut note_json);
            }
        }

        notes.push(note_json);
        wrote += 1;
    }

    let summary = serde_json::json!({ "wrote": wrote, "notes": notes });
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn resolve_paths(paths: &[String], depth: Option<u64>) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = Vec::new();

    for path in paths {
        if path == "-" {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            let tmp = std::env::temp_dir().join("nark_stdin.md");
            std::fs::write(&tmp, &buf)?;
            files.push(tmp);
            continue;
        }

        let p = PathBuf::from(path);
        if p.is_file() {
            files.push(p);
        } else if p.is_dir() {
            let mut walker = WalkDir::new(&p);
            if let Some(d) = depth {
                walker = walker.max_depth(d as usize);
            }
            for entry in walker.into_iter().filter_map(|e| e.ok()) {
                let ep = entry.path();
                if ep.is_file() && ep.extension().is_some_and(|ext| ext == "md") {
                    files.push(ep.to_path_buf());
                }
            }
        } else {
            anyhow::bail!("path not found: {}", path);
        }
    }

    Ok(files)
}
