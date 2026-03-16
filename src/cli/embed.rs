use anyhow::{bail, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

use crate::config;
use crate::db;
use crate::embed::{self, build_embed_input};
use crate::registry::{embeddings, resolve};
use crate::vault::fs::Vault;

pub fn run_init(vault_dir: &Path) -> Result<()> {
    embed::download::run_init(vault_dir)
}

pub fn run_build(vault_dir: &Path) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());
    let cfg = config::load(vault_dir)?;

    let mut provider = match embed::init_provider(vault_dir, &cfg.embedding) {
        Some(p) => p,
        None => bail!("embedding not available. Run `nark embed init` first."),
    };

    // Detect model mismatch — if any existing embeddings used a different model, re-embed all
    let model_name = provider.model_name().to_string();
    let needs_full_rebuild = has_model_mismatch(&conn, &model_name);

    let note_ids = if needs_full_rebuild {
        eprintln!("Model changed to {model_name} — re-embedding all notes.");
        get_all_active_note_ids(&conn)?
    } else {
        embeddings::get_notes_without_embeddings(&conn)?
    };

    if note_ids.is_empty() {
        eprintln!("All notes already embedded with {model_name}.");
        return Ok(());
    }

    eprintln!("Found {} notes to embed", note_ids.len());

    let pb = ProgressBar::new(note_ids.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("  Embedding... [{bar:40}] {pos}/{len}")
        .unwrap()
        .progress_chars("##-"));

    for note_id in &note_ids {
        let meta = resolve::get_meta(&conn, note_id)?;
        let refs = resolve::get_ref(&conn, note_id)?;
        let body = vault.read_object("objects/md", &refs.md_hash, "md")?;
        let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
        let fm: crate::types::markdown::Frontmatter = serde_yaml::from_str(&fm_raw)?;

        let input = build_embed_input(
            &meta.title, &meta.domain, &meta.kind, &meta.intent,
            &fm.tags, &fm.aliases, &body,
        );

        let embedding = provider.embed_document(&input)?;
        embeddings::upsert_embedding(&conn, note_id, &embedding, &model_name)?;

        pb.inc(1);
    }

    pb.finish_and_clear();
    eprintln!("Done. {} notes embedded with {model_name}.", note_ids.len());

    // Sanity check: cosine self-similarity of first embedded note should ≈ 1.0
    if let Some(first_id) = note_ids.first() {
        if let Some(vec) = embeddings::get_embedding(&conn, first_id)? {
            let self_sim = embed::cosine_similarity(&vec, &vec);
            if (self_sim - 1.0).abs() > 0.001 {
                eprintln!("Warning: self-similarity sanity check failed ({self_sim:.4} != 1.0)");
            } else {
                eprintln!("Self-similarity sanity check passed ({self_sim:.6}).");
            }
        }
    }

    Ok(())
}

const OLD_MODEL_DIRS: &[&str] = &["bge-base-en-v1.5"];

pub fn run_migrate(vault_dir: &Path) -> Result<()> {
    eprintln!("Migrating embeddings to nomic-embed-text-v1.5...\n");

    // Step 1: Download new model (idempotent)
    eprintln!("Step 1/3: Downloading new model...");
    run_init(vault_dir)?;
    eprintln!();

    // Step 2: Remove old model directories
    eprintln!("Step 2/3: Cleaning up old model files...");
    let mut freed = false;
    for old_model in OLD_MODEL_DIRS {
        let old_dir = vault_dir.join("models").join(old_model);
        if old_dir.exists() {
            let size = dir_size(&old_dir);
            std::fs::remove_dir_all(&old_dir)?;
            eprintln!("  Removed {} ({:.0} MB)", old_model, size as f64 / 1_000_000.0);
            freed = true;
        }
    }
    if !freed {
        eprintln!("  No old model files found.");
    }
    eprintln!();

    // Step 3: Re-embed all notes
    eprintln!("Step 3/3: Re-embedding all notes...");
    run_build(vault_dir)?;

    eprintln!("\nMigration complete.");
    Ok(())
}

fn dir_size(path: &Path) -> u64 {
    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// Check if any existing embeddings were produced by a different model.
fn has_model_mismatch(conn: &rusqlite::Connection, current_model: &str) -> bool {
    let mut stmt = match conn.prepare("SELECT DISTINCT model FROM note_embeddings") {
        Ok(s) => s,
        Err(_) => return false,
    };
    let models: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .ok()
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default();

    // Mismatch if any stored model differs from the current one
    models.iter().any(|m| m != current_model)
}

/// Get all active (non-retracted) note IDs for a full re-embed.
fn get_all_active_note_ids(conn: &rusqlite::Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT note_id FROM current_notes WHERE namespace = 'ark' AND status != 'retracted'"
    )?;
    let rows: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}
