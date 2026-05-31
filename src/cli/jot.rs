use anyhow::{bail, Result};
use std::io::Read;
use std::path::Path;

use crate::config;
use crate::db;
use crate::embed::{self, build_embed_input};
use crate::registry::{embeddings, resolve, similarity, write::commit_version};
use crate::vault::fs::Vault;

pub fn run(
    vault_dir: &Path,
    title: Option<String>,
    author: &str,
    domain: Option<&str>,
    kind: Option<&str>,
    intent: Option<&str>,
    status: Option<&str>,
    tags: &[String],
    body_inline: Option<&str>,
    from: Option<&str>,
    auto_link: bool,
) -> Result<()> {
    // Write command: take the advisory write lock for the whole mutation. A
    // conflicting RW open (another process holds the lock) is refused here with
    // the plain write-locked error before any work. The handle derefs to the
    // `Connection`, so the mutation below is unchanged; it is dropped (releasing
    // the lock) at the end of the function.
    let conn = db::open_registry_guarded(vault_dir)?;
    let cfg = config::load(vault_dir)?;

    // Load template metadata if --from is provided
    let template = match from {
        Some(id) => Some(resolve::get_meta(&conn, id)?),
        None => None,
    };

    // Merge: explicit flags > template values > hardcoded defaults
    let domain = match domain {
        Some(d) => d.to_string(),
        None => match &template {
            Some(t) => t.domain.clone(),
            None => bail!("--domain is required (or use --from to inherit from an existing note)"),
        },
    };
    let intent = match intent {
        Some(s) => s.to_string(),
        None => template.as_ref().map(|t| t.intent.clone()).unwrap_or_else(|| "research".to_string()),
    };
    let kind = match kind {
        Some(s) => {
            let valid = cfg.taxonomy.valid_kinds();
            if !valid.contains(&s) {
                bail!("invalid --kind: '{}' (valid: {})", s, valid.join(", "));
            }
            s.to_string()
        }
        None => template.as_ref().map(|t| t.kind.clone()).unwrap_or_else(|| "reference".to_string()),
    };
    let status = match status {
        Some(s) => {
            validate_enum(s, &["active", "deprecated", "retracted", "draft"], "status")?;
            s.to_string()
        }
        None => template.as_ref().map(|t| t.status.clone()).unwrap_or_else(|| "active".to_string()),
    };
    let tags: Vec<String> = if !tags.is_empty() {
        tags.to_vec()
    } else {
        template.as_ref().map(|t| t.tags.clone()).unwrap_or_default()
    };

    // Read body from --body flag or stdin
    let body = match body_inline {
        Some(b) => b.to_string(),
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    let body = body.trim().to_string();
    if body.is_empty() {
        bail!("body is empty — provide via --body or pipe to stdin");
    }

    // Infer title from first line if not provided
    let title = title.unwrap_or_else(|| {
        let first_line = body.lines().next().unwrap_or("untitled");
        let t = first_line.trim_start_matches('#').trim();
        match t.char_indices().nth(80) {
            Some((idx, _)) => t[..idx].to_string(),
            None => t.to_string(),
        }
    });

    let tags_yaml = if tags.is_empty() {
        "[]".to_string()
    } else {
        let items: Vec<String> = tags.iter().map(|t| format!("  - {}", t)).collect();
        format!("\n{}", items.join("\n"))
    };

    let note = format!(
        "---\n\
         title: \"{}\"\n\
         author: \"{}\"\n\
         domain: \"{}\"\n\
         intent: \"{}\"\n\
         kind: \"{}\"\n\
         status: \"{}\"\n\
         tags: {}\n\
         ---\n\
         {}",
        title.replace('"', "\\\""),
        author.replace('"', "\\\""),
        domain.replace('"', "\\\""),
        intent.replace('"', "\\\""),
        kind.replace('"', "\\\""),
        status.replace('"', "\\\""),
        tags_yaml,
        body
    );

    let vault = Vault::new(vault_dir.to_path_buf());
    let result = vault.ingest(&note, None)?;
    commit_version(&conn, &result)?;

    let last_embedding = if let Some(ref mut prov) = embed::init_provider(vault_dir, &cfg.embedding) {
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

    let mut output = serde_json::json!({
        "id": result.note_id,
        "title": result.frontmatter.title,
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
                similarity::append_to_json(&sim_result, &mut output);
            }
        }
    }

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

fn validate_enum(value: &str, valid: &[&str], field: &str) -> Result<()> {
    if !valid.contains(&value) {
        bail!("invalid --{}: '{}' (valid: {})", field, value, valid.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fresh, unique temp vault dir (matches the repo's temp-dir + pid + uuid
    /// convention; no `tempfile` crate).
    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-jot-lock-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        crate::vault::fs::Vault::new(dir.clone())
            .init_dirs()
            .expect("init vault dirs");
        dir
    }

    fn note_count(vault_dir: &Path) -> i64 {
        let conn = db::open_registry(vault_dir).expect("open registry for count");
        conn.query_row("SELECT COUNT(*) FROM notes", [], |r| r.get(0))
            .expect("count notes")
    }

    /// Invoke `jot` with an inline body (so it never touches stdin) and the
    /// minimal flags. Exercises the real command path end to end.
    fn run_jot(vault_dir: &Path, body: &str) -> Result<()> {
        run(
            vault_dir,
            Some("Lock Test".to_string()),
            "tester",
            Some("engineering"),
            None,
            None,
            None,
            &[],
            Some(body),
            None,
            false,
        )
    }

    /// With no lock held, `jot` succeeds (today's behavior) and actually commits
    /// the note — i.e. the guarded open did not change the success path.
    #[test]
    fn jot_succeeds_and_commits_when_unlocked() {
        let dir = fresh_vault();
        assert_eq!(note_count(&dir), 0, "fresh vault has no notes");

        run_jot(&dir, "First note body.").expect("jot succeeds when unlocked");

        assert_eq!(
            note_count(&dir),
            1,
            "jot must have committed exactly one note"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// When the registry write lock is already held by a separate handle, `jot`
    /// is refused with the plain write-locked error and must NOT mutate the
    /// registry (no note committed).
    #[test]
    fn jot_refuses_and_does_not_mutate_when_write_locked() {
        let dir = fresh_vault();
        // Materialize the db + lock file via a first guarded open, then keep the
        // lock held for the duration of the jot attempt.
        let held = db::open_registry_guarded(&dir).expect("pre-hold the write lock");
        assert_eq!(
            note_count(&dir),
            0,
            "no notes before the locked jot attempt"
        );

        let err = run_jot(&dir, "Should not be written.")
            .expect_err("jot must be refused while the write lock is held");
        let msg = err.to_string();
        assert_eq!(
            msg, "registry is write-locked by another process",
            "conflict must surface the plain honest error"
        );
        assert!(
            !msg.contains("serve"),
            "safety-net error must not mention serve"
        );

        assert_eq!(
            note_count(&dir),
            0,
            "a refused jot must not have committed anything"
        );

        drop(held);
        // After the lock is released, jot works again.
        run_jot(&dir, "Now it works.").expect("jot succeeds after lock released");
        assert_eq!(note_count(&dir), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Reads must NEVER be gated by the write lock: while the lock is held, a
    /// plain `open_registry` (the read path peek/read/etc. use) still opens and
    /// queries the registry.
    #[test]
    fn read_works_while_write_locked() {
        let dir = fresh_vault();
        run_jot(&dir, "A readable note.").expect("seed one note");

        let held = db::open_registry_guarded(&dir).expect("hold the write lock");

        // Read path: open_registry (NOT guarded) must succeed and see the note.
        let conn = db::open_registry(&dir).expect("read open while write-locked");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM notes", [], |r| r.get(0))
            .expect("query while write-locked");
        assert_eq!(count, 1, "read must see the committed note while locked");

        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
