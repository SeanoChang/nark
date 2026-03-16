use anyhow::{bail, Result};
use std::io::Read;
use std::path::Path;

use crate::config;
use crate::db;
use crate::embed::{self, build_embed_input};
use crate::registry::{embeddings, resolve, write::commit_version};
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
) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
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

    // Auto-embed if provider available
    if let Some(ref mut prov) = embed::init_provider(vault_dir, &cfg.embedding) {
        let fm = &result.frontmatter;
        let input = build_embed_input(
            &fm.title, &fm.domain, &fm.kind,
            &fm.intent, &fm.tags, &fm.aliases, &result.body,
        );
        if let Ok(embedding) = prov.embed_document(&input) {
            let _ = embeddings::upsert_embedding(
                &conn, &result.note_id, &embedding, prov.model_name(),
            );
        }
    }

    let output = serde_json::json!({
        "id": result.note_id,
        "title": result.frontmatter.title,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

fn validate_enum(value: &str, valid: &[&str], field: &str) -> Result<()> {
    if !valid.contains(&value) {
        bail!("invalid --{}: '{}' (valid: {})", field, value, valid.join(", "));
    }
    Ok(())
}
