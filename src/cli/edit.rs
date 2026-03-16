use anyhow::{bail, Result};
use std::io::Read;
use std::path::Path;

use crate::config;
use crate::db;
use crate::embed::{self, build_embed_input};
use crate::registry::{embeddings, resolve, write::commit_version};
use crate::vault::fs::Vault;

#[derive(Debug)]
enum EditOp {
    Set { content: String },
    Replace { old: String, new: String, mode: ReplaceMode },
    Append { content: String },
    Prepend { content: String },
}

#[derive(Debug)]
enum ReplaceMode {
    Unique,
    All,
    Count(usize),
}

/// Read stdin eagerly once, returning the content. Called at most once per
/// invocation so that batch ops sharing "-" all get the same content.
fn read_stdin() -> Result<String> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

fn parse_operations(args: &[String], batch: bool) -> Result<Vec<EditOp>> {
    if args.is_empty() {
        bail!("no edit operation specified. Use: set, replace, append, or prepend");
    }

    // Eagerly read stdin once if any arg is "-"
    let stdin_content = if args.iter().any(|a| a == "-") {
        Some(read_stdin()?)
    } else {
        None
    };

    if batch {
        parse_batch_operations(args, stdin_content.as_deref())
    } else {
        let op = parse_single_operation(args, stdin_content.as_deref())?;
        Ok(vec![op])
    }
}

fn parse_batch_operations(args: &[String], stdin: Option<&str>) -> Result<Vec<EditOp>> {
    let mut ops = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for arg in args {
        if arg == "," {
            if !current.is_empty() {
                ops.push(parse_single_operation(&current, stdin)?);
                current.clear();
            }
        } else {
            current.push(arg.clone());
        }
    }
    if !current.is_empty() {
        ops.push(parse_single_operation(&current, stdin)?);
    }

    if ops.is_empty() {
        bail!("no operations found in --batch args");
    }
    Ok(ops)
}

fn resolve_content(value: &str, stdin: Option<&str>) -> Result<String> {
    if value == "-" {
        match stdin {
            Some(s) => Ok(s.to_string()),
            None => bail!("stdin content not available"),
        }
    } else {
        Ok(value.to_string())
    }
}

fn parse_single_operation(args: &[String], stdin: Option<&str>) -> Result<EditOp> {
    if args.is_empty() {
        bail!("empty operation");
    }

    match args[0].as_str() {
        "set" => {
            if args.len() != 2 {
                bail!("set requires exactly one argument: set <path-or-dash>");
            }
            let content = if args[1] == "-" {
                resolve_content("-", stdin)?
            } else {
                std::fs::read_to_string(&args[1])
                    .map_err(|e| anyhow::anyhow!("failed to read file '{}': {}", args[1], e))?
            };
            Ok(EditOp::Set { content })
        }
        "replace" => {
            let mut mode = ReplaceMode::Unique;
            let mut positional = Vec::new();

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--all" => mode = ReplaceMode::All,
                    "--count" => {
                        i += 1;
                        if i >= args.len() {
                            bail!("--count requires a number");
                        }
                        let n: usize = args[i].parse()
                            .map_err(|_| anyhow::anyhow!("--count value must be a number: {}", args[i]))?;
                        mode = ReplaceMode::Count(n);
                    }
                    _ => positional.push(args[i].clone()),
                }
                i += 1;
            }

            if positional.len() != 2 {
                bail!("replace requires exactly two positional arguments: replace <old> <new>");
            }
            Ok(EditOp::Replace {
                old: positional[0].clone(),
                new: positional[1].clone(),
                mode,
            })
        }
        "append" => {
            if args.len() != 2 {
                bail!("append requires exactly one argument: append <content-or-dash>");
            }
            let content = resolve_content(&args[1], stdin)?;
            Ok(EditOp::Append { content })
        }
        "prepend" => {
            if args.len() != 2 {
                bail!("prepend requires exactly one argument: prepend <content-or-dash>");
            }
            let content = resolve_content(&args[1], stdin)?;
            Ok(EditOp::Prepend { content })
        }
        other => bail!("unknown edit operation: '{}'. Use: set, replace, append, or prepend", other),
    }
}

fn apply_operation(full_doc: &str, fm_raw: &str, body: &str, op: &EditOp) -> Result<String> {
    match op {
        EditOp::Set { content } => Ok(content.clone()),
        EditOp::Replace { old, new, mode } => {
            let count = full_doc.matches(old.as_str()).count();
            match mode {
                ReplaceMode::Unique => {
                    if count == 0 {
                        bail!("replace: '{}' not found in document", old);
                    }
                    if count > 1 {
                        bail!("replace: '{}' found {} times (expected unique match). Use --all or --count {}", old, count, count);
                    }
                    Ok(full_doc.replacen(old.as_str(), new.as_str(), 1))
                }
                ReplaceMode::All => {
                    if count == 0 {
                        bail!("replace --all: '{}' not found in document", old);
                    }
                    Ok(full_doc.replace(old.as_str(), new.as_str()))
                }
                ReplaceMode::Count(expected) => {
                    if count != *expected {
                        bail!("replace --count {}: '{}' found {} times (expected {})", expected, old, count, expected);
                    }
                    Ok(full_doc.replace(old.as_str(), new.as_str()))
                }
            }
        }
        EditOp::Append { content } => {
            let new_body = format!("{}\n{}", body.trim_end(), content);
            Ok(format!("---\n{}---\n{}", fm_raw, new_body))
        }
        EditOp::Prepend { content } => {
            let new_body = format!("{}\n{}", content, body);
            Ok(format!("---\n{}---\n{}", fm_raw, new_body))
        }
    }
}

pub fn run(vault_dir: &Path, id: &str, batch: bool, args: Vec<String>) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());

    // Validate note exists and resolve prefix
    let meta = resolve::get_meta(&conn, id)
        .map_err(|_| anyhow::anyhow!("note not found: {}", id))?;

    // Parse operations (reads stdin eagerly if any arg is "-")
    let ops = parse_operations(&args, batch)?;

    // Read current content
    let refs = resolve::get_ref(&conn, &meta.note_id)?;
    let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
    let body = vault.read_object("objects/md", &refs.md_hash, "md")?;

    // Build the full document
    let mut full_doc = format!("---\n{}---\n{}", fm_raw, body);
    let mut current_fm = fm_raw;
    let mut current_body = body;

    // Apply operations sequentially
    let mut op_names: Vec<String> = Vec::new();
    for op in &ops {
        let op_name = match op {
            EditOp::Set { .. } => "set",
            EditOp::Replace { .. } => "replace",
            EditOp::Append { .. } => "append",
            EditOp::Prepend { .. } => "prepend",
        };
        op_names.push(op_name.to_string());

        full_doc = apply_operation(&full_doc, &current_fm, &current_body, op)?;

        // Re-split for next operation (needed for append/prepend which use fm_raw/body)
        let (fm, bd) = Vault::split_doc(&full_doc)?;
        current_fm = fm;
        current_body = bd;
    }

    // Validate the result parses correctly
    let (test_fm, _) = Vault::split_doc(&full_doc)?;
    let _: crate::types::markdown::Frontmatter = serde_yaml::from_str(&test_fm)
        .map_err(|e| anyhow::anyhow!("edit produced invalid frontmatter: {}", e))?;

    // Re-ingest as new version
    let result = vault.ingest(&full_doc, Some(&meta.note_id))?;
    commit_version(&conn, &result)?;

    // Auto-embed if provider available
    let cfg = config::load(vault_dir)?;
    let mut provider = embed::init_provider(vault_dir, &cfg.embedding);
    if let Some(ref mut prov) = provider {
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

    // Output
    let operation = if ops.len() == 1 {
        serde_json::Value::String(op_names[0].clone())
    } else {
        serde_json::Value::Array(op_names.into_iter().map(serde_json::Value::String).collect())
    };

    let out = serde_json::json!({
        "note_id": result.note_id,
        "version_id": result.version_id,
        "prev_version_id": result.prev_version_id,
        "title": result.frontmatter.title,
        "operation": operation,
        "batch": batch,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> String {
        "---\ntitle: Test Note\nauthor: agent\ndomain: systems\nintent: build\nkind: spec\nstatus: active\ntags:\n- test\n---\nThis is the body.\nIt has multiple lines.\nTODO: fix this.\nTODO: fix that.".to_string()
    }

    fn split(doc: &str) -> (String, String) {
        Vault::split_doc(doc).unwrap()
    }

    // --- split_doc ---

    #[test]
    fn test_split_doc_roundtrip() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let reconstructed = format!("---\n{}---\n{}", fm, body);
        assert_eq!(doc, reconstructed);
    }

    #[test]
    fn test_split_doc_no_frontmatter() {
        assert!(Vault::split_doc("no frontmatter here").is_err());
    }

    #[test]
    fn test_split_doc_no_closing_delimiter() {
        assert!(Vault::split_doc("---\ntitle: X\nbody without closing").is_err());
    }

    // --- replace: unique mode ---

    #[test]
    fn test_replace_unique_success() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Replace {
            old: "multiple lines".into(),
            new: "several lines".into(),
            mode: ReplaceMode::Unique,
        };
        let result = apply_operation(&doc, &fm, &body, &op).unwrap();
        assert!(result.contains("several lines"));
        assert!(!result.contains("multiple lines"));
    }

    #[test]
    fn test_replace_unique_not_found() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Replace {
            old: "nonexistent string".into(),
            new: "replacement".into(),
            mode: ReplaceMode::Unique,
        };
        let err = apply_operation(&doc, &fm, &body, &op).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_replace_unique_ambiguous() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Replace {
            old: "TODO".into(),
            new: "DONE".into(),
            mode: ReplaceMode::Unique,
        };
        let err = apply_operation(&doc, &fm, &body, &op).unwrap_err();
        assert!(err.to_string().contains("found 2 times"));
    }

    // --- replace: --all mode ---

    #[test]
    fn test_replace_all_success() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Replace {
            old: "TODO".into(),
            new: "DONE".into(),
            mode: ReplaceMode::All,
        };
        let result = apply_operation(&doc, &fm, &body, &op).unwrap();
        assert!(!result.contains("TODO"));
        assert_eq!(result.matches("DONE").count(), 2);
    }

    #[test]
    fn test_replace_all_not_found() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Replace {
            old: "NOPE".into(),
            new: "YEP".into(),
            mode: ReplaceMode::All,
        };
        let err = apply_operation(&doc, &fm, &body, &op).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    // --- replace: --count mode ---

    #[test]
    fn test_replace_count_success() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Replace {
            old: "TODO".into(),
            new: "DONE".into(),
            mode: ReplaceMode::Count(2),
        };
        let result = apply_operation(&doc, &fm, &body, &op).unwrap();
        assert!(!result.contains("TODO"));
        assert_eq!(result.matches("DONE").count(), 2);
    }

    #[test]
    fn test_replace_count_mismatch() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Replace {
            old: "TODO".into(),
            new: "DONE".into(),
            mode: ReplaceMode::Count(3),
        };
        let err = apply_operation(&doc, &fm, &body, &op).unwrap_err();
        assert!(err.to_string().contains("found 2 times"));
        assert!(err.to_string().contains("expected 3"));
    }

    // --- append ---

    #[test]
    fn test_append_preserves_frontmatter() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Append { content: "New finding added.".into() };
        let result = apply_operation(&doc, &fm, &body, &op).unwrap();

        // Frontmatter is preserved
        let (new_fm, new_body) = split(&result);
        assert_eq!(fm, new_fm);
        assert!(new_body.ends_with("New finding added."));
        assert!(new_body.contains("This is the body."));
    }

    // --- prepend ---

    #[test]
    fn test_prepend_preserves_frontmatter() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Prepend { content: "IMPORTANT:".into() };
        let result = apply_operation(&doc, &fm, &body, &op).unwrap();

        let (new_fm, new_body) = split(&result);
        assert_eq!(fm, new_fm);
        assert!(new_body.starts_with("IMPORTANT:"));
        assert!(new_body.contains("This is the body."));
    }

    // --- set ---

    #[test]
    fn test_set_replaces_entire_doc() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let new_doc = "---\ntitle: Replaced\nauthor: agent\ndomain: systems\nintent: build\nkind: spec\nstatus: active\ntags:\n- replaced\n---\nBrand new body.".to_string();
        let op = EditOp::Set { content: new_doc.clone() };
        let result = apply_operation(&doc, &fm, &body, &op).unwrap();
        assert_eq!(result, new_doc);
    }

    // --- replace on frontmatter ---

    #[test]
    fn test_replace_frontmatter_field() {
        let doc = sample_doc();
        let (fm, body) = split(&doc);
        let op = EditOp::Replace {
            old: "status: active".into(),
            new: "status: deprecated".into(),
            mode: ReplaceMode::Unique,
        };
        let result = apply_operation(&doc, &fm, &body, &op).unwrap();
        assert!(result.contains("status: deprecated"));
        assert!(!result.contains("status: active"));
    }

    // --- batch: sequential ops produce correct result ---

    #[test]
    fn test_batch_sequential_operations() {
        let doc = sample_doc();
        let (mut current_fm, mut current_body) = split(&doc);
        let mut full_doc = doc.clone();

        let ops = vec![
            EditOp::Replace {
                old: "TODO: fix this.".into(),
                new: "DONE: fixed this.".into(),
                mode: ReplaceMode::Unique,
            },
            EditOp::Append { content: "Extra line.".into() },
        ];

        for op in &ops {
            full_doc = apply_operation(&full_doc, &current_fm, &current_body, op).unwrap();
            let (fm, bd) = split(&full_doc);
            current_fm = fm;
            current_body = bd;
        }

        assert!(full_doc.contains("DONE: fixed this."));
        assert!(full_doc.contains("Extra line."));
        // Only one TODO remains
        assert_eq!(full_doc.matches("TODO").count(), 1);
    }

    // --- batch: failure rolls back (no partial application) ---

    #[test]
    fn test_batch_fails_atomically() {
        let doc = sample_doc();
        let (mut current_fm, mut current_body) = split(&doc);
        let mut full_doc = doc.clone();

        let ops = vec![
            EditOp::Replace {
                old: "This is the body.".into(),
                new: "Modified body.".into(),
                mode: ReplaceMode::Unique,
            },
            // This will fail — "nonexistent" doesn't exist
            EditOp::Replace {
                old: "nonexistent".into(),
                new: "x".into(),
                mode: ReplaceMode::Unique,
            },
        ];

        let mut failed = false;
        for op in &ops {
            match apply_operation(&full_doc, &current_fm, &current_body, op) {
                Ok(new_doc) => {
                    full_doc = new_doc;
                    let (fm, bd) = split(&full_doc);
                    current_fm = fm;
                    current_body = bd;
                }
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }

        assert!(failed);
        // In the real run() function, the original document is untouched because
        // we bail before vault.ingest(). Here we verify the error is raised.
    }

    // --- parse_single_operation ---

    #[test]
    fn test_parse_replace_all_flag() {
        let args: Vec<String> = vec!["replace", "--all", "old", "new"]
            .into_iter().map(String::from).collect();
        let op = parse_single_operation(&args, None).unwrap();
        match op {
            EditOp::Replace { old, new, mode } => {
                assert_eq!(old, "old");
                assert_eq!(new, "new");
                assert!(matches!(mode, ReplaceMode::All));
            }
            _ => panic!("expected Replace"),
        }
    }

    #[test]
    fn test_parse_replace_count_flag() {
        let args: Vec<String> = vec!["replace", "--count", "3", "old", "new"]
            .into_iter().map(String::from).collect();
        let op = parse_single_operation(&args, None).unwrap();
        match op {
            EditOp::Replace { mode, .. } => {
                assert!(matches!(mode, ReplaceMode::Count(3)));
            }
            _ => panic!("expected Replace"),
        }
    }

    #[test]
    fn test_parse_unknown_operation() {
        let args: Vec<String> = vec!["frobnicate", "x"]
            .into_iter().map(String::from).collect();
        assert!(parse_single_operation(&args, None).is_err());
    }

    #[test]
    fn test_parse_replace_missing_args() {
        let args: Vec<String> = vec!["replace", "only_one"]
            .into_iter().map(String::from).collect();
        assert!(parse_single_operation(&args, None).is_err());
    }
}
