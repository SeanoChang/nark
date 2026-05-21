//! nark adapter — drives the actual nark CLI as a subprocess.
//!
//! We pass --vault-dir to point at an isolated workdir per benchmark run, write
//! markdown files to a staging directory inside the workdir, and ingest them via
//! `nark write <path>`. Search calls `nark search <query>` and parses the JSON.
//!
//! The bench tracks documents by file-stem (e.g. "n01"); nark assigns its own
//! UUIDs. We parse the `nark write` JSON output to learn the assigned UUID per
//! document and keep a `nark_uuid → bench_id` map so search results translate
//! back to bench IDs before being returned to the harness.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::protocol::{Adapter, Document, SearchMetrics, SearchResult, WriteMetrics};

/// Pinned version of the `nark` crate this adapter targets. Kept in sync with
/// the workspace `nark` crate's Cargo.toml via the `nark_version_matches_workspace`
/// test below. Bump both together when nark releases.
const NARK_VERSION: &str = "0.13.0";

pub struct NarkAdapter {
    workdir: Option<PathBuf>,
    staging: Option<PathBuf>,
    /// Path to the nark binary. Resolved at setup time.
    nark_bin: Option<PathBuf>,
    /// nark-assigned UUID → bench-managed document id.
    uuid_to_bench_id: HashMap<String, String>,
    /// Path to the shared model cache. If `None`, nark runs in BM25-only
    /// mode (no embedding files staged into workdir).
    model_cache: Option<PathBuf>,
}

impl NarkAdapter {
    pub fn new(model_cache: Option<PathBuf>) -> Self {
        Self {
            workdir: None,
            staging: None,
            nark_bin: None,
            uuid_to_bench_id: HashMap::new(),
            model_cache,
        }
    }

    fn run_nark(&self, args: &[&str]) -> Result<std::process::Output> {
        let bin = self.nark_bin.as_ref().ok_or_else(|| anyhow!("nark binary not located"))?;
        let workdir = self.workdir.as_ref().ok_or_else(|| anyhow!("setup not called"))?;
        let workdir_str = workdir.to_string_lossy().to_string();
        let mut cmd = Command::new(bin);
        cmd.arg("--vault-dir").arg(&workdir_str);
        for a in args {
            cmd.arg(a);
        }
        let output = cmd.output().with_context(|| format!("failed to spawn nark with args {:?}", args))?;
        if !output.status.success() {
            return Err(anyhow!(
                "nark {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(output)
    }
}

impl Default for NarkAdapter {
    fn default() -> Self { Self::new(None) }
}

#[derive(Debug, Deserialize)]
struct NarkWriteOut {
    #[serde(default)]
    wrote: u64,
    #[serde(default)]
    notes: Vec<NarkWriteNote>,
}

#[derive(Debug, Deserialize)]
struct NarkWriteNote {
    id: String,
    #[allow(dead_code)]
    title: String,
    /// Source file path nark ingested from. Populated by `nark write --json`
    /// and used to map nark UUIDs back to bench doc IDs on batch writes.
    #[serde(default)]
    file: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NarkSearchOut {
    #[allow(dead_code)]
    #[serde(default)]
    query: String,
    #[allow(dead_code)]
    #[serde(default)]
    hits: usize,
    #[serde(default)]
    results: Vec<NarkHit>,
}

#[derive(Debug, Deserialize)]
struct NarkHit {
    id: String,
    #[serde(default)]
    snippet: String,
    rank: f64,
}

impl Adapter for NarkAdapter {
    fn name(&self) -> &str { "nark" }

    fn version(&self) -> Result<String> {
        // The current nark CLI does not implement `--version`. We pin the nark
        // version as a constant here; a test (`nark_version_matches_workspace`)
        // verifies it stays in sync with the parent crate's Cargo.toml.
        // When nark adds --version, swap this for `self.run_nark(&["--version"])`.
        Ok(format!("nark {}", NARK_VERSION))
    }

    fn setup(&mut self, workdir: &Path) -> Result<()> {
        let workdir = workdir.to_path_buf();
        let staging = workdir.join("_staging");
        std::fs::create_dir_all(&staging)?;
        let nark_bin = locate_nark_bin()?;
        self.workdir = Some(workdir.clone());
        self.staging = Some(staging);
        self.nark_bin = Some(nark_bin);
        self.uuid_to_bench_id.clear();

        // Stage model files into workdir so nark write produces embeddings
        // inline. If staging fails (cache missing, download failed, etc),
        // fall back to BM25-only — adapter still functions, just without
        // the cosine component. We log to stderr so operator sees what happened.
        if let Some(cache) = &self.model_cache {
            if let Err(e) = crate::model_cache::stage_into(cache, &workdir) {
                eprintln!("nark adapter: model staging failed, continuing without embeddings: {}", e);
            }
        }

        // Initialize the vault
        self.run_nark(&["init"])?;
        Ok(())
    }

    fn write(&mut self, doc: &Document) -> Result<WriteMetrics> {
        let t0 = Instant::now();
        let staging = self.staging.as_ref().ok_or_else(|| anyhow!("setup not called"))?;
        let path = staging.join(format!("{}.md", doc.id));
        let body = ensure_frontmatter(&doc.id, &doc.body);
        std::fs::write(&path, &body)?;
        let path_str = path.to_string_lossy().to_string();
        let out = self.run_nark(&["write", &path_str])?;
        let parsed: NarkWriteOut = serde_json::from_slice(&out.stdout)
            .with_context(|| format!(
                "failed to parse nark write JSON for bench_id={}: {}",
                doc.id, String::from_utf8_lossy(&out.stdout)
            ))?;
        if parsed.wrote != 1 || parsed.notes.len() != 1 {
            return Err(anyhow!(
                "nark write returned unexpected count: wrote={}, notes.len={}",
                parsed.wrote, parsed.notes.len()
            ));
        }
        let uuid = parsed.notes.into_iter().next().unwrap().id;
        self.uuid_to_bench_id.insert(uuid, doc.id.clone());
        Ok(WriteMetrics {
            latency_us: t0.elapsed().as_micros() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        })
    }

    fn write_batch(&mut self, docs: &[Document]) -> Result<WriteMetrics> {
        // Bulk ingest in a single subprocess call. nark write accepts a
        // directory and walks it for *.md, which sidesteps the ~700ms/turn
        // process-spawn cost that made LongMemEval-S unworkable per-turn.
        let t0 = Instant::now();
        let staging = self.staging.as_ref()
            .ok_or_else(|| anyhow!("setup not called"))?
            .clone();

        // Stage every doc to disk. Track filename → bench_id so we can
        // reconstruct the mapping from the `file` field nark returns.
        let mut filename_to_bench_id: HashMap<String, String> = HashMap::new();
        for doc in docs {
            let filename = format!("{}.md", doc.id);
            let path = staging.join(&filename);
            let body = ensure_frontmatter(&doc.id, &doc.body);
            std::fs::write(&path, &body)?;
            filename_to_bench_id.insert(filename, doc.id.clone());
        }

        let staging_str = staging.to_string_lossy().to_string();
        let out = self.run_nark(&["write", &staging_str])?;
        let parsed: NarkWriteOut = serde_json::from_slice(&out.stdout)
            .with_context(|| format!(
                "failed to parse batch nark write JSON ({} docs): {}",
                docs.len(),
                String::from_utf8_lossy(&out.stdout)
            ))?;

        for note in parsed.notes {
            let basename = note.file.as_deref()
                .and_then(|f| std::path::Path::new(f).file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if let Some(bench_id) = filename_to_bench_id.get(basename) {
                self.uuid_to_bench_id.insert(note.id, bench_id.clone());
            }
        }

        Ok(WriteMetrics {
            latency_us: t0.elapsed().as_micros() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        })
    }

    fn search(&mut self, query: &str, k: usize) -> Result<(Vec<SearchResult>, SearchMetrics)> {
        let t0 = Instant::now();
        let limit = k.to_string();
        let out = self.run_nark(&["search", query, "--limit", &limit])?;
        let parsed: NarkSearchOut = serde_json::from_slice(&out.stdout)
            .with_context(|| format!("failed to parse nark search JSON: {}", String::from_utf8_lossy(&out.stdout)))?;
        let results = parsed.results.into_iter()
            .filter_map(|h| {
                // Drop any hit whose UUID we did not ingest (defensive: shouldn't happen
                // in practice since each adapter has its own vault, but a stray hit
                // would otherwise score as a phantom miss).
                let bench_id = self.uuid_to_bench_id.get(&h.id).cloned()?;
                Some(SearchResult {
                    doc_id: bench_id,
                    score: h.rank as f32,
                    snippet: if h.snippet.is_empty() { None } else { Some(h.snippet) },
                })
            })
            .collect();
        Ok((results, SearchMetrics {
            latency_us: t0.elapsed().as_micros() as u64,
            llm_tokens_in: 0,
            llm_tokens_out: 0,
        }))
    }

    fn teardown(&mut self) -> Result<()> {
        self.workdir = None;
        self.staging = None;
        self.nark_bin = None;
        self.uuid_to_bench_id.clear();
        Ok(())
    }
}

/// Wrap a raw body in nark's required YAML frontmatter unless it already has one.
/// nark refuses notes without frontmatter, so the bench has to synthesize defaults
/// for corpora (LongMemEval/LOCOMO turns) that don't carry one. Detection: if the
/// body starts with `---\n` it's assumed to already be a full note.
fn ensure_frontmatter(doc_id: &str, body: &str) -> String {
    if body.starts_with("---\n") || body.starts_with("---\r\n") {
        return body.to_string();
    }
    format!(
        "---\n\
         title: bench {doc_id}\n\
         author: bench\n\
         domain: conversation\n\
         intent: reference\n\
         kind: note\n\
         status: active\n\
         tags: []\n\
         ---\n\n{body}\n",
        doc_id = doc_id,
        body = body,
    )
}

/// Find the nark binary built in this workspace. Prefers the release binary
/// over debug because most bench invocations run `cargo build -p nark --release`
/// and we want consistent latency measurements between local runs and CI.
/// Falls back to `nark` on PATH if neither exists.
fn locate_nark_bin() -> Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().ok_or_else(|| anyhow!("no workspace root"))?;
    let target_release = workspace_root.join("target/release/nark");
    if target_release.exists() {
        return Ok(target_release);
    }
    let target_debug = workspace_root.join("target/debug/nark");
    if target_debug.exists() {
        return Ok(target_debug);
    }
    Ok(PathBuf::from("nark"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    /// Guards against `NARK_VERSION` drifting from the actual nark crate version.
    /// Reads the parent `Cargo.toml` and parses the `[package].version` line.
    #[test]
    fn nark_version_matches_workspace() {
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest.parent().unwrap();
        let nark_cargo_toml = std::fs::read_to_string(workspace_root.join("Cargo.toml")).unwrap();
        let mut in_package = false;
        let mut found = None;
        for line in nark_cargo_toml.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                in_package = trimmed == "[package]";
                continue;
            }
            if in_package {
                if let Some(rest) = trimmed.strip_prefix("version") {
                    let rest = rest.trim_start().strip_prefix('=').unwrap().trim();
                    let version = rest.trim_matches('"').trim_matches('\'');
                    found = Some(version.to_string());
                    break;
                }
            }
        }
        let actual = found.expect("could not find [package].version in workspace Cargo.toml");
        assert_eq!(
            actual, NARK_VERSION,
            "NARK_VERSION constant ({}) is out of sync with workspace nark Cargo.toml ({}). \
             Bump NARK_VERSION in bench/src/adapters/nark.rs to match.",
            NARK_VERSION, actual
        );
    }

    /// Smoke test — this is gated on the nark binary being built. The integration
    /// test (`tests/smoke.rs`) builds nark first; here we only verify that if the
    /// binary is found, ingest+search round-trips with the UUID→bench-id mapping.
    #[test]
    #[ignore = "requires built nark binary; run via the integration smoke test"]
    fn nark_smoke_write_then_search() {
        let dir = tempdir().unwrap();
        let mut a = NarkAdapter::new(None);
        a.setup(dir.path()).unwrap();

        let body = "---\ntitle: Test note\nauthor: bench\ndomain: games\nintent: reference\nkind: note\nstatus: active\ntags: [chess]\n---\n\nThe sicilian defense begins with 1.e4 c5.\n";
        let d = Document { id: "n01".into(), body: body.into(), metadata: json!({}) };
        a.write(&d).unwrap();

        let (results, _) = a.search("sicilian defense", 5).unwrap();
        assert!(!results.is_empty(), "expected at least one nark hit");
        assert_eq!(results[0].doc_id, "n01", "expected bench-id translation; got {}", results[0].doc_id);
        a.teardown().unwrap();
    }
}
