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

pub struct NarkAdapter {
    workdir: Option<PathBuf>,
    staging: Option<PathBuf>,
    /// Path to the nark binary. Resolved at setup time.
    nark_bin: Option<PathBuf>,
    /// nark-assigned UUID → bench-managed document id.
    uuid_to_bench_id: HashMap<String, String>,
}

impl NarkAdapter {
    pub fn new() -> Self {
        Self {
            workdir: None,
            staging: None,
            nark_bin: None,
            uuid_to_bench_id: HashMap::new(),
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
    fn default() -> Self { Self::new() }
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
        // The current nark CLI does not implement `--version`; report our pinned
        // crate version of nark instead, taken from the binary's parent workspace.
        // (If a later nark adds --version, replace this with `self.run_nark(&["--version"])`
        // and trim the output.)
        Ok(format!("nark {}", env!("CARGO_PKG_VERSION")))
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

        // Initialize the vault
        self.run_nark(&["init"])?;
        Ok(())
    }

    fn write(&mut self, doc: &Document) -> Result<WriteMetrics> {
        let t0 = Instant::now();
        let staging = self.staging.as_ref().ok_or_else(|| anyhow!("setup not called"))?;
        let path = staging.join(format!("{}.md", doc.id));
        std::fs::write(&path, &doc.body)?;
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
            latency_ms: t0.elapsed().as_millis() as u64,
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
            latency_ms: t0.elapsed().as_millis() as u64,
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

/// Find the nark binary built in this workspace. Looks at `target/debug/nark`
/// relative to CARGO_MANIFEST_DIR, then `target/release/nark`, then falls back
/// to `nark` on PATH.
fn locate_nark_bin() -> Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().ok_or_else(|| anyhow!("no workspace root"))?;
    let target_debug = workspace_root.join("target/debug/nark");
    if target_debug.exists() {
        return Ok(target_debug);
    }
    let target_release = workspace_root.join("target/release/nark");
    if target_release.exists() {
        return Ok(target_release);
    }
    Ok(PathBuf::from("nark"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    /// Smoke test — this is gated on the nark binary being built. The integration
    /// test (`tests/smoke.rs`) builds nark first; here we only verify that if the
    /// binary is found, ingest+search round-trips with the UUID→bench-id mapping.
    #[test]
    #[ignore = "requires built nark binary; run via the integration smoke test"]
    fn nark_smoke_write_then_search() {
        let dir = tempdir().unwrap();
        let mut a = NarkAdapter::new();
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
