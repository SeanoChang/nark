//! Integration smoke test — runs the full harness against synthetic-tiny
//! for all three adapters (fts5, nark, vector). This is what CI Lane 1
//! (the PR check) actually executes.
//!
//! On first run after the model cache is empty, this downloads the
//! ONNX runtime + nomic-embed model (~270MB, 30-60s). Subsequent runs
//! are fast.

use assert_cmd::Command;
use serde_json::Value;
use std::path::PathBuf;
use tempfile::tempdir;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

#[test]
fn smoke_fts5_nark_vector_all_run_against_synthetic_tiny() {
    let out_dir = tempdir().unwrap();
    let mut cmd = Command::cargo_bin("nark-bench").unwrap();
    cmd.current_dir(workspace_root())
        .args(["run", "--task", "ir",
               "--systems", "fts5,nark,vector",
               "--corpus", "synthetic-tiny",
               "--output"]).arg(out_dir.path());

    let out = cmd.assert().success().get_output().clone();
    let stderr = String::from_utf8_lossy(&out.stderr);
    for system in ["fts5", "nark", "vector"] {
        assert!(
            stderr.contains(&format!("ir-{}-default.json", system)),
            "stderr did not mention {} result: {}", system, stderr
        );
    }

    for system in ["fts5", "nark", "vector"] {
        let path = out_dir.path().join(format!("ir-{}-default.json", system));
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("could not read {}: {}", path.display(), e)
        });
        let v: Value = serde_json::from_str(&content).unwrap();

        // Schema bumped to "2" in Task 3
        assert_eq!(v["schema_version"], "2", "{} schema_version mismatch", system);
        assert_eq!(v["task"], "ir");
        assert_eq!(v["system"], system);
        assert!(v["ir"]["recall_at_5"].is_number());

        // Metrics in valid range
        let r5 = v["ir"]["recall_at_5"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&r5), "{} recall_at_5 out of range: {}", system, r5);

        // Per-class breakdowns
        assert!(v["ir_per_class"]["single_hop"].is_object(),
            "{} missing single_hop breakdown", system);

        // All three adapters should produce zero errors. FTS5's q10
        // sanitizer (Phase 1a fix) handles the column-prefix issue;
        // nark with embeddings and vector both run all queries cleanly.
        let errors = v["errors"].as_array().expect("errors field required");
        assert!(
            errors.is_empty(),
            "{} run had unexpected errors (regression?): {:?}", system, errors
        );
    }
}
