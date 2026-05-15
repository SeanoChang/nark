//! Integration smoke test — runs the full harness against synthetic-tiny.
//! This is what CI Lane 1 (the PR check) actually executes.

use assert_cmd::Command;
use serde_json::Value;
use std::path::PathBuf;
use tempfile::tempdir;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

#[test]
fn smoke_fts5_and_nark_run_against_synthetic_tiny() {
    let out_dir = tempdir().unwrap();
    let mut cmd = Command::cargo_bin("nark-bench").unwrap();
    cmd.current_dir(workspace_root())
        .args(["run", "--task", "ir",
               "--systems", "fts5,nark",
               "--corpus", "synthetic-tiny",
               "--output"]).arg(out_dir.path());

    let out = cmd.assert().success().get_output().clone();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("ir-fts5-default.json"), "stderr did not mention fts5 result: {}", stderr);
    assert!(stderr.contains("ir-nark-default.json"), "stderr did not mention nark result: {}", stderr);

    for system in ["fts5", "nark"] {
        let path = out_dir.path().join(format!("ir-{}-default.json", system));
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("could not read {}: {}", path.display(), e)
        });
        let v: Value = serde_json::from_str(&content).unwrap();

        // Required top-level fields
        assert_eq!(v["schema_version"], "1");
        assert_eq!(v["task"], "ir");
        assert_eq!(v["system"], system);
        assert!(v["ir"]["recall_at_5"].is_number());

        // Metrics in valid range
        let r5 = v["ir"]["recall_at_5"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&r5), "recall_at_5 out of range: {}", r5);

        // Per-class breakdowns exist
        assert!(v["ir_per_class"]["single_hop"].is_object(), "missing single_hop breakdown");

        // Both adapters should now run all 10 queries cleanly. Previously FTS5
        // erroed on q10 due to hyphen-as-column-prefix; the FTS5 adapter now
        // sanitizes queries before MATCH, so this assertion catches any future
        // regression that re-introduces silent query failures.
        let errors = v["errors"].as_array().expect("errors field must be present");
        assert!(
            errors.is_empty(),
            "{} run had unexpected errors (regression?): {:?}",
            system, errors
        );
    }
}
