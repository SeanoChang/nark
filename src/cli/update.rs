use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;

pub fn run() -> Result<()> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // git pull
    let pull = Command::new("git")
        .args(["pull", "origin", "main"])
        .current_dir(&repo)
        .output()?;

    if !pull.status.success() {
        let msg = String::from_utf8_lossy(&pull.stderr);
        let out = serde_json::json!({"status": "error", "step": "git pull", "message": msg.trim()});
        println!("{}", serde_json::to_string_pretty(&out)?);
        anyhow::bail!("git pull failed");
    }

    // cargo build --release
    let build = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&repo)
        .output()?;

    if !build.status.success() {
        let msg = String::from_utf8_lossy(&build.stderr);
        let out = serde_json::json!({"status": "error", "step": "cargo build", "message": msg.trim()});
        println!("{}", serde_json::to_string_pretty(&out)?);
        anyhow::bail!("cargo build failed");
    }

    // get commit hash
    let rev = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(&repo)
        .output()?;
    let commit = String::from_utf8_lossy(&rev.stdout).trim().to_string();

    let out = serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "commit": commit,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);

    Ok(())
}
