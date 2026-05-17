//! Shared model cache for adapters that need nark's ONNX embedding model.
//!
//! Default cache location: `~/.cache/nark-bench/models/`. Override with the
//! `NARK_BENCH_MODEL_CACHE` env var. First call to `ensure_ready` downloads
//! the ORT dylib + nomic-embed model (~270MB, 30–60s on a fast connection).
//! Subsequent calls are instant (marker-file existence check).
//!
//! Adapters that need embeddings call `stage_into(cache, workdir)` during
//! `setup()` to hard-link the model files into their per-run workdir, where
//! nark's `init_embedding` expects them.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const CACHE_ENV: &str = "NARK_BENCH_MODEL_CACHE";

/// Resolve the cache root. Honors `NARK_BENCH_MODEL_CACHE` env var; falls
/// back to `~/.cache/nark-bench/models/`.
pub fn cache_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var(CACHE_ENV) {
        return Ok(PathBuf::from(p));
    }
    let home = dirs::home_dir().context("no home directory")?;
    Ok(home.join(".cache/nark-bench/models"))
}

/// Idempotent: downloads ORT dylib + nomic-embed model into `cache` if the
/// expected marker files aren't already present.
pub fn ensure_ready(cache: &Path) -> Result<()> {
    let lib_marker = cache.join("lib").join(nark::embed::onnx_dylib_name());
    let model_marker = cache
        .join("models")
        .join(nark::embed::MODEL_NAME)
        .join("model.onnx");
    if lib_marker.exists() && model_marker.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(cache)?;
    nark::embed::download::install_into(cache)
        .context("failed to download embedding model into cache")?;
    Ok(())
}

/// Stage model files from `cache` into `workdir` via hard-links (cheap, no
/// copying). Falls back to copy if hard-link fails (e.g. cross-volume).
///
/// Staged files are treated as read-only. Currently safe only for serial
/// callers — the `!to.exists()` check in `link_tree` is TOCTOU under
/// concurrent invocation. The runner dispatches adapters sequentially so
/// this is fine today.
pub fn stage_into(cache: &Path, workdir: &Path) -> Result<()> {
    for sub in &["lib", "models"] {
        link_tree(&cache.join(sub), &workdir.join(sub))?;
    }
    Ok(())
}

fn link_tree(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            link_tree(&from, &to)?;
        } else if !to.exists() {
            std::fs::hard_link(&from, &to)
                .or_else(|_| std::fs::copy(&from, &to).map(|_| ()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn cache_root_honors_env_var() {
        // SAFETY: tests in the same process share env; isolate via tempdir-derived path
        let tmp = tempdir().unwrap();
        let path_str = tmp.path().to_string_lossy().to_string();
        // SAFETY: single-threaded test, no other test reads CACHE_ENV
        unsafe { std::env::set_var(CACHE_ENV, &path_str); }
        let resolved = cache_root().unwrap();
        unsafe { std::env::remove_var(CACHE_ENV); }
        assert_eq!(resolved, PathBuf::from(&path_str));
    }

    #[test]
    fn stage_into_hardlinks_files_recursively() {
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();

        // Build a small fixture tree:
        //   src/lib/foo.txt
        //   src/models/nested/bar.txt
        std::fs::create_dir_all(src.path().join("lib")).unwrap();
        std::fs::create_dir_all(src.path().join("models").join("nested")).unwrap();
        std::fs::write(src.path().join("lib").join("foo.txt"), b"hello").unwrap();
        std::fs::write(src.path().join("models").join("nested").join("bar.txt"), b"world").unwrap();

        stage_into(src.path(), dst.path()).unwrap();

        assert!(dst.path().join("lib").join("foo.txt").exists());
        assert!(dst.path().join("models").join("nested").join("bar.txt").exists());
        // Content matches
        assert_eq!(std::fs::read(dst.path().join("lib").join("foo.txt")).unwrap(), b"hello");
    }

    #[test]
    fn stage_into_is_idempotent() {
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();
        std::fs::create_dir_all(src.path().join("lib")).unwrap();
        std::fs::create_dir_all(src.path().join("models")).unwrap();
        std::fs::write(src.path().join("lib").join("a"), b"x").unwrap();

        // First call creates
        stage_into(src.path(), dst.path()).unwrap();
        // Second call should not error on existing files
        stage_into(src.path(), dst.path()).unwrap();

        assert!(dst.path().join("lib").join("a").exists());
    }
}
