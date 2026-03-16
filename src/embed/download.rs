use std::fs;
use std::io::{self, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use tar::Archive;

use super::MODEL_NAME;

const ORT_VERSION: &str = "1.24.2";

// ONNX Runtime download URLs by platform
struct PlatformAsset {
    url: &'static str,
    archive_prefix: &'static str,
    dylib_name: &'static str,
    sha256: Option<&'static str>,
}

fn ort_asset() -> Result<PlatformAsset> {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;

    match (os, arch) {
        ("macos", "aarch64") => Ok(PlatformAsset {
            url: "https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-osx-arm64-1.24.2.tgz",
            archive_prefix: "onnxruntime-osx-arm64-1.24.2",
            dylib_name: "libonnxruntime.1.24.2.dylib",
            sha256: None, // TODO: pin after first verified download
        }),
        ("linux", "x86_64") => Ok(PlatformAsset {
            url: "https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz",
            archive_prefix: "onnxruntime-linux-x64-1.24.2",
            dylib_name: "libonnxruntime.so.1.24.2",
            sha256: None, // TODO: pin after first verified download
        }),
        _ => bail!("unsupported platform: {os}-{arch}"),
    }
}

struct ModelFile {
    url: &'static str,
    dest: &'static str,
    sha256: Option<&'static str>,
}

const MODEL_FILES: &[ModelFile] = &[
    ModelFile {
        url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx",
        dest: "model.onnx",
        sha256: None, // ~137MB — hash populated on first verified download
    },
    ModelFile {
        url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json",
        dest: "tokenizer.json",
        sha256: None,
    },
    ModelFile {
        url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/config.json",
        dest: "config.json",
        sha256: None,
    },
];

/// Download ONNX Runtime and model files. Idempotent — skips files that exist.
pub fn run_init(vault_dir: &Path) -> Result<()> {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    eprintln!("Detecting platform... {arch}-{os}");

    download_ort(vault_dir)?;
    download_model(vault_dir)?;

    eprintln!("Embedding ready. Run `nark embed build` to backfill existing notes.");
    Ok(())
}

fn download_ort(vault_dir: &Path) -> Result<()> {
    let asset = ort_asset()?;
    let lib_dir = vault_dir.join("lib");
    let dylib_path = lib_dir.join(asset.dylib_name);

    if dylib_path.exists() {
        eprintln!("ONNX Runtime v{ORT_VERSION} already present, skipping.");
        return Ok(());
    }

    fs::create_dir_all(&lib_dir)?;

    eprintln!("Downloading ONNX Runtime v{ORT_VERSION}...");
    let data = download_with_progress(asset.url)?;

    verify_checksum(&data, asset.sha256, "ONNX Runtime archive")?;

    eprintln!("Extracting to {}...", lib_dir.display());
    extract_ort_dylib(&data, asset.archive_prefix, asset.dylib_name, &lib_dir)?;

    if dylib_path.exists() {
        eprintln!("ONNX Runtime ready.");
    } else {
        bail!("extraction failed — {} not found", dylib_path.display());
    }
    Ok(())
}

fn download_model(vault_dir: &Path) -> Result<()> {
    let model_dir = vault_dir.join("models").join(MODEL_NAME);
    fs::create_dir_all(&model_dir)?;

    for file in MODEL_FILES {
        let dest = model_dir.join(file.dest);
        if dest.exists() {
            eprintln!("{} already present, skipping.", file.dest);
            continue;
        }

        eprintln!("Downloading {}...", file.dest);
        let data = download_with_progress(file.url)?;
        verify_checksum(&data, file.sha256, file.dest)?;
        fs::write(&dest, &data)?;
        eprintln!("{} saved.", file.dest);
    }

    Ok(())
}

fn download_with_progress(url: &str) -> Result<Vec<u8>> {
    let resp = ureq::get(url).call()
        .context("HTTP request failed")?;

    let total: u64 = resp.header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let pb = if total > 0 {
        let pb = ProgressBar::new(total);
        pb.set_style(ProgressStyle::default_bar()
            .template("  [{bar:40}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("##-"));
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner()
            .template("  {spinner} {bytes}")
            .unwrap());
        pb
    };

    let mut reader = resp.into_reader();
    let mut buf = Vec::with_capacity(total as usize);
    let mut chunk = [0u8; 32768];

    loop {
        match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                pb.set_position(buf.len() as u64);
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }

    pb.finish_and_clear();
    Ok(buf)
}

fn extract_ort_dylib(
    tgz_data: &[u8],
    archive_prefix: &str,
    dylib_name: &str,
    dest_dir: &Path,
) -> Result<()> {
    let decoder = GzDecoder::new(tgz_data);
    let mut archive = Archive::new(decoder);

    let target_path = format!("{archive_prefix}/lib/{dylib_name}");

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().to_string();
        if path == target_path {
            let dest = dest_dir.join(dylib_name);
            let mut file = fs::File::create(&dest)?;
            io::copy(&mut entry, &mut file)?;
            return Ok(());
        }
    }

    bail!("dylib not found in archive at {target_path}");
}

/// Verify a downloaded file against an expected SHA-256 checksum.
/// If no expected hash is provided, logs the computed hash for future pinning.
fn verify_checksum(data: &[u8], expected: Option<&str>, label: &str) -> Result<()> {
    let actual = sha256_hex(data);
    match expected {
        Some(expected_hash) => {
            if actual != expected_hash {
                bail!(
                    "checksum mismatch for {label}\n  expected: {expected_hash}\n  got:      {actual}"
                );
            }
            eprintln!("Checksum verified for {label}.");
        }
        None => {
            eprintln!("SHA-256 for {label}: {actual} (no pinned hash — consider hardcoding)");
        }
    }
    Ok(())
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}
