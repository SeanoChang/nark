use anyhow::Result;
use std::path::Path;

use crate::vault::fs::Vault;
use crate::db;

pub fn run(vault_dir: &Path) -> Result<()> {
    let vault = Vault::new(vault_dir.to_path_buf());
    vault.init_dirs()?;

    db::open_registry(vault_dir)?;

    println!("Initialized vault at {}", vault_dir.display());
    Ok(())
}
