use anyhow::Result;
use std::path::Path;

use crate::vault::fs::Vault;
use crate::db;

pub fn run(vault_dir: &Path) -> Result<()> {
    let vault = Vault::new(vault_dir.to_path_buf());
    vault.init_dirs()?;

    // EXEMPT from the write lock: `init` is the bootstrap that *creates* the
    // registry (and the vault dirs). It runs before any concurrent writer could
    // exist, and gating it behind the advisory lock would add no protection
    // (and the dedicated lock file lives in the very vault being initialized).
    // So it stays on the plain `open_registry`. Every *mutating* command
    // (jot/write/edit/append/delete/tag/link/retract/rollback/embed build/reset)
    // routes through `open_registry_guarded` instead.
    db::open_registry(vault_dir)?;

    println!("Initialized vault at {}", vault_dir.display());
    Ok(())
}
