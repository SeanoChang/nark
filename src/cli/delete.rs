use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::delete::{self, DeletedNote};
use crate::vault::fs::Vault;

pub fn run(vault_dir: &Path, ids: Vec<String>, force: bool, recursive: bool) -> Result<()> {
    // Write command: hold the advisory write lock for the whole delete. A
    // conflicting RW open is refused with the plain write-locked error; the
    // handle derefs to the `Connection` so the logic below is unchanged, and
    // the lock is released when the handle drops at end of function.
    let conn = db::open_registry_guarded(vault_dir)?;

    let notes = delete::validate_ids(&conn, &ids)?;

    let mode = if force && recursive {
        purge(&conn, &notes, vault_dir)?;
        "purge"
    } else if force {
        delete::hard_delete(&conn, &notes)?;
        "hard_delete"
    } else {
        delete::soft_delete(&conn, &notes)?;
        "retract"
    };

    let out = serde_json::json!({
        "deleted": notes.len(),
        "mode": mode,
        "notes": notes.iter().map(|n| serde_json::json!({
            "id": n.note_id,
            "title": n.title,
        })).collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn purge(conn: &rusqlite::Connection, notes: &[DeletedNote], vault_dir: &Path) -> Result<()> {
    delete::hard_delete(conn, notes)?;

    let vault = Vault::new(vault_dir.to_path_buf());
    for note in notes {
        vault.remove_object("objects/fm", &note.fm_hash, "yaml")?;
        vault.remove_object("objects/md", &note.md_hash, "md")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_vault() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nark-delete-lock-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp vault");
        dir
    }

    /// Spot-check: `delete` routes through the guarded open, so while the write
    /// lock is held it is refused with the plain write-locked error (before any
    /// id validation).
    #[test]
    fn delete_refuses_when_write_locked() {
        let dir = fresh_vault();
        let held = db::open_registry_guarded(&dir).expect("hold the write lock");

        let err = run(&dir, vec!["whatever".to_string()], false, false)
            .expect_err("delete must be refused while the write lock is held");
        assert_eq!(
            err.to_string(),
            "registry is write-locked by another process"
        );

        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
