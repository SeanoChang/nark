use rusqlite::Connection;
use rusqlite_migration::Migrations;
use std::path::Path;
use anyhow::Result;
use std::sync::LazyLock;
use include_dir::{include_dir, Dir};

pub const DEFAULT_AGENT_ID: &str = "noah";
pub const DEFAULT_CLIENT_ID: &str = "cli_default";

static MIGRATIONS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/migrations");

static MIGRATIONS: LazyLock<Migrations<'static>> = LazyLock::new(|| {
    Migrations::from_directory(&MIGRATIONS_DIR).unwrap()
});

pub fn open_registry(vault_dir: &Path) -> Result<Connection> {
    let db_path = vault_dir.join("registry.db");
    let mut conn = Connection::open(db_path)?;

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA foreign_keys=ON;"
    )?;

    MIGRATIONS.to_latest(&mut conn)?;

    seed_defaults(&conn)?;

    Ok(conn)
}

fn seed_defaults(conn: &Connection) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();

    conn.execute(
        "INSERT OR IGNORE INTO clients (client_id, name, api_key_hash, is_admin, created_at)
         VALUES (?1, ?2, '', 1, ?3)",
        rusqlite::params![DEFAULT_CLIENT_ID, DEFAULT_CLIENT_ID, now],
    )?;

    conn.execute(
        "INSERT OR IGNORE INTO agents (agent_id, name, namespace, role, can_write_public, registered_by, registered_at)
         VALUES (?1, ?2, 'lib', 'admin', 1, ?3, ?4)",
        rusqlite::params![DEFAULT_AGENT_ID, DEFAULT_AGENT_ID, DEFAULT_CLIENT_ID, now],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_migration() {
        let mut conn = Connection::open_in_memory().unwrap();
        MIGRATIONS.to_latest(&mut conn).unwrap();

        let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type='table'").unwrap();
        let table_names: Vec<String> = stmt.query_map([], |row| row.get(0)).unwrap()
            .map(|res| res.unwrap())
            .collect();

        assert!(table_names.contains(&"notes".to_string()));
        assert!(table_names.contains(&"note_versions".to_string()));
    }
}
