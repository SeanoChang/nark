use rusqlite::Connection;
use rusqlite_migration::{Migrations, M};

const MIGRATIONS: Migrations = Migrations::new(vec![
    M::up(include_str!("../../migrations/001_create_tables.sql")),
]);

pub fn migrate(conn: &mut Connection) -> Result<(), rusqlite_migration::Error> {
    MIGRATIONS.to_latest(conn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_migration() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();

        // Check if tables exist
        let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type='table'").unwrap();
        let table_names: Vec<String> = stmt.query_map([], |row| row.get(0)).unwrap()
            .map(|res| res.unwrap())
            .collect();

        assert!(table_names.contains(&"notes".to_string()));
        assert!(table_names.contains(&"versions".to_string()));
    }
}
