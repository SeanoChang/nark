use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::stats;

pub fn run(vault_dir: &Path) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let s = stats::overview(&conn)?;

    let out = serde_json::json!({
        "total_notes": s.total_notes,
        "total_versions": s.total_versions,
        "by_domain": s.by_domain.iter().map(|f| {
            serde_json::json!({ "domain": f.label, "count": f.count })
        }).collect::<Vec<_>>(),
        "by_kind": s.by_kind.iter().map(|f| {
            serde_json::json!({ "kind": f.label, "count": f.count })
        }).collect::<Vec<_>>(),
        "by_trust": s.by_trust.iter().map(|f| {
            serde_json::json!({ "trust": f.label, "count": f.count })
        }).collect::<Vec<_>>(),
        "recent": s.recent.iter().map(|n| {
            serde_json::json!({
                "id": n.note_id,
                "title": n.title,
                "domain": n.domain,
                "intent": n.intent,
                "kind": n.kind,
                "updated_at": n.updated_at,
            })
        }).collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
