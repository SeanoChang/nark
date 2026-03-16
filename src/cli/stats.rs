use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::stats;

pub fn run(vault_dir: &Path) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let s = stats::overview(&conn)?;

    let most_accessed = s.access.most_accessed.as_ref().map(|m| {
        serde_json::json!({ "title": m.title, "count": m.count })
    });

    let out = serde_json::json!({
        "total_notes": s.total_notes,
        "total_versions": s.total_versions,
        "by_domain": s.by_domain.iter().map(|f| {
            serde_json::json!({ "domain": f.label, "count": f.count })
        }).collect::<Vec<_>>(),
        "by_kind": s.by_kind.iter().map(|f| {
            serde_json::json!({ "kind": f.label, "count": f.count })
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
        "access": {
            "total_reads": s.access.total_reads,
            "most_accessed": most_accessed,
            "never_read": s.access.never_read,
        },
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
