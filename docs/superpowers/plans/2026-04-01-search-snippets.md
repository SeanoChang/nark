# Search Snippet Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `nark search` return non-empty snippets for every hit, regardless of discovery method (BM25, cosine, graph).

**Architecture:** CLI-layer hybrid fallback in `cli/search.rs`. After the search pipeline returns hits, fill empty snippets by (1) trying FTS5 `snippet()` with the original query, then (2) falling back to first 150 chars of note body from vault. Pipeline (`registry/search.rs`) is unchanged.

**Tech Stack:** Rust, rusqlite (FTS5 snippet function), existing vault I/O

---

### Task 1: Extract `truncate_at_word` to shared util

**Files:**
- Create: `src/cli/util.rs`
- Modify: `src/cli/mod.rs` (add `pub mod util;`)
- Modify: `src/cli/orient.rs:1-9` (replace local fn with import)

- [ ] **Step 1: Create `src/cli/util.rs` with the function**

```rust
/// Truncate a string at a word boundary, respecting multi-byte UTF-8.
pub fn truncate_at_word(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let boundary = s.char_indices()
        .take_while(|(i, _)| *i < max)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    match s[..boundary].rfind(' ') {
        Some(i) => &s[..i],
        None => &s[..boundary],
    }
}
```

- [ ] **Step 2: Add `pub mod util;` to `src/cli/mod.rs`**

Add after the existing module declarations at the top of `src/cli/mod.rs`. Find the block of `mod` statements and add `pub mod util;` there.

- [ ] **Step 3: Update `src/cli/orient.rs` to use the shared function**

Remove the local `truncate_at_word` function (lines 127-141) and add this import at the top:

```rust
use crate::cli::util::truncate_at_word;
```

- [ ] **Step 4: Verify it compiles and tests pass**

Run: `cargo test`
Expected: All existing tests pass. No behavior change.

- [ ] **Step 5: Commit**

```bash
git add src/cli/util.rs src/cli/mod.rs src/cli/orient.rs
git commit -m "refactor: extract truncate_at_word to shared cli util"
```

---

### Task 2: Add `fill_missing_snippets` to `cli/search.rs`

**Files:**
- Modify: `src/cli/search.rs:1-9` (add imports)
- Modify: `src/cli/search.rs:80-93` (call fill function, make hits mutable)

- [ ] **Step 1: Add imports to `src/cli/search.rs`**

Add these imports to the existing import block at the top of the file:

```rust
use crate::cli::util::truncate_at_word;
use crate::registry::resolve;
use crate::vault::fs::Vault;
```

- [ ] **Step 2: Add the `fill_missing_snippets` function**

Add this function at the bottom of `src/cli/search.rs`, before the `#[cfg(test)]` block:

```rust
/// For search hits with empty snippets, try FTS5 snippet with the original query,
/// then fall back to first 150 chars of note body from vault.
fn fill_missing_snippets(
    conn: &rusqlite::Connection,
    vault: &Vault,
    query: &str,
    hits: &mut [search::SearchHit],
) {
    for hit in hits.iter_mut() {
        if !hit.snippet.is_empty() {
            continue;
        }

        // Try FTS5 snippet if we have a query
        if !query.is_empty() {
            if let Ok(snippet) = try_fts_snippet(conn, query, &hit.note_id) {
                hit.snippet = snippet;
                continue;
            }
        }

        // Fallback: first 150 chars of body from vault
        if let Ok(body) = read_body_preview(conn, vault, &hit.note_id) {
            hit.snippet = body;
        }
    }
}

/// Try to get an FTS5 snippet for a specific note using the original query.
fn try_fts_snippet(conn: &rusqlite::Connection, query: &str, note_id: &str) -> Result<String> {
    let mut stmt = conn.prepare(
        "SELECT snippet(note_text, 2, '[', ']', '...', 32)
         FROM note_text
         WHERE note_text MATCH ?1 AND note_id = ?2"
    )?;
    let snippet: String = stmt.query_row(rusqlite::params![query, note_id], |row| row.get(0))?;
    if snippet.is_empty() {
        bail!("empty snippet");
    }
    Ok(snippet)
}

/// Read the first 150 chars of a note's body from the vault.
fn read_body_preview(conn: &rusqlite::Connection, vault: &Vault, note_id: &str) -> Result<String> {
    let refs = resolve::get_ref(conn, note_id)?;
    let body = vault.read_object("objects/md", &refs.md_hash, "md")?;
    Ok(truncate_at_word(&body, 150).trim().to_string())
}
```

- [ ] **Step 3: Wire it into the `run` function**

In the `run` function, change the section after `search::search()` returns (around line 80-93). Make `hits` mutable, create a `Vault`, and call `fill_missing_snippets`:

Replace:
```rust
    let hits = search::search(&conn, query, &filters, &cfg.search, cosine_ctx.as_ref(), mode)?;

    let results: Vec<serde_json::Value> = hits.iter().map(|h| {
```

With:
```rust
    let mut hits = search::search(&conn, query, &filters, &cfg.search, cosine_ctx.as_ref(), mode)?;

    let vault = Vault::new(vault_dir.to_path_buf());
    fill_missing_snippets(&conn, &vault, query, &mut hits);

    let results: Vec<serde_json::Value> = hits.iter().map(|h| {
```

- [ ] **Step 4: Verify it compiles and tests pass**

Run: `cargo test`
Expected: All existing tests pass.

- [ ] **Step 5: Manual test with the live vault**

Run: `cargo run -- search "backup" --limit 5 2>/dev/null`
Expected: Every result in the JSON output has a non-empty `"snippet"` field.

Run: `cargo run -- search --domain systems --limit 5 2>/dev/null`
Expected: Filter-only results also have non-empty snippets (body previews).

- [ ] **Step 6: Commit**

```bash
git add src/cli/search.rs
git commit -m "feat: add snippet fallback for non-FTS search hits"
```
