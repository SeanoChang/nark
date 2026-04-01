# Search Snippet Fallback

**Date:** 2026-04-01
**Status:** Approved
**Scope:** `cli/search.rs` only — no search pipeline changes

## Problem

`nark search` returns empty `snippet: ""` for most results. The FTS5 `snippet()` function works for BM25-matched candidates, but notes discovered via cosine similarity or graph expansion get hardcoded `'' AS snippet` in the SQL queries. In Normal search mode (the default), this means cosine-only and graph-discovered hits — often highly relevant — have no snippet.

Alice's #1 frustration: "Every search becomes a multi-step process: search -> scan titles -> read 2-3 notes -> find the one I needed."

## Solution

Hybrid snippet fallback at the CLI layer, after the search pipeline returns results.

### Flow

For each `SearchHit` with an empty snippet:

1. **If query is non-empty:** Try FTS5 `snippet()` via `SELECT snippet(note_text, 2, '[', ']', '...', 32) FROM note_text WHERE note_text MATCH ?1 AND note_id = ?2`. This produces keyword-highlighted snippets for notes that contain the query terms literally.
2. **If FTS returns no row** (semantic match where query terms don't appear literally): Read the note body from the vault via `resolve::get_ref()` + `vault.read_object()`, truncate to 150 chars at word boundary.
3. **If query is empty** (filter-only search): Read the note body from the vault, truncate to 150 chars at word boundary.

### Snippet Format

- FTS snippets: `[matched term]...surrounding context...` (existing bracket format)
- Vault fallback: First ~150 chars of body at word boundary, no brackets

### Architecture

- **File changed:** `cli/search.rs`
- **New imports:** `crate::registry::resolve`, `crate::vault::fs::Vault`
- **New function:** `fill_missing_snippets(conn, vault, query, hits)` — iterates hits, applies hybrid fallback
- **Shared util:** `truncate_at_word` — same logic as `orient.rs` line 127, either inlined or extracted to a shared location
- **Pipeline (`registry/search.rs`):** Unchanged. `SearchHit.snippet` remains `String`.
- **No config:** Hardcoded 150 char limit for vault fallback.

### Performance

- FTS5 indexed lookup by `note_id` + `MATCH`: microseconds per query
- Vault body read: one file open per note
- Both apply only to final results (5-10 notes), not the full candidate pool
- Negligible overhead at nark's scale

### Testing

- Existing search pipeline tests in `registry/search.rs` are unaffected
- New behavior is presentation-layer — verify manually that `nark search "query"` returns non-empty snippets for all result modes (BM25, cosine, graph)

### Success Criteria

- `nark search "query"` returns non-empty snippets for every hit, regardless of how the note was discovered
- FTS-matched notes get keyword-highlighted snippets (bracket markers)
- Cosine/graph-only notes get body preview snippets (first 150 chars)
- Filter-only searches (`--domain X --tag Y` with no query) get body preview snippets
