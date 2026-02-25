# nark — Noah's Ark

Structured memory for AI agents. A local-first knowledge vault that stores markdown notes as content-addressed objects and indexes them in SQLite for fast search and browsing.

## Install

### From GitHub Releases

```bash
# macOS (Apple Silicon)
gh release download --repo SeanoChang/ironvault --pattern '*aarch64-apple-darwin'
chmod +x nark-* && mv nark-* ~/.local/bin/nark

# Linux
gh release download --repo SeanoChang/ironvault --pattern '*x86_64-unknown-linux-gnu'
chmod +x nark-* && mv nark-* ~/.local/bin/nark
```

### From source

```bash
git clone https://github.com/SeanoChang/ironvault.git
cd ironvault
cargo build --release
ln -sf "$(pwd)/target/release/nark" ~/.local/bin/nark
```

## Quick start

```bash
# Initialize the vault
nark init

# Write a note
nark write path/to/note.md

# Write all markdown files in a directory (recursive)
nark write path/to/notes/

# Search for notes
nark search "capability tokens"

# Browse the knowledge tree
nark ls
nark ls systems/build/spec

# Quick research — search + body previews
nark about "BLAKE3 hashing"

# Inspect a note's metadata
nark peek <note-id>

# Read full note content
nark read <note-id>
```

## Commands

| Command | What it does | Cost |
|---|---|---|
| `nark search <query> [--tag ...] [--domain ...]` | FTS5 ranked search, filterable by tags and domain | Cheap — registry only |
| `nark ls [path] [--tags]` | Browse domain/intent/kind tree | Cheap — registry only |
| `nark about <topic>` | Search + body previews in one call | Medium — registry + vault reads |
| `nark peek <id>` | Note metadata (title, domain, tags, etc.) | Cheap — registry only |
| `nark read <id>` | Full note content (frontmatter + body) | Heavy — vault CAS read |
| `nark write <paths...>` | Ingest markdown notes | Write — vault + registry |
| `nark delete <ids...> [-f] [-rf]` | Soft-delete (retract), hard-delete, or full purge | Write — registry (+ vault for -rf) |
| `nark tag <id> +add -remove` | Add/remove tags without creating a new version | Write — registry only |
| `nark tag --list` | List all tags with usage counts | Cheap — registry only |
| `nark tag --find <tags...>` | Find notes by tag (AND logic) | Cheap — registry only |
| `nark reset [--confirm]` | Destroy and recreate registry (vault objects kept) | Destructive |
| `nark init` | Create vault dirs + registry database | One-time setup |
| `nark update` | Download latest release binary from GitHub | Maintenance |

### Agent workflow

```
search/ls → peek → read → write
  cheap      cheap   heavy   write
```

Start broad, narrow down, commit to reading only what matters.

## Note format

Notes are markdown files with YAML frontmatter:

```markdown
---
title: "CAS Write Discipline"
author: "noah"
domain: "systems"
intent: "build"
kind: "spec"
trust: "verified"
status: "active"
tags: ["cas", "storage", "blake3"]
---
# CAS Write Discipline

Content goes here...
```

### Frontmatter fields

| Field | Purpose | Allowed values |
|---|---|---|
| `domain` | Knowledge area | systems, security, finance, ai_ml, data, programming, math, writing, product |
| `intent` | Why it exists | build, debug, operate, design, research, evaluate, decide |
| `kind` | What it is | spec, decision, runbook, report, reference, incident, experiment, dataset |
| `trust` | Confidence level | hypothesis, reviewed, verified |
| `status` | Lifecycle state | active, deprecated, retracted, draft |

Intent, kind, trust, and status are enforced enums — invalid values are rejected at parse time.

## Architecture

```
~/.ark/
├── registry.db          # SQLite — indexes, FTS5, metadata
├── objects/
│   ├── fm/              # Content-addressed frontmatter (YAML)
│   └── md/              # Content-addressed bodies (Markdown)
├── notes/
│   └── <note-id>/
│       ├── head         # Current version pointer
│       └── versions/    # Version history (.ref + .json)
└── tmp/                 # Atomic write staging
```

- **Content-addressed storage** — files stored by BLAKE3 hash. Deduplication is automatic.
- **Append-only versions** — every write creates a new version. Old versions are never overwritten.
- **SQLite registry** — `current_notes` materialized view for fast queries, `note_text` FTS5 table for search, `note_versions` for history.

## Search syntax

nark uses SQLite FTS5. Plain words work, but you can also use:

| Syntax | Example | Meaning |
|---|---|---|
| plain words | `BLAKE3 hashing` | Both words must appear (implicit AND) |
| `"phrase"` | `"content addressed"` | Exact phrase match |
| `OR` | `BLAKE3 OR SHA256` | Either word |
| `NOT` | `BLAKE3 NOT deprecated` | Exclude matches |
| `column:` | `title:BLAKE3` | Match in specific column |
| `prefix*` | `blake*` | Prefix match |

## Release

```bash
# Manual release (current platform only)
./scripts/release.sh 0.1.0

# Automated: push a tag to trigger CI builds for all platforms
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```
