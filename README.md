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
| `nark search <query> [--domain] [--kind] [--intent] [--tag]` | Ranked search (BM25 + cosine + graph) | Cheap — registry only |
| `nark search <query> --bm25` | BM25-only mode — skip cosine and graph | Cheap — registry only |
| `nark search <query> --semantic` | Semantic mode — bypass BM25, cosine against all notes | Medium — needs embeddings |
| `nark stats` | Vault overview — counts, distributions, recent notes | Cheap — registry only |
| `nark ls [path] [--tags]` | Browse domain/intent/kind tree | Cheap — registry only |
| `nark about <topic>` | Search + body previews in one call | Medium — registry + vault reads |
| `nark peek <id>` | Note metadata (title, domain, tags, etc.) | Cheap — registry only |
| `nark read <id>` | Full note content (frontmatter + body) | Heavy — vault CAS read |
| `nark write <paths...>` | Ingest markdown notes | Write — vault + registry |
| `nark delete <ids...> [-f] [-rf]` | Soft-delete (retract), hard-delete, or full purge | Write — registry (+ vault for -rf) |
| `nark tag <id> +add -remove` | Add/remove tags without creating a new version | Write — registry only |
| `nark tag --list` | List all tags with usage counts | Cheap — registry only |
| `nark tag --find <tags...>` | Find notes by tag (AND logic) | Cheap — registry only |
| `nark link <sources...> --target <id> [--rel <type>]` | Create typed links between notes | Write — vault + registry |
| `nark links <id>` | Show a note's link neighborhood | Cheap — registry only |
| `nark embed init` | Download ONNX Runtime + nomic-embed-text-v1.5 model | Setup |
| `nark embed build` | Backfill embeddings for all notes | Write — registry only |
| `nark embed migrate` | Upgrade from bge to nomic (download + cleanup + re-embed) | Setup + Write |
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
aliases: ["CAS", "content-addressed store"]
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
| `tags` | Free-form labels | Any lowercase alphanumeric + hyphens |
| `aliases` | Search synonyms (3x FTS5 weight) | Free-form strings, optional |

Domain, intent, kind, trust, and status are enforced enums — invalid values are rejected at parse time. Tags and aliases are optional (default to `[]`).

## Architecture

```
~/.ark/
├── config.toml          # Optional — search tuning knobs
├── registry.db          # SQLite — indexes, FTS5, edges, embeddings
├── objects/
│   ├── fm/              # Content-addressed frontmatter (YAML)
│   └── md/              # Content-addressed bodies (Markdown)
├── notes/
│   └── <note-id>/
│       ├── head         # Current version pointer
│       └── versions/    # Version history (.ref + .json)
├── onnxruntime/         # ONNX Runtime dylib (nark embed init)
├── models/
│   └── nomic-embed-text-v1.5/  # Embedding model (nark embed init)
└── tmp/                 # Atomic write staging
```

- **Content-addressed storage** — files stored by BLAKE3 hash. Deduplication is automatic.
- **Append-only versions** — every write creates a new version. Old versions are never overwritten.
- **SQLite registry** — `current_notes` materialized view for fast queries, `note_text` FTS5 table for search, `note_versions` for history, `note_edges` for typed links, `note_embeddings` for vector search.

## Search pipeline

Search runs a 6-step ranked pipeline:

```
pre-filter → BM25 candidates → graph expand → cosine rank → blend → threshold
```

1. **Pre-filter** — apply `--domain`, `--kind`, `--intent`, `--tag` filters
2. **BM25 candidates** — FTS5 full-text search returns top-k candidates (recall, not ranking)
3. **Graph expand** — follow note edges to discover related notes not in the BM25 set
4. **Cosine rank** — score candidates against the query embedding (primary ranking signal)
5. **Blend** — combine three signals: `cosine * 0.50 + graph * 0.25 + activation * 0.25`
6. **Threshold** — drop results below the minimum score, return top-n

The pipeline degrades gracefully:
- **No embeddings, no graph** — BM25 rank + activation only
- **No embeddings, with graph** — graph scores + activation only
- **Full pipeline** — all three signals blended

### Search modes

| Flag | Mode | What it does |
|---|---|---|
| *(default)* | Normal | Full 6-step pipeline |
| `--bm25` | BM25-only | Skip cosine + graph. Fast exact-term search. |
| `--semantic` | Semantic | Bypass BM25, cosine against all notes. Requires embeddings. |

`--bm25` and `--semantic` are mutually exclusive.

### FTS5 syntax

nark uses SQLite FTS5. Plain words work, but you can also use:

| Syntax | Example | Meaning |
|---|---|---|
| plain words | `BLAKE3 hashing` | Both words must appear (implicit AND) |
| `"phrase"` | `"content addressed"` | Exact phrase match |
| `OR` | `BLAKE3 OR SHA256` | Either word |
| `NOT` | `BLAKE3 NOT deprecated` | Exclude matches |
| `column:` | `title:BLAKE3` | Match in specific column |
| `prefix*` | `blake*` | Prefix match |

## Edges

Notes can be linked with typed, weighted edges:

| Edge type | Weight | Direction | Meaning |
|---|---|---|---|
| `references` | 1.0 | bidirectional | General citation |
| `depends-on` | 2.0 | bidirectional | Hard dependency |
| `supersedes` | 3.0 | old → new only | Replacement (directional) |
| `contradicts` | 1.5 | bidirectional | Conflicting information |
| `extends` | 1.5 | bidirectional | Builds upon |
| `informed-by` | 1.0 | bidirectional | Loosely inspired by |

Edges are created via frontmatter `links:` fields or the `nark link` command. Graph expansion during search uses these edges to surface related notes.

## Embeddings

Embeddings are optional but unlock cosine-ranked search and semantic mode.

```bash
# Download ONNX Runtime + nomic-embed-text-v1.5 model
nark embed init

# Backfill embeddings for all notes
nark embed build

# Upgrading from bge-base-en-v1.5? One command handles everything:
nark embed migrate
```

By default, embeddings are computed locally via ONNX (no API calls). Optionally, configure OpenAI as the embedding provider in `config.toml`:

```toml
[embedding]
provider = "openai"                    # default: "local"
api_model = "text-embedding-3-small"   # optional override
```

Requires `OPENAI_API_KEY` in the environment. Without embeddings, search falls back to BM25 + activation scoring.

## Configuration

Place a `config.toml` in your vault directory (`~/.ark/config.toml`). All fields are optional — missing values use defaults.

```toml
[embedding]
provider = "local"    # "local" (ONNX) or "openai"
# api_model = "text-embedding-3-small"  # only used when provider = "openai"

[search]
threshold = 0.10      # minimum score to return a result
top_n = 20            # max results

[search.bm25]
top_k = 100           # BM25 candidate pool size
weight_title = 5.0    # FTS5 column weights
weight_body = 1.0
weight_spine = 2.0
weight_aliases = 3.0
weight_keywords = 10.0

[search.weights]
cosine = 0.50         # blend weights (must sum to 1.0)
graph = 0.25
activation = 0.25

[search.graph]
decay = 0.5           # graph score decay per hop
max_hops = 1          # max graph traversal depth
respect_domain_filter = false  # restrict graph expansion to filtered domain
```

## Release

```bash
# Manual release (current platform only)
./scripts/release.sh 0.1.0

# Automated: push a tag to trigger CI builds for all platforms
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```
