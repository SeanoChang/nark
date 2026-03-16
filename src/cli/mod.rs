use clap::{Parser, Subcommand};

pub mod about;
pub mod delete;
pub mod diff;
pub mod edit;
pub mod embed;
pub mod history;
pub mod init;
pub mod jot;
pub mod link;
pub mod links;
pub mod ls;
pub mod peek;
pub mod read;
pub mod reset;
pub mod rollback;
pub mod search;
pub mod stats;
pub mod tag;
pub mod update;
pub mod write;

#[derive(Parser)]
#[command(
    name = "nark",
    about = "Noah's Ark — structured memory for AI agents",
    long_about = "Noah's Ark (nark) is a local-first knowledge vault for AI agents.\n\n\
        Notes are markdown files with YAML frontmatter, stored as content-addressed\n\
        objects and indexed in a SQLite registry for fast search and browsing.\n\n\
        Agent workflow: search/ls → peek → read → write\n\n\
        All output is JSON on stdout — designed to be consumed by agents directly.",
    version
)]
pub struct Cli {
    /// Vault directory (default: ~/.ark)
    #[arg(long, global = true)]
    pub vault_dir: Option<String>,

    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize vault directories and registry database
    Init,

    /// Quick-capture a note with minimal ceremony
    ///
    /// Only author + domain required (or --from to inherit taxonomy).
    /// Body comes from stdin or --body. Title inferred from first line if omitted.
    Jot {
        /// Note author (required)
        #[arg(long)]
        author: String,

        /// Domain (required unless --from is provided)
        #[arg(long)]
        domain: Option<String>,

        /// Intent (default: research)
        #[arg(long)]
        intent: Option<String>,

        /// Kind (default: reference)
        #[arg(long)]
        kind: Option<String>,

        /// Status (default: active)
        #[arg(long)]
        status: Option<String>,

        /// Importance 0-10 (default: 5)
        #[arg(long)]
        importance: Option<u8>,

        /// Title (inferred from first body line if omitted)
        #[arg(long)]
        title: Option<String>,

        /// Tag (repeatable)
        #[arg(long)]
        tag: Vec<String>,

        /// Inline body text (alternative to stdin)
        #[arg(long)]
        body: Option<String>,

        /// Clone taxonomy from an existing note's head
        #[arg(long)]
        from: Option<String>,
    },

    /// Ingest markdown notes into the vault
    ///
    /// Accepts files, directories (recursive *.md), or "-" for stdin.
    /// Each note must have YAML frontmatter with: title, author, domain,
    /// intent, kind, trust, status, tags.
    Write {
        /// Paths to files or directories, or "-" for stdin
        paths: Vec<String>,

        /// Max directory recursion depth (unlimited if omitted)
        #[arg(long)]
        depth: Option<u64>,
    },

    /// Edit an existing note (replace, append, prepend, set)
    ///
    /// Creates a new MVCC version for each edit. Supports surgical find/replace,
    /// body append/prepend, and full document replacement.
    Edit {
        /// Note ID (UUID)
        id: String,

        /// Run multiple operations as one atomic version
        #[arg(long)]
        batch: bool,

        /// Edit operations and arguments
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Show note metadata from the registry (cheap, no vault read)
    ///
    /// Returns: id, title, domain, intent, kind, trust, status, tags, updated_at.
    /// Use this to inspect a note before committing to a full read.
    Peek {
        /// Note ID (UUID)
        id: String,
    },

    /// Read full note content from the vault (frontmatter + body)
    ///
    /// Resolves the head version, reads content-addressed objects from disk.
    /// Heavier than peek — use only when you need the actual content.
    Read {
        /// Note ID (UUID)
        id: String,
    },

    /// Full-text search across all notes (BM25 ranked)
    ///
    /// Searches title (5x), keywords/tags (10x), aliases (3x), spine (2x),
    /// and body (1x). Returns ranked results with match snippets.
    /// Supports FTS5 syntax: "exact phrase", OR, NOT, prefix*, column:term.
    Search {
        /// Search query (FTS5 syntax). Optional if filters are provided.
        query: Option<String>,

        /// Filter by domain
        #[arg(long)]
        domain: Option<String>,

        /// Filter by kind (spec, decision, runbook, report, reference, incident, experiment, dataset)
        #[arg(long)]
        kind: Option<String>,

        /// Filter by intent (build, debug, operate, design, research, evaluate, decide)
        #[arg(long)]
        intent: Option<String>,

        /// Filter by tag (AND logic). Accepts multiple: --tag cas vault
        #[arg(long, num_args = 1..)]
        tag: Vec<String>,

        /// Max results to return
        #[arg(long, default_value = "10")]
        limit: usize,

        /// BM25-only mode: skip cosine ranking and graph expansion.
        /// Useful for debugging recall or exact-term searches.
        #[arg(long, conflicts_with = "semantic")]
        bm25: bool,

        /// Semantic mode: bypass BM25 filter, run cosine against all notes.
        /// Requires embeddings (nark embed init + build).
        #[arg(long, conflicts_with = "bm25")]
        semantic: bool,

        /// Filter to notes updated since (e.g. "1d", "7d", "24h", "1w", "1mo")
        #[arg(long)]
        since: Option<String>,

        /// Filter to notes updated before (e.g. "3d", "1w", "1mo")
        #[arg(long)]
        before: Option<String>,
    },

    /// Browse the knowledge tree: domain → intent → kind → notes
    ///
    /// Navigate the hierarchy one level at a time.
    /// No path = list domains. "systems" = list intents. "systems/build/spec" = list notes.
    Ls {
        /// Tree path, e.g. "systems", "systems/build", "systems/build/spec"
        path: Option<String>,

        /// Include tags for each note (leaf level only)
        #[arg(long)]
        tags: bool,
    },

    /// Quick research — search + read previews in one call
    ///
    /// Finds the top N matching notes and returns a ~500 char body preview
    /// for each. Saves multiple round-trips vs search → peek → read.
    About {
        /// Topic to research
        topic: String,

        /// Number of notes to return
        #[arg(long, default_value = "3")]
        limit: usize,

        /// Filter to notes updated since (e.g. "1d", "7d", "24h", "1w", "1mo")
        #[arg(long)]
        since: Option<String>,

        /// Filter to notes updated before (e.g. "3d", "1w", "1mo")
        #[arg(long)]
        before: Option<String>,
    },

    /// Delete notes from the vault
    ///
    /// Default: soft delete (sets status = 'retracted'). Note disappears from
    /// browse/search but all data survives — reversible via re-ingest.
    /// -f: hard delete (removes registry rows, vault objects become orphans).
    /// -rf: full purge (removes registry rows + vault CAS objects).
    Delete {
        /// Note IDs to delete
        ids: Vec<String>,

        /// Hard delete — remove all registry rows
        #[arg(short, long)]
        force: bool,

        /// Recursive — also remove vault CAS objects (requires -f)
        #[arg(short, long, requires = "force")]
        recursive: bool,
    },

    /// Add, remove, list, or find tags on notes
    ///
    /// Mutate: nark tag <id> [<id>...] +add -remove
    /// Read:   nark tag <id> (no modifiers = show tags)
    /// List:   nark tag --list
    /// Find:   nark tag --find <tag> [--find <tag>...]
    Tag {
        /// Note IDs and +tag/-tag modifiers
        #[arg(allow_hyphen_values = true)]
        args: Vec<String>,

        /// List all tags with usage counts
        #[arg(long)]
        list: bool,

        /// Find notes by tag (AND logic). Accepts multiple: --find cas vault
        #[arg(long, num_args = 1..)]
        find: Vec<String>,
    },

    /// Create typed links from one or more source notes to a single target
    ///
    /// Adds a frontmatter link entry and a [[wikilink]] in each source note body.
    /// Idempotent — re-running with the same args is a no-op.
    Link {
        /// Source note IDs (one or more)
        #[arg(required = true, num_args = 1..)]
        sources: Vec<String>,

        /// Target note ID
        #[arg(long)]
        target: String,

        /// Relationship type (references, depends-on, supersedes, contradicts, extends, informed-by)
        #[arg(long, default_value = "references")]
        rel: String,
    },

    /// Show a note's link neighborhood (outgoing + incoming edges)
    Links {
        /// Note ID (UUID)
        id: String,
    },

    /// List version history for a note
    ///
    /// Walks the version chain from head to the original version via
    /// prev_version_id links. Shows version_id, content_hash, and timestamps.
    History {
        /// Note ID (UUID)
        id: String,
    },

    /// Compare two versions of a note (unified diff)
    ///
    /// Defaults to comparing the previous version against the current head.
    /// Use --from and --to to compare arbitrary versions.
    Diff {
        /// Note ID (UUID)
        id: String,

        /// Version ID to diff from (default: prev of head)
        #[arg(long)]
        from: Option<String>,

        /// Version ID to diff to (default: head)
        #[arg(long)]
        to: Option<String>,
    },

    /// Restore an old version as a new head version
    ///
    /// Creates a new version with the content of the specified old version.
    /// Non-destructive — the old version chain is preserved.
    Rollback {
        /// Note ID (UUID)
        id: String,

        /// Version ID to restore
        version_id: String,
    },

    /// Vault overview — note counts, distributions, recent activity
    Stats,

    /// Reset the registry database
    ///
    /// Deletes registry.db and recreates it with fresh schema + seed data.
    /// Vault CAS objects are untouched — re-ingest with `nark write` after reset.
    /// Requires --confirm to prevent accidental data loss.
    Reset {
        /// Actually perform the reset (without this, shows a dry-run summary)
        #[arg(long)]
        confirm: bool,
    },

    /// Manage local embeddings (ONNX + bge-base-en-v1.5)
    ///
    /// Subcommands: init (download runtime + model), build (backfill embeddings).
    /// Embedding is optional — search falls back to BM25 if not initialized.
    Embed {
        #[command(subcommand)]
        action: EmbedAction,
    },

    /// Pull latest code and rebuild the binary
    Update,
}

#[derive(Subcommand)]
pub enum EmbedAction {
    /// Download ONNX Runtime and bge-base-en-v1.5 model
    Init,
    /// Backfill embeddings for all notes that don't have one
    Build,
}
