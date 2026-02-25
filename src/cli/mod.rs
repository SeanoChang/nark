use clap::{Parser, Subcommand};

pub mod init;
pub mod write;
pub mod peek;
pub mod read;
pub mod search;
pub mod ls;
pub mod about;
pub mod update;
pub mod delete;
pub mod reset;
pub mod stats;
pub mod tag;

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

    /// Pull latest code and rebuild the binary
    Update,
}
