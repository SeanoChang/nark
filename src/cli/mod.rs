use clap::{Parser, Subcommand};

pub mod init;
pub mod write;
pub mod peek;
pub mod search;
pub mod ls;
pub mod about;
pub mod update;

#[derive(Parser)]
#[command(name = "nark", about = "Noah's Ark - agent memory CLI", version)]
pub struct Cli {
    #[arg(long, global = true)]
    pub vault_dir: Option<String>,

    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Boostrap a new vault dirs + registry database
    Init,
    
    /// Ingest a new note (markdown file with frontmatter)
    Write {
        /// Files, directories, or "-" for stdin
        paths: Vec<String>,

        /// Max directory recursion depth (unlimited if omitted)
        #[arg(long)]
        depth: Option<u64>,
    },

    /// Peek at a note (full body or just frontmatter)
    Peek {
        /// Note ID 
        id: String,

        /// Show only frontmatter
        #[arg(long)]
        meta: bool,
    },

    /// FTS5 ranked search
    Search {
        /// Search query
        query: String,

        /// Filter by domain
        domain: Option<String>,

        /// Max results
        #[arg(long, default_value = "10")]
        limit: usize,
    },

    /// Browse knowledge tree (domain/intent/kind/notes)
    Ls {
        /// e.g. "systems/build/spec"
        path: Option<String>,
    },

    /// Search + return top N note summaries
    About {
        /// Topic to search for
        topic: String,

        /// Number of notes to returnß
        #[arg(long, default_value = "3")]
        limit: usize,
    },

    /// Self update from GitHub releases
    Update,
}


