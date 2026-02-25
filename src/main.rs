use anyhow::Result;
use clap::Parser;

mod cli;
mod db;
mod registry;
mod types;
mod vault;

use crate::cli::Commands::{
    Init,
    Write,
    Peek,
    Read,
    Search,
    Ls,
    About,
    Delete,
    Tag,
    Stats,
    Reset,
    Update
};
use cli::init::run;

fn main() -> Result<()> {
    let args = cli::Cli::parse();

    let vault_dir = match args.vault_dir {
        Some(p) => std::path::PathBuf::from(p),
        None => dirs::home_dir()
            .expect("Could not find home directory")
            .join(".ark"),
    };

    match args.command {
        Init => run(&vault_dir),
        Write { paths, depth } => cli::write::run(&vault_dir, paths, depth),
        Peek { id } => cli::peek::run(&vault_dir, &id),
        Read { id } => cli::read::run(&vault_dir, &id),
        Search { query, domain, kind, intent, tag, limit } => cli::search::run(&vault_dir, query.as_deref().unwrap_or(""), domain.as_deref(), kind.as_deref(), intent.as_deref(), &tag, limit),
        Ls { path, tags } => cli::ls::run(&vault_dir, path.as_deref(), tags),
        About { topic, limit } => cli::about::run(&vault_dir, &topic, limit),
        Delete { ids, force, recursive } => cli::delete::run(&vault_dir, ids, force, recursive),
        Tag { args, list, find } => cli::tag::run(&vault_dir, args, list, find),
        Stats => cli::stats::run(&vault_dir),
        Reset { confirm } => cli::reset::run(&vault_dir, confirm),
        Update => cli::update::run(),
    }
}
