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
        Search { query, domain, limit } => cli::search::run(&vault_dir, &query, domain.as_deref(), limit),
        Ls { path } => cli::ls::run(&vault_dir, path.as_deref()),
        About { topic, limit } => cli::about::run(&vault_dir, &topic, limit),
        Update => cli::update::run(),
    }
}
