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
        Peek { id, meta } => todo!(),
        Search { query, domain, limit } => todo!(),
        Ls { path } => todo!(),
        About{ topic, limit } => todo!(),
        Update => todo!(),
    }
}
