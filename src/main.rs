use anyhow::Result;
use clap::Parser;

mod cli;
mod config;
mod db;
mod embed;
mod registry;
mod types;
mod vault;

use crate::cli::Commands::{
    Init,
    Jot,
    Write,
    Edit,
    Peek,
    Read,
    Search,
    Ls,
    About,
    Delete,
    Related,
    Embed,
    Tag,
    Link,
    Links,
    History,
    Diff,
    Rollback,
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
        Jot { author, domain, intent, kind, status, title, tag, body, from, auto_link } => cli::jot::run(&vault_dir, title, &author, domain.as_deref(), kind.as_deref(), intent.as_deref(), status.as_deref(), &tag, body.as_deref(), from.as_deref(), auto_link),
        Write { paths, depth, auto_link } => cli::write::run(&vault_dir, paths, depth, auto_link),
        Edit { id, batch, auto_link, args } => cli::edit::run(&vault_dir, &id, batch, auto_link, args),
        Peek { id } => cli::peek::run(&vault_dir, &id),
        Read { id } => cli::read::run(&vault_dir, &id),
        Search { query, domain, kind, intent, tag, limit, bm25, semantic, since, before } => cli::search::run(&vault_dir, query.as_deref().unwrap_or(""), domain.as_deref(), kind.as_deref(), intent.as_deref(), &tag, limit, bm25, semantic, since.as_deref(), before.as_deref()),
        Ls { path, tags } => cli::ls::run(&vault_dir, path.as_deref(), tags),
        About { topic, limit, since, before } => cli::about::run(&vault_dir, &topic, limit, since.as_deref(), before.as_deref()),
        Related { id, limit, link } => cli::related::run(&vault_dir, &id, limit, link),
        Delete { ids, force, recursive } => cli::delete::run(&vault_dir, ids, force, recursive),
        Tag { args, list, find } => cli::tag::run(&vault_dir, args, list, find),
        Link { sources, target, rel } => cli::link::run(&vault_dir, sources, &target, &rel),
        Links { id } => cli::links::run(&vault_dir, &id),
        History { id } => cli::history::run(&vault_dir, &id),
        Diff { id, from, to } => cli::diff::run(&vault_dir, &id, from.as_deref(), to.as_deref()),
        Rollback { id, version_id } => cli::rollback::run(&vault_dir, &id, &version_id),
        Stats => cli::stats::run(&vault_dir),
        Embed { action } => match action {
            cli::EmbedAction::Init => cli::embed::run_init(&vault_dir),
            cli::EmbedAction::Build => cli::embed::run_build(&vault_dir),
            cli::EmbedAction::Migrate => cli::embed::run_migrate(&vault_dir),
        },
        Reset { confirm } => cli::reset::run(&vault_dir, confirm),
        Update => cli::update::run(),
    }
}
