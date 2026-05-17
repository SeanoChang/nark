use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod protocol;
mod metrics;
mod adapters;
mod result;
mod tasks;
mod model_cache;

#[derive(Parser)]
#[command(name = "nark-bench", about = "Benchmark harness for nark")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a benchmark task
    Run {
        /// Task name (currently only "ir" supported)
        #[arg(long)]
        task: String,
        /// Comma-separated systems to benchmark (e.g. "fts5,nark")
        #[arg(long)]
        systems: String,
        /// Corpus name relative to bench/datasets/ir/ (e.g. "synthetic-tiny")
        #[arg(long)]
        corpus: String,
        /// Output directory for result JSON files
        #[arg(long, default_value = "bench/results/local")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { task, systems, corpus, output } => {
            if task != "ir" {
                anyhow::bail!("only --task ir is supported in Phase 1a");
            }
            let corpus_root = PathBuf::from("bench/datasets/ir").join(&corpus);
            if !corpus_root.exists() {
                anyhow::bail!("corpus not found at {:?}", corpus_root);
            }
            let cache = model_cache::cache_root()?;
            model_cache::ensure_ready(&cache)?;
            for system in systems.split(',') {
                let system = system.trim();
                if system.is_empty() { continue; }
                let mut adapter = adapters::make_adapter(system, Some(&cache))?;
                let result = tasks::ir::run_ir_task(adapter.as_mut(), &corpus_root, "default")?;
                let path = result.write_to_disk(&output)?;
                eprintln!("wrote {}", path.display());
            }
            Ok(())
        }
    }
}
