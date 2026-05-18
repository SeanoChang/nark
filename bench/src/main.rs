use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod protocol;
mod metrics;
mod adapters;
mod result;
mod tasks;
mod model_cache;
mod llm;

use crate::llm::api::ApiBackend;
use crate::llm::cache::LlmCache;
use crate::llm::claude_cli::ClaudeCliBackend;
use crate::llm::LlmBackend;

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
        /// Task name: "ir" | "longmemeval" | "locomo"
        #[arg(long)]
        task: String,
        /// Comma-separated systems (e.g. "fts5,nark,vector")
        #[arg(long)]
        systems: String,
        /// For --task ir: corpus name (e.g. "synthetic-tiny"). Ignored for Task B.
        #[arg(long, default_value = "synthetic-tiny")]
        corpus: String,
        /// Output directory for result JSON files
        #[arg(long, default_value = "bench/results/local")]
        output: PathBuf,

        // Task B (longmemeval / locomo) flags
        /// Generation backend: claude-cli | api  (codex-cli intentionally not supported in Phase 2 — see bench/src/llm/mod.rs notes)
        #[arg(long)]
        gen_backend: Option<String>,
        /// Generation model id (e.g. "claude-opus-4-7")
        #[arg(long)]
        gen_model: Option<String>,
        /// Judge backend: claude-cli | api
        #[arg(long)]
        judge_backend: Option<String>,
        /// Judge model id
        #[arg(long)]
        judge_model: Option<String>,
        /// Concurrency cap for parallel LLM calls (per backend)
        #[arg(long, default_value = "4")]
        llm_concurrency: usize,
        /// Skip the pre-run confirmation prompt
        #[arg(long)]
        yes: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { task, systems, corpus, output, gen_backend, gen_model, judge_backend, judge_model, llm_concurrency, yes } => {
            match task.as_str() {
                "ir" => run_ir(systems, corpus, output),
                "longmemeval" => run_task_b("longmemeval", systems, output, gen_backend, gen_model, judge_backend, judge_model, llm_concurrency, yes),
                "locomo" => run_task_b("locomo", systems, output, gen_backend, gen_model, judge_backend, judge_model, llm_concurrency, yes),
                other => anyhow::bail!("unknown --task value: {}", other),
            }
        }
    }
}

fn run_ir(systems: String, corpus: String, output: PathBuf) -> Result<()> {
    let corpus_root = PathBuf::from("bench/datasets/ir").join(&corpus);
    if !corpus_root.exists() {
        anyhow::bail!("corpus not found at {:?}", corpus_root);
    }
    let cache = model_cache::cache_root()?;
    let needs_model = systems.split(',').any(|s| matches!(s.trim(), "nark" | "vector"));
    if needs_model {
        model_cache::ensure_ready(&cache)?;
    }
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

fn build_backend(name: Option<&str>, model: Option<&str>, concurrency: usize, role: &str) -> Result<Box<dyn LlmBackend>> {
    let name = name.ok_or_else(|| anyhow!("--{}-backend is required for Task B", role))?;
    let model = model.ok_or_else(|| anyhow!("--{}-model is required for Task B", role))?;
    match name {
        "claude-cli" => Ok(Box::new(ClaudeCliBackend::new(model).with_concurrency(concurrency))),
        "api" => Ok(Box::new(ApiBackend::from_env(model)?.with_concurrency(concurrency))),
        // "codex-cli" intentionally rejected — see bench/src/llm/mod.rs notes about
        // codex's --dangerously-bypass-approvals-and-sandbox security concern.
        other => anyhow::bail!("unknown {}-backend: {} (valid: claude-cli, api)", role, other),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_task_b(
    task_name: &str,
    systems: String,
    output: PathBuf,
    gen_backend_name: Option<String>,
    gen_model: Option<String>,
    judge_backend_name: Option<String>,
    judge_model: Option<String>,
    concurrency: usize,
    yes: bool,
) -> Result<()> {
    let model_cache_path = model_cache::cache_root()?;
    let needs_model = systems.split(',').any(|s| matches!(s.trim(), "nark" | "vector"));
    if needs_model {
        model_cache::ensure_ready(&model_cache_path)?;
    }

    let mut gen_backend = build_backend(gen_backend_name.as_deref(), gen_model.as_deref(), concurrency, "gen")?;
    let mut judge_backend = build_backend(judge_backend_name.as_deref(), judge_model.as_deref(), concurrency, "judge")?;
    let llm_cache_path = PathBuf::from("bench/cache/llm.db");
    std::fs::create_dir_all(llm_cache_path.parent().unwrap())?;
    let llm_cache = LlmCache::open(&llm_cache_path)?;

    let dataset_path = match task_name {
        "longmemeval" => PathBuf::from("bench/datasets/longmemeval/upstream/data/longmemeval_s_cleaned.json"),
        "locomo" => PathBuf::from("bench/datasets/locomo/upstream/repo/data/locomo10.json"),
        _ => unreachable!(),
    };
    if !dataset_path.exists() {
        anyhow::bail!(
            "dataset not found at {:?}. Run bench/datasets/{}/fetch.sh first.",
            dataset_path, task_name
        );
    }

    let gen_template_path = PathBuf::from(format!("bench/llm/prompts/{}-generate.md", task_name));
    let judge_template_path = PathBuf::from(format!("bench/llm/prompts/{}-judge.md", task_name));

    let system_count = systems.split(',').filter(|s| !s.trim().is_empty()).count();
    // Question counts per dataset — keep in sync with what fetch.sh + loaders pull.
    // The estimate exists to prevent accidental subscription burn, so err high.
    let est_questions = match task_name {
        "longmemeval" => 500,
        "locomo" => 1986,
        _ => 500,
    };
    let est_calls = est_questions * system_count * 2;
    if !yes {
        eprintln!(
            "About to run {} on {} systems with up to ~{} LLM calls (gen + judge per question).\n\
             First-run cost: subscription limits or API spend.\n\
             Subsequent runs use cache (bench/cache/llm.db) and finish in seconds.\n\
             \n\
             Continue? [y/N]: ",
            task_name, system_count, est_calls
        );
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).ok();
        if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
            eprintln!("aborted.");
            std::process::exit(1);
        }
    }

    for system in systems.split(',') {
        let system = system.trim();
        if system.is_empty() { continue; }

        let cache_path = model_cache_path.clone();
        let sys_owned = system.to_string();
        let mut factory = move || -> Result<Box<dyn crate::protocol::Adapter>> {
            adapters::make_adapter(&sys_owned, Some(&cache_path))
        };

        let result = match task_name {
            "longmemeval" => tasks::longmemeval::run_longmemeval_task(
                &mut factory,
                &dataset_path,
                gen_backend.as_mut(),
                &gen_template_path,
                judge_backend.as_mut(),
                &judge_template_path,
                &llm_cache,
                "default",
            )?,
            "locomo" => tasks::locomo::run_locomo_task(
                &mut factory,
                &dataset_path,
                gen_backend.as_mut(),
                &gen_template_path,
                judge_backend.as_mut(),
                &judge_template_path,
                &llm_cache,
                "default",
            )?,
            _ => unreachable!(),
        };
        let path = result.write_to_disk(&output)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
