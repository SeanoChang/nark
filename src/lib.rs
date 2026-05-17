//! nark library — re-exports modules for cross-crate consumers (currently
//! the in-workspace `bench/` crate). `main.rs` is the primary entry point
//! and does not depend on this lib; the lib exists so other crates can
//! `use nark::embed::OnnxProvider` etc. without forking the code.

pub mod cli;
pub mod config;
pub mod db;
pub mod embed;
pub mod registry;
pub mod types;
pub mod vault;
