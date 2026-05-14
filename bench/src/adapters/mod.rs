pub mod fts5;
pub mod nark;

use crate::protocol::Adapter;
use anyhow::{bail, Result};

pub fn make_adapter(name: &str) -> Result<Box<dyn Adapter>> {
    match name {
        "fts5" => Ok(Box::new(fts5::Fts5Adapter::new())),
        "nark" => Ok(Box::new(nark::NarkAdapter::new())),
        other => bail!("unknown adapter: {}", other),
    }
}
