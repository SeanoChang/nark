use crate::types::markdown::Frontmatter;

pub struct IngestResult {
    pub note_id: String,
    pub version_id: String,
    pub prev_version_id: Option<String>,
    pub fm_hash: String,
    pub md_hash: String,
    pub content_hash: String,
    pub frontmatter: Frontmatter,
    pub body: String,
}
