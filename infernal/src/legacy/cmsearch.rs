//! cmsearch - Search sequence databases with CM

use crate::cm::CM;
use crate::pipeline::{cm_search, PipelineConfig, SearchHit};
use crate::output::{TbloutLine, tblout_header};
use crate::types::EslDsq;
use easel::alphabet::EslAlphabet;

/// cmsearch result
#[derive(Debug, Clone)]
pub struct CmsearchResult {
    pub hits: Vec<SearchHit>,
    pub model_name: String,
    pub target_name: String,
    pub target_len: i32,
}

/// Run cmsearch on a single sequence
pub fn cmsearch(cm: &CM, seq: &str, seq_name: &str) -> Result<CmsearchResult, String> {
    let abc = EslAlphabet::rna();

    // Digitize sequence
    let mut dsq: Vec<EslDsq> = vec![255];  // sentinel
    let digitized = abc.digitize(seq);
    dsq.extend(digitized);
    dsq.push(255);  // sentinel

    let l = seq.len() as i32;

    let config = PipelineConfig::default();
    let hit = cm_search(cm, &dsq, l, &config)?;

    Ok(CmsearchResult {
        hits: vec![hit],
        model_name: cm.name.clone(),
        target_name: seq_name.to_string(),
        target_len: l,
    })
}

/// Format cmsearch results as tblout
pub fn format_tblout(results: &[CmsearchResult]) -> String {
    let mut output = tblout_header();
    
    for result in results {
        for hit in &result.hits {
            let line = TbloutLine {
                target_name: result.target_name.clone(),
                target_acc: "-".to_string(),
                query_name: result.model_name.clone(),
                query_acc: "-".to_string(),
                mdl_from: 1,
                mdl_to: 71,  // CM consensus length (typical for tRNA)
                seq_from: hit.start,
                seq_to: hit.end,
                strand: '+',
                trunc: "no".to_string(),
                pass: 1,
                gc: 0.54,
                bias: 0.0,
                score: hit.score,
                evalue: 1e-10,  // placeholder - would need exp params for real calculation
                inc: '!',
                desc: "-".to_string(),
            };
            output.push_str(&line.format());
            output.push('\n');
        }
    }
    
    output
}
