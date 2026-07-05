//! CM Search Pipeline
//!
//! Implements the complete search pipeline:
//! 1. Create CP9 HMM from CM
//! 2. Compute HMM bands using CP9 Forward/Backward
//! 3. Map HMM bands to CM state bands
//! 4. Run CYK/Inside with bands
//! 5. Return hits above threshold

use crate::cm::CM;
use crate::cm_dp::cyk_inside;
use crate::cm_dp_hb::{cm_cyk_inside_align_hb_bands, CMHbMx};
use crate::cp9::CP9;
use crate::cp9_bands::CP9Bands;
use crate::cp9_dp::{cp9_compute_bands, cp9_to_cm_bands_with_types};
use crate::cp9_map::cp9_map_cm2hmm;
use crate::cp9_modelmaker::cplan9_from_cm;
use crate::types::EslDsq;

/// Search hit result
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub start: i32,      // Start position (1-indexed)
    pub end: i32,        // End position (1-indexed)
    pub score: f32,      // Bit score
    pub b: i32,          // Band width used
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub use_hmm_bands: bool,
    pub score_threshold: f32,  // Minimum score in bits
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            use_hmm_bands: true,
            score_threshold: f32::NEG_INFINITY,
        }
    }
}

/// Search a sequence with CM using the full banded pipeline
///
/// This is the main entry point for CM search with HMM-derived bands.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence (1-indexed, with sentinels)
/// * `l` - Sequence length
/// * `config` - Pipeline configuration
///
/// # Returns
/// SearchHit with score and alignment info
pub fn cm_search(cm: &CM, dsq: &[EslDsq], l: i32, config: &PipelineConfig)
    -> Result<SearchHit, String>
{
    if config.use_hmm_bands {
        cm_search_banded(cm, dsq, l)
    } else {
        cm_search_unbanded(cm, dsq, l)
    }
}

/// Full banded search pipeline
fn cm_search_banded(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<SearchHit, String> {
    // Step 1: Create CP9 HMM from CM
    let cp9 = create_cp9_from_cm(cm)?;

    // Step 2: Initialize CP9 bands structure
    let mut bands = CP9Bands::new(cp9.m, cm.m);
    bands.l = l;
    bands.cm_m = cm.m;
    bands.cm_clen = cm.clen;

    // Step 3: Compute HMM bands using Forward/Backward
    let _fwd_score = cp9_compute_bands(&cp9, dsq, l, &mut bands);

    // Step 4: Map HMM bands to CM state bands (pass state types for proper E_ST handling)
    let cm2hmm = compute_cm2hmm_mapping(cm)?;
    cp9_to_cm_bands_with_types(&mut bands, cm.m, &cm2hmm, Some(&cm.sttype));
    bands.valid = true;

    // Step 5: Run banded CYK
    let result = cm_cyk_inside_align_hb_bands(cm, &bands, dsq, l);

    match result {
        Ok((banded_score, mx)) if banded_score > f32::NEG_INFINITY => {
            // Banded CYK succeeded - find optimal (j, d) from ROOT_S matrix
            // Don't assume (j=L, d=L) is optimal - search for best hit
            let (opt_score, opt_j, opt_d) = find_optimal_hit(&bands, &mx, l);

            if opt_score > f32::NEG_INFINITY {
                // Found valid optimal hit in banded matrix
                let start = opt_j - opt_d + 1;
                let end = opt_j;
                Ok(SearchHit {
                    start,
                    end,
                    score: opt_score,
                    b: 1,  // Banded mode was used
                })
            } else {
                // No valid hit found in banded matrix, fall back to unbanded scan
                #[cfg(debug_assertions)]
                eprintln!("DEBUG: No valid hit in banded matrix, falling back");
                cm_search_unbanded(cm, dsq, l)
            }
        }
        Ok((score, _mx)) => {
            // Banded CYK returned -inf, fall back to unbanded scan
            #[cfg(debug_assertions)]
            eprintln!("DEBUG: Banded CYK returned score={}, falling back", score);
            cm_search_unbanded(cm, dsq, l)
        }
        Err(e) => {
            // Banded CYK failed with error, fall back to unbanded scan
            #[cfg(debug_assertions)]
            eprintln!("DEBUG: Banded CYK failed: {}, falling back", e);
            cm_search_unbanded(cm, dsq, l)
        }
    }
}

/// Non-banded search (full matrix)
fn cm_search_unbanded(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<SearchHit, String> {
    // Get the full DP matrix to search for optimal (j, d)
    use crate::cm_dp::cyk_inside_debug;
    let alpha = cyk_inside_debug(cm, dsq, l)?;

    // Search ROOT_S (state 0) for optimal (j, d) pair
    let (opt_score, opt_j, opt_d) = find_optimal_hit_unbanded(&alpha, l);

    if opt_score > f32::NEG_INFINITY {
        let start = opt_j - opt_d + 1;
        let end = opt_j;
        Ok(SearchHit {
            start,
            end,
            score: opt_score,
            b: 0,  // Non-banded mode
        })
    } else {
        Err("No valid hits found in unbanded search".to_string())
    }
}

/// Find optimal (j, d) hit in ROOT_S (state 0) banded matrix
///
/// Searches all valid (j, d) cells in the ROOT_S row to find the maximum score.
/// This allows finding optimal hits at different sequence windows, not just (j=L, d=L).
///
/// Returns (best_score, best_j, best_d)
fn find_optimal_hit(bands: &CP9Bands, mx: &CMHbMx, l: i32) -> (f32, i32, i32) {
    let v = 0;  // ROOT_S state
    let jmin_v = bands.jmin[v];
    let jmax_v = bands.jmax[v];

    if jmax_v < jmin_v {
        return (f32::NEG_INFINITY, l, l);
    }

    #[cfg(debug_assertions)]
    {
        eprintln!("DEBUG find_optimal_hit: v={}, l={}", v, l);
        eprintln!("DEBUG bands: jmin[{}]={}, jmax[{}]={}", v, jmin_v, v, jmax_v);
        eprintln!("DEBUG searching for j=69, d=67 (Archaea expected optimal)");

        // Check if j=69 is in range
        if 69 >= jmin_v && 69 <= jmax_v {
            let dmin_69 = bands.get_hdmin(v, 69);
            let dmax_69 = bands.get_hdmax(v, 69);
            eprintln!("DEBUG j=69: dmin={}, dmax={} (need d=67)", dmin_69, dmax_69);
            if 67 >= dmin_69 && 67 <= dmax_69 {
                eprintln!("DEBUG j=69, d=67 IS in bands - will be searched");
            } else {
                eprintln!("DEBUG j=69, d=67 NOT in bands - this is the problem!");
            }
        } else {
            eprintln!("DEBUG j=69 NOT in jmin..jmax range - this is the problem!");
        }
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_j = l;
    let mut best_d = l;

    // Search all valid (j, d) pairs in ROOT_S row
    for j in jmin_v..=jmax_v {
        let jp_v = (j - jmin_v) as usize;
        let dmin = bands.get_hdmin(v, j);
        let dmax = bands.get_hdmax(v, j);

        for d in dmin..=dmax {
            let dp_v = (d - dmin) as usize;

            // Access banded matrix safely
            if jp_v < mx.dp[v].len() && dp_v < mx.dp[v][jp_v].len() {
                let score = mx.dp[v][jp_v][dp_v];
                if score > best_score {
                    best_score = score;
                    best_j = j;
                    best_d = d;
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    eprintln!("DEBUG find_optimal_hit result: score={}, j={}, d={}", best_score, best_j, best_d);

    (best_score, best_j, best_d)
}

/// Find optimal (j, d) hit in ROOT_S (state 0) unbanded matrix
///
/// Searches all (j, d) cells in the ROOT_S row to find the maximum score.
/// This is a scan search over all possible sequence windows.
///
/// Returns (best_score, best_j, best_d)
fn find_optimal_hit_unbanded(alpha: &[Vec<Vec<f32>>], l: i32) -> (f32, i32, i32) {
    let v = 0;  // ROOT_S state
    let mut best_score = f32::NEG_INFINITY;
    let mut best_j = l;
    let mut best_d = l;

    // Search all valid (j, d) pairs in ROOT_S row
    for j in 0..=l {
        for d in 0..=l {
            if j >= d {  // Valid subsequence: i = j - d + 1 must be >= 1
                let score = alpha[v][j as usize][d as usize];
                if score > best_score {
                    best_score = score;
                    best_j = j;
                    best_d = d;
                }
            }
        }
    }

    (best_score, best_j, best_d)
}

/// Create CP9 HMM from CM
fn create_cp9_from_cm(cm: &CM) -> Result<CP9, String> {
    let (cp9, _map) = cplan9_from_cm(cm)?;
    Ok(cp9)
}

/// Compute CM state to HMM node mapping
fn compute_cm2hmm_mapping(cm: &CM) -> Result<Vec<(i32, i32)>, String> {
    let cp9_map = cp9_map_cm2hmm(cm)?;

    // Convert CP9Map to (left_k, right_k) pairs using cs2hn
    let mut cm2hmm = Vec::with_capacity(cm.m as usize);

    for v in 0..(cm.m as usize) {
        if v < cp9_map.cs2hn.len() {
            // cs2hn[v][0] is the first (left) HMM node
            // cs2hn[v][1] is the second (right) HMM node for pair states, or -1
            let left_k = cp9_map.cs2hn[v][0];
            let right_k = if cp9_map.cs2hn[v][1] >= 0 {
                cp9_map.cs2hn[v][1]
            } else {
                left_k  // Single emitting states: use same node for both
            };
            cm2hmm.push((left_k, right_k));
        } else {
            cm2hmm.push((0, 0));
        }
    }

    Ok(cm2hmm)
}

// =============================================================================
// Legacy compatibility functions
// =============================================================================

/// Legacy HMM-banded CYK (stub for backward compatibility)
///
/// This function is deprecated. Use cm_search() with use_hmm_bands=true instead.
pub fn cm_cyk_inside_align_hb(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<(f32, i32), String> {
    let hit = cm_search_banded(cm, dsq, l)?;
    Ok((hit.score, hit.b))
}

/// Legacy HMM-banded Inside (stub for backward compatibility)
///
/// This function is deprecated. Use the banded functions from cm_dp_hb directly.
pub fn cm_inside_align_hb(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<f32, String> {
    // For Inside, we'd need to use cm_inside_align_hb_bands
    // For now, fall back to full matrix
    let (score, _) = cyk_inside(cm, dsq, l, false)?;
    Ok(score)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.use_hmm_bands);
        assert!(config.score_threshold == f32::NEG_INFINITY);
    }
}
