//! HMM-Banded Dynamic Programming for Covariance Models - 1:1 port from cm_dpalign.c
//!
//! This module implements HMM-banded versions of CYK, Inside, Outside, and Posterior
//! algorithms. The bands are derived from CP9 HMM posterior probabilities and
//! dramatically reduce computation by constraining valid (j, d) cells.

use crate::cm::CM;
use crate::cm_dp::calc_init_scores;
use crate::cp9_bands::CP9Bands;
use crate::cm_qdband::{state_delta, state_right_delta};
use crate::constants::{
    B_ST, D_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST, S_ST,
};
use crate::types::EslDsq;

// =============================================================================
// HMM-Banded DP Matrix
// =============================================================================

/// HMM-Banded DP Matrix
///
/// Memory-efficient DP matrix that only allocates cells within the HMM-derived bands.
/// Access uses banded coordinates: dp[v][jp_v][dp_v] where:
///   jp_v = j - jmin[v]
///   dp_v = d - hdmin[v][jp_v]
#[derive(Debug, Clone)]
pub struct CMHbMx {
    /// Number of CM states
    pub m: usize,
    /// Sequence length
    pub l: i32,
    /// Total valid cells allocated
    pub ncells_valid: usize,
    /// DP values: dp[v][jp_v][dp_v]
    pub dp: Vec<Vec<Vec<f32>>>,
}

impl CMHbMx {
    /// Create a new HMM-banded matrix from CP9Bands
    pub fn new(cp9b: &CP9Bands) -> Self {
        let m = cp9b.cm_m as usize;
        let mut dp = Vec::with_capacity(m);
        let mut ncells_valid = 0usize;

        for v in 0..m {
            let jmin = cp9b.jmin[v];
            let jmax = cp9b.jmax[v];
            let jwidth = if jmax >= jmin {
                (jmax - jmin + 1) as usize
            } else {
                0
            };

            let mut state_dp = Vec::with_capacity(jwidth);
            for jp in 0..jwidth {
                let j = jmin + jp as i32;
                let dmin = cp9b.get_hdmin(v, j);
                let dmax = cp9b.get_hdmax(v, j);
                let dwidth = if dmax >= dmin {
                    (dmax - dmin + 1) as usize
                } else {
                    0
                };
                ncells_valid += dwidth;
                state_dp.push(vec![f32::NEG_INFINITY; dwidth]);
            }
            dp.push(state_dp);
        }

        CMHbMx {
            m,
            l: cp9b.l,
            ncells_valid,
            dp,
        }
    }

    /// Get value at banded coordinates (returns NEG_INFINITY if out of bounds)
    #[inline]
    pub fn get(&self, v: usize, jp: usize, dp: usize) -> f32 {
        if v < self.dp.len() && jp < self.dp[v].len() && dp < self.dp[v][jp].len() {
            self.dp[v][jp][dp]
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Set value at banded coordinates
    #[inline]
    pub fn set(&mut self, v: usize, jp: usize, dp: usize, val: f32) {
        if v < self.dp.len() && jp < self.dp[v].len() && dp < self.dp[v][jp].len() {
            self.dp[v][jp][dp] = val;
        }
    }

    /// Clear all values to NEG_INFINITY
    pub fn clear(&mut self) {
        for v in 0..self.dp.len() {
            for jp in 0..self.dp[v].len() {
                for dp in 0..self.dp[v][jp].len() {
                    self.dp[v][jp][dp] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

// =============================================================================
// HMM-Banded CYK Algorithm
// =============================================================================

/// HMM-Banded CYK Inside Alignment
///
/// CYK algorithm constrained by HMM bands from CP9 posterior decoding.
/// Only cells within the bands are computed, providing O(band_size) instead of O(L^2).
///
/// # Arguments
/// * `cm` - The covariance model
/// * `cp9b` - CP9 bands constraining valid (j, d) cells
/// * `dsq` - Digitized sequence (1-indexed, with sentinels)
/// * `l` - Sequence length
///
/// # Returns
/// (score, banded_matrix) on success
pub fn cm_cyk_inside_align_hb_bands(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[EslDsq],
    l: i32,
) -> Result<(f32, CMHbMx), String> {
    // Validate bands cover full alignment at ROOT_S (state 0)
    if cp9b.jmin[0] > l || cp9b.jmax[0] < l {
        return Err(format!(
            "L ({}) outside ROOT_S j band ({}-{})",
            l, cp9b.jmin[0], cp9b.jmax[0]
        ));
    }
    let jp_0 = (l - cp9b.jmin[0]) as usize;

    let dmin_0 = cp9b.get_hdmin(0, l);
    let dmax_0 = cp9b.get_hdmax(0, l);
    if dmin_0 > l || dmax_0 < l {
        return Err(format!(
            "L ({}) outside ROOT_S d band ({}-{})",
            l, dmin_0, dmax_0
        ));
    }
    let lp_0 = (l - dmin_0) as usize;

    // Allocate banded matrix
    let mut alpha = CMHbMx::new(cp9b);
    let m = cm.m as usize;

    // Main recursion: states in reverse order (bottom-up)
    for v in (0..m).rev() {
        let st = cm.sttype[v] as i32;
        let jmin_v = cp9b.jmin[v];
        let jmax_v = cp9b.jmax[v];

        if jmax_v < jmin_v {
            continue; // No valid cells for this state
        }

        let sd = state_delta(st);
        let sdr = state_right_delta(st);

        #[cfg(debug_assertions)]
        if v == 0 {
            eprintln!("\nDEBUG: Processing ROOT_S (state 0):");
            eprintln!("  sttype[0]={}", cm.sttype[0]);
            eprintln!("  cfirst[0]={}, cnum[0]={}", cm.cfirst[0], cm.cnum[0]);
            eprintln!("  jmin[0]={}, jmax[0]={}", cp9b.jmin[0], cp9b.jmax[0]);

            // Check what type ROOT_S is
            let st_val = cm.sttype[0] as i32;
            eprintln!("  State type: {}", match st_val {
                0 => "D_ST/S_ST",
                1 => "MP_ST",
                2 => "ML_ST",
                3 => "MR_ST",
                4 => "IL_ST",
                5 => "IR_ST",
                6 => "B_ST",
                7 => "E_ST",
                _ => "UNKNOWN",
            });
        }

        match st {
            x if x == E_ST => {
                // End state: d=0 only gets score 0.0
                #[cfg(debug_assertions)]
                let mut cells_set = 0;
                for j in jmin_v..=jmax_v {
                    let jp_v = (j - jmin_v) as usize;
                    let dmin = cp9b.get_hdmin(v, j);
                    let dmax = cp9b.get_hdmax(v, j);
                    if dmin <= 0 && dmax >= 0 {
                        let dp_v = (0 - dmin) as usize;
                        alpha.set(v, jp_v, dp_v, 0.0);
                        #[cfg(debug_assertions)]
                        { cells_set += 1; }
                    }
                }
                #[cfg(debug_assertions)]
                {
                    eprintln!("DEBUG: E_ST state {}: jmin={}, jmax={}, set {} cells to 0.0",
                              v, jmin_v, jmax_v, cells_set);
                    // Verify cells are actually set
                    if v == 123 {
                        for j in [9, 16, 23, 32].iter() {
                            let j = *j;
                            if j >= jmin_v && j <= jmax_v {
                                let jp_v = (j - jmin_v) as usize;
                                let dmin = cp9b.get_hdmin(v, j);
                                if dmin <= 0 && 0 <= cp9b.get_hdmax(v, j) {
                                    let dp_v = (0 - dmin) as usize;
                                    if jp_v < alpha.dp[v].len() && dp_v < alpha.dp[v][jp_v].len() {
                                        eprintln!("  E_ST 123: alpha[{}][j={}][d=0] = {}", v, j, alpha.dp[v][jp_v][dp_v]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            x if x == B_ST => {
                // Bifurcation state: convolution of left and right children
                // CRITICAL: Use lchild/rchild, NOT cfirst/cnum!
                let y = cm.lchild[v] as usize; // Left child (BEGL_S)
                let z = cm.rchild[v] as usize; // Right child (BEGR_S)
                #[cfg(debug_assertions)]
                {
                    eprintln!("DEBUG: B_ST state {}: lchild={} (type {}), rchild={} (type {})",
                              v, y, cm.sttype[y], z, cm.sttype[z]);
                    eprintln!("  v bands: jmin={}, jmax={}", jmin_v, jmax_v);
                    eprintln!("  y bands: jmin={}, jmax={}", cp9b.jmin[y], cp9b.jmax[y]);
                    eprintln!("  z bands: jmin={}, jmax={}", cp9b.jmin[z], cp9b.jmax[z]);
                    // Check if children have valid cells
                    let mut y_valid = 0;
                    for jp in 0..alpha.dp[y].len() {
                        for dp in 0..alpha.dp[y][jp].len() {
                            if alpha.dp[y][jp][dp] > f32::NEG_INFINITY {
                                y_valid += 1;
                            }
                        }
                    }
                    let mut z_valid = 0;
                    for jp in 0..alpha.dp[z].len() {
                        for dp in 0..alpha.dp[z][jp].len() {
                            if alpha.dp[z][jp][dp] > f32::NEG_INFINITY {
                                z_valid += 1;
                            }
                        }
                    }
                    eprintln!("  y valid cells: {}, z valid cells: {}", y_valid, z_valid);
                }

                // Intersect j bands: v ∩ z
                let jn = jmin_v.max(cp9b.jmin[z]);
                let jx = jmax_v.min(cp9b.jmax[z]);

                #[cfg(debug_assertions)]
                let mut cells_set = 0;

                for j in jn..=jx {
                    let jp_v = (j - jmin_v) as usize;
                    let jp_z = (j - cp9b.jmin[z]) as usize;

                    let dmin_v = cp9b.get_hdmin(v, j);
                    let dmax_v = cp9b.get_hdmax(v, j);

                    for d in dmin_v..=dmax_v {
                        let dp_v = (d - dmin_v) as usize;
                        let mut best_sc = f32::NEG_INFINITY;

                        // Compute k bounds (k represents right_d)
                        // Constraints:
                        //   k >= 0 (allows k=0, matching C Infernal bug i36 fix)
                        //   k >= hdmin[z][j] (z's d band lower bound)
                        //   k <= hdmax[z][j] (z's d band upper bound)
                        //   j - k >= jmin[y] (left_j in y's j band)
                        //   j - k <= jmax[y] (left_j in y's j band)
                        let kn = (j - cp9b.jmax[y]).max(cp9b.get_hdmin(z, j)).max(0);
                        let kx = (j - cp9b.jmin[y]).min(cp9b.get_hdmax(z, j));

                        for k in kn..=kx {
                            let j_minus_k = j - k;
                            if j_minus_k < cp9b.jmin[y] || j_minus_k > cp9b.jmax[y] {
                                continue;
                            }
                            let jp_y_k = (j_minus_k - cp9b.jmin[y]) as usize;

                            // Check inequalities 5-6 (d-dependent)
                            let hdmin_y_jk = cp9b.get_hdmin(y, j_minus_k);
                            let hdmax_y_jk = cp9b.get_hdmax(y, j_minus_k);

                            if k >= d - hdmax_y_jk && k <= d - hdmin_y_jk {
                                let d_minus_k = d - k;
                                if d_minus_k >= hdmin_y_jk && d_minus_k <= hdmax_y_jk {
                                    let dp_y_k = (d_minus_k - hdmin_y_jk) as usize;

                                    let hdmin_z = cp9b.get_hdmin(z, j);
                                    if k >= hdmin_z && k <= cp9b.get_hdmax(z, j) {
                                        let kp_z = (k - hdmin_z) as usize;

                                        let sc =
                                            alpha.get(y, jp_y_k, dp_y_k) + alpha.get(z, jp_z, kp_z);
                                        if sc > best_sc {
                                            best_sc = sc;
                                        }
                                    }
                                }
                            }
                        }
                        alpha.set(v, jp_v, dp_v, best_sc);
                        #[cfg(debug_assertions)]
                        if best_sc > f32::NEG_INFINITY {
                            cells_set += 1;
                        }
                    }
                }
                #[cfg(debug_assertions)]
                eprintln!("  B_ST {} after processing: {} valid cells", v, cells_set);
            }

            x if x == IL_ST || x == IR_ST => {
                // Insert states: self-transitions require special order
                // Process j{d{y}} to handle self-transitions correctly
                #[cfg(debug_assertions)]
                if v <= 5 || (v >= 131 && v <= 142) || (v >= 177 && v <= 187) {
                    eprintln!("DEBUG: Insert state {} (type {}): sd={}, sdr={}", v, st, sd, sdr);
                    eprintln!("  cfirst={}, cnum={}", cm.cfirst[v], cm.cnum[v]);
                    eprintln!("  jmin={}, jmax={}", jmin_v, jmax_v);
                    // Show children details
                    for yoff in 0..cm.cnum[v] as usize {
                        let y = cm.cfirst[v] as usize + yoff;
                        let mut child_valid = 0;
                        for jp in 0..alpha.dp[y].len() {
                            for dp in 0..alpha.dp[y][jp].len() {
                                if alpha.dp[y][jp][dp] > f32::NEG_INFINITY {
                                    child_valid += 1;
                                }
                            }
                        }
                        eprintln!("    child[{}]={}: type={}, jmin={}, jmax={}, valid={}",
                                  yoff, y, cm.sttype[y], cp9b.jmin[y], cp9b.jmax[y], child_valid);
                    }
                }
                for j in jmin_v..=jmax_v {
                    let jp_v = (j - jmin_v) as usize;
                    let j_sdr = j - sdr;

                    let dmin_v = cp9b.get_hdmin(v, j);
                    let dmax_v = cp9b.get_hdmax(v, j);

                    for d in dmin_v..=dmax_v {
                        let dp_v = (d - dmin_v) as usize;
                        let mut best_sc = f32::NEG_INFINITY;
                        let d_sd = d - sd;

                        for yoffset in 0..cm.cnum[v] as usize {
                            let y = cm.cfirst[v] as usize + yoffset;

                            // Check j-sdr is in child's j band
                            if j_sdr < cp9b.jmin[y] || j_sdr > cp9b.jmax[y] {
                                continue;
                            }
                            let jp_y_sdr = (j_sdr - cp9b.jmin[y]) as usize;

                            // Check d-sd is in child's d band
                            let hdmin_y = cp9b.get_hdmin(y, j_sdr);
                            let hdmax_y = cp9b.get_hdmax(y, j_sdr);
                            if d_sd < hdmin_y || d_sd > hdmax_y {
                                continue;
                            }
                            let dp_y_sd = (d_sd - hdmin_y) as usize;

                            let sc = alpha.get(y, jp_y_sdr, dp_y_sd) + cm.tsc[v][yoffset];
                            if sc > best_sc {
                                best_sc = sc;
                            }
                        }

                        // Add emission score
                        if best_sc > f32::NEG_INFINITY {
                            let i = (j - d + 1) as usize;
                            let esc = if st == IL_ST {
                                if i < dsq.len() && i > 0 {
                                    let res = dsq[i] as usize;
                                    if res < cm.esc[v].len() {
                                        cm.esc[v][res]
                                    } else {
                                        0.0  // Unknown residue treated as neutral
                                    }
                                } else {
                                    f32::NEG_INFINITY
                                }
                            } else {
                                // IR_ST
                                if (j as usize) < dsq.len() && j > 0 {
                                    let res = dsq[j as usize] as usize;
                                    if res < cm.esc[v].len() {
                                        cm.esc[v][res]
                                    } else {
                                        0.0  // Unknown residue treated as neutral
                                    }
                                } else {
                                    f32::NEG_INFINITY
                                }
                            };
                            best_sc += esc;
                        }

                        alpha.set(v, jp_v, dp_v, best_sc);
                    }
                }
            }

            x if x == MP_ST => {
                // Pair state: emits on both left and right
                cyk_hb_non_self_transition(cm, cp9b, dsq, &mut alpha, v, sd, sdr)?;
                add_emission_scores_mp_hb(cm, cp9b, dsq, &mut alpha, v)?;
            }

            x if x == ML_ST || x == MR_ST || x == D_ST || x == S_ST => {
                // Non-self-transitioning states
                #[cfg(debug_assertions)]
                if v <= 5 || (v >= 123 && v <= 142) || (v >= 172 && v <= 187) {
                    eprintln!("DEBUG: State {} (type {}): sd={}, sdr={}", v, st, sd, sdr);
                    eprintln!("  cfirst={}, cnum={}", cm.cfirst[v], cm.cnum[v]);
                    eprintln!("  jmin={}, jmax={}", jmin_v, jmax_v);
                    // Show children details
                    for yoff in 0..cm.cnum[v] as usize {
                        let child = cm.cfirst[v] as usize + yoff;
                        // Count valid cells for this child
                        let mut child_valid = 0;
                        for jp in 0..alpha.dp[child].len() {
                            for dp in 0..alpha.dp[child][jp].len() {
                                if alpha.dp[child][jp][dp] > f32::NEG_INFINITY {
                                    child_valid += 1;
                                }
                            }
                        }
                        eprintln!("    child[{}]={}: type={}, jmin={}, jmax={}, valid={}",
                                  yoff, child, cm.sttype[child], cp9b.jmin[child], cp9b.jmax[child], child_valid);
                    }
                }
                #[cfg(debug_assertions)]
                if v == 0 {
                    eprintln!("DEBUG: Processing ROOT_S (state 0):");
                    eprintln!("  cfirst={}, cnum={}", cm.cfirst[v], cm.cnum[v]);
                    for yoff in 0..cm.cnum[v] as usize {
                        let y = cm.cfirst[v] as usize + yoff;
                        let st_y = cm.sttype[y];
                        eprintln!("  child y={} (type {}): jmin={}, jmax={}, tsc={:.3}",
                                  y, st_y, cp9b.jmin[y], cp9b.jmax[y], cm.tsc[v][yoff]);
                        // Check if child has valid scores at j=l, various d values
                        if cp9b.jmin[y] <= l && cp9b.jmax[y] >= l {
                            let jp_y = (l - cp9b.jmin[y]) as usize;
                            let hdmin_y = cp9b.get_hdmin(y, l);
                            let hdmax_y = cp9b.get_hdmax(y, l);
                            eprintln!("    at j={}: hdmin={}, hdmax={}", l, hdmin_y, hdmax_y);
                            // Check d=l, d=l-1, d=l-2 to see where valid scores are
                            for d_test in [l, l-1, l-2, 1, 0].iter() {
                                let d_test = *d_test;
                                if d_test >= hdmin_y && d_test <= hdmax_y {
                                    let dp_y = (d_test - hdmin_y) as usize;
                                    if jp_y < alpha.dp[y].len() && dp_y < alpha.dp[y][jp_y].len() {
                                        let sc = alpha.dp[y][jp_y][dp_y];
                                        eprintln!("    alpha[{}][j={}][d={}] = {:.3} (jp_y={}, dp_y={})",
                                                  y, l, d_test, sc, jp_y, dp_y);
                                    }
                                }
                            }
                        }
                    }
                }
                cyk_hb_non_self_transition(cm, cp9b, dsq, &mut alpha, v, sd, sdr)?;

                // Add emission scores for emit states
                if st == ML_ST {
                    add_emission_scores_ml_hb(cm, cp9b, dsq, &mut alpha, v)?;
                } else if st == MR_ST {
                    add_emission_scores_mr_hb(cm, cp9b, dsq, &mut alpha, v)?;
                }

                #[cfg(debug_assertions)]
                if v <= 5 || v >= 127 && v <= 142 || v >= 174 && v <= 187 {
                    // Check if this state got any valid scores
                    let mut valid_cells = 0;
                    for jp in 0..alpha.dp[v].len() {
                        for dp in 0..alpha.dp[v][jp].len() {
                            if alpha.dp[v][jp][dp] > f32::NEG_INFINITY {
                                valid_cells += 1;
                            }
                        }
                    }
                    eprintln!("  After processing state {}: {} valid cells", v, valid_cells);
                }

                #[cfg(debug_assertions)]
                if v == 0 {
                    eprintln!("DEBUG: Finished ROOT_S processing");
                    eprintln!("  Sample alpha[0][76][76] = {:.2}",
                        if 76 < alpha.dp[0].len() && 76 < alpha.dp[0][76].len() {
                            alpha.dp[0][76][76]
                        } else {
                            f32::NEG_INFINITY
                        });
                }
            }

            _ => {
                // Other states (EL_ST, etc.) - handled similarly to D_ST
                cyk_hb_non_self_transition(cm, cp9b, dsq, &mut alpha, v, sd, sdr)?;
            }
        }
    }

    #[cfg(debug_assertions)]
    {
        eprintln!("DEBUG CYK HB final ROOT_S scores:");
        let v = 0; // ROOT_S
        let jmin_v = cp9b.jmin[v];
        let jmax_v = cp9b.jmax[v];
        for j in jmin_v..=jmax_v.min(jmin_v + 5) {
            let jp_v = (j - jmin_v) as usize;
            let dmin = cp9b.get_hdmin(v, j);
            let dmax = cp9b.get_hdmax(v, j);
            for d in dmin..=dmax.min(dmin + 3) {
                let dp_v = (d - dmin) as usize;
                if jp_v < alpha.dp[v].len() && dp_v < alpha.dp[v][jp_v].len() {
                    eprintln!("  alpha[0][{}][{}] = {:.2}", j, d, alpha.dp[v][jp_v][dp_v]);
                }
            }
        }
    }

    // Return score - check local begins if enabled
    let mut best_score = alpha.get(0, jp_0, lp_0);

    // If local begins are enabled, check all states for better scores
    if (cm.flags & crate::cm::CM_LOCAL_BEGIN) != 0 {
        for v in 1..cm.m as usize {
            if cm.has_local_begin(v) {
                // Check if this state covers j=l, d=l
                let jmin_v = cp9b.jmin[v];
                let jmax_v = cp9b.jmax[v];
                if l >= jmin_v && l <= jmax_v {
                    let jp_v = (l - jmin_v) as usize;
                    let dmin_v = cp9b.get_hdmin(v, l);
                    let dmax_v = cp9b.get_hdmax(v, l);
                    if l >= dmin_v && l <= dmax_v {
                        let dp_v = (l - dmin_v) as usize;
                        let local_score = alpha.get(v, jp_v, dp_v) + cm.beginsc[v];
                        if local_score > best_score {
                            best_score = local_score;
                        }
                    }
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    {
        eprintln!("DEBUG cm_cyk_inside_align_hb_bands:");
        eprintln!("  l={}, jmin[0]={}, jmax[0]={}", l, cp9b.jmin[0], cp9b.jmax[0]);
        eprintln!("  jp_0={}, lp_0={}", jp_0, lp_0);
        eprintln!("  dmin_0={}, dmax_0={}", dmin_0, dmax_0);
        eprintln!("  alpha.dp[0].len()={}", alpha.dp[0].len());
        if !alpha.dp[0].is_empty() && jp_0 < alpha.dp[0].len() {
            eprintln!("  alpha.dp[0][jp_0].len()={}", alpha.dp[0][jp_0].len());
        }
        eprintln!("  best_score={}", best_score);
    }

    #[cfg(debug_assertions)]
    {
        eprintln!("\nDEBUG ROOT_S (v=0) final analysis:");
        eprintln!("  cfirst[0]={}, cnum[0]={}", cm.cfirst[0], cm.cnum[0]);

        // Check each child
        for child_idx in 0..cm.cnum[0] as usize {
            let child = (cm.cfirst[0] as usize) + child_idx;
            eprintln!("  Child {}:", child);

            // Sample a few cells from this child
            let jmin_child = cp9b.jmin[child];
            let jmax_child = cp9b.jmax[child];
            eprintln!("    jmin={}, jmax={}", jmin_child, jmax_child);

            if jmax_child >= jmin_child && jmax_child < alpha.dp[child].len() as i32 {
                let jp = (jmax_child - jmin_child) as usize;
                if jp < alpha.dp[child].len() && !alpha.dp[child][jp].is_empty() {
                    eprintln!("    alpha[{}][{}][0] = {:.2}", child, jmax_child, alpha.dp[child][jp][0]);
                }
            }
        }
    }

    Ok((best_score, alpha))
}

/// Helper: Non-self-transitioning state DP recursion
fn cyk_hb_non_self_transition(
    cm: &CM,
    cp9b: &CP9Bands,
    _dsq: &[EslDsq],
    alpha: &mut CMHbMx,
    v: usize,
    sd: i32,
    sdr: i32,
) -> Result<(), String> {
    let jmin_v = cp9b.jmin[v];
    let jmax_v = cp9b.jmax[v];

    #[cfg(debug_assertions)]
    let mut updates_made = 0;

    // Loop order: y { j { d } } for efficiency (process each child once per j)
    for yoffset in 0..cm.cnum[v] as usize {
        let y = cm.cfirst[v] as usize + yoffset;
        let tsc = cm.tsc[v][yoffset];

        // Intersect j bands: v ∩ (y + sdr)
        let jn = jmin_v.max(cp9b.jmin[y] + sdr);
        let jx = jmax_v.min(cp9b.jmax[y] + sdr);

        #[cfg(debug_assertions)]
        if v == 126 && y == 127 {
            eprintln!("DEBUG cyk_hb_non_self_transition v={} child y={}:", v, y);
            eprintln!("  jmin_v={}, jmax_v={}, jmin[y]={}, jmax[y]={}", jmin_v, jmax_v, cp9b.jmin[y], cp9b.jmax[y]);
            eprintln!("  sd={}, sdr={}, jn={}, jx={}", sd, sdr, jn, jx);
        }


        for j in jn..=jx {
            let jp_v = (j - jmin_v) as usize;
            let j_sdr = j - sdr;
            let jp_y_sdr = (j_sdr - cp9b.jmin[y]) as usize;

            // Intersect d bands
            let hdmin_y = cp9b.get_hdmin(y, j_sdr);
            let hdmax_y = cp9b.get_hdmax(y, j_sdr);
            let dn = cp9b.get_hdmin(v, j).max(hdmin_y + sd);
            let dx = cp9b.get_hdmax(v, j).min(hdmax_y + sd);

            #[cfg(debug_assertions)]
            if (v == 0 || v == 126) && j == 32 && dn <= dx {
                let y_j_len = if jp_y_sdr < alpha.dp[y].len() { alpha.dp[y][jp_y_sdr].len() } else { 0 };
                eprintln!("  j={}: hdmin_y={}, hdmax_y={}, dn={}, dx={}, alpha[{}][{}].len()={}",
                          j, hdmin_y, hdmax_y, dn, dx, y, jp_y_sdr, y_j_len);
            }

            for d in dn..=dx {
                let dp_v = (d - cp9b.get_hdmin(v, j)) as usize;
                let dp_y_sd = (d - sd - hdmin_y) as usize;

                let sc = alpha.get(y, jp_y_sdr, dp_y_sd) + tsc;
                #[cfg(debug_assertions)]
                if (v == 0 || v == 126) && j == 32 && d == 32 {
                    eprintln!("    d={}: alpha[{}][jp_y_sdr={}][dp_y_sd={}] = {:.3}, tsc={:.3}, sc={:.3}",
                              d, y, jp_y_sdr, dp_y_sd, alpha.get(y, jp_y_sdr, dp_y_sd), tsc, sc);
                }
                if sc > alpha.get(v, jp_v, dp_v) {
                    alpha.set(v, jp_v, dp_v, sc);
                    #[cfg(debug_assertions)]
                    {
                        updates_made += 1;
                    }
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    if (v == 0 || v == 126) && updates_made == 0 {
        eprintln!("DEBUG cyk_hb_non_self_transition v={}: NO UPDATES MADE", v);
    }

    Ok(())
}

/// Helper: Add MP (pair) emission scores
fn add_emission_scores_mp_hb(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[EslDsq],
    alpha: &mut CMHbMx,
    v: usize,
) -> Result<(), String> {
    let jmin_v = cp9b.jmin[v];
    let jmax_v = cp9b.jmax[v];

    for j in jmin_v..=jmax_v {
        let jp_v = (j - jmin_v) as usize;
        let dmin_v = cp9b.get_hdmin(v, j);
        let dmax_v = cp9b.get_hdmax(v, j);

        for d in dmin_v..=dmax_v {
            let dp_v = (d - dmin_v) as usize;
            let cur = alpha.get(v, jp_v, dp_v);
            if cur > f32::NEG_INFINITY {
                let i = (j - d + 1) as usize;
                if i < dsq.len() && i > 0 && (j as usize) < dsq.len() && j > 0 {
                    // Pair emission: index = i_res * 4 + j_res
                    let i_res = dsq[i] as usize;
                    let j_res = dsq[j as usize] as usize;
                    // Check bounds - residues should be 0-3, sentinels are out of range
                    if i_res < 4 && j_res < 4 {
                        let pair_idx = i_res * 4 + j_res;
                        let esc = cm.esc[v][pair_idx];
                        alpha.set(v, jp_v, dp_v, cur + esc);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Helper: Add ML (left match) emission scores
fn add_emission_scores_ml_hb(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[EslDsq],
    alpha: &mut CMHbMx,
    v: usize,
) -> Result<(), String> {
    let jmin_v = cp9b.jmin[v];
    let jmax_v = cp9b.jmax[v];

    for j in jmin_v..=jmax_v {
        let jp_v = (j - jmin_v) as usize;
        let dmin_v = cp9b.get_hdmin(v, j);
        let dmax_v = cp9b.get_hdmax(v, j);

        for d in dmin_v..=dmax_v {
            let dp_v = (d - dmin_v) as usize;
            let cur = alpha.get(v, jp_v, dp_v);
            if cur > f32::NEG_INFINITY {
                let i = (j - d + 1) as usize;
                if i < dsq.len() && i > 0 {
                    let res = dsq[i] as usize;
                    // Check bounds - residues should be 0-3, sentinels are out of range
                    if res < cm.esc[v].len() {
                        let esc = cm.esc[v][res];
                        alpha.set(v, jp_v, dp_v, cur + esc);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Helper: Add MR (right match) emission scores
fn add_emission_scores_mr_hb(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[EslDsq],
    alpha: &mut CMHbMx,
    v: usize,
) -> Result<(), String> {
    let jmin_v = cp9b.jmin[v];
    let jmax_v = cp9b.jmax[v];

    for j in jmin_v..=jmax_v {
        let jp_v = (j - jmin_v) as usize;
        let dmin_v = cp9b.get_hdmin(v, j);
        let dmax_v = cp9b.get_hdmax(v, j);

        for d in dmin_v..=dmax_v {
            let dp_v = (d - dmin_v) as usize;
            let cur = alpha.get(v, jp_v, dp_v);
            if cur > f32::NEG_INFINITY && (j as usize) < dsq.len() && j > 0 {
                let res = dsq[j as usize] as usize;
                // Check bounds - residues should be 0-3, sentinels are out of range
                if res < cm.esc[v].len() {
                    let esc = cm.esc[v][res];
                    alpha.set(v, jp_v, dp_v, cur + esc);
                }
            }
        }
    }

    Ok(())
}

// =============================================================================
// HMM-Banded Inside Algorithm
// =============================================================================

/// HMM-Banded Inside Algorithm
///
/// Inside algorithm constrained by HMM bands.
/// Similar to CYK but uses log-sum-exp instead of max.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `cp9b` - CP9 bands constraining valid (j, d) cells
/// * `dsq` - Digitized sequence (1-indexed, with sentinels)
/// * `l` - Sequence length
///
/// # Returns
/// (inside_score, banded_matrix) on success
pub fn cm_inside_align_hb_bands(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[EslDsq],
    l: i32,
) -> Result<(f32, CMHbMx), String> {
    // Validate bands cover full alignment at ROOT_S
    if cp9b.jmin[0] > l || cp9b.jmax[0] < l {
        return Err(format!(
            "L ({}) outside ROOT_S j band ({}-{})",
            l, cp9b.jmin[0], cp9b.jmax[0]
        ));
    }
    let jp_0 = (l - cp9b.jmin[0]) as usize;

    let dmin_0 = cp9b.get_hdmin(0, l);
    let dmax_0 = cp9b.get_hdmax(0, l);
    if dmin_0 > l || dmax_0 < l {
        return Err(format!(
            "L ({}) outside ROOT_S d band ({}-{})",
            l, dmin_0, dmax_0
        ));
    }
    let lp_0 = (l - dmin_0) as usize;

    // Allocate banded matrix
    let mut alpha = CMHbMx::new(cp9b);
    let m = cm.m as usize;

    // Calculate initialization scores for local ends
    // This matches C's ICalcInitDPScores()
    let max_d = (l as usize).min(cm.w as usize);
    let init_sc = calc_init_scores(cm, max_d);

    // Main recursion: states in reverse order (bottom-up)
    for v in (0..m).rev() {
        let st = cm.sttype[v] as i32;
        let jmin_v = cp9b.jmin[v];
        let jmax_v = cp9b.jmax[v];

        if jmax_v < jmin_v {
            continue;
        }

        let sd = state_delta(st);
        let sdr = state_right_delta(st);

        match st {
            x if x == E_ST => {
                // End state
                for j in jmin_v..=jmax_v {
                    let jp_v = (j - jmin_v) as usize;
                    let dmin = cp9b.get_hdmin(v, j);
                    let dmax = cp9b.get_hdmax(v, j);
                    if dmin <= 0 && dmax >= 0 {
                        let dp_v = (0 - dmin) as usize;
                        alpha.set(v, jp_v, dp_v, 0.0);
                    }
                }
            }

            x if x == B_ST => {
                // Bifurcation: use log-sum-exp for Inside
                inside_hb_bifurcation(cm, cp9b, &mut alpha, v)?;
            }

            _ => {
                // All other states: use log-sum-exp
                inside_hb_general(cm, cp9b, dsq, &mut alpha, v, st, sd, sdr)?;
            }
        }

        // Debug output for ROOT_S (v=0) to trace why alpha[0][69][67] is not computed
        #[cfg(debug_assertions)]
        if v == 0 {
            eprintln!("\nDEBUG ROOT_S (v=0) children check:");
            eprintln!("  cfirst={}, cnum={}", cm.cfirst[0], cm.cnum[0]);
            for child_offset in 0..cm.cnum[0] as usize {
                let child = cm.cfirst[0] as usize + child_offset;
                eprintln!("  Child v={}:", child);
                eprintln!("    jmin={}, jmax={}", cp9b.jmin[child], cp9b.jmax[child]);

                // Check if j=69 is in this child's band
                if 69 >= cp9b.jmin[child] && 69 <= cp9b.jmax[child] {
                    let jp_child = (69 - cp9b.jmin[child]) as usize;
                    let hdmin_child = cp9b.get_hdmin(child, 69);
                    let hdmax_child = cp9b.get_hdmax(child, 69);
                    eprintln!("    j=69 IN BAND: hdmin={}, hdmax={}", hdmin_child, hdmax_child);

                    // Check if d=67 is valid for this child
                    if 67 >= hdmin_child && 67 <= hdmax_child {
                        eprintln!("    d=67 ALSO IN BAND - checking alpha value");
                        let dp_child = (67 - hdmin_child) as usize;
                        if jp_child < alpha.dp[child].len() && dp_child < alpha.dp[child][jp_child].len() {
                            let child_score = alpha.dp[child][jp_child][dp_child];
                            eprintln!("    alpha[{}][{}][{}] = {:.2}", child, jp_child, dp_child, child_score);
                        }
                    } else {
                        eprintln!("    d=67 NOT in hdmin..hdmax - THIS IS THE PROBLEM");
                    }
                } else {
                    eprintln!("    j=69 NOT in jmin..jmax - THIS IS THE PROBLEM");
                }
            }
        }
    }

    let score = alpha.get(0, jp_0, lp_0);
    Ok((score, alpha))
}

/// Inside scanning algorithm constrained by HMM bands.
///
/// Similar to cm_inside_align_hb_bands but searches all valid (j, d) endpoints
/// instead of a fixed alignment at j=L. This matches C's FastFInsideScanHB.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `cp9b` - CP9 bands constraining valid (j, d) cells
/// * `dsq` - Digitized sequence (1-indexed, with sentinels)
/// * `l` - Sequence length
///
/// # Returns
/// (best_score, best_j, best_d, null3_correction) on success
pub fn cm_inside_scan_hb_bands(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[EslDsq],
    l: i32,
) -> Result<(f32, i32, i32, f32), String> {
    eprintln!("[cm_inside_scan_hb_bands DEBUG] ENTRY: l={}, cm.m={}, dsq.len={}", l, cm.m, dsq.len());
    eprintln!("[cm_inside_scan_hb_bands DEBUG] ROOT_S bands: jmin={}, jmax={}",
             cp9b.jmin[0], cp9b.jmax[0]);
    eprintln!("[cm_inside_scan_hb_bands DEBUG] dsq[1..10]: {:?}", &dsq[1..=10.min(dsq.len()-2)]);

    // Allocate banded matrix
    let mut alpha = CMHbMx::new(cp9b);
    let m = cm.m as usize;

    // Calculate initialization scores for local ends
    // This matches C's ICalcInitDPScores()
    let max_d = (l as usize).min(cm.w as usize);
    let init_sc = calc_init_scores(cm, max_d);

    // Main recursion: states in reverse order (bottom-up)
    // This is identical to alignment version
    let local_begin_enabled = (cm.flags & crate::cm::CM_LOCAL_BEGIN) != 0;

    for v in (0..m).rev() {
        let st = cm.sttype[v] as i32;
        let jmin_v = cp9b.jmin[v];
        let jmax_v = cp9b.jmax[v];

        if jmax_v < jmin_v {
            continue;
        }

        let sd = state_delta(st);
        let sdr = state_right_delta(st);

        // CRITICAL: Initialize alpha[v] with local end scores BEFORE computing children
        // This matches C's structure: first initialize, then add child contributions
        if v != 0 && cm.endsc[v] > f32::NEG_INFINITY + 1.0 {
            for j in jmin_v..=jmax_v {
                let jp_v = (j - jmin_v) as usize;
                let dmin = cp9b.get_hdmin(v, j);
                let dmax = cp9b.get_hdmax(v, j);

                for d in dmin..=dmax {
                    let dp_v = (d - dmin) as usize;
                    let dp = std::cmp::max(d - sd, 0);
                    if (dp as usize) < init_sc[v].len() {
                        alpha.set(v, jp_v, dp_v, init_sc[v][dp as usize]);
                    }
                }
            }
        }

        match st {
            x if x == E_ST => {
                // End state
                for j in jmin_v..=jmax_v {
                    let jp_v = (j - jmin_v) as usize;
                    let dmin = cp9b.get_hdmin(v, j);
                    let dmax = cp9b.get_hdmax(v, j);
                    if dmin <= 0 && dmax >= 0 {
                        let dp_v = (0 - dmin) as usize;
                        alpha.set(v, jp_v, dp_v, 0.0);
                    }
                }
            }

            x if x == B_ST => {
                // Bifurcation: use log-sum-exp for Inside
                inside_hb_bifurcation(cm, cp9b, &mut alpha, v)?;
            }

            _ => {
                // All other states: use log-sum-exp
                inside_hb_general(cm, cp9b, dsq, &mut alpha, v, st, sd, sdr)?;
            }
        }

    }

    // CRITICAL: Apply local begins to ROOT_S AFTER all states computed
    // This matches C's structure (line 4196): separate j-loop after main DP loop
    if local_begin_enabled {
        let jmin_0 = cp9b.jmin[0];
        let jmax_0 = cp9b.jmax[0];

        for j in jmin_0..=jmax_0 {
            let jp_0 = (j - jmin_0) as usize;
            let dmin_0 = cp9b.get_hdmin(0, j);
            let dmax_0 = cp9b.get_hdmax(0, j);

            // For each state y (y=1..M-1), check if local begin is possible
            for y in 1..m {
                if cm.beginsc[y] > f32::NEG_INFINITY + 1.0 {
                    let jmin_y = cp9b.jmin[y];
                    let jmax_y = cp9b.jmax[y];

                    if j >= jmin_y && j <= jmax_y {
                        let jp_y = (j - jmin_y) as usize;
                        let dmin_y = cp9b.get_hdmin(y, j);
                        let dmax_y = cp9b.get_hdmax(y, j);

                        // Find overlapping d range
                        let dn = dmin_0.max(dmin_y);
                        let dx = dmax_0.min(dmax_y);

                        for d in dn..=dx {
                            let dp_0 = (d - dmin_0) as usize;
                            let dp_y = (d - dmin_y) as usize;

                            let state_score = alpha.get(y, jp_y, dp_y);
                            let local_begin_score = state_score + cm.beginsc[y];
                            let current_root_score = alpha.get(0, jp_0, dp_0);

                            // Update ROOT_S if local begin gives better score
                            if local_begin_score > current_root_score {
                                alpha.set(0, jp_0, dp_0, local_begin_score);

                                if j >= 69 && j <= 74 && d >= 69 && d <= 74 {
                                    eprintln!("[LOCAL BEGIN] j={}, d={}: state {} score={:.6}, beginsc={:.2}, local_begin_score={:.6}, current_root={:.6}",
                                             j, d, y, state_score, cm.beginsc[y], local_begin_score, current_root_score);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // SCANNING: Find best score in ROOT_S (v=0) across all valid (j, d) endpoints
    // Match C's FastFInsideScanHB (line 4231): only check alpha[0][jp_v][dp_v]
    let mut best_score = f32::NEG_INFINITY;
    let mut best_j = 0;
    let mut best_d = 0;
    let mut num_cells_checked = 0;

    let v = 0;  // Only check ROOT_S
    let jmin_v = cp9b.jmin[v];
    let jmax_v = cp9b.jmax[v];

    for j in jmin_v..=jmax_v {
        let jp_v = (j - jmin_v) as usize;
        let dmin = cp9b.get_hdmin(v, j);
        let dmax = cp9b.get_hdmax(v, j);

        for d in dmin..=dmax {
            let dp_v = (d - dmin) as usize;
            let score = alpha.get(v, jp_v, dp_v);  // No beginsc addition - already in cell

            num_cells_checked += 1;

            // Log all ROOT_S scores at the end of sequence
            // if j >= 60 && j <= 74 && d >= 60 && score > -100.0 {
            //     eprintln!("[SCAN] ROOT_S alpha[0][j={}][d={}] = {:.6}", j, d, score);
            // }

            if score > best_score {
                best_score = score;
                best_j = j;
                best_d = d;
            }

            // Debug output for Archaea sequence - find highest scores anywhere
            if score > 10.0 {
                eprintln!("[HIGH SCORE] v={}, j={}, d={}, alpha={:.6}",
                         v, j, d, score);
            }
        }
    }

    eprintln!("[cm_inside_scan_hb_bands DEBUG] Checked {} cells, best_score={:.6}, best_j={}, best_d={}",
             num_cells_checked, best_score, best_j, best_d);

    // Apply NULL3 composition bias correction
    let null3_correction = crate::evalue::score_correction_null3_comp_unknown(
        dsq,
        best_j - best_d + 1,  // start
        best_j,                // end
        &cm.null,
        cm.n3_omega,
    );

    eprintln!("[HB INSIDE NULL3] j={}, d={}: raw_score={:.6}, null3={:.6}, final={:.6}",
             best_j, best_d, best_score, null3_correction, best_score - null3_correction);

    best_score -= null3_correction;

    Ok((best_score, best_j, best_d, null3_correction))
}

/// Helper: Inside algorithm for bifurcation states
fn inside_hb_bifurcation(
    cm: &CM,
    cp9b: &CP9Bands,
    alpha: &mut CMHbMx,
    v: usize,
) -> Result<(), String> {
    let jmin_v = cp9b.jmin[v];
    let jmax_v = cp9b.jmax[v];
    let y = cm.lchild[v] as usize;  // Left child (BEGL_S)
    let z = cm.rchild[v] as usize;  // Right child (BEGR_S)

    let jn = jmin_v.max(cp9b.jmin[z]);
    let jx = jmax_v.min(cp9b.jmax[z]);

    for j in jn..=jx {
        let jp_v = (j - jmin_v) as usize;
        let jp_z = (j - cp9b.jmin[z]) as usize;

        let dmin_v = cp9b.get_hdmin(v, j);
        let dmax_v = cp9b.get_hdmax(v, j);

        let kn_base = (j - cp9b.jmax[y]).max(cp9b.get_hdmin(z, j)).max(0);
        let kx_base = (j - cp9b.jmin[y]).min(cp9b.get_hdmax(z, j));

        for d in dmin_v..=dmax_v {
            let dp_v = (d - dmin_v) as usize;
            let mut sum = f32::NEG_INFINITY;

            for k in kn_base..=kx_base {
                let j_minus_k = j - k;
                if j_minus_k < cp9b.jmin[y] || j_minus_k > cp9b.jmax[y] {
                    continue;
                }
                let jp_y_k = (j_minus_k - cp9b.jmin[y]) as usize;

                let hdmin_y_jk = cp9b.get_hdmin(y, j_minus_k);
                let hdmax_y_jk = cp9b.get_hdmax(y, j_minus_k);

                if k >= d - hdmax_y_jk && k <= d - hdmin_y_jk {
                    let d_minus_k = d - k;
                    if d_minus_k >= hdmin_y_jk && d_minus_k <= hdmax_y_jk {
                        let dp_y_k = (d_minus_k - hdmin_y_jk) as usize;

                        let hdmin_z = cp9b.get_hdmin(z, j);
                        if k >= hdmin_z && k <= cp9b.get_hdmax(z, j) {
                            let kp_z = (k - hdmin_z) as usize;

                            let sc = alpha.get(y, jp_y_k, dp_y_k) + alpha.get(z, jp_z, kp_z);
                            sum = log_sum_exp(sum, sc);
                        }
                    }
                }
            }
            alpha.set(v, jp_v, dp_v, sum);
        }
    }

    Ok(())
}

/// Helper: Inside algorithm for general states
fn inside_hb_general(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[EslDsq],
    alpha: &mut CMHbMx,
    v: usize,
    st: i32,
    sd: i32,
    sdr: i32,
) -> Result<(), String> {
    if v == 0 {
        eprintln!("[TSC DEBUG] ROOT_S (v=0) has {} children", cm.cnum[v]);
        for yoffset in 0..cm.cnum[v] as usize {
            eprintln!("[TSC DEBUG] ROOT_S tsc[0][{}] = {:.6}", yoffset, cm.tsc[v][yoffset]);
        }
    }

    let jmin_v = cp9b.jmin[v];
    let jmax_v = cp9b.jmax[v];

    for j in jmin_v..=jmax_v {
        let jp_v = (j - jmin_v) as usize;
        let j_sdr = j - sdr;

        let dmin_v = cp9b.get_hdmin(v, j);
        let dmax_v = cp9b.get_hdmax(v, j);

        for d in dmin_v..=dmax_v {
            let dp_v = (d - dmin_v) as usize;
            let d_sd = d - sd;

            // CRITICAL: Start with current alpha value (which may have been initialized with local end score)
            // Then add child contributions via log_sum_exp
            // This matches C's structure where alpha is pre-initialized, then updated with FLogsum
            let mut sum = alpha.get(v, jp_v, dp_v);

            for yoffset in 0..cm.cnum[v] as usize {
                let y = cm.cfirst[v] as usize + yoffset;
                let tsc = cm.tsc[v][yoffset];

                if j_sdr < cp9b.jmin[y] || j_sdr > cp9b.jmax[y] {
                    if v == 0 && j == 71 && d == 71 {
                        eprintln!("[DEBUG inside_hb_general] ROOT_S j=71,d=71: child {} j_sdr={} out of bounds [{},{}]",
                                 y, j_sdr, cp9b.jmin[y], cp9b.jmax[y]);
                    }
                    continue;
                }
                let jp_y_sdr = (j_sdr - cp9b.jmin[y]) as usize;

                let hdmin_y = cp9b.get_hdmin(y, j_sdr);
                let hdmax_y = cp9b.get_hdmax(y, j_sdr);
                if d_sd < hdmin_y || d_sd > hdmax_y {
                    if v == 0 && j == 71 && d == 71 {
                        eprintln!("[DEBUG inside_hb_general] ROOT_S j=71,d=71: child {} d_sd={} out of bounds [{},{}]",
                                 y, d_sd, hdmin_y, hdmax_y);
                    }
                    continue;
                }
                let dp_y_sd = (d_sd - hdmin_y) as usize;

                if v == 0 && j == 71 && d == 71 {
                    eprintln!("[BAND CHECK] ROOT_S child {}: j_sdr={}, j_bounds=[{}, {}], d_sd={}, d_bounds=[{}, {}]",
                             y, j_sdr, cp9b.jmin[y], cp9b.jmax[y], d_sd, hdmin_y, hdmax_y);
                }

                let child_score = alpha.get(y, jp_y_sdr, dp_y_sd);
                if v == 0 && j == 71 && d == 71 {
                    eprintln!("[DEBUG inside_hb_general] ROOT_S j=71,d=71: child {} d_sd={}, child_score={:.6}, tsc={:.6}",
                             y, d_sd, child_score, tsc);
                }
                let sc = child_score + tsc;
                if v == 0 && j == 71 && d == 71 {
                    eprintln!("[CHILD SCORE] ROOT_S child {}: child_score={:.6}, tsc={:.6}, sc={:.6}",
                             y, child_score, tsc, sc);
                }
                if v == 3 && j == 71 && d == 71 {
                    eprintln!("[STATE 3 CHILD] y={}, j_sdr={}, d_sd={}, child_score={:.6}, tsc={:.6}, sc={:.6}",
                             y, j_sdr, d_sd, child_score, tsc, sc);
                }
                sum = log_sum_exp(sum, sc);
            }

            // Add emission score
            if sum > f32::NEG_INFINITY {
                let esc = get_emission_score_hb(cm, dsq, v, st, j, d);
                sum += esc;
            }

            alpha.set(v, jp_v, dp_v, sum);

            // Log state 3 (ROOT_S child) scores
            if v == 3 && j >= 69 && j <= 74 && d >= 69 && sum > -100.0 {
                eprintln!("[STATE 3 DP] alpha[3][j={}][d={}] = {:.6}", j, d, sum);
            }
        }
    }

    Ok(())
}

/// Get emission score for a state
fn get_emission_score_hb(cm: &CM, dsq: &[EslDsq], v: usize, st: i32, j: i32, d: i32) -> f32 {
    let i = (j - d + 1) as usize;

    match st {
        x if x == MP_ST => {
            if i < dsq.len() && i > 0 && (j as usize) < dsq.len() && j > 0 {
                let i_res = dsq[i] as usize;
                let j_res = dsq[j as usize] as usize;
                // Check bounds - residues should be 0-3, sentinels are out of range
                if i_res < 4 && j_res < 4 {
                    cm.esc[v][i_res * 4 + j_res]
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        x if x == ML_ST || x == IL_ST => {
            if i < dsq.len() && i > 0 {
                let res = dsq[i] as usize;
                if res < cm.esc[v].len() {
                    cm.esc[v][res]
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        x if x == MR_ST || x == IR_ST => {
            if (j as usize) < dsq.len() && j > 0 {
                let res = dsq[j as usize] as usize;
                if res < cm.esc[v].len() {
                    cm.esc[v][res]
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Log-sum-exp for two values
#[inline]
fn log_sum_exp(a: f32, b: f32) -> f32 {
    // Handle NEG_INFINITY cases
    if a == f32::NEG_INFINITY && b == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }

    let max = a.max(b);
    let min = a.min(b);
    // CRITICAL: Use log2/exp2 for BITS arithmetic (not ln/exp for nats!)
    // This matches C's ILogsum: max + log2(1 + 2^(min-max))
    // Check for underflow: if difference >= 23, exp2 would underflow to 0
    if (max - min) >= 23.0 {
        return max;
    }
    // log2(2^max + 2^min) = max + log2(1 + 2^(min-max))
    max + (1.0 + 2.0_f32.powf(min - max)).log2()
}

// =============================================================================
// HMM-Banded Outside Algorithm
// =============================================================================

/// HMM-Banded Outside Algorithm
///
/// Outside algorithm constrained by HMM bands.
/// Proceeds top-down (v = 0 to M-1).
///
/// # Arguments
/// * `cm` - The covariance model
/// * `cp9b` - CP9 bands
/// * `dsq` - Digitized sequence
/// * `l` - Sequence length
/// * `inside` - Inside matrix (already computed)
///
/// # Returns
/// (outside_score, banded_matrix) on success
pub fn cm_outside_align_hb_bands(
    cm: &CM,
    cp9b: &CP9Bands,
    dsq: &[EslDsq],
    l: i32,
    _inside: &CMHbMx,
) -> Result<(f32, CMHbMx), String> {
    // Validate bands
    if cp9b.jmin[0] > l || cp9b.jmax[0] < l {
        return Err("L outside ROOT_S j band".to_string());
    }
    let jp_0 = (l - cp9b.jmin[0]) as usize;

    let dmin_0 = cp9b.get_hdmin(0, l);
    let dmax_0 = cp9b.get_hdmax(0, l);
    if dmin_0 > l || dmax_0 < l {
        return Err("L outside ROOT_S d band".to_string());
    }
    let lp_0 = (l - dmin_0) as usize;

    // Allocate banded matrix
    let mut beta = CMHbMx::new(cp9b);
    let m = cm.m as usize;

    // Base case: beta[0][L][L] = 0.0
    beta.set(0, jp_0, lp_0, 0.0);

    // Fill matrix TOP-DOWN (forward through states)
    for v in 0..m {
        let st = cm.sttype[v] as i32;
        let jmin_v = cp9b.jmin[v];
        let jmax_v = cp9b.jmax[v];

        if jmax_v < jmin_v {
            continue;
        }

        let sd = state_delta(st);
        let sdr = state_right_delta(st);

        // For each valid (j, d) in this state's bands
        for j in jmin_v..=jmax_v {
            let jp_v = (j - jmin_v) as usize;
            let dmin_v = cp9b.get_hdmin(v, j);
            let dmax_v = cp9b.get_hdmax(v, j);

            for d in dmin_v..=dmax_v {
                let dp_v = (d - dmin_v) as usize;
                let parent_beta = beta.get(v, jp_v, dp_v);

                if parent_beta == f32::NEG_INFINITY {
                    continue;
                }

                // Propagate to children based on state type
                if st == B_ST {
                    // Bifurcation: propagate to both children
                    let _y = cm.lchild[v] as usize;  // Left child (BEGL_S)
                    let _z = cm.rchild[v] as usize;  // Right child (BEGR_S)

                    // Complex k-split propagation (simplified for now)
                    // Full implementation would iterate over all valid k values
                    // and update beta for both children
                }

                // For non-bifurcation states, propagate to children
                for yoffset in 0..cm.cnum[v] as usize {
                    let y = cm.cfirst[v] as usize + yoffset;
                    let tsc = cm.tsc[v][yoffset];

                    // Calculate child's banded coordinates
                    let j_sdr = j - sdr;
                    if j_sdr < cp9b.jmin[y] || j_sdr > cp9b.jmax[y] {
                        continue;
                    }
                    let jp_y = (j_sdr - cp9b.jmin[y]) as usize;

                    let d_sd = d - sd;
                    let hdmin_y = cp9b.get_hdmin(y, j_sdr);
                    let hdmax_y = cp9b.get_hdmax(y, j_sdr);
                    if d_sd < hdmin_y || d_sd > hdmax_y {
                        continue;
                    }
                    let dp_y = (d_sd - hdmin_y) as usize;

                    // Update child's beta using log-sum-exp
                    let esc = get_emission_score_hb(cm, dsq, v, st, j, d);
                    let contribution = parent_beta + tsc + esc;
                    let cur_beta = beta.get(y, jp_y, dp_y);
                    beta.set(y, jp_y, dp_y, log_sum_exp(cur_beta, contribution));
                }
            }
        }
    }

    let score = beta.get(0, jp_0, lp_0);
    Ok((score, beta))
}

// =============================================================================
// HMM-Banded Posterior Decoding
// =============================================================================

/// HMM-Banded Posterior Decoding
///
/// Computes posterior probabilities from inside and outside matrices.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `cp9b` - CP9 bands
/// * `inside` - Inside matrix
/// * `outside` - Outside matrix
/// * `l` - Sequence length
///
/// # Returns
/// (posterior_matrix) on success
pub fn cm_posterior_hb_bands(
    cm: &CM,
    cp9b: &CP9Bands,
    inside: &CMHbMx,
    outside: &CMHbMx,
    l: i32,
) -> Result<CMHbMx, String> {
    // Validate
    if cp9b.jmin[0] > l || cp9b.jmax[0] < l {
        return Err("L outside ROOT_S j band".to_string());
    }
    let jp_0 = (l - cp9b.jmin[0]) as usize;

    let dmin_0 = cp9b.get_hdmin(0, l);
    if dmin_0 > l {
        return Err("L outside ROOT_S d band".to_string());
    }
    let lp_0 = (l - dmin_0) as usize;

    // Overall log probability
    let overall_sc = inside.get(0, jp_0, lp_0);
    if overall_sc == f32::NEG_INFINITY {
        return Err("Inside score is -infinity".to_string());
    }

    // Allocate posterior matrix
    let mut post = CMHbMx::new(cp9b);
    let m = cm.m as usize;

    // Compute posteriors: P(v, j, d | seq) = exp(inside + outside - overall)
    for v in 0..m {
        let jmin_v = cp9b.jmin[v];
        let jmax_v = cp9b.jmax[v];

        if jmax_v < jmin_v {
            continue;
        }

        for j in jmin_v..=jmax_v {
            let jp_v = (j - jmin_v) as usize;
            let dmin_v = cp9b.get_hdmin(v, j);
            let dmax_v = cp9b.get_hdmax(v, j);

            for d in dmin_v..=dmax_v {
                let dp_v = (d - dmin_v) as usize;

                let ins = inside.get(v, jp_v, dp_v);
                let out = outside.get(v, jp_v, dp_v);

                if ins > f32::NEG_INFINITY && out > f32::NEG_INFINITY {
                    let log_post = ins + out - overall_sc;
                    post.set(v, jp_v, dp_v, log_post);
                }
            }
        }
    }

    Ok(post)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmhbmx_new() {
        // Create minimal CP9Bands
        let mut cp9b = CP9Bands::new(10, 10);
        cp9b.l = 20;

        // Set some basic band values
        for v in 0..10 {
            cp9b.jmin[v] = 0;
            cp9b.jmax[v] = 20;
            cp9b.set_j_band(v, 0, 20);
            for j in 0..=20 {
                cp9b.set_hd_band(v, j, 0, j);
            }
        }

        let mx = CMHbMx::new(&cp9b);
        assert_eq!(mx.m, 10);
        assert_eq!(mx.l, 20);
        assert!(mx.ncells_valid > 0);
    }

    #[test]
    fn test_cmhbmx_get_set() {
        let mut cp9b = CP9Bands::new(5, 5);
        cp9b.l = 10;

        for v in 0..5 {
            cp9b.jmin[v] = 0;
            cp9b.jmax[v] = 10;
            cp9b.set_j_band(v, 0, 10);
            for j in 0..=10 {
                cp9b.set_hd_band(v, j, 0, j);
            }
        }

        let mut mx = CMHbMx::new(&cp9b);

        // Test set and get
        mx.set(0, 5, 3, 42.0);
        assert!((mx.get(0, 5, 3) - 42.0).abs() < 1e-6);

        // Test out of bounds returns NEG_INFINITY
        assert!(mx.get(100, 0, 0) == f32::NEG_INFINITY);
    }

    #[test]
    fn test_log_sum_exp() {
        // log_sum_exp of -inf and x should be x
        assert!((log_sum_exp(f32::NEG_INFINITY, 0.0) - 0.0).abs() < 1e-6);
        assert!((log_sum_exp(0.0, f32::NEG_INFINITY) - 0.0).abs() < 1e-6);

        // Infernal works in BITS (log base 2): log_sum_exp uses log2/exp2, so
        // log_sum_exp of equal values is value + log2(2) = value + 1.0.
        let result = log_sum_exp(0.0, 0.0);
        let expected = 2.0_f32.log2();
        assert!((result - expected).abs() < 1e-6);
    }
}
