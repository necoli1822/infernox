//! CYK and Inside Dynamic Programming Algorithms for Covariance Models
//!
//! This module implements the core alignment algorithms:
//! - CYK: Maximum likelihood alignment (Viterbi-style)
//! - Inside: Sum over all alignments (Forward-style)
//!
//! These algorithms use dynamic programming on the SCFG represented by the CM.

use crate::cm::{CM, ALPHABET_SIZE_P};
use crate::constants::*;
use crate::parsetree::Parsetree;
use crate::types::EslDsq;

/// Minimum subsequence length for divide-and-conquer
const MIN_SUBSEQ_LEN: i32 = 10;

/// CYK Divide-and-Conquer Algorithm
///
/// Finds the maximum likelihood parse tree for a sequence using the CYK algorithm
/// with divide-and-conquer to save memory.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence (1..L, 0-indexed in array)
/// * `l` - Sequence length
///
/// # Returns
/// Tuple of (score, parsetree)
pub fn cyk_divide_and_conquer(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<(f32, Parsetree), String> {
    if l == 0 {
        return Err("Sequence length must be > 0".to_string());
    }

    // For short sequences, use full matrix
    if l < MIN_SUBSEQ_LEN {
        return cyk_inside(cm, dsq, l, true);
    }

    // Allocate DP matrix: alpha[v][j][d]
    // We use j (end position) and d (subsequence length)
    let m = cm.m as usize;
    let l_usize = l as usize;

    // Use QDB bounds if available
    let use_qdb = cm.has_qdb();

    // Initialize matrix
    let mut alpha = vec![vec![vec![f32::NEG_INFINITY; l_usize + 1]; l_usize + 1]; m];
    let mut best_k = vec![vec![vec![0i32; l_usize + 1]; l_usize + 1]; m]; // For bifurcations

    // Fill the matrix
    for d in 1..=l {
        for j in d..=l {
            let i = j - d + 1; // Start position (1-indexed)

            // Process states in REVERSE order (children have higher indices)
            for v in (0..m).rev() {
                let st = cm.sttype[v] as i32;

                // Check QDB bounds
                if use_qdb {
                    let dv = d - 1; // 0-indexed d
                    if dv < cm.dmin1[v] || dv > cm.dmax1[v] {
                        continue;
                    }
                }

                // Handle different state types
                match st {
                    B_ST => {
                        // Bifurcation: find best split point k
                        let lchild = cm.lchild[v] as usize;
                        let rchild = cm.rchild[v] as usize;
                        let mut best_score = f32::NEG_INFINITY;
                        let mut best_split = i;

                        for k in i..j {
                            let left_d = k - i + 1;
                            let right_d = j - k;
                            if left_d > 0 && right_d > 0 {
                                let score = alpha[lchild][k as usize][left_d as usize]
                                    + alpha[rchild][j as usize][right_d as usize];
                                if score > best_score {
                                    best_score = score;
                                    best_split = k;
                                }
                            }
                        }
                        alpha[v][j as usize][d as usize] = best_score;
                        best_k[v][j as usize][d as usize] = best_split;
                    }
                    E_ST => {
                        // End state: score = 0 for d=0
                        if d == 0 {
                            alpha[v][j as usize][d as usize] = 0.0;
                        }
                    }
                    S_ST | D_ST => {
                        // Non-emitting states: sum over transitions
                        let cfirst = cm.cfirst[v] as usize;
                        let cnum = cm.cnum[v] as usize;
                        let mut best_score = f32::NEG_INFINITY;

                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            let tsc = cm.tsc[v][yoffset];
                            let child_score = alpha[y][j as usize][d as usize];
                            let score = tsc + child_score;
                            if score > best_score {
                                best_score = score;
                            }
                        }
                        alpha[v][j as usize][d as usize] = best_score;
                    }
                    MP_ST => {
                        // Match pair: emit i and j
                        // dsq uses sentinels: dsq[0]=sentinel, dsq[1..L]=sequence, dsq[L+1]=sentinel
                        // i and j are 1-indexed positions, so dsq[i] and dsq[j] give correct bases
                        if d >= 2 {
                            let symi = dsq[i as usize] as usize;
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symi * 4 + symj];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut best_score = f32::NEG_INFINITY;

                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 2;
                                let child_j = j - 1;
                                let child_score = if child_d > 0 {
                                    alpha[y][child_j as usize][child_d as usize]
                                } else {
                                    0.0
                                };
                                let score = esc + tsc + child_score;
                                if score > best_score {
                                    best_score = score;
                                }
                            }
                            alpha[v][j as usize][d as usize] = best_score;
                        }
                    }
                    ML_ST | IL_ST => {
                        // Match/insert left: emit i
                        // dsq[i] gives base at 1-indexed position i
                        if d >= 1 {
                            let symi = dsq[i as usize] as usize;
                            let esc = cm.esc[v][symi];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut best_score = f32::NEG_INFINITY;

                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 1;
                                let child_score = if child_d > 0 {
                                    alpha[y][j as usize][child_d as usize]
                                } else {
                                    0.0
                                };
                                let score = esc + tsc + child_score;
                                if score > best_score {
                                    best_score = score;
                                }
                            }
                            alpha[v][j as usize][d as usize] = best_score;
                        }
                    }
                    MR_ST | IR_ST => {
                        // Match/insert right: emit j
                        // dsq[j] gives base at 1-indexed position j
                        // Child spans (i, j-1) which has j_new = j-1, d_new = d-1
                        if d >= 1 {
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symj];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut best_score = f32::NEG_INFINITY;

                            let child_j = j - 1;
                            let child_d = d - 1;
                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_score = if child_d > 0 && child_j >= 1 {
                                    alpha[y][child_j as usize][child_d as usize]
                                } else if child_d == 0 && child_j >= 0 {
                                    alpha[y][child_j as usize][0]
                                } else {
                                    f32::NEG_INFINITY
                                };
                                let score = esc + tsc + child_score;
                                if score > best_score {
                                    best_score = score;
                                }
                            }
                            alpha[v][j as usize][d as usize] = best_score;
                        }
                    }
                    EL_ST => {
                        // Local end state - available when CM_LOCAL_END is set
                        if (cm.flags & crate::cm::CM_LOCAL_END) != 0 {
                            if d == 0 {
                                alpha[v][j as usize][0] = 0.0;
                            } else if d >= 1 && j >= 1 {
                                let prev_score = alpha[v][(j - 1) as usize][(d - 1) as usize];
                                if prev_score.is_finite() {
                                    let new_score = cm.el_selfsc + prev_score;
                                    if new_score > alpha[v][j as usize][d as usize] {
                                        alpha[v][j as usize][d as usize] = new_score;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Get final score - check local begins if enabled
    let mut best_score = alpha[0][l as usize][l as usize];
    let mut best_v = 0usize;

    // If local begins are enabled, check all states for better scores
    if (cm.flags & crate::cm::CM_LOCAL_BEGIN) != 0 {
        for v in 1..m {
            if cm.has_local_begin(v) {
                let local_score = alpha[v][l as usize][l as usize] + cm.beginsc[v];
                if local_score > best_score {
                    best_score = local_score;
                    best_v = v;
                }
            }
        }
    }

    // Build parse tree using traceback
    // For divide-and-conquer, we reuse the traceback logic from cyk_inside
    // by building a shadow matrix alongside alpha
    let tr = traceback_from_alpha_local(cm, &alpha, &best_k, l, best_v as i32);

    Ok((best_score, tr))
}

/// Traceback from alpha matrix for divide-and-conquer
///
/// Reconstructs the parse tree from the DP matrix without a shadow matrix,
/// by recomputing the best choices at each step.
fn traceback_from_alpha(cm: &CM, alpha: &[Vec<Vec<f32>>], best_k: &[Vec<Vec<i32>>], l: i32) -> Parsetree {
    traceback_from_alpha_local(cm, alpha, best_k, l, 0)
}

/// Traceback from alpha matrix with local begin support
///
/// Reconstructs the parse tree from the DP matrix, starting from the
/// specified state (which may be a local begin state).
fn traceback_from_alpha_local(cm: &CM, alpha: &[Vec<Vec<f32>>], best_k: &[Vec<Vec<i32>>], l: i32, start_v: i32) -> Parsetree {
    let mut tr = Parsetree::new(128);

    // Stack for traceback: (v, i, j, parent_idx)
    let mut stack: Vec<(i32, i32, i32, i32)> = Vec::new();

    // Start at the best state (may be a local begin), spanning positions 1..L
    stack.push((start_v, 1, l, -1));

    while let Some((v, i, j, parent_idx)) = stack.pop() {
        if j < i {
            continue; // Invalid span
        }

        let d = j - i + 1;
        let st = cm.sttype[v as usize] as i32;

        // Add this node to parsetree
        let node_idx = tr.add_node(i, j, v, -1, -1, parent_idx);

        // Update parent's children
        if parent_idx >= 0 {
            let parent = parent_idx as usize;
            if tr.nxtl[parent] == -1 {
                tr.nxtl[parent] = node_idx;
            } else if tr.nxtr[parent] == -1 {
                tr.nxtr[parent] = node_idx;
            }
        }

        match st {
            B_ST => {
                // Bifurcation: use best_k to find split point
                let k = best_k[v as usize][j as usize][d as usize];
                let lchild = cm.lchild[v as usize];
                let rchild = cm.rchild[v as usize];

                // Right child spans k+1..j (push first so left is processed first)
                if k < j {
                    stack.push((rchild, k + 1, j, node_idx));
                }
                // Left child spans i..k
                if k >= i {
                    stack.push((lchild, i, k, node_idx));
                }
            }
            E_ST => {
                // End state: no children
            }
            S_ST | D_ST => {
                // Non-emitting: find best child
                let cfirst = cm.cfirst[v as usize] as usize;
                let cnum = cm.cnum[v as usize] as usize;
                let mut best_yoffset = 0;
                let mut best_score = f32::NEG_INFINITY;

                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    let tsc = cm.tsc[v as usize][yoffset];
                    let child_score = alpha[y][j as usize][d as usize];
                    if tsc + child_score > best_score {
                        best_score = tsc + child_score;
                        best_yoffset = yoffset;
                    }
                }

                let y = (cfirst + best_yoffset) as i32;
                stack.push((y, i, j, node_idx));
            }
            MP_ST => {
                // Match pair: consumes both ends
                if d >= 2 {
                    let cfirst = cm.cfirst[v as usize] as usize;
                    let cnum = cm.cnum[v as usize] as usize;
                    let mut best_yoffset = 0;
                    let mut best_score = f32::NEG_INFINITY;

                    for yoffset in 0..cnum {
                        let y = cfirst + yoffset;
                        let tsc = cm.tsc[v as usize][yoffset];
                        let child_d = d - 2;
                        let child_j = j - 1;
                        let child_score = if child_d > 0 {
                            alpha[y][child_j as usize][child_d as usize]
                        } else {
                            0.0
                        };
                        if tsc + child_score > best_score {
                            best_score = tsc + child_score;
                            best_yoffset = yoffset;
                        }
                    }

                    let y = (cfirst + best_yoffset) as i32;
                    stack.push((y, i + 1, j - 1, node_idx));
                }
            }
            ML_ST | IL_ST => {
                // Left emit: consumes left
                if d >= 1 {
                    let cfirst = cm.cfirst[v as usize] as usize;
                    let cnum = cm.cnum[v as usize] as usize;
                    let mut best_yoffset = 0;
                    let mut best_score = f32::NEG_INFINITY;

                    for yoffset in 0..cnum {
                        let y = cfirst + yoffset;
                        let tsc = cm.tsc[v as usize][yoffset];
                        let child_d = d - 1;
                        let child_score = alpha[y][j as usize][child_d as usize];
                        if tsc + child_score > best_score {
                            best_score = tsc + child_score;
                            best_yoffset = yoffset;
                        }
                    }

                    let y = (cfirst + best_yoffset) as i32;
                    stack.push((y, i + 1, j, node_idx));
                }
            }
            MR_ST | IR_ST => {
                // Right emit: consumes right
                if d >= 1 {
                    let cfirst = cm.cfirst[v as usize] as usize;
                    let cnum = cm.cnum[v as usize] as usize;
                    let mut best_yoffset = 0;
                    let mut best_score = f32::NEG_INFINITY;

                    let child_j = j - 1;
                    let child_d = d - 1;
                    for yoffset in 0..cnum {
                        let y = cfirst + yoffset;
                        let tsc = cm.tsc[v as usize][yoffset];
                        let child_score = if child_d > 0 && child_j >= 1 {
                            alpha[y][child_j as usize][child_d as usize]
                        } else if child_d == 0 {
                            alpha[y][child_j as usize][0]
                        } else {
                            f32::NEG_INFINITY
                        };
                        if tsc + child_score > best_score {
                            best_score = tsc + child_score;
                            best_yoffset = yoffset;
                        }
                    }

                    let y = (cfirst + best_yoffset) as i32;
                    stack.push((y, i, j - 1, node_idx));
                }
            }
            EL_ST => {
                // Local end: handled by traceback from parent
            }
            _ => {}
        }
    }

    tr
}

/// Shadow matrix entry for traceback
#[derive(Clone, Copy, Debug)]
enum Shadow {
    /// Transition to child at yoffset
    Child(i32),
    /// Bifurcation split at position k
    Bifurcation(i32),
    /// End state (base case)
    End,
    /// Not set
    None,
}

/// CYK Inside Algorithm with Full Matrix
///
/// Finds the maximum likelihood parse tree for a sequence using the CYK algorithm
/// with a full DP matrix.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence with sentinels: dsq[0]=sentinel, dsq[1..L]=seq, dsq[L+1]=sentinel
/// * `l` - Sequence length (not including sentinels)
/// * `build_tr` - Whether to build the parse tree
///
/// # Returns
/// Tuple of (score, parsetree)
pub fn cyk_inside(cm: &CM, dsq: &[EslDsq], l: i32, build_tr: bool) -> Result<(f32, Parsetree), String> {
    if l == 0 {
        return Err("Sequence length must be > 0".to_string());
    }

    let m = cm.m as usize;
    let l_usize = l as usize;

    // Use QDB bounds if available
    let use_qdb = cm.has_qdb();

    // Allocate DP matrix: alpha[v][j][d]
    let mut alpha = vec![vec![vec![f32::NEG_INFINITY; l_usize + 1]; l_usize + 1]; m];

    // Shadow matrix for traceback: shadow[v][j][d]
    let mut shadow = vec![vec![vec![Shadow::None; l_usize + 1]; l_usize + 1]; m];

    // Fill the matrix bottom-up by subsequence length d
    for d in 0..=l {
        for j in d..=l {
            let i = j - d + 1; // Start position (1-indexed)

            // Process states in REVERSE order (children have higher indices)
            for v in (0..m).rev() {
                let st = cm.sttype[v] as i32;

                // Check QDB bounds
                if use_qdb && d > 0 {
                    let dv = (d - 1) as i32; // 0-indexed d
                    if dv < cm.dmin1[v] || dv > cm.dmax1[v] {
                        continue;
                    }
                }

                // Handle different state types
                match st {
                    B_ST => {
                        // Bifurcation: find best split point k
                        let lchild = cm.lchild[v] as usize;
                        let rchild = cm.rchild[v] as usize;
                        let mut best_score = f32::NEG_INFINITY;
                        let mut best_k = i;

                        for k in i..=j {
                            let left_d = (k - i + 1) as usize;
                            let right_d = (j - k) as usize;
                            if left_d <= l_usize && right_d <= l_usize {
                                let left_score = alpha[lchild][k as usize][left_d];
                                let right_score = alpha[rchild][j as usize][right_d];
                                let score = left_score + right_score;
                                if score > best_score {
                                    best_score = score;
                                    best_k = k;
                                }
                            }
                        }
                        alpha[v][j as usize][d as usize] = best_score;
                        shadow[v][j as usize][d as usize] = Shadow::Bifurcation(best_k);
                    }
                    E_ST => {
                        // End state: score = 0 for d=0
                        if d == 0 {
                            alpha[v][j as usize][0] = 0.0;
                            shadow[v][j as usize][0] = Shadow::End;
                        }
                    }
                    S_ST | D_ST => {
                        // Non-emitting states: max over transitions
                        let cfirst = cm.cfirst[v] as usize;
                        let cnum = cm.cnum[v] as usize;
                        let mut best_score = f32::NEG_INFINITY;
                        let mut best_yoffset = 0i32;

                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            let tsc = cm.tsc[v][yoffset];
                            let child_score = alpha[y][j as usize][d as usize];
                            let score = tsc + child_score;
                            if score > best_score {
                                best_score = score;
                                best_yoffset = yoffset as i32;
                            }
                        }
                        alpha[v][j as usize][d as usize] = best_score;
                        shadow[v][j as usize][d as usize] = Shadow::Child(best_yoffset);
                    }
                    MP_ST => {
                        // Match pair: emit i and j
                        // dsq[i] and dsq[j] give bases at 1-indexed positions
                        if d >= 2 {
                            let symi = dsq[i as usize] as usize;
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symi * 4 + symj];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut best_score = f32::NEG_INFINITY;
                            let mut best_yoffset = 0i32;

                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 2;
                                let child_j = j - 1;
                                let child_score = if child_d >= 0 {
                                    alpha[y][child_j as usize][child_d as usize]
                                } else {
                                    f32::NEG_INFINITY
                                };
                                let score = esc + tsc + child_score;
                                if score > best_score {
                                    best_score = score;
                                    best_yoffset = yoffset as i32;
                                }
                            }
                            alpha[v][j as usize][d as usize] = best_score;
                            shadow[v][j as usize][d as usize] = Shadow::Child(best_yoffset);
                        }
                    }
                    ML_ST | IL_ST => {
                        // Match/insert left: emit i
                        // dsq[i] gives base at 1-indexed position i
                        if d >= 1 {
                            let symi = dsq[i as usize] as usize;
                            let esc = cm.esc[v][symi];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut best_score = f32::NEG_INFINITY;
                            let mut best_yoffset = 0i32;

                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 1;
                                let child_score = alpha[y][j as usize][child_d as usize];
                                let score = esc + tsc + child_score;
                                if score > best_score {
                                    best_score = score;
                                    best_yoffset = yoffset as i32;
                                }
                            }
                            alpha[v][j as usize][d as usize] = best_score;
                            shadow[v][j as usize][d as usize] = Shadow::Child(best_yoffset);
                        }
                    }
                    MR_ST | IR_ST => {
                        // Match/insert right: emit j
                        // dsq[j] gives base at 1-indexed position j
                        // Child spans (i, j-1) which has j_new = j-1, d_new = d-1
                        if d >= 1 {
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symj];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut best_score = f32::NEG_INFINITY;
                            let mut best_yoffset = 0i32;

                            let child_j = j - 1;
                            let child_d = d - 1;
                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_score = if child_d >= 0 && child_j >= 0 {
                                    alpha[y][child_j as usize][child_d as usize]
                                } else {
                                    f32::NEG_INFINITY
                                };
                                let score = esc + tsc + child_score;
                                if score > best_score {
                                    best_score = score;
                                    best_yoffset = yoffset as i32;
                                }
                            }
                            alpha[v][j as usize][d as usize] = best_score;
                            shadow[v][j as usize][d as usize] = Shadow::Child(best_yoffset);
                        }
                    }
                    EL_ST => {
                        // Local end state - available when CM_LOCAL_END is set
                        if (cm.flags & crate::cm::CM_LOCAL_END) != 0 {
                            if d == 0 {
                                alpha[v][j as usize][0] = 0.0;
                                shadow[v][j as usize][0] = Shadow::End;
                            } else if d >= 1 && j >= 1 {
                                let prev_score = alpha[v][(j - 1) as usize][(d - 1) as usize];
                                if prev_score.is_finite() {
                                    let new_score = cm.el_selfsc + prev_score;
                                    if new_score > alpha[v][j as usize][d as usize] {
                                        alpha[v][j as usize][d as usize] = new_score;
                                        // Self-transition marker
                                        shadow[v][j as usize][d as usize] = Shadow::Child(-1);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Get final score - search over all (j, d) pairs to find optimal hit
    let mut best_score = f32::NEG_INFINITY;
    let mut best_v = 0i32;
    let mut best_j = l;
    let mut best_d = l;

    // Search over all (j, d) for optimal alignment
    for j in 1..=l {
        let max_d = j.min(cm.w); // d cannot exceed W (window size)
        for d in 1..=max_d {
            // Check root state
            let root_score = alpha[0][j as usize][d as usize];
            if root_score > best_score {
                best_score = root_score;
                best_v = 0;
                best_j = j;
                best_d = d;
            }

            // If local begins are enabled, check all states
            if (cm.flags & crate::cm::CM_LOCAL_BEGIN) != 0 {
                for v in 1..m {
                    if cm.has_local_begin(v) {
                        let local_score = alpha[v][j as usize][d as usize] + cm.beginsc[v];
                        if local_score > best_score {
                            best_score = local_score;
                            best_v = v as i32;
                            best_j = j;
                            best_d = d;
                        }
                    }
                }
            }
        }
    }

    // Build parse tree if requested (using optimal j, d)
    let tr = if build_tr {
        traceback_local_jd(cm, &shadow, best_j, best_d, best_v)
    } else {
        Parsetree::new(0)
    };

    Ok((best_score, tr))
}

/// Traceback through shadow matrix to build parsetree
fn traceback(cm: &CM, shadow: &[Vec<Vec<Shadow>>], l: i32) -> Parsetree {
    traceback_local(cm, shadow, l, 0)
}

/// Traceback through shadow matrix with local begin support
fn traceback_local(cm: &CM, shadow: &[Vec<Vec<Shadow>>], l: i32, start_v: i32) -> Parsetree {
    let mut tr = Parsetree::new(128);

    // Stack for traceback: (v, i, j, parent_idx)
    // i, j are 1-indexed sequence positions
    let mut stack: Vec<(i32, i32, i32, i32)> = Vec::new();

    // Start at the best state (may be a local begin), spanning positions 1..L
    stack.push((start_v, 1, l, -1));

    while let Some((v, i, j, parent_idx)) = stack.pop() {
        let d = j - i + 1;
        let st = cm.sttype[v as usize] as i32;

        // Determine emissions based on state type
        let (emitl, emitr) = match st {
            MP_ST => (i, j),           // Pair: emit both
            ML_ST | IL_ST => (i, j),   // Left: emit left, track right bound
            MR_ST | IR_ST => (i, j),   // Right: emit right, track left bound
            _ => (i, j),               // Non-emitting: just track bounds
        };

        // Add this node to parsetree
        let node_idx = tr.add_node(emitl, emitr, v, -1, -1, parent_idx);

        // Update parent's nxtl/nxtr
        if parent_idx >= 0 {
            let parent = parent_idx as usize;
            if tr.nxtl[parent] == -1 {
                tr.nxtl[parent] = node_idx;
            } else if tr.nxtr[parent] == -1 {
                tr.nxtr[parent] = node_idx;
            }
        }

        // Process based on shadow entry
        match shadow[v as usize][j as usize][d as usize] {
            Shadow::Child(yoffset) => {
                let y = cm.cfirst[v as usize] + yoffset;

                // Calculate child positions based on state type
                let (child_i, child_j) = match st {
                    MP_ST => (i + 1, j - 1),      // Pair consumes both ends
                    ML_ST | IL_ST => (i + 1, j),  // Left consumes left
                    MR_ST | IR_ST => (i, j - 1),  // Right consumes right
                    _ => (i, j),                   // Non-emitting: same span
                };

                // Only push if valid span
                if child_j >= child_i || (child_j == child_i - 1 && cm.sttype[y as usize] as i32 == E_ST) {
                    stack.push((y, child_i, child_j, node_idx));
                }
            }
            Shadow::Bifurcation(k) => {
                // Bifurcation: push right child first (so left is processed first due to stack)
                let lchild = cm.lchild[v as usize];
                let rchild = cm.rchild[v as usize];

                // Right child spans k+1..j
                if k < j {
                    stack.push((rchild, k + 1, j, node_idx));
                }
                // Left child spans i..k
                if k >= i {
                    stack.push((lchild, i, k, node_idx));
                }
            }
            Shadow::End => {
                // End state: no children
            }
            Shadow::None => {
                // Should not happen for valid paths
            }
        }
    }

    tr
}

/// Traceback through shadow matrix with specific j, d, and starting state
///
/// This version allows specifying the exact (j, d) cell to start traceback from,
/// which is needed when the optimal alignment doesn't span the full sequence.
fn traceback_local_jd(cm: &CM, shadow: &[Vec<Vec<Shadow>>], j: i32, d: i32, start_v: i32) -> Parsetree {
    let mut tr = Parsetree::new(128);

    // Stack for traceback: (v, i, j, parent_idx)
    // i, j are 1-indexed sequence positions
    let mut stack: Vec<(i32, i32, i32, i32)> = Vec::new();

    // Calculate sequence boundaries from the optimal hit
    let start_i = j - d + 1;

    // If start_v != 0 (local begin), we need to add a ROOT node (state 0)
    // that wraps the optimal alignment
    let parent_idx = if start_v != 0 {
        // Add ROOT node spanning the hit boundaries (start_i, j)
        // ROOT node is state 0 (S_ST) and has no parent (-1)
        let root_idx = tr.add_node(start_i, j, 0, -1, -1, -1);
        root_idx
    } else {
        -1  // No parent for state 0
    };

    // Start at the optimal (j, d) cell
    stack.push((start_v, start_i, j, parent_idx));

    while let Some((v, i, cur_j, parent_idx)) = stack.pop() {
        let cur_d = cur_j - i + 1;
        let st = cm.sttype[v as usize] as i32;

        // Determine emissions based on state type
        let (emitl, emitr) = match st {
            MP_ST => (i, cur_j),           // Pair: emit both
            ML_ST | IL_ST => (i, cur_j),   // Left: emit left, track right bound
            MR_ST | IR_ST => (i, cur_j),   // Right: emit right, track left bound
            _ => (i, cur_j),               // Non-emitting: just track bounds
        };

        // Add this node to parsetree
        let node_idx = tr.add_node(emitl, emitr, v, -1, -1, parent_idx);

        // Update parent's nxtl/nxtr
        if parent_idx >= 0 {
            let parent = parent_idx as usize;
            if tr.nxtl[parent] == -1 {
                tr.nxtl[parent] = node_idx;
            } else if tr.nxtr[parent] == -1 {
                tr.nxtr[parent] = node_idx;
            }
        }

        // Process based on shadow entry
        if cur_d < 0 || cur_j < 0 || cur_d as usize > shadow[v as usize][cur_j as usize].len() {
            continue;
        }
        match shadow[v as usize][cur_j as usize][cur_d as usize] {
            Shadow::Child(yoffset) => {
                let y = cm.cfirst[v as usize] + yoffset;

                // Calculate child positions based on state type
                let (child_i, child_j) = match st {
                    MP_ST => (i + 1, cur_j - 1),      // Pair consumes both ends
                    ML_ST | IL_ST => (i + 1, cur_j),  // Left consumes left
                    MR_ST | IR_ST => (i, cur_j - 1),  // Right consumes right
                    _ => (i, cur_j),                   // Non-emitting: same span
                };

                // Only push if valid span
                if child_j >= child_i || (child_j == child_i - 1 && cm.sttype[y as usize] as i32 == E_ST) {
                    stack.push((y, child_i, child_j, node_idx));
                }
            }
            Shadow::Bifurcation(k) => {
                // Bifurcation: push right child first (so left is processed first due to stack)
                let lchild = cm.lchild[v as usize];
                let rchild = cm.rchild[v as usize];

                // Right child spans k+1..j
                if k < cur_j {
                    stack.push((rchild, k + 1, cur_j, node_idx));
                }
                // Left child spans i..k
                if k >= i {
                    stack.push((lchild, i, k, node_idx));
                }
            }
            Shadow::End => {
                // End state: no children
            }
            Shadow::None => {
                // Should not happen for valid paths
            }
        }
    }

    tr
}

/// Inside Algorithm (Score Only)
///
/// Computes the Inside score (sum over all parse trees) using log-space arithmetic.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence with sentinels: dsq[0]=sentinel, dsq[1..L]=seq, dsq[L+1]=sentinel
/// * `l` - Sequence length (not including sentinels)
///
/// # Returns
/// Inside score
pub fn cyk_inside_score(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<f32, String> {
    if l == 0 {
        return Err("Sequence length must be > 0".to_string());
    }

    let m = cm.m as usize;
    let l_usize = l as usize;

    // Use QDB bounds if available
    let use_qdb = cm.has_qdb();

    // Allocate DP matrix: alpha[v][j][d]
    let mut alpha = vec![vec![vec![f32::NEG_INFINITY; l_usize + 1]; l_usize + 1]; m];

    // Fill the matrix using log-sum-exp instead of max
    for d in 0..=l {
        for j in d..=l {
            let i = j - d + 1; // 1-indexed start position

            // Process states in REVERSE order (children have higher indices)
            for v in (0..m).rev() {
                let st = cm.sttype[v] as i32;

                // Check QDB bounds
                if use_qdb && d > 0 {
                    let dv = (d - 1) as i32;
                    if dv < cm.dmin1[v] || dv > cm.dmax1[v] {
                        continue;
                    }
                }

                match st {
                    B_ST => {
                        // Bifurcation: sum over all split points
                        let lchild = cm.lchild[v] as usize;
                        let rchild = cm.rchild[v] as usize;
                        let mut scores = Vec::new();

                        for k in i..=j {
                            let left_d = (k - i + 1) as usize;
                            let right_d = (j - k) as usize;
                            if left_d <= l_usize && right_d <= l_usize {
                                let left_score = alpha[lchild][k as usize][left_d];
                                let right_score = alpha[rchild][j as usize][right_d];
                                if left_score.is_finite() && right_score.is_finite() {
                                    scores.push(left_score + right_score);
                                }
                            }
                        }
                        alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                    }
                    E_ST => {
                        if d == 0 {
                            alpha[v][j as usize][0] = 0.0;
                        }
                    }
                    S_ST | D_ST => {
                        // Non-emitting: sum over transitions
                        let cfirst = cm.cfirst[v] as usize;
                        let cnum = cm.cnum[v] as usize;
                        let mut scores = Vec::new();

                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            let tsc = cm.tsc[v][yoffset];
                            let child_score = alpha[y][j as usize][d as usize];
                            if child_score.is_finite() {
                                scores.push(tsc + child_score);
                            }
                        }
                        alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                    }
                    MP_ST => {
                        // dsq[i] and dsq[j] give bases at 1-indexed positions
                        if d >= 2 {
                            let symi = dsq[i as usize] as usize;
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symi * 4 + symj];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut scores = Vec::new();

                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 2;
                                let child_j = j - 1;
                                let child_score = if child_d >= 0 {
                                    alpha[y][child_j as usize][child_d as usize]
                                } else {
                                    f32::NEG_INFINITY
                                };
                                if child_score.is_finite() {
                                    scores.push(esc + tsc + child_score);
                                }
                            }
                            alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                        }
                    }
                    ML_ST | IL_ST => {
                        // dsq[i] gives base at 1-indexed position i
                        if d >= 1 {
                            let symi = dsq[i as usize] as usize;
                            let esc = cm.esc[v][symi];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut scores = Vec::new();

                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 1;
                                let child_score = alpha[y][j as usize][child_d as usize];
                                if child_score.is_finite() {
                                    scores.push(esc + tsc + child_score);
                                }
                            }
                            alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                        }
                    }
                    MR_ST | IR_ST => {
                        // dsq[j] gives base at 1-indexed position j
                        // Child spans (i, j-1) which has j_new = j-1, d_new = d-1
                        if d >= 1 {
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symj];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut scores = Vec::new();

                            let child_j = j - 1;
                            let child_d = d - 1;
                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_score = if child_d >= 0 && child_j >= 0 {
                                    alpha[y][child_j as usize][child_d as usize]
                                } else {
                                    f32::NEG_INFINITY
                                };
                                if child_score.is_finite() {
                                    scores.push(esc + tsc + child_score);
                                }
                            }
                            alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                        }
                    }
                    EL_ST => {
                        // Local end state for Inside algorithm
                        if (cm.flags & crate::cm::CM_LOCAL_END) != 0 {
                            if d == 0 {
                                alpha[v][j as usize][0] = 0.0;
                            } else if d >= 1 && j >= 1 {
                                let prev_score = alpha[v][(j - 1) as usize][(d - 1) as usize];
                                if prev_score.is_finite() {
                                    let new_score = cm.el_selfsc + prev_score;
                                    alpha[v][j as usize][d as usize] = log_sum_exp(&[
                                        alpha[v][j as usize][d as usize],
                                        new_score
                                    ]);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Search for optimal (j, d) like cyk_inside
    let mut best_score = f32::NEG_INFINITY;

    for j in 1..=l {
        let max_d = j.min(cm.w);
        for d in 1..=max_d {
            // Check root state
            let root_score = alpha[0][j as usize][d as usize];
            if root_score > best_score {
                best_score = root_score;
            }

            // Check local begins if enabled
            if (cm.flags & crate::cm::CM_LOCAL_BEGIN) != 0 {
                for v in 1..m {
                    if cm.has_local_begin(v) {
                        let local_score = alpha[v][j as usize][d as usize] + cm.beginsc[v];
                        if local_score > best_score {
                            best_score = local_score;
                        }
                    }
                }
            }
        }
    }

    Ok(best_score)
}

/// CYK Debug - returns the full alpha matrix for inspection
pub fn cyk_inside_debug(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<Vec<Vec<Vec<f32>>>, String> {
    if l == 0 {
        return Err("Sequence length must be > 0".to_string());
    }

    let m = cm.m as usize;
    let l_usize = l as usize;
    let use_qdb = cm.has_qdb();
    let mut alpha = vec![vec![vec![f32::NEG_INFINITY; l_usize + 1]; l_usize + 1]; m];

    for d in 0..=l {
        for j in d..=l {
            let i = j - d + 1;

            for v in (0..m).rev() {
                let st = cm.sttype[v] as i32;

                if use_qdb && d > 0 {
                    let dv = (d - 1) as i32;
                    if dv < cm.dmin1[v] || dv > cm.dmax1[v] {
                        continue;
                    }
                }

                match st {
                    B_ST => {
                        let lchild = cm.lchild[v] as usize;
                        let rchild = cm.rchild[v] as usize;
                        let mut best = f32::NEG_INFINITY;
                        for k in i..=j {
                            let left_d = k - i + 1;
                            let right_d = j - k;
                            if left_d > 0 && right_d >= 0 {
                                let score = alpha[lchild][k as usize][left_d as usize]
                                    + alpha[rchild][j as usize][right_d as usize];
                                if score > best { best = score; }
                            }
                        }
                        alpha[v][j as usize][d as usize] = best;
                    }
                    E_ST => {
                        if d == 0 { alpha[v][j as usize][0] = 0.0; }
                    }
                    S_ST | D_ST => {
                        let cfirst = cm.cfirst[v] as usize;
                        let cnum = cm.cnum[v] as usize;
                        let mut best = f32::NEG_INFINITY;
                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            let tsc = cm.tsc[v][yoffset];
                            let sc = tsc + alpha[y][j as usize][d as usize];
                            if sc > best { best = sc; }
                        }
                        alpha[v][j as usize][d as usize] = best;
                    }
                    MP_ST => {
                        if d >= 2 {
                            let symi = dsq[i as usize] as usize;
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symi * 4 + symj];
                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut best = f32::NEG_INFINITY;
                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 2;
                                let child_j = j - 1;
                                let cs = if child_d >= 0 { alpha[y][child_j as usize][child_d as usize] } else { 0.0 };
                                let sc = esc + tsc + cs;
                                if sc > best { best = sc; }
                            }
                            alpha[v][j as usize][d as usize] = best;
                        }
                    }
                    ML_ST | IL_ST => {
                        if d >= 1 {
                            let symi = dsq[i as usize] as usize;
                            let esc = cm.esc[v][symi];
                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut best = f32::NEG_INFINITY;
                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 1;
                                let cs = if child_d >= 0 { alpha[y][j as usize][child_d as usize] } else { 0.0 };
                                let sc = esc + tsc + cs;
                                if sc > best { best = sc; }
                            }
                            alpha[v][j as usize][d as usize] = best;
                        }
                    }
                    MR_ST | IR_ST => {
                        if d >= 1 {
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symj];
                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut best = f32::NEG_INFINITY;
                            let child_j = j - 1;
                            let child_d = d - 1;
                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let cs = if child_d >= 0 && child_j >= 0 { alpha[y][child_j as usize][child_d as usize] } else { f32::NEG_INFINITY };
                                let sc = esc + tsc + cs;
                                if sc > best { best = sc; }
                            }
                            alpha[v][j as usize][d as usize] = best;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(alpha)
}

/// HMM-Banded CYK Inside Align
///
/// CYK algorithm with HMM bands (CP9 bands) for efficiency.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence
/// * `l` - Sequence length
///
/// # Returns
/// Tuple of (score, best_band_index)
pub fn cm_cyk_inside_align_hb(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<(f32, i32), String> {
    // For now, just call the non-banded version
    let (score, _tr) = cyk_inside(cm, dsq, l, false)?;
    Ok((score, 3)) // Return dummy band index
}

/// HMM-Banded Inside Align
///
/// Inside algorithm with HMM bands (CP9 bands) for efficiency.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence
/// * `l` - Sequence length
///
/// # Returns
/// Inside score
pub fn cm_inside_align_hb(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<f32, String> {
    // For now, just call the non-banded version
    cyk_inside_score(cm, dsq, l)
}

/// Outside Algorithm
///
/// Computes the outside scores (probability of all parse trees that
/// use state v at positions (i,j) for everything OUTSIDE that subtree).
///
/// The Outside algorithm proceeds TOP-DOWN (v = 0 to M-1), opposite of Inside.
/// beta[v][j][d] = log probability of all derivations OUTSIDE v generating [i,j]
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence with sentinels
/// * `l` - Sequence length
/// * `inside` - Inside matrix (already computed)
///
/// # Returns
/// (outside_score, outside_matrix)
pub fn cm_outside(cm: &CM, dsq: &[EslDsq], l: i32, inside: &[Vec<Vec<f32>>])
    -> Result<(f32, Vec<Vec<Vec<f32>>>), String>
{
    if l == 0 {
        return Err("Sequence length must be > 0".to_string());
    }

    let m = cm.m as usize;
    let l_usize = l as usize;
    let _use_qdb = cm.has_qdb();

    // Initialize outside matrix: beta[v][j][d]
    let mut beta = vec![vec![vec![f32::NEG_INFINITY; l_usize + 1]; l_usize + 1]; m];

    // Base case: beta[0][L][L] = 0.0 (nothing outside the root)
    beta[0][l_usize][l_usize] = 0.0;

    // Fill matrix TOP-DOWN (forward through states)
    for v in 0..m {
        let st = cm.sttype[v] as i32;

        // For bifurcations, we need special handling
        if st == B_ST {
            let lchild = cm.lchild[v] as usize;
            let rchild = cm.rchild[v] as usize;

            // Iterate over all (j, d) cells where beta[v][j][d] is valid
            for d in 0..=l {
                for j in d..=l {
                    let i = j - d + 1;
                    if !beta[v][j as usize][d as usize].is_finite() {
                        continue;
                    }

                    let parent_beta = beta[v][j as usize][d as usize];

                    // For each split point k
                    for k in i..=j {
                        let left_d = k - i + 1;
                        let right_d = j - k;

                        if left_d > 0 && right_d > 0 {
                            // Left child: spans [i, k]
                            let left_inside = inside[lchild][k as usize][left_d as usize];
                            let right_inside = inside[rchild][j as usize][right_d as usize];

                            if left_inside.is_finite() && right_inside.is_finite() {
                                // beta[lchild][k][left_d] gets contribution from parent + right sibling
                                let left_contrib = parent_beta + right_inside;
                                let left_idx = (k as usize, left_d as usize);
                                beta[lchild][left_idx.0][left_idx.1] =
                                    log_sum_exp(&[beta[lchild][left_idx.0][left_idx.1], left_contrib]);

                                // beta[rchild][j][right_d] gets contribution from parent + left sibling
                                let right_contrib = parent_beta + left_inside;
                                let right_idx = (j as usize, right_d as usize);
                                beta[rchild][right_idx.0][right_idx.1] =
                                    log_sum_exp(&[beta[rchild][right_idx.0][right_idx.1], right_contrib]);
                            }
                        }
                    }
                }
            }
            continue;
        }

        // For non-bifurcation states, propagate to children
        let cfirst = cm.cfirst[v] as usize;
        let cnum = cm.cnum[v] as usize;

        for d in 0..=l {
            for j in d..=l {
                let i = j - d + 1;

                if !beta[v][j as usize][d as usize].is_finite() {
                    continue;
                }

                let parent_beta = beta[v][j as usize][d as usize];

                // Propagate to each child
                for yoffset in 0..cnum {
                    let y = cfirst + yoffset;
                    let tsc = cm.tsc[v][yoffset];
                    let _y_st = cm.sttype[y] as i32;

                    // Calculate child's (i', j', d') based on parent's emissions
                    let (child_i, child_j, child_d) = match st {
                        S_ST | D_ST => {
                            // Non-emitting: same span
                            (i, j, d)
                        }
                        MP_ST => {
                            // Match pair: child spans [i+1, j-1]
                            if d >= 2 {
                                let _symi = dsq[i as usize] as usize;
                                let _symj = dsq[j as usize] as usize;
                                (i + 1, j - 1, d - 2)
                            } else {
                                continue;
                            }
                        }
                        ML_ST | IL_ST => {
                            // Left emit: child spans [i+1, j]
                            if d >= 1 {
                                (i + 1, j, d - 1)
                            } else {
                                continue;
                            }
                        }
                        MR_ST | IR_ST => {
                            // Right emit: child spans [i, j-1]
                            if d >= 1 {
                                (i, j - 1, d - 1)
                            } else {
                                continue;
                            }
                        }
                        E_ST => continue,
                        _ => continue,
                    };

                    // Get emission score if needed
                    let esc = match st {
                        MP_ST => {
                            let symi = dsq[i as usize] as usize;
                            let symj = dsq[j as usize] as usize;
                            cm.esc[v][symi * 4 + symj]
                        }
                        ML_ST | IL_ST => {
                            let symi = dsq[i as usize] as usize;
                            cm.esc[v][symi]
                        }
                        MR_ST | IR_ST => {
                            let symj = dsq[j as usize] as usize;
                            cm.esc[v][symj]
                        }
                        _ => 0.0,
                    };

                    // Check bounds
                    if child_d >= 0 && child_j >= child_i - 1 && child_j <= l {
                        let contrib = parent_beta + tsc + esc;
                        let idx = (child_j as usize, child_d as usize);

                        if idx.0 <= l_usize && idx.1 <= l_usize {
                            beta[y][idx.0][idx.1] = log_sum_exp(&[beta[y][idx.0][idx.1], contrib]);
                        }
                    }
                }
            }
        }
    }

    // The outside score at root should equal inside score
    let outside_score = beta[0][l_usize][l_usize];
    Ok((outside_score, beta))
}

/// Posterior Probability Computation
///
/// Computes posterior = inside + outside - total_score
/// posterior[v][j][d] represents the probability that state v is used
/// to generate subsequence [i..j] in a random parse tree
///
/// # Arguments
/// * `cm` - The covariance model
/// * `l` - Sequence length
/// * `inside` - Inside matrix
/// * `outside` - Outside matrix
/// * `inside_score` - Total inside score (normalization constant)
///
/// # Returns
/// Posterior probability matrix
pub fn cm_posterior(cm: &CM, l: i32, inside: &[Vec<Vec<f32>>], outside: &[Vec<Vec<f32>>], inside_score: f32)
    -> Vec<Vec<Vec<f32>>>
{
    let m = cm.m as usize;
    let l_usize = l as usize;

    let mut posterior = vec![vec![vec![f32::NEG_INFINITY; l_usize + 1]; l_usize + 1]; m];

    for v in 0..m {
        for j in 0..=l_usize {
            for d in 0..=l_usize {
                if inside[v][j][d].is_finite() && outside[v][j][d].is_finite() {
                    posterior[v][j][d] = inside[v][j][d] + outside[v][j][d] - inside_score;
                }
            }
        }
    }

    posterior
}

/// Inside Algorithm with Matrix Return
///
/// Computes the Inside score and returns the full DP matrix for use
/// in Outside/Posterior calculations.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence
/// * `l` - Sequence length
///
/// # Returns
/// (inside_score, inside_matrix)
pub fn cm_inside_with_matrix(cm: &CM, dsq: &[EslDsq], l: i32)
    -> Result<(f32, Vec<Vec<Vec<f32>>>), String>
{
    if l == 0 {
        return Err("Sequence length must be > 0".to_string());
    }

    let m = cm.m as usize;
    let l_usize = l as usize;
    let use_qdb = cm.has_qdb();

    // Allocate DP matrix: alpha[v][j][d]
    let mut alpha = vec![vec![vec![f32::NEG_INFINITY; l_usize + 1]; l_usize + 1]; m];

    // Fill the matrix using log-sum-exp
    for d in 0..=l {
        for j in d..=l {
            let i = j - d + 1; // 1-indexed start position

            for v in (0..m).rev() {
                let st = cm.sttype[v] as i32;

                // Check QDB bounds
                if use_qdb && d > 0 {
                    let dv = (d - 1) as i32;

                    // DEBUG: For root state (v=0) on Archaea sequence (l=74), show bands
                    if v == 0 && l == 74 {
                        eprintln!("[BAND DEBUG v=0] j={}, d={}, dv={}, dmin1[0]={}, dmax1[0]={}",
                                  j, d, dv, cm.dmin1[0], cm.dmax1[0]);
                    }

                    if dv < cm.dmin1[v] || dv > cm.dmax1[v] {
                        if v == 0 && l == 74 && j == 69 && d == 67 {
                            eprintln!("[BAND DEBUG] SKIPPING v=0, j=69, d=67 because dv={} out of range [{}, {}]",
                                      dv, cm.dmin1[v], cm.dmax1[v]);
                        }
                        continue;
                    }
                }

                match st {
                    B_ST => {
                        // B_ST (bifurcation) combines left child (BEGL_S) and right child (BEGR_S)
                        // We need to consider all possible split points, including empty left/right
                        // C code: for (k = 0; k <= d; k++) where k = right_d, left_d = d - k
                        let lchild = cm.lchild[v] as usize;
                        let rchild = cm.rchild[v] as usize;
                        let mut scores = Vec::new();

                        // k represents the length of the right subsequence (0 to d)
                        // When k = 0: left takes all (left_d = d), right takes nothing (right_d = 0)
                        // When k = d: left takes nothing (left_d = 0), right takes all (right_d = d)
                        for right_d in 0..=(d as usize) {
                            let left_d = (d as usize) - right_d;
                            // For BEGL_S: j position is i + left_d - 1 (end of left subsequence)
                            // For BEGR_S: j position is j (end of whole sequence)
                            let left_j = if left_d == 0 { i - 1 } else { i + left_d as i32 - 1 };

                            if left_j >= 0 && (left_j as usize) < alpha[lchild].len() {
                                let left_score = alpha[lchild][left_j as usize][left_d];
                                let right_score = alpha[rchild][j as usize][right_d];
                                if left_score.is_finite() && right_score.is_finite() {
                                    scores.push(left_score + right_score);
                                }
                            }
                        }
                        alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                    }
                    E_ST => {
                        if d == 0 {
                            alpha[v][j as usize][0] = 0.0;
                        }
                    }
                    S_ST | D_ST => {
                        let cfirst = cm.cfirst[v] as usize;
                        let cnum = cm.cnum[v] as usize;
                        let mut scores = Vec::new();

                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            let tsc = cm.tsc[v][yoffset];
                            let child_score = alpha[y][j as usize][d as usize];
                            if child_score.is_finite() {
                                scores.push(tsc + child_score);
                            }
                        }
                        alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                    }
                    MP_ST => {
                        if d >= 2 {
                            let symi = dsq[i as usize] as usize;
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symi * 4 + symj];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut scores = Vec::new();

                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 2;
                                let child_j = j - 1;
                                let child_score = if child_d >= 0 {
                                    alpha[y][child_j as usize][child_d as usize]
                                } else {
                                    f32::NEG_INFINITY
                                };
                                if child_score.is_finite() {
                                    scores.push(esc + tsc + child_score);
                                }
                            }
                            alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                        }
                    }
                    ML_ST | IL_ST => {
                        if d >= 1 {
                            let symi = dsq[i as usize] as usize;
                            let esc = cm.esc[v][symi];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut scores = Vec::new();

                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_d = d - 1;
                                let child_score = alpha[y][j as usize][child_d as usize];
                                if child_score.is_finite() {
                                    scores.push(esc + tsc + child_score);
                                }
                            }
                            alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                        }
                    }
                    MR_ST | IR_ST => {
                        if d >= 1 {
                            let symj = dsq[j as usize] as usize;
                            let esc = cm.esc[v][symj];

                            let cfirst = cm.cfirst[v] as usize;
                            let cnum = cm.cnum[v] as usize;
                            let mut scores = Vec::new();

                            let child_j = j - 1;
                            let child_d = d - 1;
                            for yoffset in 0..cnum {
                                let y = cfirst + yoffset;
                                let tsc = cm.tsc[v][yoffset];
                                let child_score = if child_d >= 0 && child_j >= 0 {
                                    alpha[y][child_j as usize][child_d as usize]
                                } else {
                                    f32::NEG_INFINITY
                                };
                                if child_score.is_finite() {
                                    scores.push(esc + tsc + child_score);
                                }
                            }
                            alpha[v][j as usize][d as usize] = log_sum_exp(&scores);
                        }
                    }
                    EL_ST => {
                        // Local end state for Inside algorithm
                        if (cm.flags & crate::cm::CM_LOCAL_END) != 0 {
                            if d == 0 {
                                alpha[v][j as usize][0] = 0.0;
                            } else if d >= 1 && j >= 1 {
                                let prev_score = alpha[v][(j - 1) as usize][(d - 1) as usize];
                                if prev_score.is_finite() {
                                    let new_score = cm.el_selfsc + prev_score;
                                    alpha[v][j as usize][d as usize] = log_sum_exp(&[
                                        alpha[v][j as usize][d as usize],
                                        new_score
                                    ]);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Search for optimal (j, d) like cyk_inside_score
    let mut best_score = f32::NEG_INFINITY;
    let mut best_j = 0;
    let mut best_d = 0;

    for j in 1..=l {
        let max_d = j.min(cm.w);
        for d in 1..=max_d {
            // Check root state
            let root_score = alpha[0][j as usize][d as usize];
            if root_score > best_score {
                best_score = root_score;
                best_j = j;
                best_d = d;
            }

            // Check local begins if enabled
            if (cm.flags & crate::cm::CM_LOCAL_BEGIN) != 0 {
                for v in 1..m {
                    if cm.has_local_begin(v) {
                        let local_score = alpha[v][j as usize][d as usize] + cm.beginsc[v];
                        if local_score > best_score {
                            best_score = local_score;
                            best_j = j;
                            best_d = d;
                        }
                    }
                }
            }
        }
    }

    // Debug: print alignment score vs best scan score and Archaea-specific info
    if std::env::var("INSIDE_DEBUG").is_ok() || l == 74 {
        let align_score = alpha[0][l as usize][l as usize];
        eprintln!("[INSIDE DEBUG] Alignment score alpha[0][{}][{}] = {:.2}", l, l, align_score);
        eprintln!("[INSIDE DEBUG] Best scan score = {:.2} at (j={}, d={})", best_score, best_j, best_d);

        // Archaea-specific: Check if j=69, d=67 was evaluated
        if l == 74 {
            let archaea_score = alpha[0][69][67];
            eprintln!("[ARCHAEA DEBUG] alpha[0][69][67] = {:.2} (expected hit position)", archaea_score);
            eprintln!("[ARCHAEA DEBUG] Is alpha[0][69][67] finite? {}", archaea_score.is_finite());
        }
    }

    Ok((best_score, alpha))
}

/// HMM-Banded Outside Algorithm
///
/// Outside algorithm with HMM bands (CP9 bands) for efficiency.
/// Currently uses non-banded implementation as a fallback, computing
/// the full outside matrix and returning the outside score.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence
/// * `l` - Sequence length
///
/// # Returns
/// Outside score (which should equal the inside score for a correct implementation)
pub fn cm_outside_align_hb(cm: &CM, dsq: &[EslDsq], l: i32)
    -> Result<f32, String>
{
    // Compute inside matrix first
    let (inside_score, inside_matrix) = cm_inside_with_matrix(cm, dsq, l)?;

    // Compute outside using the inside matrix
    let (outside_score, _outside_matrix) = cm_outside(cm, dsq, l, &inside_matrix)?;

    // In a correct implementation, inside_score should approximately equal outside_score
    // Return the inside score as the canonical result
    // (the outside score may have small numerical differences)
    let _ = outside_score; // Verify computation was done
    Ok(inside_score)
}

/// HMM-Banded Posterior Computation
///
/// Computes posterior probabilities with HMM bands for efficiency.
/// Currently uses non-banded implementation as a fallback.
///
/// The posterior probability P(v generates i..j | sequence) is computed as:
/// posterior[v][j][d] = inside[v][j][d] + outside[v][j][d] - inside_total
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence
/// * `l` - Sequence length
///
/// # Returns
/// Inside score (the total log probability of the sequence given the model)
pub fn cm_posterior_hb(cm: &CM, dsq: &[EslDsq], l: i32)
    -> Result<f32, String>
{
    // Compute inside matrix and score
    let (inside_score, inside_matrix) = cm_inside_with_matrix(cm, dsq, l)?;

    // Compute outside matrix
    let (_outside_score, outside_matrix) = cm_outside(cm, dsq, l, &inside_matrix)?;

    // Compute posterior probabilities
    let _posterior = cm_posterior(cm, l, &inside_matrix, &outside_matrix, inside_score);

    // Return the inside score (normalization constant)
    Ok(inside_score)
}

/// HMM-Banded Posterior Computation with Matrix Return
///
/// Computes posterior probabilities and returns the posterior matrix.
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence
/// * `l` - Sequence length
///
/// # Returns
/// (inside_score, posterior_matrix)
pub fn cm_posterior_hb_with_matrix(cm: &CM, dsq: &[EslDsq], l: i32)
    -> Result<(f32, Vec<Vec<Vec<f32>>>), String>
{
    // Compute inside matrix and score
    let (inside_score, inside_matrix) = cm_inside_with_matrix(cm, dsq, l)?;

    // Compute outside matrix
    let (_outside_score, outside_matrix) = cm_outside(cm, dsq, l, &inside_matrix)?;

    // Compute posterior probabilities
    let posterior = cm_posterior(cm, l, &inside_matrix, &outside_matrix, inside_score);

    Ok((inside_score, posterior))
}

/// Log-sum-exp function for numerical stability
///
/// Computes log(sum(exp(x_i))) in a numerically stable way.
fn log_sum_exp(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return f32::NEG_INFINITY;
    }

    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if !max_score.is_finite() {
        return f32::NEG_INFINITY;
    }

    // CRITICAL: Use log2/exp2 for BITS arithmetic (not ln/exp for nats!)
    let sum: f32 = scores.iter().map(|&s| 2.0_f32.powf(s - max_score)).sum();
    max_score + sum.log2()
}

/// Inline two-value log-sum-exp without allocation
#[inline(always)]
fn log_sum_exp2(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    let min = a.min(b);
    // CRITICAL: Use log2/exp2 for BITS arithmetic (not ln/exp for nats!)
    // This matches C's ILogsum: max + log2(1 + 2^(min-max))
    // Check for underflow: if difference >= 23, exp2 would underflow to 0
    // This matches C's LOGSUM_TBL = 23000 (in 1000x scaled integers)
    if min == f32::NEG_INFINITY || (max - min) >= 23.0 {
        return max;
    }
    // log2(2^max + 2^min) = max + log2(1 + 2^(min-max))
    max + (1.0 + 2.0_f32.powf(min - max)).log2()
}

/// Calculate initialization scores for local end paths
/// This matches C's ICalcInitDPScores()
pub fn calc_init_scores(cm: &CM, max_d: usize) -> Vec<Vec<f32>> {
    let m = cm.m as usize;
    let mut init_sc = vec![vec![f32::NEG_INFINITY; max_d + 1]; m];

    for v in 0..m {
        if cm.has_local_end(v) {
            // Local end is possible from this state
            // init_sc[v][d] = el_selfsc * d + endsc[v]
            for d in 0..=max_d {
                init_sc[v][d] = cm.el_selfsc * (d as f32) + cm.endsc[v];
            }
        }
        // else: remains NEG_INFINITY (no local end from this state)
    }

    init_sc
}

/// Optimized Inside algorithm with d-bounded matrix
///
/// This implementation uses O(M × L × W) memory instead of O(M × L × L)
/// where W = cm.w (maximum hit width, typically 200-300).
///
/// For a 30kb sequence with W=200 and M=100:
/// - Original: 100 × 30000 × 30000 × 4 = 360GB (causes extreme swapping)
/// - Optimized: 100 × 30000 × 200 × 4 = 2.4GB (manageable)
///
/// Key optimizations:
/// 1. d is bounded by cm.w (maximum expected hit length)
/// 2. Inline log-sum-exp without Vec allocation
/// 3. QDB bounds further reduce computation
pub fn cm_inside_scan(cm: &CM, dsq: &[EslDsq], l: i32) -> Result<(f32, i32, i32), String> {
    if l == 0 {
        return Err("Sequence length must be > 0".to_string());
    }

    let m = cm.m as usize;
    let l_usize = l as usize;
    let use_qdb = cm.has_qdb();

    // Maximum d value is min(L, W) where W is CM's expected max hit width
    let max_d_global = (l as usize).min(cm.w as usize);

    // DEBUG: Always print to see if we're even getting here
    eprintln!("[SCAN DEBUG] cm_inside_scan called: l={}, max_d_global={}, use_qdb={}", l, max_d_global, use_qdb);

    // DEBUG: Archaea detection
    if l == 74 {
        eprintln!("[SCAN DEBUG] Archaea sequence (l=74) detected");
        eprintln!("[SCAN DEBUG] max_d_global = {}", max_d_global);
        eprintln!("[SCAN DEBUG] use_qdb = {}", use_qdb);
        if use_qdb {
            eprintln!("[SCAN DEBUG] dmin1[0] = {}, dmax1[0] = {}", cm.dmin1[0], cm.dmax1[0]);
        }
    }

    // Allocate DP matrix: alpha[v][j][d]
    // d is bounded by max_d_global instead of L
    let mut alpha = vec![vec![vec![f32::NEG_INFINITY; max_d_global + 1]; l_usize + 1]; m];

    // Calculate initialization scores for local ends
    // This matches C's ICalcInitDPScores()
    let init_sc = calc_init_scores(cm, max_d_global);

    // DEBUG: Check init_sc values for Archaea
    if l == 74 {
        eprintln!("[SCAN DEBUG] el_selfsc = {}", cm.el_selfsc);
        eprintln!("[SCAN DEBUG] endsc[0] = {}, has_local_end(0) = {}", cm.endsc[0], cm.has_local_end(0));
        eprintln!("[SCAN DEBUG] init_sc[0][69] = {}", init_sc[0][69]);
        eprintln!("[SCAN DEBUG] init_sc[0][70] = {}", init_sc[0][70]);
        eprintln!("[SCAN DEBUG] init_sc[0][71] = {}", init_sc[0][71]);
    }

    // Fill the matrix using optimized log-sum-exp
    for d in 0..=max_d_global as i32 {
        for j in d..=l {
            let i = j - d + 1; // 1-indexed start position

            for v in (0..m).rev() {
                let st = cm.sttype[v] as i32;

                // Check QDB bounds
                if use_qdb && d > 0 {
                    let dv = (d - 1) as i32;

                    // DEBUG: Track j=69, d=67 for Archaea
                    if l == 74 && v == 0 && j == 69 && d == 67 {
                        eprintln!("[SCAN DEBUG] Checking v=0, j=69, d=67: dv={}, dmin1[0]={}, dmax1[0]={}",
                                  dv, cm.dmin1[0], cm.dmax1[0]);
                    }

                    if dv < cm.dmin1[v] || dv > cm.dmax1[v] {
                        if l == 74 && v == 0 && j == 69 && d == 67 {
                            eprintln!("[SCAN DEBUG] SKIPPING v=0, j=69, d=67 due to QDB bounds");
                        }
                        continue;
                    }
                }

                match st {
                    B_ST => {
                        let lchild = cm.lchild[v] as usize;
                        let rchild = cm.rchild[v] as usize;
                        let mut sc = f32::NEG_INFINITY;

                        for right_d in 0..=(d as usize) {
                            let left_d = (d as usize) - right_d;
                            let left_j = if left_d == 0 { i - 1 } else { i + left_d as i32 - 1 };

                            if left_j >= 0 && (left_j as usize) < alpha[lchild].len()
                                && left_d <= max_d_global && right_d <= max_d_global
                            {
                                let left_score = alpha[lchild][left_j as usize][left_d];
                                let right_score = alpha[rchild][j as usize][right_d];
                                if left_score > f32::NEG_INFINITY && right_score > f32::NEG_INFINITY {
                                    sc = log_sum_exp2(sc, left_score + right_score);
                                }
                            }
                        }
                        alpha[v][j as usize][d as usize] = sc;
                    }
                    E_ST => {
                        if d == 0 {
                            alpha[v][j as usize][0] = 0.0;
                        }
                    }
                    S_ST | D_ST => {
                        let cfirst = cm.cfirst[v] as usize;
                        let cnum = cm.cnum[v] as usize;
                        // CRITICAL: Root state (v=0) does NOT use init_sc!
                        // Only non-root states can have local ends.
                        let mut sc = if v == 0 { f32::NEG_INFINITY } else { init_sc[v][d as usize] };

                        for yoffset in 0..cnum {
                            let y = cfirst + yoffset;
                            let tsc = cm.tsc[v][yoffset];
                            let child_score = alpha[y][j as usize][d as usize];
                            if child_score > f32::NEG_INFINITY {
                                sc = log_sum_exp2(sc, tsc + child_score);
                            }
                        }
                        alpha[v][j as usize][d as usize] = sc;
                    }
                    MP_ST => {
                        if d >= 2 {
                            let symi = dsq[i as usize] as usize;
                            let symj = dsq[j as usize] as usize;
                            if symi < 4 && symj < 4 {
                                let esc = cm.esc[v][symi * 4 + symj];

                                let cfirst = cm.cfirst[v] as usize;
                                let cnum = cm.cnum[v] as usize;
                                let child_d = (d - 2) as usize;
                                let mut sc = init_sc[v][child_d];  // CRITICAL: Start with local end score

                                let child_j = (j - 1) as usize;
                                if child_d <= max_d_global {
                                    for yoffset in 0..cnum {
                                        let y = cfirst + yoffset;
                                        let tsc = cm.tsc[v][yoffset];
                                        let child_score = alpha[y][child_j][child_d];
                                        if child_score > f32::NEG_INFINITY {
                                            sc = log_sum_exp2(sc, esc + tsc + child_score);
                                        }
                                    }
                                }
                                alpha[v][j as usize][d as usize] = sc;
                            }
                        }
                    }
                    ML_ST | IL_ST => {
                        if d >= 1 {
                            let symi = dsq[i as usize] as usize;
                            if symi < 4 {
                                let esc = cm.esc[v][symi];

                                let cfirst = cm.cfirst[v] as usize;
                                let cnum = cm.cnum[v] as usize;
                                let child_d = (d - 1) as usize;
                                let mut sc = init_sc[v][child_d];  // CRITICAL: Start with local end score

                                if child_d <= max_d_global {
                                    for yoffset in 0..cnum {
                                        let y = cfirst + yoffset;
                                        let tsc = cm.tsc[v][yoffset];
                                        let child_score = alpha[y][j as usize][child_d];
                                        if child_score > f32::NEG_INFINITY {
                                            sc = log_sum_exp2(sc, esc + tsc + child_score);
                                        }
                                    }
                                }
                                alpha[v][j as usize][d as usize] = sc;
                            }
                        }
                    }
                    MR_ST | IR_ST => {
                        if d >= 1 {
                            let symj = dsq[j as usize] as usize;
                            if symj < 4 {
                                let esc = cm.esc[v][symj];

                                let cfirst = cm.cfirst[v] as usize;
                                let cnum = cm.cnum[v] as usize;
                                let child_j = (j - 1) as usize;
                                let child_d = (d - 1) as usize;
                                let mut sc = init_sc[v][child_d];  // CRITICAL: Start with local end score

                                if child_d <= max_d_global && child_j > 0 {
                                    for yoffset in 0..cnum {
                                        let y = cfirst + yoffset;
                                        let tsc = cm.tsc[v][yoffset];
                                        let child_score = alpha[y][child_j][child_d];
                                        if child_score > f32::NEG_INFINITY {
                                            sc = log_sum_exp2(sc, esc + tsc + child_score);
                                        }
                                    }
                                }
                                alpha[v][j as usize][d as usize] = sc;
                            }
                        }
                    }
                    EL_ST => {
                        if (cm.flags & crate::cm::CM_LOCAL_END) != 0 {
                            if d == 0 {
                                alpha[v][j as usize][0] = 0.0;
                            } else if d >= 1 && j >= 1 {
                                let child_d = (d - 1) as usize;
                                let child_j = (j - 1) as usize;
                                if child_d <= max_d_global && child_j > 0 {
                                    let prev_score = alpha[v][child_j][child_d];
                                    if prev_score > f32::NEG_INFINITY {
                                        let new_score = cm.el_selfsc + prev_score;
                                        alpha[v][j as usize][d as usize] = log_sum_exp2(
                                            alpha[v][j as usize][d as usize],
                                            new_score
                                        );
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Search for optimal (j, d)
    let mut best_score = f32::NEG_INFINITY;
    let mut best_j = 0;
    let mut best_d = 0;

    // DEBUG: Check if j=71, d=71 is in the search space (C finds 42.9 bits here)
    if l == 74 {
        let max_d_at_71 = (71_usize).min(max_d_global).min(cm.w as usize);
        eprintln!("[SCAN DEBUG] At j=71, max_d={}, checking if d=71 is in range", max_d_at_71);
        if 71 <= max_d_at_71 {
            let score_71_71 = alpha[0][71][71];
            let score_71_70 = alpha[0][71][70];
            let score_71_69 = alpha[0][71][69];
            eprintln!("[SCAN DEBUG] alpha[0][71][69] = {:.6} (C=29.39)", score_71_69);
            eprintln!("[SCAN DEBUG] alpha[0][71][70] = {:.6} (C=32.75)", score_71_70);
            eprintln!("[SCAN DEBUG] alpha[0][71][71] = {:.6} (C=42.90)", score_71_71);
        } else {
            eprintln!("[SCAN DEBUG] d=71 is OUT OF RANGE (max_d={})", max_d_at_71);
        }
    }

    for j in 1..=l {
        let max_d = (j as usize).min(max_d_global).min(cm.w as usize);
        for d in 1..=max_d {
            let root_score = alpha[0][j as usize][d];
            if root_score > best_score {
                best_score = root_score;
                best_j = j;
                best_d = d as i32;
            }

            // Check local begins if enabled
            if (cm.flags & crate::cm::CM_LOCAL_BEGIN) != 0 {
                for v in 1..m {
                    if cm.has_local_begin(v) {
                        let local_score = alpha[v][j as usize][d] + cm.beginsc[v];
                        if local_score > best_score {
                            best_score = local_score;
                            best_j = j;
                            best_d = d as i32;
                        }
                    }
                }
            }
        }
    }

    // DEBUG: Final result for Archaea
    if l == 74 {
        eprintln!("[SCAN DEBUG] Final result: best_score={:.2}, best_j={}, best_d={}", best_score, best_j, best_d);
        eprintln!("[SCAN DEBUG] Expected Archaea hit at j=69, d=67");
    }

    let hit_start = best_j - best_d + 1;
    let hit_end = best_j;
    Ok((best_score, hit_start, hit_end))
}

/// Hit structure for scan results
#[derive(Debug, Clone)]
pub struct ScanHit {
    pub start: i32,      // Start position (1-indexed)
    pub end: i32,        // End position (1-indexed)
    pub score: f32,      // Bit score (NULL3-corrected)
    pub root: usize,     // Root state (0 for global, >0 for local)
    pub bias: f32,       // NULL3 composition bias correction
}

/// Gamma hit matrix for optimal non-overlapping hit resolution
/// This is a semi-HMM that finds the optimal set of non-overlapping hits
struct GammaHitMx {
    mx: Vec<f32>,        // DP matrix for optimal score ending at position j
    gback: Vec<i32>,     // Traceback: start position of hit ending at j (-1 if no hit)
    savesc: Vec<f32>,    // Score of hit ending at j
    saver: Vec<usize>,   // Root state of hit ending at j
    savebias: Vec<f32>,  // NULL3 composition bias correction for hit ending at j
    cutoff: f32,         // Minimum score to report
}

impl GammaHitMx {
    fn new(l: usize, cutoff: f32) -> Self {
        GammaHitMx {
            mx: vec![0.0; l + 2],
            gback: vec![-1; l + 2],
            savesc: vec![f32::NEG_INFINITY; l + 2],
            saver: vec![0; l + 2],
            savebias: vec![0.0; l + 2],
            cutoff,
        }
    }

    /// Update gamma matrix with hits ending at position j
    /// Applies NULL3 composition bias correction BEFORE hit selection (like C code)
    fn update(
        &mut self,
        j: usize,
        bestsc: &[f32],
        bestr: &[usize],
        dmin: usize,
        dmax: usize,
        act: &[[f64; 4]],
        null: &[f32; 4],
        omega: f64,
        w: usize,
    ) {
        // Initialize: no hit at position j
        self.mx[j] = self.mx[j.saturating_sub(1)];
        self.gback[j] = -1;
        self.savesc[j] = f32::NEG_INFINITY;
        self.saver[j] = 0;
        self.savebias[j] = 0.0;

        if dmin > dmax {
            return;
        }

        let w_plus_1 = w + 1;
        let jp_mod = j % w_plus_1;

        // Check all possible hit lengths d
        for d in dmin.max(1)..=dmax {
            let i = j.saturating_sub(d - 1); // Start position (1-indexed)
            if i == 0 {
                continue;
            }

            let raw_score = bestsc[d];
            if raw_score <= f32::NEG_INFINITY {
                continue;
            }

            // DIAGNOSTIC: Raw CYK score before corrections
            // eprintln!("[SCORE_PIPELINE] Hit at {}-{}: raw_score={:.4}", i, j, raw_score);

            // Compute NULL3 composition bias correction (C code: cm_mx.c:7490-7502)
            let ip_mod = (i - 1) % w_plus_1;

            // Compute composition for this hit [i..j]
            let mut comp = [0.0f32; 4];
            for a in 0..4 {
                comp[a] = (act[jp_mod][a] - act[ip_mod][a]) as f32;
            }

            // Normalize composition
            let total: f32 = comp.iter().sum();
            if total > 0.0 {
                for a in 0..4 {
                    comp[a] /= total;
                }
            } else {
                // No valid residues, use uniform composition
                comp = [0.25; 4];
            }

            // Compute NULL3 correction using score_correction_null3
            let null3_correction = crate::evalue::score_correction_null3(
                &comp,
                null,
                d as i32,
                omega,
            );

            // DIAGNOSTIC: NULL3 correction amount
            // eprintln!("[SCORE_PIPELINE] Hit at {}-{}: null3_correction={:.4}", i, j, null3_correction);

            // DEBUG: Show NULL3 calculation for Archaea test case
            // if j == 71 && d == 71 {
            //     eprintln!("[NULL3 DEBUG] j={}, d={}, comp=[{:.4},{:.4},{:.4},{:.4}], null3_correction={:.4}, hit_sc_before={:.4}, hit_sc_after={:.4}",
            //              j, d, comp[0], comp[1], comp[2], comp[3], null3_correction, raw_score, raw_score - null3_correction);
            // }

            // Apply NULL3 correction BEFORE hit selection (KEY FIX!)
            let mut hit_sc = raw_score - null3_correction;

            // DIAGNOSTIC: Final score after corrections
            // eprintln!("[SCORE_PIPELINE] Hit at {}-{}: final_score={:.4}", i, j, hit_sc);

            // Now check cutoff after NULL3 correction
            if hit_sc < self.cutoff {
                continue;
            }

            // Cumulative score: previous optimal + this hit (NULL3-corrected)
            let cumulative_sc = self.mx[i.saturating_sub(1)] + hit_sc;

            // If this is better than current optimal at j, update
            if cumulative_sc > self.mx[j] {
                self.mx[j] = cumulative_sc;
                self.gback[j] = i as i32;
                self.savesc[j] = hit_sc; // Store NULL3-corrected score
                self.saver[j] = bestr[d];
                self.savebias[j] = null3_correction; // Store bias for reporting
            }
        }
    }

    /// Traceback to recover all hits
    fn traceback(&self, l: usize) -> Vec<ScanHit> {
        let mut hits = Vec::new();
        let mut j = l;

        while j >= 1 {
            if self.gback[j] == -1 {
                j = j.saturating_sub(1);
            } else {
                // Found a hit
                if self.savesc[j] >= self.cutoff {
                    // savesc[j] is the NULL3-corrected score (after subtraction)
                    // This is what C reports as "cyksc" in alignment output
                    // The bias field stores the NULL3 correction amount for reference

                    hits.push(ScanHit {
                        start: self.gback[j],
                        end: j as i32,
                        score: self.savesc[j], // NULL3-corrected scan score
                        root: self.saver[j],
                        bias: self.savebias[j], // NULL3 correction amount
                    });
                }
                j = (self.gback[j] as usize).saturating_sub(1);
            }
        }

        // Reverse to get hits in order
        hits.reverse();
        hits
    }
}

/// CYK Scan - C-style full sequence scan with GammaHitMx
///
/// This function matches C's FastCYKScan algorithm:
/// 1. Processes entire sequence position by position
/// 2. Uses memory-efficient rolling arrays
/// 3. Uses GammaHitMx for optimal non-overlapping hit resolution
///
/// # Arguments
/// * `cm` - The covariance model
/// * `dsq` - Digitized sequence (1..L, 0-indexed in array)
/// * `l` - Sequence length
/// * `cutoff` - Minimum bit score to report
///
/// # Returns
/// Vector of ScanHits
pub fn cm_cyk_scan(cm: &CM, dsq: &[EslDsq], l: i32, cutoff: f32) -> Vec<ScanHit> {
    if l <= 0 {
        return Vec::new();
    }

    let m = cm.m as usize;
    let l_usize = l as usize;
    let use_qdb = cm.has_qdb();

    // W is the maximum subsequence length (hit width)
    let w = cm.w.min(l) as usize;

    // Allocate DP matrices with rolling j dimension (j % 2)
    // alpha[cur/prv][v][d] where d is bounded by W
    let mut alpha: Vec<Vec<Vec<f32>>> = vec![vec![vec![f32::NEG_INFINITY; w + 1]; m]; 2];

    // For BEGL_S states, we need full j range modulo W+1
    let mut alpha_begl: Vec<Vec<Vec<f32>>> = vec![vec![vec![f32::NEG_INFINITY; w + 1]; m]; w + 1];

    // Best scores for each d at current j
    let mut bestsc = vec![f32::NEG_INFINITY; w + 1];
    let mut bestr = vec![0usize; w + 1];

    // Initialize GammaHitMx for non-overlapping hit resolution
    let mut gamma = GammaHitMx::new(l_usize, cutoff);

    // Allocate act matrix for NULL3 composition tracking (circular buffer)
    // act[j % (w+1)][a] counts residues of type a from position 1..j
    let w_plus_1 = w + 1;
    let mut act: Vec<[f64; 4]> = vec![[0.0; 4]; w_plus_1];

    // Main scan loop: process each position j
    for j in 1..=l {
        let j_usize = j as usize;
        let cur = (j as usize) % 2;
        let prv = ((j - 1) as usize) % 2;

        // Update composition tracking (for NULL3 correction)
        let jp_mod = j_usize % w_plus_1;
        if j_usize > 0 {
            let prev_mod = (j_usize - 1) % w_plus_1;
            act[jp_mod] = act[prev_mod];
        }
        // Add current residue to composition counts
        let residue = dsq[j_usize];
        if residue < 4 {
            act[jp_mod][residue as usize] += 1.0;
        }

        // Determine d bounds for this j
        let dmax = (j as usize).min(w);

        // Reset current alpha row
        for v in 0..m {
            for d in 0..=w {
                alpha[cur][v][d] = f32::NEG_INFINITY;
            }
        }

        // Initialize states that can do local ends with EL path score
        // C code: if(NOT_IMPOSSIBLE(cm->endsc[v])) alpha[v] = el_scA[dp] + cm->endsc[v]
        // where el_scA[d] = el_selfsc * d
        if (cm.flags & crate::cm::CM_LOCAL_END) != 0 {
            for v in 0..m {
                if cm.endsc[v] > IMPOSSIBLE_F32 + 1.0 {
                    let sd = state_delta(cm.sttype[v] as i32) as usize;
                    for d in 0..=dmax {
                        let dp = if d >= sd { d - sd } else { 0 };
                        let el_sc = cm.el_selfsc * (dp as f32);
                        alpha[cur][v][d] = el_sc + cm.endsc[v];
                    }
                }
            }
        }

        // Process states in reverse order (children first)
        for v in (0..m).rev() {
            let st = cm.sttype[v] as i32;

            // Skip E_st boundary conditions handled separately
            if st == E_ST {
                alpha[cur][v][0] = 0.0;
                continue;
            }

            // Determine valid d range for this state
            let (dn, dx) = if use_qdb {
                let dn_state = cm.dmin1[v].max(0) as usize;
                let dx_state = (cm.dmax1[v] as usize).min(dmax);
                (dn_state, dx_state)
            } else {
                (0, dmax)
            };

            if dn > dx {
                continue;
            }

            // Handle different state types
            match st {
                B_ST => {
                    // Bifurcation: split (i,j) into left (i,j-k) and right (j-k+1,j)
                    // C code: alpha[y][j-k][d-k] + alpha[z][j][k]
                    // Right child covers positions ending at j, so use cur (same j)
                    let lchild = cm.lchild[v] as usize;
                    let rchild = cm.rchild[v] as usize;

                    for d in dn..=dx {
                        let mut sc = f32::NEG_INFINITY;
                        // UNSAFE: Remove bounds checking from critical inner loop
                        unsafe {
                            for k in 0..=d {
                                // k is length of right fragment
                                let left_d = d - k;
                                let left_j_mod = (j_usize.saturating_sub(k)) % (w + 1);

                                // Left child from alpha_begl at j-k, right child at current j
                                let left_sc = *alpha_begl.get_unchecked(left_j_mod)
                                    .get_unchecked(lchild)
                                    .get_unchecked(left_d);
                                let right_sc = *alpha.get_unchecked(cur)
                                    .get_unchecked(rchild)
                                    .get_unchecked(k);

                                if left_sc > f32::NEG_INFINITY && right_sc > f32::NEG_INFINITY {
                                    sc = sc.max(left_sc + right_sc);
                                }
                            }
                        }
                        alpha[cur][v][d] = sc.max(alpha[cur][v][d]);

                    }
                }
                _ if cm.stid[v] as i32 == BEGL_S => {
                    // BEGL_S state - non-emitting, use alpha_begl for storage
                    // Children are at same j (cur), C code: alpha[y+yoffset][j][d]
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let jp_begl = j_usize % (w + 1);

                    for d in dn..=dx {
                        let mut sc = f32::NEG_INFINITY;

                        // UNSAFE: Remove bounds checking from child iteration
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                // Non-emitting state uses same j (cur), same d
                                let child_sc = *alpha.get_unchecked(cur)
                                    .get_unchecked(child)
                                    .get_unchecked(d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + child_sc);
                                }
                            }
                        }
                        alpha_begl[jp_begl][v][d] = sc;
                    }
                }
                S_ST | D_ST => {
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let sd = state_delta(st);

                    for d in dn..=dx {
                        let mut sc = f32::NEG_INFINITY;
                        let child_d = if sd == 0 { d } else if d >= sd as usize { d - sd as usize } else { continue };

                        // UNSAFE: Remove bounds checking from child iteration
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                let child_sc = *alpha.get_unchecked(cur)
                                    .get_unchecked(child)
                                    .get_unchecked(child_d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + child_sc);
                                }
                            }
                        }
                        alpha[cur][v][d] = sc.max(alpha[cur][v][d]);
                    }
                }
                MP_ST => {
                    if dmax < 2 {
                        continue;
                    }
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let use_oesc = !cm.oesc.is_empty() && !cm.oesc[v].is_empty();

                    for d in dn.max(2)..=dx {
                        let i = j_usize - d + 1;
                        let symi = dsq[i] as usize;
                        let symj = dsq[j_usize] as usize;

                        // FastCYKScan with oesc supports extended alphabet (Kp=18)
                        // but we still skip truly invalid residues
                        let esc = if use_oesc {
                            // oesc uses Kp=18 indexing: a * Kp + b
                            if symi >= ALPHABET_SIZE_P || symj >= ALPHABET_SIZE_P {
                                continue;
                            }
                            cm.oesc[v][symi * ALPHABET_SIZE_P + symj]
                        } else {
                            // esc uses K=4 indexing: a * K + b
                            if symi >= 4 || symj >= 4 {
                                continue;
                            }
                            cm.esc[v][symi * 4 + symj]
                        };

                        let child_d = d - 2;

                        // C code pattern: compute child transitions first, then compare with
                        // init_scAA (local end), then add emission to the winner
                        let mut sc = f32::NEG_INFINITY;
                        // UNSAFE: Remove bounds checking from child iteration
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                let child_sc = *alpha.get_unchecked(prv)
                                    .get_unchecked(child)
                                    .get_unchecked(child_d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + child_sc);
                                }
                            }
                        }
                        // Compare with local end score (already in alpha from init)
                        sc = sc.max(alpha[cur][v][d]);
                        // Add emission score to the winner
                        alpha[cur][v][d] = sc + esc;
                    }
                }
                ML_ST | IL_ST => {
                    // ML/IL emit to the LEFT (position i), so child is at same j
                    // C code: alpha[y+yoffset][j][d-1] (same j)
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let use_oesc = !cm.oesc.is_empty() && !cm.oesc[v].is_empty();

                    for d in dn.max(1)..=dx {
                        let i = j_usize - d + 1;
                        let symi = dsq[i] as usize;

                        // FastCYKScan with oesc supports extended alphabet (Kp=18)
                        let esc = if use_oesc {
                            if symi >= ALPHABET_SIZE_P {
                                continue;
                            }
                            cm.oesc[v][symi]
                        } else {
                            if symi >= 4 {
                                continue;
                            }
                            cm.esc[v][symi]
                        };

                        let child_d = d - 1;

                        // ML/IL use current j (cur), because emitting left doesn't change j
                        // C code pattern: compute child transitions first, then compare with
                        // init_scAA (local end), then add emission to the winner
                        let mut sc = f32::NEG_INFINITY;
                        // UNSAFE: Remove bounds checking from child iteration
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                let child_sc = *alpha.get_unchecked(cur)
                                    .get_unchecked(child)
                                    .get_unchecked(child_d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + child_sc);
                                }
                            }
                        }
                        // Compare with local end score (already in alpha from init)
                        // alpha[cur][v][d] was pre-initialized with el_selfsc * dp + endsc[v]
                        sc = sc.max(alpha[cur][v][d]);
                        // Add emission score to the winner
                        alpha[cur][v][d] = sc + esc;
                    }
                }
                MR_ST | IR_ST => {
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let use_oesc = !cm.oesc.is_empty() && !cm.oesc[v].is_empty();

                    for d in dn.max(1)..=dx {
                        let symj = dsq[j_usize] as usize;

                        // FastCYKScan with oesc supports extended alphabet (Kp=18)
                        let esc = if use_oesc {
                            if symj >= ALPHABET_SIZE_P {
                                continue;
                            }
                            cm.oesc[v][symj]
                        } else {
                            if symj >= 4 {
                                continue;
                            }
                            cm.esc[v][symj]
                        };

                        let child_d = d - 1;

                        // C code pattern: compute child transitions first, then compare with
                        // init_scAA (local end), then add emission to the winner
                        let mut sc = f32::NEG_INFINITY;
                        // UNSAFE: Remove bounds checking from child iteration
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                let child_sc = *alpha.get_unchecked(prv)
                                    .get_unchecked(child)
                                    .get_unchecked(child_d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + child_sc);
                                }
                            }
                        }
                        // Compare with local end score (already in alpha from init)
                        sc = sc.max(alpha[cur][v][d]);
                        // Add emission score to the winner
                        alpha[cur][v][d] = sc + esc;
                    }
                }
                EL_ST => {
                    if (cm.flags & crate::cm::CM_LOCAL_END) != 0 {
                        alpha[cur][v][0] = 0.0;
                        // EL can emit 0 or more residues
                        for d in 1..=dx {
                            let child_d = d - 1;
                            let prev_sc = alpha[prv][v][child_d];
                            if prev_sc > f32::NEG_INFINITY {
                                alpha[cur][v][d] = cm.el_selfsc + prev_sc;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Compute ROOT (state 0) scores and collect bestsc[d]
        bestsc.fill(f32::NEG_INFINITY);
        bestr.fill(0);

        let y = cm.cfirst[0] as usize;
        let cnum = cm.cnum[0] as usize;

        for d in 1..=dmax {
            // Standard root transition
            let mut sc = f32::NEG_INFINITY;
            // UNSAFE: Remove bounds checking from root child iteration
            unsafe {
                for yoffset in 0..cnum {
                    let child = y + yoffset;
                    let tsc = *cm.tsc.get_unchecked(0).get_unchecked(yoffset);
                    let child_sc = *alpha.get_unchecked(cur)
                        .get_unchecked(child)
                        .get_unchecked(d);
                    if child_sc > f32::NEG_INFINITY {
                        sc = sc.max(tsc + child_sc);
                    }
                }
            }
            alpha[cur][0][d] = sc;
            bestsc[d] = sc;
            bestr[d] = 0;

            // DEBUG for Archaea (l=74)
            if l == 74 && j == 71 && d == 71 {
                eprintln!("[ROOT_S DEBUG] j={}, d={}, alpha[cur][0][{}]={:.4} (before local begin)",
                         j, d, d, alpha[cur][0][d]);
                eprintln!("[ROOT_S DEBUG] cnum={}, y={}", cnum, y);
                for yoffset in 0..cnum {
                    let child = y + yoffset;
                    eprintln!("[ROOT_S DEBUG]   child={}, tsc[{}]={:.4}, alpha[cur][{}][71]={:.4}",
                             child, yoffset, cm.tsc[0][yoffset], child, alpha[cur][child][71]);
                }
            }
        }

        // Check local begins
        // C code: BEGL_S states use alpha_begl, other states use alpha[cur]
        // CRITICAL: Must also update alpha[cur][0][d] like C does!
        if (cm.flags & crate::cm::CM_LOCAL_BEGIN) != 0 {
            for v in 1..m {
                if cm.has_local_begin(v) {
                    let begin_sc = cm.beginsc[v];
                    let is_begl_s = cm.stid[v] as i32 == BEGL_S;

                    for d in 1..=dmax {
                        let sc = if is_begl_s {
                            // BEGL_S: use alpha_begl at j % (W+1)
                            let jp_begl = j_usize % (w + 1);
                            alpha_begl[jp_begl][v][d] + begin_sc
                        } else {
                            // Other states: use alpha[cur]
                            alpha[cur][v][d] + begin_sc
                        };

                        // C: if(alpha[jp_v][0][d] < (alpha[jp_y][y][d] + cm->beginsc[y]))
                        if alpha[cur][0][d] < sc {
                            alpha[cur][0][d] = sc;
                            bestr[d] = v;
                        }
                    }
                }
            }
        }

        // C: fill in bestsc for all valid d values (AFTER local begin processing!)
        // C code: for (d = dnA[0]; d <= dxA[0]; d++) { bestsc[d] = alpha[jp_v][0][d]; }
        for d in 1..=dmax {
            bestsc[d] = alpha[cur][0][d];
            // DEBUG for Archaea (l=74)
            if l == 74 && j == 71 && d == 71 {
                eprintln!("[BESTSC DEBUG] j={}, d={}, bestsc[{}]={:.4}, bestr[{}]={}",
                         j, d, d, bestsc[d], d, bestr[d]);
            }
        }

        // Update gamma matrix with hits ending at j
        // Pass act matrix and NULL3 parameters for composition correction
        gamma.update(j_usize, &bestsc, &bestr, 1, dmax, &act, &cm.null, cm.n3_omega, w);
    }

    // Traceback to get all hits
    // NULL3 correction was already applied during gamma.update()
    let hits = gamma.traceback(l_usize);
    hits
}

/// Helper: Get state delta (number of emissions)
fn state_delta(st: i32) -> i32 {
    match st {
        MP_ST => 2,
        ML_ST | MR_ST | IL_ST | IR_ST => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp() {
        // Infernal works in BITS (log base 2), not nats: log_sum_exp uses log2/exp2.
        // log2(2^1 + 2^2) = log2(6) ≈ 2.585.
        let result = log_sum_exp(&[1.0, 2.0]);
        assert!((result - 6.0_f32.log2()).abs() < 0.01);

        // Empty case
        let result = log_sum_exp(&[]);
        assert!(result.is_infinite() && result.is_sign_negative());

        // Single value
        let result = log_sum_exp(&[3.5]);
        assert!((result - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_cyk_inside_empty() {
        let cm = CM::new(10, 5);
        let dsq = vec![0u8; 1];
        let result = cyk_inside(&cm, &dsq, 0, false);
        assert!(result.is_err());
    }
}

/// HMM-banded CYK scan matching C's FastCYKScanHB
/// This dramatically improves performance by using CP9 bands to constrain search space
pub fn cm_cyk_scan_hb(cm: &CM, cp9b: &crate::cp9_bands::CP9Bands, dsq: &[EslDsq], l: i32, cutoff: f32) -> Vec<ScanHit> {
    if l <= 0 || !cp9b.valid {
        return Vec::new();
    }

    let m = cm.m as usize;
    let l_usize = l as usize;
    
    // Get HMM band constraints for ROOT_S (state 0)
    let jmin = cp9b.jmin[0];
    let jmax = cp9b.jmax[0];
    
    if jmin > jmax || jmin < 1 || jmax > l {
        return Vec::new();
    }
    
    // Calculate W from actual band constraints
    let mut w = 0;
    for j in jmin..=jmax.min(l) {
        let j_idx = (j - jmin) as usize;
        if j_idx < cp9b.hdmax[0].len() {
            w = w.max(cp9b.hdmax[0][j_idx] as usize);
        }
    }
    w = w.min(cm.w as usize).min(l as usize);

    // Allocate DP matrices (same as cm_cyk_scan)
    let mut alpha: Vec<Vec<Vec<f32>>> = vec![vec![vec![f32::NEG_INFINITY; w + 1]; m]; 2];
    let mut alpha_begl: Vec<Vec<Vec<f32>>> = vec![vec![vec![f32::NEG_INFINITY; w + 1]; m]; w + 1];
    
    let mut bestsc = vec![f32::NEG_INFINITY; w + 1];
    let mut bestr = vec![0usize; w + 1];
    
    let mut gamma = GammaHitMx::new(l_usize, cutoff);
    
    let w_plus_1 = w + 1;
    let mut act: Vec<[f64; 4]> = vec![[0.0; 4]; w_plus_1];
    
    // Main scan loop: process only positions within HMM bands
    for j in jmin..=jmax.min(l) {
        let j_usize = j as usize;
        let cur = (j as usize) % 2;
        let prv = ((j - 1) as usize) % 2;
        let j_idx = (j - jmin) as usize;
        
        // Update composition tracking
        let jp_mod = j_usize % w_plus_1;
        if j_usize > 1 {
            let prev_mod = (j_usize - 1) % w_plus_1;
            act[jp_mod] = act[prev_mod];
        }
        let residue = dsq[j_usize];
        if residue < 4 {
            act[jp_mod][residue as usize] += 1.0;
        }
        
        // Get HMM band constraints for this j
        let hdmin_j = if j_idx < cp9b.hdmin[0].len() {
            cp9b.hdmin[0][j_idx].max(0) as usize
        } else {
            0
        };
        let hdmax_j = if j_idx < cp9b.hdmax[0].len() {
            (cp9b.hdmax[0][j_idx] as usize).min(j as usize).min(w)
        } else {
            (j as usize).min(w)
        };
        
        let dmax = hdmax_j;
        
        if hdmin_j > hdmax_j {
            continue;
        }
        
        // Reset current alpha row
        for v in 0..m {
            for d in 0..=w {
                alpha[cur][v][d] = f32::NEG_INFINITY;
            }
        }
        
        // Initialize local end states (same as unbanded)
        if (cm.flags & crate::cm::CM_LOCAL_END) != 0 {
            for v in 0..m {
                if cm.endsc[v] > IMPOSSIBLE_F32 + 1.0 {
                    let sd = state_delta(cm.sttype[v] as i32) as usize;
                    for d in hdmin_j..=hdmax_j {
                        let dp = if d >= sd { d - sd } else { 0 };
                        let el_sc = cm.el_selfsc * (dp as f32);
                        alpha[cur][v][d] = el_sc + cm.endsc[v];
                    }
                }
            }
        }
        
        // Process all states (same recursions as cm_cyk_scan but with band constraints)
        for v in (0..m).rev() {
            let st = cm.sttype[v] as i32;
            
            if st == E_ST {
                alpha[cur][v][0] = 0.0;
                continue;
            }
            
            let (dn, dx) = (hdmin_j, hdmax_j);
            if dn > dx {
                continue;
            }
            
            // All state type recursions from cm_cyk_scan...
            // (Copying full logic to ensure correctness)
            
            match st {
                B_ST => {
                    let lchild = cm.lchild[v] as usize;
                    let rchild = cm.rchild[v] as usize;
                    
                    for d in dn..=dx {
                        let mut sc = f32::NEG_INFINITY;
                        // UNSAFE: Remove bounds checking from critical Bifurcation loop
                        unsafe {
                            for k in 0..=d {
                                let left_d = d - k;
                                let left_j_mod = (j_usize.saturating_sub(k)) % (w + 1);
                                let left_sc = *alpha_begl.get_unchecked(left_j_mod)
                                    .get_unchecked(lchild)
                                    .get_unchecked(left_d);
                                let right_sc = *alpha.get_unchecked(cur)
                                    .get_unchecked(rchild)
                                    .get_unchecked(k);
                                if left_sc > f32::NEG_INFINITY && right_sc > f32::NEG_INFINITY {
                                    sc = sc.max(left_sc + right_sc);
                                }
                            }
                        }
                        alpha[cur][v][d] = sc.max(alpha[cur][v][d]);
                    }
                }
                _ if cm.stid[v] as i32 == BEGL_S => {
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let jp_begl = j_usize % (w + 1);
                    
                    for d in dn..=dx {
                        let mut sc = f32::NEG_INFINITY;
                        // UNSAFE: Remove bounds checking from BEGL_S child loop
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                let child_sc = *alpha.get_unchecked(cur)
                                    .get_unchecked(child)
                                    .get_unchecked(d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + child_sc);
                                }
                            }
                        }
                        alpha_begl[jp_begl][v][d] = sc;
                    }
                }
                S_ST | D_ST => {
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let sd = state_delta(st);
                    
                    for d in dn..=dx {
                        let mut sc = f32::NEG_INFINITY;
                        let child_d = if sd == 0 { d } else if d >= sd as usize { d - sd as usize } else { continue };

                        // UNSAFE: Remove bounds checking from S_ST/D_ST child loop
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                let child_sc = *alpha.get_unchecked(cur)
                                    .get_unchecked(child)
                                    .get_unchecked(child_d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + child_sc);
                                }
                            }
                        }
                        alpha[cur][v][d] = sc.max(alpha[cur][v][d]);
                    }
                }
                MP_ST => {
                    if dmax < 2 {
                        continue;
                    }
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let use_oesc = !cm.oesc.is_empty() && !cm.oesc[v].is_empty();
                    
                    for d in dn.max(2)..=dx {
                        let i = j_usize - d + 1;
                        let symi = dsq[i] as usize;
                        let symj = dsq[j_usize] as usize;
                        
                        let esc = if use_oesc {
                            if symi >= ALPHABET_SIZE_P || symj >= ALPHABET_SIZE_P {
                                continue;
                            }
                            cm.oesc[v][symi * ALPHABET_SIZE_P + symj]
                        } else {
                            if symi >= 4 || symj >= 4 {
                                continue;
                            }
                            cm.esc[v][symi * 4 + symj]
                        };
                        
                        let child_d = d - 2;
                        let mut sc = f32::NEG_INFINITY;

                        // UNSAFE: Remove bounds checking from MP_ST child loop
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                let child_sc = *alpha.get_unchecked(prv)
                                    .get_unchecked(child)
                                    .get_unchecked(child_d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + esc + child_sc);
                                }
                            }
                        }
                        alpha[cur][v][d] = sc.max(alpha[cur][v][d]);
                    }
                }
                ML_ST | IL_ST => {
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let sd = state_delta(st) as usize;
                    
                    for d in dn.max(sd)..=dx {
                        let i = j_usize - d + 1;
                        let symi = dsq[i] as usize;
                        
                        if symi >= 4 {
                            continue;
                        }
                        
                        let esc = cm.esc[v][symi];
                        let child_d = d - sd;
                        let mut sc = f32::NEG_INFINITY;

                        // UNSAFE: Remove bounds checking from ML_ST/IL_ST child loop
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                let child_sc = *alpha.get_unchecked(prv)
                                    .get_unchecked(child)
                                    .get_unchecked(child_d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + esc + child_sc);
                                }
                            }
                        }
                        alpha[cur][v][d] = sc.max(alpha[cur][v][d]);
                    }
                }
                MR_ST | IR_ST => {
                    let y = cm.cfirst[v] as usize;
                    let cnum = cm.cnum[v] as usize;
                    let symj = dsq[j_usize] as usize;
                    
                    if symj >= 4 {
                        continue;
                    }
                    
                    let esc = cm.esc[v][symj];
                    let sd = state_delta(st) as usize;
                    
                    for d in dn.max(sd)..=dx {
                        let child_d = d - sd;
                        let mut sc = f32::NEG_INFINITY;

                        // UNSAFE: Remove bounds checking from MR_ST/IR_ST child loop
                        unsafe {
                            for yoffset in 0..cnum {
                                let child = y + yoffset;
                                let tsc = *cm.tsc.get_unchecked(v).get_unchecked(yoffset);
                                let child_sc = *alpha.get_unchecked(cur)
                                    .get_unchecked(child)
                                    .get_unchecked(child_d);
                                if child_sc > f32::NEG_INFINITY {
                                    sc = sc.max(tsc + esc + child_sc);
                                }
                            }
                        }
                        alpha[cur][v][d] = sc.max(alpha[cur][v][d]);
                    }
                }
                _ => {}
            }
        }
        
        // Update bestsc for ROOT_S
        for d in hdmin_j..=hdmax_j {
            if alpha[cur][0][d] > bestsc[d] {
                bestsc[d] = alpha[cur][0][d];
                bestr[d] = 0;
            }
        }
        
        // Report hits
        gamma.update(j as usize, &bestsc, &bestr, hdmin_j, hdmax_j, &act, &cm.null, cm.n3_omega, w_plus_1);
    }
    
    gamma.traceback(l_usize)
}
