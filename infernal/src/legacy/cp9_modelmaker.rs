//! CP9 Model Maker - Build CP9 HMM from CM
//!
//! Port of cp9_modelmaker.c:CPlan9FromCM() from Infernal 1.1.5
//!
//! This module creates a CP9 profile HMM from a covariance model.
//! The CP9 HMM is used for efficient banding in CM alignment.

use crate::cm::{CM, ALPHABET_SIZE};
use crate::cm_emitmap::{create_emit_map, CMEmitMap};
use crate::cm_subinfo::cm_expected_state_occupancy;
use crate::constants::*;
use crate::cp9::{
    CP9, CP9_NTRANS, CTDD, CTDI, CTDM, CTID, CTII, CTIM, CTMD, CTMEL, CTMI, CTMM,
    CP9_HASNULL,
};
use crate::cp9_map::{cp9_map_cm2hmm, CP9Map, HmmStateType};

/// Build a CP9 HMM from a CM.
///
/// This is the main entry point for building a CP9 HMM from a CM.
/// It creates the mapping, computes expected state occupancies (psi),
/// and fills in the HMM transition and emission probabilities.
///
/// # Arguments
/// * `cm` - The covariance model
///
/// # Returns
/// * `Ok((CP9, CP9Map))` - The CP9 HMM and the mapping structure
/// * `Err(String)` - Error message on failure
pub fn cplan9_from_cm(cm: &CM) -> Result<(CP9, CP9Map), String> {
    // Get consensus length
    let hmm_m = cm.clen;
    if hmm_m <= 0 {
        return Err("CM has no consensus length".to_string());
    }

    // Create the CP9 HMM
    let mut cp9 = CP9::new(hmm_m);

    // Create emit map
    let emap = create_emit_map(cm).ok_or("Failed to create emit map")?;

    // Create CM-HMM mapping
    let cp9map = cp9_map_cm2hmm(cm)?;

    // Compute expected state occupancy (psi)
    let psi = cm_expected_state_occupancy(cm)?;

    // Fill in the HMM parameters
    fill_psi_based_params(cm, &mut cp9, &cp9map, &psi, &emap)?;

    // Normalize the HMM probabilities
    cplan9_normalize(&mut cp9)?;

    // Copy null model from CM
    for a in 0..ALPHABET_SIZE {
        cp9.null[a] = cm.null[a];
    }
    cp9.flags |= CP9_HASNULL;

    // Compute log-odds scores
    cp9.logoddsify();

    Ok((cp9, cp9map))
}

/// Fill CP9 HMM parameters based on CM psi values
fn fill_psi_based_params(
    cm: &CM,
    cp9: &mut CP9,
    cp9map: &CP9Map,
    psi: &[f64],
    emap: &CMEmitMap,
) -> Result<(), String> {
    let hmm_m = cp9.m as usize;

    // Initialize all parameters to zero
    for k in 0..=hmm_m + 1 {
        for t in 0..CP9_NTRANS {
            cp9.t[k][t] = 0.0;
        }
    }
    for k in 0..=hmm_m {
        for a in 0..ALPHABET_SIZE {
            cp9.mat[k][a] = 0.0;
            cp9.ins[k][a] = 0.0;
        }
        cp9.begin[k] = 0.0;
        cp9.end[k] = 0.0;
    }

    // Fill node 0 (special root handling)
    fill_node_0(cm, cp9, psi)?;

    // Fill nodes 1 to M
    for k in 1..=hmm_m {
        fill_node_k(cm, cp9, cp9map, psi, emap, k)?;
    }

    // Fill transitions out of final node
    fill_final_transitions(cm, cp9, cp9map, psi, emap)?;

    Ok(())
}

/// Fill node 0 parameters (special root handling)
fn fill_node_0(cm: &CM, cp9: &mut CP9, psi: &[f64]) -> Result<(), String> {
    // Node 0 handles ROOT_IL insert state emissions
    // ROOT_IL is state 1 in the CM

    let root_il = 1_usize; // ROOT_IL state index

    // Insert emissions at node 0 from ROOT_IL
    if psi[root_il] > 0.0 {
        for a in 0..ALPHABET_SIZE {
            cp9.ins[0][a] = cm.e[root_il][a];
        }
    } else {
        // If ROOT_IL is never visited, use uniform
        for a in 0..ALPHABET_SIZE {
            cp9.ins[0][a] = 0.25;
        }
    }

    // Transition parameters for node 0
    // t[0][CTIM] = prob of ROOT_IL -> next match
    // t[0][CTII] = prob of ROOT_IL -> ROOT_IL (self loop)

    // Get ROOT_IL transition probabilities
    if cm.cnum[root_il] > 0 {
        // Self-loop probability (first child of ROOT_IL is itself)
        cp9.t[0][CTII] = cm.t[root_il][0];

        // Probability to move on (sum of non-self transitions)
        let mut t_im = 0.0_f32;
        let cnum = cm.cnum[root_il] as usize;
        for k in 1..cnum {
            t_im += cm.t[root_il][k];
        }
        cp9.t[0][CTIM] = t_im;
    }

    Ok(())
}

/// Fill parameters for HMM node k (1..M)
fn fill_node_k(
    cm: &CM,
    cp9: &mut CP9,
    cp9map: &CP9Map,
    psi: &[f64],
    _emap: &CMEmitMap,
    k: usize,
) -> Result<(), String> {
    let m = cm.m as usize;

    // Get CM states mapping to this HMM node
    let (v_match1, v_match2) = cp9map.get_cm_states(k, HmmStateType::Match);
    let (v_insert, _) = cp9map.get_cm_states(k, HmmStateType::Insert);
    let (v_delete1, _v_delete2) = cp9map.get_cm_states(k, HmmStateType::Delete);

    // === Match emissions ===
    fill_match_emissions(cm, cp9, psi, k, v_match1, v_match2)?;

    // === Insert emissions ===
    fill_insert_emissions(cm, cp9, psi, k, v_insert)?;

    // === Transitions ===
    // Fill transition probabilities based on psi-weighted CM transitions

    // From Match state
    if let Some(v_m) = v_match1 {
        if v_m < m {
            fill_match_transitions(cm, cp9, cp9map, psi, k, v_m)?;
        }
    }

    // From Insert state
    if let Some(v_i) = v_insert {
        if v_i < m {
            fill_insert_transitions(cm, cp9, psi, k, v_i)?;
        }
    }

    // From Delete state
    if let Some(v_d) = v_delete1 {
        if v_d < m {
            fill_delete_transitions(cm, cp9, cp9map, psi, k, v_d)?;
        }
    }

    Ok(())
}

/// Fill match emission probabilities for node k
fn fill_match_emissions(
    cm: &CM,
    cp9: &mut CP9,
    psi: &[f64],
    k: usize,
    v1: Option<usize>,
    v2: Option<usize>,
) -> Result<(), String> {
    let mut total_psi = 0.0_f64;

    // Accumulate weighted emissions from CM match states
    if let Some(v) = v1 {
        let sttype = cm.sttype[v] as i32;
        if sttype == MP_ST {
            // Pair state - marginalize over pairs for left/right
            // For left: sum over second residue
            // For right: sum over first residue
            // Here we use a simplified approach: average over pair
            let psi_v = psi[v];
            total_psi += psi_v;
            for a in 0..ALPHABET_SIZE {
                // Sum emissions where this residue appears
                let mut emit_prob = 0.0_f32;
                for b in 0..ALPHABET_SIZE {
                    // Left emission: pair index = a*4 + b
                    emit_prob += cm.e[v][a * 4 + b];
                }
                cp9.mat[k][a] += emit_prob * psi_v as f32;
            }
        } else if sttype == ML_ST || sttype == MR_ST {
            // Single emission state
            let psi_v = psi[v];
            total_psi += psi_v;
            for a in 0..ALPHABET_SIZE {
                cp9.mat[k][a] += cm.e[v][a] * psi_v as f32;
            }
        }
    }

    if let Some(v) = v2 {
        let sttype = cm.sttype[v] as i32;
        if sttype == MP_ST {
            let psi_v = psi[v];
            total_psi += psi_v;
            for a in 0..ALPHABET_SIZE {
                let mut emit_prob = 0.0_f32;
                for b in 0..ALPHABET_SIZE {
                    emit_prob += cm.e[v][a * 4 + b];
                }
                cp9.mat[k][a] += emit_prob * psi_v as f32;
            }
        } else if sttype == ML_ST || sttype == MR_ST {
            let psi_v = psi[v];
            total_psi += psi_v;
            for a in 0..ALPHABET_SIZE {
                cp9.mat[k][a] += cm.e[v][a] * psi_v as f32;
            }
        }
    }

    // Normalize
    if total_psi > 0.0 {
        for a in 0..ALPHABET_SIZE {
            cp9.mat[k][a] /= total_psi as f32;
        }
    } else {
        // If no psi, use uniform
        for a in 0..ALPHABET_SIZE {
            cp9.mat[k][a] = 0.25;
        }
    }

    Ok(())
}

/// Fill insert emission probabilities for node k
fn fill_insert_emissions(
    cm: &CM,
    cp9: &mut CP9,
    _psi: &[f64],
    k: usize,
    v_insert: Option<usize>,
) -> Result<(), String> {
    if let Some(v) = v_insert {
        let sttype = cm.sttype[v] as i32;
        if sttype == IL_ST || sttype == IR_ST {
            // Copy insert emissions directly
            for a in 0..ALPHABET_SIZE {
                cp9.ins[k][a] = cm.e[v][a];
            }
            return Ok(());
        }
    }

    // Default to uniform if no insert state
    for a in 0..ALPHABET_SIZE {
        cp9.ins[k][a] = 0.25;
    }

    Ok(())
}

/// Fill match state transitions for node k
fn fill_match_transitions(
    cm: &CM,
    cp9: &mut CP9,
    cp9map: &CP9Map,
    psi: &[f64],
    k: usize,
    v: usize,
) -> Result<(), String> {
    let hmm_m = cp9.m as usize;
    let cnum = cm.cnum[v] as usize;
    let cfirst = cm.cfirst[v] as usize;

    let psi_v = psi[v];
    if psi_v <= 0.0 {
        return Ok(());
    }

    // Accumulate transitions to each child
    for c in 0..cnum {
        let child = cfirst + c;
        if child >= cm.m as usize {
            continue;
        }

        let t_prob = cm.t[v][c];
        if t_prob <= 0.0 {
            continue;
        }

        // Determine where this child maps in the HMM
        if let Some(hmm_k) = cp9map.get_hmm_node(child) {
            if let Some(hmm_state) = cp9map.get_hmm_state_type(child) {
                match hmm_state {
                    HmmStateType::Match => {
                        // Match to Match: M_k -> M_{k+1}
                        if hmm_k > k && hmm_k <= hmm_m {
                            cp9.t[k][CTMM] += t_prob;
                        }
                    }
                    HmmStateType::Insert => {
                        // Match to Insert: M_k -> I_k
                        if hmm_k == k {
                            cp9.t[k][CTMI] += t_prob;
                        }
                    }
                    HmmStateType::Delete => {
                        // Match to Delete: M_k -> D_{k+1}
                        if hmm_k > k && hmm_k <= hmm_m {
                            cp9.t[k][CTMD] += t_prob;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Fill insert state transitions for node k
fn fill_insert_transitions(
    cm: &CM,
    cp9: &mut CP9,
    _psi: &[f64],
    k: usize,
    v: usize,
) -> Result<(), String> {
    let cnum = cm.cnum[v] as usize;

    if cnum == 0 {
        return Ok(());
    }

    // First child is typically self-loop for insert states
    cp9.t[k][CTII] = cm.t[v][0];

    // Remaining transitions go to match
    let mut t_im = 0.0_f32;
    for c in 1..cnum {
        t_im += cm.t[v][c];
    }
    cp9.t[k][CTIM] = t_im;

    // CTID is typically 0 (insert to delete not allowed in standard profile)
    cp9.t[k][CTID] = 0.0;

    Ok(())
}

/// Fill delete state transitions for node k
fn fill_delete_transitions(
    cm: &CM,
    cp9: &mut CP9,
    cp9map: &CP9Map,
    _psi: &[f64],
    k: usize,
    v: usize,
) -> Result<(), String> {
    let hmm_m = cp9.m as usize;
    let cnum = cm.cnum[v] as usize;
    let cfirst = cm.cfirst[v] as usize;

    for c in 0..cnum {
        let child = cfirst + c;
        if child >= cm.m as usize {
            continue;
        }

        let t_prob = cm.t[v][c];
        if t_prob <= 0.0 {
            continue;
        }

        if let Some(hmm_k) = cp9map.get_hmm_node(child) {
            if let Some(hmm_state) = cp9map.get_hmm_state_type(child) {
                match hmm_state {
                    HmmStateType::Match => {
                        if hmm_k > k && hmm_k <= hmm_m {
                            cp9.t[k][CTDM] += t_prob;
                        }
                    }
                    HmmStateType::Insert => {
                        // D->I not typically used
                        cp9.t[k][CTDI] += t_prob;
                    }
                    HmmStateType::Delete => {
                        if hmm_k > k && hmm_k <= hmm_m {
                            cp9.t[k][CTDD] += t_prob;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Fill transitions out of the final node M
fn fill_final_transitions(
    _cm: &CM,
    cp9: &mut CP9,
    _cp9map: &CP9Map,
    _psi: &[f64],
    _emap: &CMEmitMap,
) -> Result<(), String> {
    let hmm_m = cp9.m as usize;

    // Final node has special handling
    // M_M -> End transition
    cp9.end[hmm_m] = 1.0; // All paths must end

    // Ensure insert at M has proper transitions
    cp9.t[hmm_m][CTIM] = 1.0 - cp9.t[hmm_m][CTII];

    Ok(())
}

/// Normalize CP9 HMM probabilities
fn cplan9_normalize(cp9: &mut CP9) -> Result<(), String> {
    let hmm_m = cp9.m as usize;

    // Normalize match emissions
    for k in 1..=hmm_m {
        let sum: f32 = cp9.mat[k].iter().sum();
        if sum > 0.0 {
            for a in 0..ALPHABET_SIZE {
                cp9.mat[k][a] /= sum;
            }
        }
    }

    // Normalize insert emissions
    for k in 0..=hmm_m {
        let sum: f32 = cp9.ins[k].iter().sum();
        if sum > 0.0 {
            for a in 0..ALPHABET_SIZE {
                cp9.ins[k][a] /= sum;
            }
        }
    }

    // Normalize transitions
    for k in 0..=hmm_m {
        // Match transitions
        let sum_m = cp9.t[k][CTMM] + cp9.t[k][CTMI] + cp9.t[k][CTMD] + cp9.t[k][CTMEL];
        if sum_m > 0.0 {
            cp9.t[k][CTMM] /= sum_m;
            cp9.t[k][CTMI] /= sum_m;
            cp9.t[k][CTMD] /= sum_m;
            cp9.t[k][CTMEL] /= sum_m;
        }

        // Insert transitions
        let sum_i = cp9.t[k][CTIM] + cp9.t[k][CTII] + cp9.t[k][CTID];
        if sum_i > 0.0 {
            cp9.t[k][CTIM] /= sum_i;
            cp9.t[k][CTII] /= sum_i;
            cp9.t[k][CTID] /= sum_i;
        }

        // Delete transitions
        let sum_d = cp9.t[k][CTDM] + cp9.t[k][CTDI] + cp9.t[k][CTDD];
        if sum_d > 0.0 {
            cp9.t[k][CTDM] /= sum_d;
            cp9.t[k][CTDI] /= sum_d;
            cp9.t[k][CTDD] /= sum_d;
        }
    }

    // Set begin probabilities (uniform for now)
    // In full implementation, these would be computed from CM
    let begin_prob = 1.0 / hmm_m as f32;
    for k in 1..=hmm_m {
        cp9.begin[k] = begin_prob;
    }

    // End probabilities (typically 1/M for global)
    for k in 1..=hmm_m {
        if cp9.end[k] == 0.0 {
            cp9.end[k] = begin_prob;
        }
    }

    Ok(())
}

/// Configure CP9 for local alignment mode
pub fn cplan9_set_local(cp9: &mut CP9) {
    use crate::cp9::CP9_LOCAL;

    let hmm_m = cp9.m as usize;

    // Set uniform begin probabilities
    let begin_prob = 1.0 / hmm_m as f32;
    for k in 1..=hmm_m {
        cp9.begin[k] = begin_prob;
    }

    // Set uniform end probabilities
    for k in 1..=hmm_m {
        cp9.end[k] = begin_prob;
    }

    cp9.flags |= CP9_LOCAL;

    // Recompute scores
    cp9.logoddsify();
}

/// Configure CP9 for global alignment mode
pub fn cplan9_set_global(cp9: &mut CP9) {
    use crate::cp9::CP9_LOCAL;

    let hmm_m = cp9.m as usize;

    // Global: begin only at position 1
    for k in 1..=hmm_m {
        cp9.begin[k] = if k == 1 { 1.0 } else { 0.0 };
    }

    // Global: end only at position M
    for k in 1..=hmm_m {
        cp9.end[k] = if k == hmm_m { 1.0 } else { 0.0 };
    }

    cp9.flags &= !CP9_LOCAL;

    // Recompute scores
    cp9.logoddsify();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cplan9_normalize() {
        let mut cp9 = CP9::new(5);

        // Set some unnormalized values
        cp9.mat[1] = [0.5, 0.3, 0.2, 0.1];
        cp9.ins[0] = [1.0, 1.0, 1.0, 1.0];

        cp9.t[1][CTMM] = 0.8;
        cp9.t[1][CTMI] = 0.1;
        cp9.t[1][CTMD] = 0.1;

        cplan9_normalize(&mut cp9).unwrap();

        // Check normalization
        let sum_mat: f32 = cp9.mat[1].iter().sum();
        assert!((sum_mat - 1.0).abs() < 0.001);

        let sum_ins: f32 = cp9.ins[0].iter().sum();
        assert!((sum_ins - 1.0).abs() < 0.001);

        let sum_t = cp9.t[1][CTMM] + cp9.t[1][CTMI] + cp9.t[1][CTMD] + cp9.t[1][CTMEL];
        assert!((sum_t - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cplan9_local_global() {
        let mut cp9 = CP9::new(10);

        // Test local configuration
        cplan9_set_local(&mut cp9);
        assert!(cp9.is_local());
        assert!((cp9.begin[1] - cp9.begin[5]).abs() < 0.001);

        // Test global configuration
        cplan9_set_global(&mut cp9);
        assert!(!cp9.is_local());
        assert!((cp9.begin[1] - 1.0).abs() < 0.001);
        assert!((cp9.begin[5] - 0.0).abs() < 0.001);
    }
}
