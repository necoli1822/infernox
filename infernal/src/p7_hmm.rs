//! P7 HMM (HMMER3-style profile) for filtering
//!
//! P7 HMM is a HMMER3-style profile used for fast filtering in Infernal.
//! It is separate from CP9 - while CP9 is derived from CM structure,
//! P7 is an independent HMM profile optimized for rapid sequence scanning.

use crate::cp9::{CP9, CTMM, CTMI, CTMD, CTIM, CTII, CTDM, CTDD};
use std::io::BufRead;

/// P7 HMM E-value parameters
///
/// These parameters are used to compute E-values for P7 HMM hits.
/// When uncalibrated, all values are set to -99999.0.
#[derive(Debug, Clone, Default)]
pub struct P7EvParams {
    pub lmmu: f64,      // Local MSV mu
    pub lmlambda: f64,  // Local MSV lambda
    pub lvmu: f64,      // Local Viterbi mu
    pub lvlambda: f64,  // Local Viterbi lambda
    pub lftau: f64,     // Local Forward tau
    pub lflambda: f64,  // Local Forward lambda
    pub gfmu: f64,      // Glocal Forward mu
    pub gflambda: f64,  // Glocal Forward lambda
}

/// P7 HMM Filter Profile
///
/// A HMMER3-style profile HMM used for fast filtering before CM alignment.
/// Contains match/insert emission probabilities and transition probabilities.
#[derive(Debug, Clone)]
pub struct P7Profile {
    /// Model name
    pub name: String,

    /// Model length (number of nodes)
    pub m: i32,

    /// Configuration flags
    pub flags: u32,

    /// Match emissions [1..M][A,C,G,U]
    /// mat[k][0..3] = probabilities for A,C,G,U at match state k
    pub mat: Vec<[f32; 4]>,

    /// Insert emissions [0..M][A,C,G,U]
    /// ins[k][0..3] = probabilities for A,C,G,U at insert state k
    /// Typically uniform (0.25 for each base)
    pub ins: Vec<[f32; 4]>,

    /// Transitions [0..M][MM,MI,MD,IM,II,DM,DD]
    /// trans[k][0..6] = transition probabilities from state k
    /// Order: MM, MI, MD, IM, II, DM, DD
    pub trans: Vec<[f32; 7]>,

    /// E-value parameters for statistical significance
    pub evparam: P7EvParams,

    /// Mean model residue composition (COMPO line), as probabilities [A,C,G,U].
    /// = C `hmm->compo`/`om->compo`; used by the F3b composition-bias filter.
    pub compo: [f32; 4],
}

impl P7Profile {
    /// Create a new P7 profile with the given model length
    ///
    /// # Arguments
    /// * `m` - Model length (number of nodes)
    ///
    /// # Returns
    /// A new P7Profile with uniform emissions and zero transitions
    pub fn new(m: i32) -> Self {
        P7Profile {
            name: String::new(),
            m,
            flags: 0,
            // Allocate M+1 elements (index 0 unused for match, used for inserts)
            mat: vec![[0.25; 4]; (m + 1) as usize],
            ins: vec![[0.25; 4]; (m + 1) as usize],
            trans: vec![[0.0; 7]; (m + 1) as usize],
            evparam: P7EvParams::default(),
            compo: [0.25; 4],
        }
    }

    /// Create P7 profile from CP9 HMM
    ///
    /// This creates a simplified P7 profile using CP9's match emissions
    /// and transitions. Used for filtering before CM alignment.
    pub fn from_cp9(cp9: &CP9) -> Self {
        let m = cp9.m;
        let mut p7 = P7Profile::new(m);
        p7.name = format!("P7_{}", m); // Simple default name

        // Copy match emissions
        for k in 1..=(m as usize) {
            for x in 0..4 {
                p7.mat[k][x] = cp9.mat[k][x];
            }
        }

        // Copy insert emissions
        for k in 0..=(m as usize) {
            for x in 0..4 {
                p7.ins[k][x] = cp9.ins[k][x];
            }
        }

        // Set transitions (CP9 to P7 mapping)
        // P7 trans order: MM, MI, MD, IM, II, DM, DD
        for k in 0..(m as usize) {
            // From match state
            p7.trans[k][0] = cp9.t[k][CTMM]; // MM
            p7.trans[k][1] = cp9.t[k][CTMI]; // MI
            p7.trans[k][2] = cp9.t[k][CTMD]; // MD

            // From insert state
            p7.trans[k][3] = cp9.t[k][CTIM]; // IM
            p7.trans[k][4] = cp9.t[k][CTII]; // II

            // From delete state
            p7.trans[k][5] = cp9.t[k][CTDM]; // DM
            p7.trans[k][6] = cp9.t[k][CTDD]; // DD
        }

        p7
    }

    /// Parse P7 profile from HMMER3 format lines
    ///
    /// Parses the embedded HMMER3/f section from CM files.
    /// Returns None if parsing fails.
    pub fn parse_hmmer3<B: BufRead>(lines: &mut std::io::Lines<B>) -> Option<Self> {
        let mut name = String::new();
        let mut m: i32 = 0;
        let mut msv_mu = 0.0f64;
        let mut msv_lambda = 0.0f64;
        let mut vit_mu = 0.0f64;
        let mut vit_lambda = 0.0f64;
        let mut fwd_tau = 0.0f64;
        let mut fwd_lambda = 0.0f64;

        // Parse header section
        loop {
            let line = match lines.next() {
                Some(Ok(l)) => l,
                _ => return None,
            };
            let line = line.trim();

            if line.starts_with("HMM ") {
                // Start of HMM matrix - read column header
                break;
            }

            if line.starts_with("//") {
                return None; // End of profile without HMM data
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "NAME" => {
                    if parts.len() >= 2 {
                        name = parts[1].to_string();
                    }
                }
                "LENG" => {
                    if parts.len() >= 2 {
                        m = parts[1].parse().unwrap_or(0);
                    }
                }
                "STATS" => {
                    // STATS LOCAL MSV -8.9335 0.71867
                    if parts.len() >= 5 && parts[1] == "LOCAL" {
                        let mu: f64 = parts[3].parse().unwrap_or(0.0);
                        let lambda: f64 = parts[4].parse().unwrap_or(0.0);
                        match parts[2] {
                            "MSV" => {
                                msv_mu = mu;
                                msv_lambda = lambda;
                            }
                            "VITERBI" => {
                                vit_mu = mu;
                                vit_lambda = lambda;
                            }
                            "FORWARD" => {
                                fwd_tau = mu;
                                fwd_lambda = lambda;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        if m <= 0 {
            return None;
        }

        let mut p7 = P7Profile::new(m);
        p7.name = name;
        p7.evparam = P7EvParams {
            lmmu: msv_mu,
            lmlambda: msv_lambda,
            lvmu: vit_mu,
            lvlambda: vit_lambda,
            lftau: fwd_tau,
            lflambda: fwd_lambda,
            gfmu: 0.0,
            gflambda: 0.0,
        };

        // Skip transition header line (m->m m->i m->d ...)
        let _ = lines.next();

        // Parse COMPO line (mean model composition). Values are -ln(prob), like
        // match emissions; store as probabilities in p7.compo (= C hmm->compo).
        if let Some(Ok(line)) = lines.next() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 && parts[0] == "COMPO" {
                for i in 0..4 {
                    if let Ok(score) = parts[i + 1].parse::<f32>() {
                        p7.compo[i] = (-score).exp();
                    }
                }
            }
        }

        // Skip insert emission line after COMPO (node-0 inserts)
        let _ = lines.next();

        // Parse the node-0 transition line (B-state begin-node transitions).
        // C's p7_ProfileConfig/p7_hmm_CalculateOccupancy needs t[0][MM,MI,DM]
        // for the occupancy-weighted local entry distribution (tBM).
        // Order in file: m->m m->i m->d i->m i->i d->m d->d
        if let Some(Ok(line)) = lines.next() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 7 {
                for i in 0..7 {
                    if parts[i] == "*" {
                        p7.trans[0][i] = 0.0;
                    } else if let Ok(score) = parts[i].parse::<f32>() {
                        p7.trans[0][i] = (-score).exp();
                    }
                }
            }
        }

        // Parse node lines (1..M)
        // Each node has 3 lines:
        // Line 1: k  mat_A mat_C mat_G mat_U  col_idx consensus ...
        // Line 2: ins_A ins_C ins_G ins_U
        // Line 3: m->m m->i m->d i->m i->i d->m d->d
        for k in 1..=m as usize {
            // Line 1: Match emissions
            let mat_line = match lines.next() {
                Some(Ok(l)) => l,
                _ => break,
            };

            if mat_line.trim().starts_with("//") {
                break;
            }

            let parts: Vec<&str> = mat_line.split_whitespace().collect();
            if parts.len() >= 5 {
                // First value is node index, then 4 emission scores (log-odds)
                for i in 0..4 {
                    if let Ok(score) = parts[i + 1].parse::<f32>() {
                        // Convert log-odds score to probability
                        // HMMER3 stores -log(prob), so prob = exp(-score)
                        p7.mat[k][i] = (-score).exp();
                    }
                }
            }

            // Line 2: Insert emissions
            let ins_line = match lines.next() {
                Some(Ok(l)) => l,
                _ => break,
            };
            let parts: Vec<&str> = ins_line.split_whitespace().collect();
            if parts.len() >= 4 {
                for i in 0..4 {
                    if let Ok(score) = parts[i].parse::<f32>() {
                        p7.ins[k][i] = (-score).exp();
                    }
                }
            }

            // Line 3: Transitions
            let trans_line = match lines.next() {
                Some(Ok(l)) => l,
                _ => break,
            };
            let parts: Vec<&str> = trans_line.split_whitespace().collect();
            if parts.len() >= 7 {
                // Order: m->m m->i m->d i->m i->i d->m d->d
                for i in 0..7 {
                    if parts[i] == "*" {
                        p7.trans[k][i] = 0.0;
                    } else if let Ok(score) = parts[i].parse::<f32>() {
                        p7.trans[k][i] = (-score).exp();
                    }
                }
            }
        }

        Some(p7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p7_profile_creation() {
        let p7 = P7Profile::new(71);
        assert_eq!(p7.m, 71);
        assert_eq!(p7.mat.len(), 72); // M+1
        assert_eq!(p7.ins.len(), 72); // M+1
        assert_eq!(p7.trans.len(), 72); // M+1
    }

    #[test]
    fn test_p7_default_emissions() {
        let p7 = P7Profile::new(10);
        // Check uniform emissions
        for k in 0..=10 {
            for base in 0..4 {
                assert!((p7.mat[k][base] - 0.25).abs() < 1e-6);
                assert!((p7.ins[k][base] - 0.25).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_p7_evparam_default() {
        let evparam = P7EvParams::default();
        assert_eq!(evparam.lmmu, 0.0);
        assert_eq!(evparam.lmlambda, 0.0);
        assert_eq!(evparam.lvmu, 0.0);
        assert_eq!(evparam.lvlambda, 0.0);
        assert_eq!(evparam.lftau, 0.0);
        assert_eq!(evparam.lflambda, 0.0);
        assert_eq!(evparam.gfmu, 0.0);
        assert_eq!(evparam.gflambda, 0.0);
    }
}
