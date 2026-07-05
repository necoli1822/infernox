//! P7 Glocal Forward/Backward and Domain Decoding
//!
//! Implements the full P7 HMM pipeline matching C Infernal's cm_pipeline.c:
//! 1. Glocal Forward - computes forward probabilities with special states (N,C,J,B,E)
//! 2. Glocal Backward - computes backward probabilities
//! 3. Domain Decoding - computes btot, etot, mocc for envelope definition
//! 4. Envelope Definition - defines domain boundaries using rt1, rt2 thresholds
//!
//! Reference: HMMER3 generic_fwdback.c, generic_decoding.c

use crate::p7_hmm::P7Profile;

/// Special state indices for P7 matrices
const P7G_E: usize = 0;  // E (end) state
const P7G_N: usize = 1;  // N state (before model)
const P7G_J: usize = 2;  // J state (between domains)
const P7G_B: usize = 3;  // B (begin) state
const P7G_C: usize = 4;  // C state (after model)
const P7G_NXCELLS: usize = 5;  // Number of special cells per row

/// Special state transition indices
const P7P_LOOP: usize = 0;  // N->N, C->C, J->J self-loop
const P7P_MOVE: usize = 1;  // N->B, C->T, J->B move

/// P7 Generic DP Matrix for Glocal algorithms
///
/// Stores full Forward or Backward matrices including special states
#[derive(Debug, Clone)]
pub struct P7GMatrix {
    /// Model length
    pub m: usize,
    /// Sequence length
    pub l: usize,
    /// Match state scores [i][k] for i=0..L, k=0..M
    pub mmx: Vec<Vec<f32>>,
    /// Insert state scores [i][k]
    pub imx: Vec<Vec<f32>>,
    /// Delete state scores [i][k]
    pub dmx: Vec<Vec<f32>>,
    /// Special state scores [i*NXCELLS + state]
    pub xmx: Vec<f32>,
}

impl P7GMatrix {
    /// Create new DP matrix for sequence length L and model length M
    pub fn new(m: usize, l: usize) -> Self {
        P7GMatrix {
            m,
            l,
            mmx: vec![vec![f32::NEG_INFINITY; m + 1]; l + 1],
            imx: vec![vec![f32::NEG_INFINITY; m + 1]; l + 1],
            dmx: vec![vec![f32::NEG_INFINITY; m + 1]; l + 1],
            xmx: vec![f32::NEG_INFINITY; (l + 1) * P7G_NXCELLS],
        }
    }
}

/// P7 Glocal Profile - profile with special state transitions
#[derive(Debug, Clone)]
pub struct P7GlocalProfile {
    /// Core profile
    pub m: usize,
    /// Match emissions [1..M][A,C,G,U] in log space
    pub mat_lod: Vec<[f32; 4]>,
    /// Insert emissions in log space
    pub ins_lod: Vec<[f32; 4]>,
    /// Core transitions [k][MM,MI,MD,IM,II,DM,DD] in log space
    pub tsc: Vec<[f32; 7]>,
    /// Special state transitions [state][LOOP,MOVE] in log space
    pub xsc: [[f32; 2]; 5],
    /// Begin transitions [1..M] in log space (glocal: uniform 1/M)
    pub bsc: Vec<f32>,
    /// End transitions [1..M] in log space
    pub esc: Vec<f32>,
    /// Background frequencies
    pub bg: [f32; 4],
}

impl P7GlocalProfile {
    /// Create glocal profile from P7Profile
    ///
    /// Configures profile for glocal alignment mode:
    /// - Uniform begin probabilities: 1/M
    /// - End only from M_M state
    /// - N->N, C->C, J->J loops configured for expected sequence length
    pub fn from_p7profile(p7: &P7Profile, target_l: usize) -> Self {
        let m = p7.m as usize;

        // Initialize in log space
        let mut mat_lod = vec![[0.0f32; 4]; m + 1];
        let mut ins_lod = vec![[0.0f32; 4]; m + 1];
        let mut tsc = vec![[f32::NEG_INFINITY; 7]; m + 1];

        // Background frequencies (uniform for RNA)
        let bg = [0.25f32; 4];

        // Convert emissions to log-odds
        for k in 1..=m {
            for x in 0..4 {
                if p7.mat[k][x] > 0.0 && bg[x] > 0.0 {
                    mat_lod[k][x] = (p7.mat[k][x] / bg[x]).ln();
                } else {
                    mat_lod[k][x] = f32::NEG_INFINITY;
                }
                if p7.ins[k][x] > 0.0 && bg[x] > 0.0 {
                    ins_lod[k][x] = (p7.ins[k][x] / bg[x]).ln();
                } else {
                    ins_lod[k][x] = f32::NEG_INFINITY;
                }
            }
        }

        // Convert transitions to log space
        for k in 0..=m {
            for t in 0..7 {
                if p7.trans[k][t] > 0.0 {
                    tsc[k][t] = p7.trans[k][t].ln();
                } else {
                    tsc[k][t] = f32::NEG_INFINITY;
                }
            }
        }

        // Configure special state transitions for glocal unihit mode
        // N->N, J->J, C->C loops: p = L/(L+1) where L = target_l
        // N->B, J->B, C->T moves: p = 1/(L+1)
        let loop_prob = (target_l as f32) / (target_l as f32 + 1.0);
        let move_prob = 1.0 / (target_l as f32 + 1.0);

        let mut xsc = [[f32::NEG_INFINITY; 2]; 5];
        xsc[P7G_N][P7P_LOOP] = loop_prob.ln();
        xsc[P7G_N][P7P_MOVE] = move_prob.ln();  // N->B
        xsc[P7G_J][P7P_LOOP] = f32::NEG_INFINITY;  // No J->J in unihit (glocal)
        xsc[P7G_J][P7P_MOVE] = f32::NEG_INFINITY;  // No J->B in unihit
        xsc[P7G_C][P7P_LOOP] = loop_prob.ln();
        xsc[P7G_C][P7P_MOVE] = move_prob.ln();  // C->T (terminal)

        // Glocal begin: uniform into any M_k with prob 1/M
        let bsc: Vec<f32> = (0..=m).map(|k| {
            if k >= 1 { (1.0 / m as f32).ln() } else { f32::NEG_INFINITY }
        }).collect();

        // Glocal end: only from M_M (last match state)
        let esc: Vec<f32> = (0..=m).map(|k| {
            if k == m { 0.0 } else { f32::NEG_INFINITY }
        }).collect();

        P7GlocalProfile {
            m,
            mat_lod,
            ins_lod,
            tsc,
            xsc,
            bsc,
            esc,
            bg,
        }
    }

    /// Reconfigure for new target length
    pub fn reconfigure_length(&mut self, target_l: usize) {
        let loop_prob = (target_l as f32) / (target_l as f32 + 1.0);
        let move_prob = 1.0 / (target_l as f32 + 1.0);

        self.xsc[P7G_N][P7P_LOOP] = loop_prob.ln();
        self.xsc[P7G_N][P7P_MOVE] = move_prob.ln();
        self.xsc[P7G_C][P7P_LOOP] = loop_prob.ln();
        self.xsc[P7G_C][P7P_MOVE] = move_prob.ln();
    }
}

// Transition indices matching p7_dp.rs
const MM: usize = 0;
const MI: usize = 1;
const MD: usize = 2;
const IM: usize = 3;
const II: usize = 4;
const DM: usize = 5;
const DD: usize = 6;

/// Log-sum-exp in natural log space
fn logsumexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY { return b; }
    if b == f32::NEG_INFINITY { return a; }
    if a > b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

/// Glocal Forward algorithm
///
/// Fills Forward matrix for full sequence using glocal (global/local hybrid) mode.
/// Returns the overall Forward score in nats.
///
/// Reference: HMMER3 generic_fwdback.c::p7_GForward()
pub fn p7_glocal_forward(
    gm: &P7GlocalProfile,
    dsq: &[u8],  // 1-indexed digital sequence
    l: usize,     // sequence length
    fwd: &mut P7GMatrix,
) -> f32 {
    let m = gm.m;

    // Initialize row 0
    fwd.xmx[P7G_N] = 0.0;  // Start in N state
    fwd.xmx[P7G_B] = gm.xsc[P7G_N][P7P_MOVE];  // N->B
    fwd.xmx[P7G_E] = f32::NEG_INFINITY;
    fwd.xmx[P7G_C] = f32::NEG_INFINITY;
    fwd.xmx[P7G_J] = f32::NEG_INFINITY;

    // Main recursion
    for i in 1..=l {
        let xbase = i * P7G_NXCELLS;
        let xprev = (i - 1) * P7G_NXCELLS;

        let res = dsq[i] as usize;
        if res >= 4 {
            // Unknown residue, skip
            fwd.xmx[xbase + P7G_E] = f32::NEG_INFINITY;
            fwd.xmx[xbase + P7G_N] = fwd.xmx[xprev + P7G_N] + gm.xsc[P7G_N][P7P_LOOP];
            fwd.xmx[xbase + P7G_J] = f32::NEG_INFINITY;
            fwd.xmx[xbase + P7G_B] = fwd.xmx[xbase + P7G_N] + gm.xsc[P7G_N][P7P_MOVE];
            fwd.xmx[xbase + P7G_C] = fwd.xmx[xprev + P7G_C] + gm.xsc[P7G_C][P7P_LOOP];
            continue;
        }

        // Initialize E, C, J for this row
        let mut e_sum = f32::NEG_INFINITY;

        // Delete state k=1 (no D_0 to come from)
        fwd.dmx[i][1] = f32::NEG_INFINITY;

        // Match and Insert states
        for k in 1..=m {
            // Match state M_k
            let emit_m = gm.mat_lod[k][res];

            // M_k can come from: M_{k-1}, I_{k-1}, D_{k-1}, or B
            let from_m = if k > 1 && fwd.mmx[i-1][k-1] > f32::NEG_INFINITY {
                fwd.mmx[i-1][k-1] + gm.tsc[k-1][MM]
            } else {
                f32::NEG_INFINITY
            };
            let from_i = if k > 1 && fwd.imx[i-1][k-1] > f32::NEG_INFINITY {
                fwd.imx[i-1][k-1] + gm.tsc[k-1][IM]
            } else {
                f32::NEG_INFINITY
            };
            let from_d = if k > 1 && fwd.dmx[i-1][k-1] > f32::NEG_INFINITY {
                fwd.dmx[i-1][k-1] + gm.tsc[k-1][DM]
            } else {
                f32::NEG_INFINITY
            };
            let from_b = fwd.xmx[xprev + P7G_B] + gm.bsc[k];

            let sum = logsumexp(logsumexp(from_m, from_i), logsumexp(from_d, from_b));
            fwd.mmx[i][k] = emit_m + sum;

            // Insert state I_k
            let emit_i = gm.ins_lod[k][res];
            let from_m_to_i = if fwd.mmx[i-1][k] > f32::NEG_INFINITY {
                fwd.mmx[i-1][k] + gm.tsc[k][MI]
            } else {
                f32::NEG_INFINITY
            };
            let from_i_to_i = if fwd.imx[i-1][k] > f32::NEG_INFINITY {
                fwd.imx[i-1][k] + gm.tsc[k][II]
            } else {
                f32::NEG_INFINITY
            };
            fwd.imx[i][k] = emit_i + logsumexp(from_m_to_i, from_i_to_i);

            // Delete state D_k (no emission)
            if k > 1 {
                let from_m_to_d = if fwd.mmx[i][k-1] > f32::NEG_INFINITY {
                    fwd.mmx[i][k-1] + gm.tsc[k-1][MD]
                } else {
                    f32::NEG_INFINITY
                };
                let from_d_to_d = if fwd.dmx[i][k-1] > f32::NEG_INFINITY {
                    fwd.dmx[i][k-1] + gm.tsc[k-1][DD]
                } else {
                    f32::NEG_INFINITY
                };
                fwd.dmx[i][k] = logsumexp(from_m_to_d, from_d_to_d);
            }

            // Accumulate to E state (glocal: only from M_M and D_M)
            if fwd.mmx[i][k] > f32::NEG_INFINITY {
                e_sum = logsumexp(e_sum, fwd.mmx[i][k] + gm.esc[k]);
            }
            if fwd.dmx[i][k] > f32::NEG_INFINITY {
                e_sum = logsumexp(e_sum, fwd.dmx[i][k] + gm.esc[k]);
            }
        }

        // Special states
        fwd.xmx[xbase + P7G_E] = e_sum;

        // J state (in glocal unihit: J is unreachable)
        fwd.xmx[xbase + P7G_J] = f32::NEG_INFINITY;

        // C state: E->C or C->C
        let c_from_e = if fwd.xmx[xbase + P7G_E] > f32::NEG_INFINITY {
            fwd.xmx[xbase + P7G_E]  // E->C (no transition cost in glocal)
        } else {
            f32::NEG_INFINITY
        };
        let c_from_c = if fwd.xmx[xprev + P7G_C] > f32::NEG_INFINITY {
            fwd.xmx[xprev + P7G_C] + gm.xsc[P7G_C][P7P_LOOP]
        } else {
            f32::NEG_INFINITY
        };
        fwd.xmx[xbase + P7G_C] = logsumexp(c_from_e, c_from_c);

        // N state: N->N
        fwd.xmx[xbase + P7G_N] = fwd.xmx[xprev + P7G_N] + gm.xsc[P7G_N][P7P_LOOP];

        // B state: N->B or J->B
        fwd.xmx[xbase + P7G_B] = fwd.xmx[xbase + P7G_N] + gm.xsc[P7G_N][P7P_MOVE];
    }

    // Final score: C->T (terminal)
    let xfinal = l * P7G_NXCELLS;
    fwd.xmx[xfinal + P7G_C] + gm.xsc[P7G_C][P7P_MOVE]
}

/// Glocal Backward algorithm
///
/// Fills Backward matrix. Must be called after Forward.
/// Reference: HMMER3 generic_fwdback.c::p7_GBackward()
pub fn p7_glocal_backward(
    gm: &P7GlocalProfile,
    dsq: &[u8],
    l: usize,
    bck: &mut P7GMatrix,
) -> f32 {
    let m = gm.m;

    // Initialize row L
    let xbase = l * P7G_NXCELLS;
    bck.xmx[xbase + P7G_C] = gm.xsc[P7G_C][P7P_MOVE];  // C->T
    bck.xmx[xbase + P7G_J] = f32::NEG_INFINITY;
    bck.xmx[xbase + P7G_N] = f32::NEG_INFINITY;
    bck.xmx[xbase + P7G_B] = f32::NEG_INFINITY;
    bck.xmx[xbase + P7G_E] = bck.xmx[xbase + P7G_C];  // E->C (glocal)

    // Initialize M, I, D for row L
    for k in 1..=m {
        // M_k -> E
        bck.mmx[l][k] = bck.xmx[xbase + P7G_E] + gm.esc[k];
        // D_k -> E
        bck.dmx[l][k] = bck.xmx[xbase + P7G_E] + gm.esc[k];
        // I_k has no outgoing to E in glocal
        bck.imx[l][k] = f32::NEG_INFINITY;
    }

    // Main backward recursion from L-1 down to 1
    for i in (1..l).rev() {
        let xbase = i * P7G_NXCELLS;
        let xnext = (i + 1) * P7G_NXCELLS;

        let res_next = dsq[i + 1] as usize;

        // Special states
        bck.xmx[xbase + P7G_C] = bck.xmx[xnext + P7G_C] + gm.xsc[P7G_C][P7P_LOOP];
        bck.xmx[xbase + P7G_J] = f32::NEG_INFINITY;
        bck.xmx[xbase + P7G_N] = bck.xmx[xnext + P7G_N] + gm.xsc[P7G_N][P7P_LOOP];

        // E state: E->C
        bck.xmx[xbase + P7G_E] = bck.xmx[xbase + P7G_C];

        // B state: sum over all B->M_k entries (weighted by M_k backward)
        let mut b_sum = f32::NEG_INFINITY;
        for k in 1..=m {
            if res_next < 4 {
                let contrib = gm.bsc[k] + gm.mat_lod[k][res_next] + bck.mmx[i+1][k];
                b_sum = logsumexp(b_sum, contrib);
            }
        }
        bck.xmx[xbase + P7G_B] = b_sum;

        // Core states
        for k in (1..=m).rev() {
            // Match state M_k
            let mut m_sum = f32::NEG_INFINITY;

            // M_k -> M_{k+1}
            if k < m && res_next < 4 {
                let contrib = gm.tsc[k][MM] + gm.mat_lod[k+1][res_next] + bck.mmx[i+1][k+1];
                m_sum = logsumexp(m_sum, contrib);
            }
            // M_k -> I_k
            if res_next < 4 {
                let contrib = gm.tsc[k][MI] + gm.ins_lod[k][res_next] + bck.imx[i+1][k];
                m_sum = logsumexp(m_sum, contrib);
            }
            // M_k -> D_{k+1}
            if k < m {
                let contrib = gm.tsc[k][MD] + bck.dmx[i][k+1];
                m_sum = logsumexp(m_sum, contrib);
            }
            // M_k -> E (glocal)
            m_sum = logsumexp(m_sum, gm.esc[k] + bck.xmx[xbase + P7G_E]);

            bck.mmx[i][k] = m_sum;

            // Insert state I_k
            let mut i_sum = f32::NEG_INFINITY;
            // I_k -> M_{k+1}
            if k < m && res_next < 4 {
                let contrib = gm.tsc[k][IM] + gm.mat_lod[k+1][res_next] + bck.mmx[i+1][k+1];
                i_sum = logsumexp(i_sum, contrib);
            }
            // I_k -> I_k
            if res_next < 4 {
                let contrib = gm.tsc[k][II] + gm.ins_lod[k][res_next] + bck.imx[i+1][k];
                i_sum = logsumexp(i_sum, contrib);
            }
            bck.imx[i][k] = i_sum;

            // Delete state D_k
            let mut d_sum = f32::NEG_INFINITY;
            // D_k -> M_{k+1}
            if k < m && res_next < 4 {
                let contrib = gm.tsc[k][DM] + gm.mat_lod[k+1][res_next] + bck.mmx[i+1][k+1];
                d_sum = logsumexp(d_sum, contrib);
            }
            // D_k -> D_{k+1}
            if k < m {
                let contrib = gm.tsc[k][DD] + bck.dmx[i][k+1];
                d_sum = logsumexp(d_sum, contrib);
            }
            // D_k -> E (glocal)
            d_sum = logsumexp(d_sum, gm.esc[k] + bck.xmx[xbase + P7G_E]);

            bck.dmx[i][k] = d_sum;
        }
    }

    // Row 0: N state
    let res1 = dsq[1] as usize;
    let mut b_sum = f32::NEG_INFINITY;
    for k in 1..=m {
        if res1 < 4 {
            let contrib = gm.bsc[k] + gm.mat_lod[k][res1] + bck.mmx[1][k];
            b_sum = logsumexp(b_sum, contrib);
        }
    }
    bck.xmx[P7G_B] = b_sum;
    bck.xmx[P7G_N] = logsumexp(
        bck.xmx[P7G_B] + gm.xsc[P7G_N][P7P_MOVE],
        bck.xmx[P7G_NXCELLS + P7G_N] + gm.xsc[P7G_N][P7P_LOOP]
    );

    bck.xmx[P7G_N]  // Return backward score (should match forward)
}

/// Domain definition result
#[derive(Debug, Clone)]
pub struct DomainDef {
    /// Sequence length
    pub l: usize,
    /// Cumulative begin probability [0..L]
    /// btot[i] = expected number of domains started at or before position i
    pub btot: Vec<f32>,
    /// Cumulative end probability [0..L]
    /// etot[i] = expected number of domains ended at or before position i
    pub etot: Vec<f32>,
    /// Match occupancy probability [0..L]
    /// mocc[i] = probability that position i is in a domain (emitted by M/I/D)
    pub mocc: Vec<f32>,
    /// Expected number of domains
    pub nexpected: f32,
    /// Number of regions defined
    pub nregions: usize,
    /// Number of envelopes defined
    pub nenvelopes: usize,
    /// Threshold for triggering region start
    pub rt1: f32,
    /// Threshold for region boundaries
    pub rt2: f32,
    /// Threshold for calling multi-domain region
    pub rt3: f32,
}

impl DomainDef {
    pub fn new(l: usize) -> Self {
        DomainDef {
            l,
            btot: vec![0.0; l + 1],
            etot: vec![0.0; l + 1],
            mocc: vec![0.0; l + 1],
            nexpected: 0.0,
            nregions: 0,
            nenvelopes: 0,
            // Default thresholds from HMMER3
            rt1: 0.25,  // trigger threshold
            rt2: 0.10,  // boundary threshold
            rt3: 0.50,  // multi-domain threshold
        }
    }
}

/// Domain Decoding - compute btot, etot, mocc
///
/// Given filled Forward and Backward matrices, compute posterior probabilities
/// for domain boundaries.
///
/// Reference: HMMER3 generic_decoding.c::p7_GDomainDecoding()
pub fn p7_domain_decoding(
    gm: &P7GlocalProfile,
    fwd: &P7GMatrix,
    bck: &P7GMatrix,
    ddef: &mut DomainDef,
) {
    let l = fwd.l;

    // Overall log probability
    let overall_logp = fwd.xmx[l * P7G_NXCELLS + P7G_C] + gm.xsc[P7G_C][P7P_MOVE];

    ddef.btot[0] = 0.0;
    ddef.etot[0] = 0.0;
    ddef.mocc[0] = 0.0;

    for i in 1..=l {
        let xbase = i * P7G_NXCELLS;
        let xprev = (i - 1) * P7G_NXCELLS;

        // btot[i] = btot[i-1] + P(B state used at i-1)
        // P(B at i-1) = fwd(B,i-1) * bck(B,i-1) / P(x)
        let b_post = (fwd.xmx[xprev + P7G_B] + bck.xmx[xprev + P7G_B] - overall_logp).exp();
        ddef.btot[i] = ddef.btot[i - 1] + b_post;

        // etot[i] = etot[i-1] + P(E state used at i)
        let e_post = (fwd.xmx[xbase + P7G_E] + bck.xmx[xbase + P7G_E] - overall_logp).exp();
        ddef.etot[i] = ddef.etot[i - 1] + e_post;

        // mocc[i] = 1 - P(residue i emitted by N, J, or C)
        let n_post = (fwd.xmx[xprev + P7G_N] + bck.xmx[xbase + P7G_N] + gm.xsc[P7G_N][P7P_LOOP] - overall_logp).exp();
        let j_post = (fwd.xmx[xprev + P7G_J] + bck.xmx[xbase + P7G_J] + gm.xsc[P7G_J][P7P_LOOP] - overall_logp).exp();
        let c_post = (fwd.xmx[xprev + P7G_C] + bck.xmx[xbase + P7G_C] + gm.xsc[P7G_C][P7P_LOOP] - overall_logp).exp();

        let njcp = n_post + j_post + c_post;
        ddef.mocc[i] = (1.0 - njcp).max(0.0).min(1.0);
    }

    ddef.nexpected = ddef.btot[l];
    ddef.l = l;
}

/// Envelope - a candidate domain region
#[derive(Debug, Clone)]
pub struct Envelope {
    /// Start position (1-indexed)
    pub start: i32,
    /// End position (1-indexed)
    pub end: i32,
    /// Bit score (from HMM envelope definition)
    pub score: f32,
}

/// Define envelopes from domain decoding results
///
/// Uses rt1, rt2 thresholds to define domain boundaries.
/// Returns list of envelopes (start, end) positions.
///
/// Reference: HMMER3 p7_domaindef.c::p7_domaindef_ByPosteriorHeuristics()
pub fn define_envelopes(ddef: &mut DomainDef) -> Vec<Envelope> {
    let mut envelopes = Vec::new();
    let l = ddef.l;

    let mut i = 0i32;  // Region start candidate
    let mut triggered = false;

    for j in 1..=l {
        if !triggered {
            // Look for region start
            let b_here = ddef.btot[j] - ddef.btot[j - 1];
            if ddef.mocc[j] - b_here < ddef.rt2 {
                i = j as i32;
            } else if i == 0 {
                i = j as i32;
            }
            if ddef.mocc[j] >= ddef.rt1 {
                triggered = true;
            }
        } else {
            // Look for region end
            let e_here = ddef.etot[j] - ddef.etot[j - 1];
            if ddef.mocc[j] - e_here < ddef.rt2 {
                // Found region i..j
                if i > 0 {
                    ddef.nregions += 1;
                    ddef.nenvelopes += 1;

                    envelopes.push(Envelope {
                        start: i,
                        end: j as i32,
                        score: 0.0,  // Score will be computed later by CYK/Inside
                    });
                }
                i = 0;
                triggered = false;
            }
        }
    }

    // Handle unterminated region at end of sequence
    if triggered && i > 0 {
        ddef.nregions += 1;
        ddef.nenvelopes += 1;
        envelopes.push(Envelope {
            start: i,
            end: l as i32,
            score: 0.0,
        });
    }

    envelopes
}

/// Full pipeline: Forward → Backward → Domain Decoding → Envelope Definition
///
/// This is the main entry point matching C Infernal's pli_p7_env_def()
pub fn p7_envelope_pipeline(
    p7: &P7Profile,
    dsq: &[u8],
    l: usize,
) -> Vec<Envelope> {
    // Create glocal profile
    let gm = P7GlocalProfile::from_p7profile(p7, l);

    // Allocate matrices
    let mut fwd = P7GMatrix::new(gm.m, l);
    let mut bck = P7GMatrix::new(gm.m, l);

    // Run Forward
    let fwd_score = p7_glocal_forward(&gm, dsq, l, &mut fwd);

    if fwd_score == f32::NEG_INFINITY {
        return Vec::new();
    }

    // Run Backward
    let _bck_score = p7_glocal_backward(&gm, dsq, l, &mut bck);

    // Domain decoding
    let mut ddef = DomainDef::new(l);
    p7_domain_decoding(&gm, &fwd, &bck, &mut ddef);

    // Define envelopes
    define_envelopes(&mut ddef)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsumexp() {
        let a = 1.0f32;
        let b = 2.0f32;
        let result = logsumexp(a, b);
        let expected = (a.exp() + b.exp()).ln();
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_glocal_forward_basic() {
        let mut p7 = P7Profile::new(3);
        // Simple uniform emissions
        for k in 1..=3 {
            p7.mat[k] = [0.25; 4];
            p7.ins[k] = [0.25; 4];
        }
        // Simple transitions
        p7.trans[0][MM] = 1.0;
        p7.trans[1][MM] = 0.9;
        p7.trans[1][MI] = 0.05;
        p7.trans[1][MD] = 0.05;
        p7.trans[2][MM] = 0.9;
        p7.trans[2][MI] = 0.05;
        p7.trans[2][MD] = 0.05;

        let gm = P7GlocalProfile::from_p7profile(&p7, 5);
        let dsq = vec![255, 0, 1, 2, 3, 0, 255];  // ACGUA with sentinels
        let l = 5;

        let mut fwd = P7GMatrix::new(gm.m, l);
        let score = p7_glocal_forward(&gm, &dsq, l, &mut fwd);

        // Just check that we get a finite score
        assert!(score.is_finite());
    }
}
