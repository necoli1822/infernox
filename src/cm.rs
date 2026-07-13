//! Covariance Model (CM) data structure - 1:1 port from cm.h
//!
//! The CM struct represents a stochastic context-free grammar (SCFG) trained
//! on an RNA structural alignment. CMs can model nested RNA secondary structures.

use crate::constants::*;
use crate::evalue::ExpParams;

/// Maximum number of transitions from any state
pub const CM_MAXCONNECT: usize = 6;

/// Alphabet size for RNA (A, C, G, U)
pub const ALPHABET_SIZE: usize = 4;

/// Extended alphabet size (Kp = 18: 4 canonical + degeneracy codes + gap + missing)
/// This matches Easel's abc->Kp for RNA alphabet
pub const ALPHABET_SIZE_P: usize = 18;

/// IUPAC RNA degeneracy table for Easel sym order "ACGU-RYMKSWHBVDN*~".
/// `RNA_DEGEN[x][i]` = true if degenerate code `x` includes canonical residue
/// `i` (0=A,1=C,2=G,3=U). Mirrors Easel's `abc->degen`. Gap(4)/nonresidue(16)/
/// missing(17) are all-false. (Same table as p7_generic; kept here as the shared
/// source of truth for marginalizing degenerate emissions.)
pub const RNA_DEGEN: [[bool; ALPHABET_SIZE]; ALPHABET_SIZE_P] = [
    [true, false, false, false],  // 0 A
    [false, true, false, false],  // 1 C
    [false, false, true, false],  // 2 G
    [false, false, false, true],  // 3 U
    [false, false, false, false], // 4 - gap
    [true, false, true, false],   // 5 R = A,G
    [false, true, false, true],   // 6 Y = C,U
    [true, true, false, false],   // 7 M = A,C
    [false, false, true, true],   // 8 K = G,U
    [false, true, true, false],   // 9 S = C,G
    [true, false, false, true],   // 10 W = A,U
    [true, true, false, true],    // 11 H = A,C,U
    [false, true, true, true],    // 12 B = C,G,U
    [true, true, true, false],    // 13 V = A,C,G
    [true, false, true, true],    // 14 D = A,G,U
    [true, true, true, true],     // 15 N = A,C,G,U
    [false, false, false, false], // 16 * nonresidue
    [false, false, false, false], // 17 ~ missing
];

/// `abc->ndegen[x]`: number of canonical residues a code maps to (0 for
/// gap/nonresidue/missing). Derived from `RNA_DEGEN`.
pub const RNA_NDEGEN: [usize; ALPHABET_SIZE_P] = {
    let mut n = [0usize; ALPHABET_SIZE_P];
    let mut x = 0;
    while x < ALPHABET_SIZE_P {
        let mut c = 0;
        let mut i = 0;
        while i < ALPHABET_SIZE {
            if RNA_DEGEN[x][i] {
                c += 1;
            }
            i += 1;
        }
        n[x] = c;
        x += 1;
    }
    n
};

/// Easel `esl_abc_XIsResidue(abc, x)`: true for canonical (x<K) or a degenerate
/// residue code (K < x < Kp-2). Gap(K), nonresidue(Kp-2), missing(Kp-1) are not.
#[inline]
pub fn abc_is_residue(x: usize) -> bool {
    x < ALPHABET_SIZE || (x > ALPHABET_SIZE && x < ALPHABET_SIZE_P - 2)
}

/// Easel `esl_abc_FCount(abc, ct, x, 1.0)` restricted to degenerate residue
/// codes: the fraction of canonical residue `l` (0..K) that code `x` maps to
/// (`1/ndegen[x]` for members, else 0). Non-residue codes yield 0.
#[inline]
pub fn abc_fcount_frac(x: usize, l: usize) -> f32 {
    if RNA_DEGEN[x][l] {
        1.0 / RNA_NDEGEN[x] as f32
    } else {
        0.0
    }
}

/// Easel `esl_abc_FAvgScore(abc, x, sc)`: UNIFORM average of the canonical
/// singlet scores over degenerate code `x`. Non-residue codes return 0.
#[inline]
pub fn abc_favg_score(x: usize, sc: &[f32]) -> f32 {
    if !abc_is_residue(x) {
        return 0.0;
    }
    let mut result = 0.0f32;
    for i in 0..ALPHABET_SIZE {
        if RNA_DEGEN[x][i] {
            result += sc[i];
        }
    }
    result / RNA_NDEGEN[x] as f32
}

/// Easel `esl_abc_IAvgScore(abc, x, sc)`: UNIFORM average of the canonical
/// integer singlet scores over degenerate code `x`, rounded half-away-from-zero.
/// Non-residue codes return 0.
#[inline]
pub fn abc_iavg_score(x: usize, sc: &[i32]) -> i32 {
    if !abc_is_residue(x) {
        return 0;
    }
    let mut result = 0.0f32;
    for i in 0..ALPHABET_SIZE {
        if RNA_DEGEN[x][i] {
            result += sc[i] as f32;
        }
    }
    result /= RNA_NDEGEN[x] as f32;
    if result < 0.0 {
        (result - 0.5) as i32
    } else {
        (result + 0.5) as i32
    }
}

/// Pair emission alphabet size (16 = 4*4)
pub const PAIR_EMIT_SIZE: usize = 16;

/// Extended pair emission alphabet size (324 = 18*18 for oesc)
pub const PAIR_EMIT_SIZE_P: usize = 324;

// =============================================================================
// CM Flags (from infernal.h)
// =============================================================================

/// Local mode is ON
pub const CM_LOCAL_BEGIN: u32 = 1 << 0;
/// Local mode ends ON
pub const CM_LOCAL_END: u32 = 1 << 1;
/// CM has exptl params
pub const CM_EMIT_COUNTS: u32 = 1 << 2;
/// CM was configured
pub const CM_CONFIG: u32 = 1 << 3;
/// QDB info valid
pub const CM_QDB: u32 = 1 << 4;
/// Model is calibrated
pub const CM_CALIBRATED: u32 = 1 << 5;
/// GA cutoff present
pub const CM_GA: u32 = 1 << 6;
/// TC cutoff present
pub const CM_TC: u32 = 1 << 7;
/// NC cutoff present
pub const CM_NC: u32 = 1 << 8;
/// W is valid
pub const CM_W: u32 = 1 << 9;
/// EL self-score is set
pub const CM_ELSELF: u32 = 1 << 10;
/// Hit scores are log2
pub const CM_LOGSUM: u32 = 1 << 11;
/// Parsed from 1.0 file
pub const CM_IS_SUB: u32 = 1 << 12;
/// Is a sub CM
pub const CM_IS_RSEARCH: u32 = 1 << 13;
/// Has RSEARCH scores
pub const CM_RSEARCHEMIT: u32 = 1 << 14;
/// RSEARCH single stranded
pub const CM_RSEARCHNULL: u32 = 1 << 15;
/// Has CP9 HMM
pub const CM_CP9: u32 = 1 << 16;
/// Has CP9 bands
pub const CM_CP9BANDS: u32 = 1 << 17;
/// Has null3 omega
pub const CM_NULL3_OMEGA: u32 = 1 << 18;
/// MSV filter score valid
pub const CM_MSV_FILTER: u32 = 1 << 19;
/// VIT filter score valid
pub const CM_VIT_FILTER: u32 = 1 << 20;
/// FWD filter score valid
pub const CM_FWD_FILTER: u32 = 1 << 21;
/// CM has E-value parameters
pub const CM_EXP: u32 = 1 << 24;
/// CM has local E-value parameters
pub const CM_EXP_LOCAL: u32 = 1 << 25;

// Header-annotation flags needed to faithfully reproduce cm_file_WriteASCII output.
// C uses CMH_RF/CMH_CONS/CMH_MAP/CMH_CHKSUM/CMH_FP7/CMH_EXPTAIL_STATS (infernal.h);
// infernox picks its own (internally-consistent) bit positions in the free range.
/// Reference (RF) annotation exists  (C: CMH_RF)
pub const CM_RF: u32 = 1 << 26;
/// Consensus (CONS) annotation exists (C: CMH_CONS)
pub const CM_CONS: u32 = 1 << 27;
/// Alignment MAP annotation exists    (C: CMH_MAP)
pub const CM_MAP: u32 = 1 << 28;
/// Training-sequence checksum exists  (C: CMH_CHKSUM)
pub const CM_CHKSUM: u32 = 1 << 29;
/// Filter p7 HMM is present           (C: CMH_FP7)
pub const CM_FP7: u32 = 1 << 22;
/// All four ECMxx exponential-tail stat lines are present (C: CMH_EXPTAIL_STATS)
pub const CM_EXPTAIL_STATS: u32 = 1 << 23;

// =============================================================================
// State Type Mapping
// =============================================================================

/// State type names for parsing/display
pub const STATE_TYPE_NAMES: [&str; 10] = ["D", "MP", "ML", "MR", "IL", "IR", "S", "E", "B", "EL"];

/// Node type names for parsing/display
pub const NODE_TYPE_NAMES: [&str; 8] = ["BIF", "MATP", "MATL", "MATR", "BEGL", "BEGR", "ROOT", "END"];

/// Convert state type string to integer
pub fn state_type_from_str(s: &str) -> Option<i8> {
    match s {
        "D" => Some(D_ST as i8),
        "MP" => Some(MP_ST as i8),
        "ML" => Some(ML_ST as i8),
        "MR" => Some(MR_ST as i8),
        "IL" => Some(IL_ST as i8),
        "IR" => Some(IR_ST as i8),
        "S" => Some(S_ST as i8),
        "E" => Some(E_ST as i8),
        "B" => Some(B_ST as i8),
        "EL" => Some(EL_ST as i8),
        _ => None,
    }
}

/// Convert state type integer to string
pub fn state_type_to_str(st: i8) -> &'static str {
    match st as i32 {
        D_ST => "D",
        MP_ST => "MP",
        ML_ST => "ML",
        MR_ST => "MR",
        IL_ST => "IL",
        IR_ST => "IR",
        S_ST => "S",
        E_ST => "E",
        B_ST => "B",
        EL_ST => "EL",
        _ => "??",
    }
}

/// Convert node type string to integer
pub fn node_type_from_str(s: &str) -> Option<i8> {
    match s {
        "BIF" => Some(BIF_ND as i8),
        "MATP" => Some(MATP_ND as i8),
        "MATL" => Some(MATL_ND as i8),
        "MATR" => Some(MATR_ND as i8),
        "BEGL" => Some(BEGL_ND as i8),
        "BEGR" => Some(BEGR_ND as i8),
        "ROOT" => Some(ROOT_ND as i8),
        "END" => Some(END_ND as i8),
        _ => None,
    }
}

/// Convert node type integer to string
pub fn node_type_to_str(nd: i8) -> &'static str {
    match nd as i32 {
        BIF_ND => "BIF",
        MATP_ND => "MATP",
        MATL_ND => "MATL",
        MATR_ND => "MATR",
        BEGL_ND => "BEGL",
        BEGR_ND => "BEGR",
        ROOT_ND => "ROOT",
        END_ND => "END",
        _ => "??",
    }
}

/// Check if state type emits on left (pairs or left match)
pub fn state_emits_left(st: i8) -> bool {
    matches!(st as i32, MP_ST | ML_ST | IL_ST)
}

/// Check if state type emits on right (pairs or right match)
pub fn state_emits_right(st: i8) -> bool {
    matches!(st as i32, MP_ST | MR_ST | IR_ST)
}

/// Check if state type emits a pair
pub fn state_emits_pair(st: i8) -> bool {
    st as i32 == MP_ST
}

/// Check if state type emits a single residue
pub fn state_emits_single(st: i8) -> bool {
    matches!(st as i32, ML_ST | MR_ST | IL_ST | IR_ST)
}

/// Get number of children for a node type
pub fn node_nchildren(ndtype: i8) -> usize {
    match ndtype as i32 {
        BIF_ND => 2,   // bifurcation has 2 children
        MATP_ND => 6,  // MATP has MP, ML, MR, D, IL, IR
        MATL_ND => 3,  // MATL has ML, D, IL
        MATR_ND => 3,  // MATR has MR, D, IR
        BEGL_ND => 1,  // BEGL has S
        BEGR_ND => 2,  // BEGR has S, IL
        ROOT_ND => 3,  // ROOT has S, IL, IR
        END_ND => 1,   // END has E
        _ => 0,
    }
}

// =============================================================================
// CM Struct
// =============================================================================

/// Covariance Model
///
/// This struct matches the CM_t structure from Infernal's cm.h.
/// It represents a stochastic context-free grammar for RNA structure modeling.
#[derive(Debug, Clone)]
pub struct CM {
    // =========================================================================
    // Model identifiers
    // =========================================================================
    /// Model name
    pub name: String,
    /// Accession number (optional)
    pub acc: Option<String>,
    /// Description (optional)
    pub desc: Option<String>,
    /// Consensus residue display line (CONS annotation), 1-based [1..clen]; index 0
    /// is a sentinel. Empty if the model has no CONS annotation. Char codes match
    /// the CM file (e.g. 'G','c'). Used for the alidisplay model/consensus line when
    /// CMH_CONS is set.
    pub consensus: Vec<u8>,
    /// Reference-annotation display line (RF), 1-based [1..clen]; index 0 sentinel.
    /// Empty if the model has no RF annotation. Used for the alidisplay RF line.
    pub rf: Vec<u8>,
    /// C: cm->map[] (cm.h). Map of consensus columns onto original alignment
    /// columns, 1-based [1..clen] with map[0]=0. Populated when CM_MAP is set.
    /// Retained so cm_file_WriteASCII can reproduce the node MAP annotation.
    pub map: Vec<i32>,
    /// C: cm->ctime (DATE line). Retained verbatim for the writer.
    pub ctime: Option<String>,
    /// C: cm->comlog (COM lines). Stored as one string with embedded '\n' between
    /// command lines; the writer re-emits it via `multiline`.
    pub comlog: Option<String>,
    /// C: cm->nseq (NSEQ line). Number of training sequences.
    pub nseq: i32,
    /// C: cm->eff_nseq (EFFN line). Effective number of sequences.
    pub eff_nseq: f32,
    /// C: cm->checksum (CKSUM line). Checksum of training sequences (uint32).
    pub checksum: u32,

    // =========================================================================
    // Model configuration
    // =========================================================================
    /// Number of states (0..M-1)
    pub m: i32,
    /// Number of nodes (0..nodes-1)
    pub nodes: i32,
    /// Consensus length
    pub clen: i32,
    /// Maximum hit length (window size)
    pub w: i32,
    /// Configuration flags
    pub flags: u32,

    // =========================================================================
    // Null model
    // =========================================================================
    /// Background/null model probabilities [ALPHABET_SIZE]
    pub null: [f32; ALPHABET_SIZE],

    // =========================================================================
    // EL state parameters
    // =========================================================================
    /// EL self-transition log-odds score
    pub el_selfsc: f32,

    // =========================================================================
    // Node information
    // =========================================================================
    /// Node types: ndtype[nd] = type of node nd (0..nodes-1)
    pub ndtype: Vec<i8>,
    /// Node->state map: nodemap[nd] = first state of node nd
    pub nodemap: Vec<i32>,

    // =========================================================================
    // State information
    // =========================================================================
    /// State types: sttype[v] = type of state v (0..M-1)
    pub sttype: Vec<i8>,
    /// State->node map: ndidx[v] = node index for state v
    pub ndidx: Vec<i32>,
    /// State id within node: stid[v] = state's position within its node
    pub stid: Vec<i8>,
    /// First child: cfirst[v] = first child state of state v
    pub cfirst: Vec<i32>,
    /// Number of children: cnum[v] = number of children
    pub cnum: Vec<i32>,
    /// Last parent: plast[v] = last parent state of state v
    pub plast: Vec<i32>,
    /// Number of parents: pnum[v] = number of parents
    pub pnum: Vec<i32>,

    // =========================================================================
    // Transition probabilities
    // =========================================================================
    /// Transition probabilities: t[v][k] = P(transition from v to child k)
    /// Dimensions: [M][MAXCONNECT]
    pub t: Vec<Vec<f32>>,

    // =========================================================================
    // Emission probabilities
    // =========================================================================
    /// Emission probabilities: e[v][x] = P(emission x from state v)
    /// For pair states: x indexes into 16 values (AA,AC,AG,AU,...,UU)
    /// For single states: x indexes into 4 values (A,C,G,U)
    /// Dimensions: [M][PAIR_EMIT_SIZE] (16 for pairs, 4 used for singles)
    pub e: Vec<Vec<f32>>,

    // =========================================================================
    // Log-odds scores
    // =========================================================================
    /// Transition scores (log-odds): tsc[v][k]
    pub tsc: Vec<Vec<f32>>,
    /// Emission scores (log-odds): esc[v][x]
    pub esc: Vec<Vec<f32>>,
    /// Optimized emission scores for FastCYKScan: oesc[v][x]
    /// Single emitters: x in 0..Kp (18 entries)
    /// Pair emitters: x in 0..(Kp*Kp) (324 entries), indexed as a*Kp+b
    pub oesc: Vec<Vec<f32>>,

    // =========================================================================
    // Marginal emission scores for truncated DP (C cm->lmesc/rmesc/ilmesc/irmesc)
    // Filled by cm_logoddsify (CMLogoddsify block cm.c:790-864). Indexed
    // [0..v..M-1][0..a..Kp-1]. For MP: left/right marginals of the pair; for
    // ML/IL: lmesc = esc, rmesc = 0; for MR/IR: lmesc = 0, rmesc = esc. Degenerate
    // codes filled via esl_abc_{F,I}ExpectScVec. Non-emitters left at 0 / IMPOSSIBLE
    // for gap/nonres/missing. Empty until cm_logoddsify runs.
    // =========================================================================
    /// Left  marginal emission scores (log-odds, float): lmesc[v][a].
    pub lmesc: Vec<Vec<f32>>,
    /// Right marginal emission scores (log-odds, float): rmesc[v][a].
    pub rmesc: Vec<Vec<f32>>,
    /// Left  marginal emission scores (integer, scaled): ilmesc[v][a].
    pub ilmesc: Vec<Vec<i32>>,
    /// Right marginal emission scores (integer, scaled): irmesc[v][a].
    pub irmesc: Vec<Vec<i32>>,

    // =========================================================================
    // Local alignment parameters
    // =========================================================================
    /// Begin probability for local alignment
    pub pbegin: f32,
    /// End probability for local alignment
    pub pend: f32,
    /// Begin probabilities: begin[v] = P(local begin at state v)
    pub begin: Vec<f32>,
    /// End probabilities: end[v] = P(local end at state v)
    pub end: Vec<f32>,
    /// Begin scores (log-odds)
    pub beginsc: Vec<f32>,
    /// End scores (log-odds)
    pub endsc: Vec<f32>,

    // =========================================================================
    // Bifurcation information
    // =========================================================================
    /// For B states: left child S state
    pub lchild: Vec<i32>,
    /// For B states: right child S state
    pub rchild: Vec<i32>,

    // =========================================================================
    // Cutoffs (Gathering, Trusted, Noise)
    // =========================================================================
    /// Gathering threshold
    pub ga: f32,
    /// Trusted cutoff
    pub tc: f32,
    /// Noise cutoff
    pub nc: f32,

    // =========================================================================
    // W-related parameters
    // =========================================================================
    /// Beta parameter for W calculation
    pub w_beta: f64,

    // =========================================================================
    // QDB (Query-Dependent Bands)
    // =========================================================================
    /// Beta for QDB1
    pub qdb_beta1: f64,
    /// Beta for QDB2
    pub qdb_beta2: f64,
    /// Min d for QDB1
    pub dmin1: Vec<i32>,
    /// Max d for QDB1
    pub dmax1: Vec<i32>,
    /// Min d for QDB2
    pub dmin2: Vec<i32>,
    /// Max d for QDB2
    pub dmax2: Vec<i32>,

    // =========================================================================
    // Null model omega parameters (v1.1)
    // =========================================================================
    /// Null2 omega parameter for composition bias
    pub n2_omega: f64,
    /// Null3 omega parameter for composition bias
    pub n3_omega: f64,

    // =========================================================================
    // P7 filter parameters (v1.1)
    // =========================================================================
    /// P7 filter glocal forward E-value parameters (tau, lambda)
    pub efp7gf_tau: f64,
    pub efp7gf_lambda: f64,

    // =========================================================================
    // E-value parameters
    // =========================================================================
    /// Exponential tail parameters for E-value computation (glocal mode)
    pub exp_params: ExpParams,
    /// Exponential tail parameters for local mode E-value computation (ECMLI)
    pub exp_params_local: ExpParams,
    /// Exponential tail parameters for local CYK (ECMLC) — used by the F6 CYK filter
    /// P-value (fcyk_cm_exp_mode = EXP_CM_LC in default local search).
    pub exp_params_local_cyk: ExpParams,
    /// Exponential tail parameters for global CYK (ECMGC) — used by the nohmm+global
    /// CYK filter P-value (fcyk_cm_exp_mode = EXP_CM_GC). Parsed from the CM file but
    /// otherwise dropped by the default reader; kept here for the `-g --nohmm` path.
    pub exp_params_global_cyk: ExpParams,
    /// C: cm->expA[EXP_NMODES] (structs.h). The four exponential-tail parameter
    /// sets, indexed by exp mode: [0]=EXP_CM_GC, [1]=EXP_CM_GI, [2]=EXP_CM_LC,
    /// [3]=EXP_CM_LI (infernal.h). Valid only when CM_EXPTAIL_STATS is set.
    /// Retained verbatim so cm_file_WriteASCII can reproduce the ECMxx lines.
    pub exp_by_mode: [ExpParams; 4],

    // =========================================================================
    // Integer (scaled) log-odds scores for the integer Inside DP (FastIInsideScan)
    // Built by cm_nohmm::cm_logoddsify_global. INTSCALE=1000, -INFTY sentinel.
    // =========================================================================
    /// itsc[v][x] = Prob2Score(t[v][x], 1.0) — integer transition scores.
    pub itsc: Vec<Vec<i32>>,
    /// ioesc[v][a] — integer optimized emission scores (singlet: Kp; pair: Kp*Kp).
    pub ioesc: Vec<Vec<i32>>,
    /// ibeginsc[v] = Prob2Score(begin[v], 1.0) — integer local-begin scores.
    pub ibeginsc: Vec<i32>,
    /// iendsc[v] = Prob2Score(end[v], 1.0) — integer local-end scores.
    pub iendsc: Vec<i32>,
    /// iel_selfsc = Prob2Score(2^el_selfsc, 1.0) — integer EL self-transition score.
    pub iel_selfsc: i32,

    // =========================================================================
    // Root transition probabilities (for local mode)
    // =========================================================================
    /// Saved global transition probs from state 0 (when local mode activated)
    pub root_trans: Option<Vec<f32>>,

    // =========================================================================
    // P7 Filter HMM
    // =========================================================================
    /// Optional P7 HMM profile for fast filtering
    pub p7: Option<crate::p7_hmm::P7Profile>,
}

impl CM {
    /// Create a new empty CM with the given number of states and nodes
    pub fn new(m: i32, nodes: i32) -> Self {
        let m_usize = m as usize;
        let nodes_usize = nodes as usize;

        CM {
            name: String::new(),
            acc: None,
            desc: None,
            consensus: Vec::new(),
            rf: Vec::new(),
            map: Vec::new(),
            ctime: None,
            comlog: None,
            nseq: 0,
            eff_nseq: 0.0,
            checksum: 0,
            m,
            nodes,
            clen: 0,
            w: 0,
            flags: 0,
            null: [0.25; ALPHABET_SIZE],
            el_selfsc: 0.0,
            ndtype: vec![0; nodes_usize],
            nodemap: vec![0; nodes_usize],
            // C CreateCMBody (cm.c:192,271-272): sttype/stid are allocated with
            // (nstates+1) entries and the EL "state" at index M is special:
            //   cm->sttype[cm->M] = EL_st;  cm->stid[cm->M] = END_EL;
            // so that parsetree code (e.g. cm_StochasticParsetree[HB],
            // cm_TrStochasticParsetreeHB) can set v = cm->M and re-read
            // cm->sttype[v]/cm->stid[v] to detect EL at the top of the traceback
            // loop. All other per-state arrays remain length M (states 0..M-1).
            sttype: {
                let mut v = vec![0i8; m_usize + 1];
                v[m_usize] = EL_ST as i8; // cm.c:271 cm->sttype[cm->M] = EL_st
                v
            },
            ndidx: vec![0; m_usize],
            stid: {
                let mut v = vec![0i8; m_usize + 1];
                v[m_usize] = EL as i8; // cm.c:272 cm->stid[cm->M] = END_EL
                v
            },
            cfirst: vec![0; m_usize],
            cnum: vec![0; m_usize],
            plast: vec![0; m_usize],
            pnum: vec![0; m_usize],
            t: vec![vec![0.0; CM_MAXCONNECT]; m_usize],
            e: vec![vec![0.0; PAIR_EMIT_SIZE]; m_usize],
            tsc: vec![vec![0.0; CM_MAXCONNECT]; m_usize],
            esc: vec![vec![0.0; PAIR_EMIT_SIZE]; m_usize],
            oesc: Vec::new(),  // Computed on demand by calc_optimized_emit_scores()
            lmesc: Vec::new(),
            rmesc: Vec::new(),
            ilmesc: Vec::new(),
            irmesc: Vec::new(),
            pbegin: DEFAULT_PBEGIN as f32,
            pend: DEFAULT_PEND as f32,
            begin: vec![0.0; m_usize],
            end: vec![0.0; m_usize],
            beginsc: vec![IMPOSSIBLE_F32; m_usize],
            endsc: vec![IMPOSSIBLE_F32; m_usize],
            lchild: vec![-1; m_usize],
            rchild: vec![-1; m_usize],
            ga: 0.0,
            tc: 0.0,
            nc: 0.0,
            w_beta: DEFAULT_BETA_W,
            qdb_beta1: DEFAULT_BETA_QDB1,
            qdb_beta2: DEFAULT_BETA_QDB2,
            dmin1: vec![0; m_usize],
            dmax1: vec![0; m_usize],
            dmin2: vec![0; m_usize],
            dmax2: vec![0; m_usize],
            n2_omega: 0.03125,  // Default: 1/32
            n3_omega: 0.03125,  // Default: 1/32
            efp7gf_tau: 0.0,
            efp7gf_lambda: 0.0,
            exp_params: ExpParams::default(),
            exp_params_local: ExpParams::default(),
            exp_params_local_cyk: ExpParams::default(),
            exp_params_global_cyk: ExpParams::default(),
            exp_by_mode: [
                ExpParams::default(),
                ExpParams::default(),
                ExpParams::default(),
                ExpParams::default(),
            ],
            itsc: Vec::new(),
            ioesc: Vec::new(),
            ibeginsc: Vec::new(),
            iendsc: Vec::new(),
            iel_selfsc: 0,
            root_trans: None,
            p7: None,
        }
    }

    /// Check if state v emits (single or pair)
    pub fn state_emits(&self, v: usize) -> bool {
        let st = self.sttype[v];
        state_emits_single(st) || state_emits_pair(st)
    }

    /// Check if state v emits a pair
    pub fn state_emits_pair(&self, v: usize) -> bool {
        state_emits_pair(self.sttype[v])
    }

    /// Check if state v emits a single residue
    pub fn state_emits_single(&self, v: usize) -> bool {
        state_emits_single(self.sttype[v])
    }

    /// Get the number of emission values for state v
    pub fn state_nemit(&self, v: usize) -> usize {
        if state_emits_pair(self.sttype[v]) {
            PAIR_EMIT_SIZE
        } else if state_emits_single(self.sttype[v]) {
            ALPHABET_SIZE
        } else {
            0
        }
    }

    /// Check if the CM has local alignment enabled
    pub fn is_local(&self) -> bool {
        (self.flags & CM_LOCAL_BEGIN) != 0
    }

    /// Check if the CM has QDB information
    pub fn has_qdb(&self) -> bool {
        (self.flags & CM_QDB) != 0
    }

    /// Get transition probability from state v to child k
    pub fn get_t(&self, v: usize, k: usize) -> f32 {
        self.t[v][k]
    }

    /// Get emission probability for state v emitting x
    pub fn get_e(&self, v: usize, x: usize) -> f32 {
        self.e[v][x]
    }

    /// Get transition score from state v to child k
    pub fn get_tsc(&self, v: usize, k: usize) -> f32 {
        self.tsc[v][k]
    }

    /// Get emission score for state v emitting x
    pub fn get_esc(&self, v: usize, x: usize) -> f32 {
        self.esc[v][x]
    }

    /// Convert scores to probabilities (from log-odds)
    /// This is the inverse of scores being log2(p/null)
    pub fn score_to_prob(score: f32) -> f64 {
        2.0_f64.powf(score as f64)
    }

    /// Convert probability to log-odds score
    /// Score = log2(p / null) where null = 0.25 for single, 0.0625 for pair
    pub fn prob_to_score(prob: f64, null: f64) -> f32 {
        if prob <= 0.0 {
            IMPOSSIBLE_F32
        } else {
            (prob / null).log2() as f32
        }
    }

    /// Configure CM for local alignment mode
    ///
    /// This function sets up local begin and end probabilities, enabling
    /// the CYK/Inside algorithms to find optimal local alignments.
    ///
    /// # Arguments
    /// * `p_internal_start` - Probability mass to spread across local begins (default: 0.05)
    /// * `p_internal_exit` - Probability mass to spread across local ends (default: 0.05)
    pub fn localize(&mut self, p_internal_start: f32, p_internal_exit: f32) {
        // =========================================================================
        // Local begins: distribute probability across internal nodes
        // =========================================================================

        // Count internal nodes that can be local begin states
        // These are: MATP, MATL, MATR, BIF (nodes 2 and above)
        let mut nstarts = 0;
        for nd in 2..self.nodes {
            let ndtype = self.ndtype[nd as usize] as i32;
            if ndtype == MATP_ND || ndtype == MATL_ND || ndtype == MATR_ND || ndtype == BIF_ND {
                nstarts += 1;
            }
        }

        // Initialize begin probs to 0
        for v in 0..self.m as usize {
            self.begin[v] = 0.0;
        }

        // Node 1 gets (1 - p_internal_start) probability
        // This is the "global" begin through the root
        if self.nodes > 1 {
            self.begin[self.nodemap[1] as usize] = 1.0 - p_internal_start;
        }

        // Distribute p_internal_start across internal nodes
        if nstarts > 0 {
            let p = p_internal_start / nstarts as f32;
            for nd in 2..self.nodes {
                let ndtype = self.ndtype[nd as usize] as i32;
                if ndtype == MATP_ND || ndtype == MATL_ND || ndtype == MATR_ND || ndtype == BIF_ND {
                    self.begin[self.nodemap[nd as usize] as usize] = p;
                }
            }
        }

        // Convert begin probabilities to log2 scores
        for v in 0..self.m as usize {
            if self.begin[v] > 0.0 {
                self.beginsc[v] = self.begin[v].log2();
            } else {
                self.beginsc[v] = IMPOSSIBLE_F32;
            }
        }

        // Set local begin flag
        self.flags |= CM_LOCAL_BEGIN;

        // =========================================================================
        // Zero out ROOT_S (state 0) transitions
        // =========================================================================
        // In local mode, alignment begins via the begin[] array, not via t[0].
        // Set all ROOT_S transitions to 0 (probability) / IMPOSSIBLE (log-space).
        // This matches C's cm_localize() behavior at lines 487-491.
        //
        // Save original ROOT_S transitions so they can be restored by globalize()
        let cnum_root = self.cnum[0] as usize;
        if self.root_trans.is_none() {
            // Only save once (if localize is called multiple times)
            self.root_trans = Some(self.t[0][..cnum_root].to_vec());
        }
        for y in 0..cnum_root {
            self.t[0][y] = 0.0;
            self.tsc[0][y] = IMPOSSIBLE_F32;
        }

        // =========================================================================
        // Local ends: distribute probability across internal nodes
        // =========================================================================

        // Count internal nodes that can be local end states
        // These are: MATP, MATL, MATR, BEGL, BEGR (not adjacent to END nodes)
        let mut nexits = 0;
        for nd in 1..(self.nodes - 1) {
            let ndtype = self.ndtype[nd as usize] as i32;
            let next_ndtype = self.ndtype[(nd + 1) as usize] as i32;
            if (ndtype == MATP_ND || ndtype == MATL_ND || ndtype == MATR_ND ||
                ndtype == BEGL_ND || ndtype == BEGR_ND) && next_ndtype != END_ND {
                nexits += 1;
            }
        }

        // Initialize end probs to 0
        for v in 0..self.m as usize {
            self.end[v] = 0.0;
        }

        // Distribute p_internal_exit across internal nodes and renormalize transitions
        if nexits > 0 {
            let p = p_internal_exit / nexits as f32;
            for nd in 1..(self.nodes - 1) {
                let ndtype = self.ndtype[nd as usize] as i32;
                let next_ndtype = self.ndtype[(nd + 1) as usize] as i32;
                if (ndtype == MATP_ND || ndtype == MATL_ND || ndtype == MATR_ND ||
                    ndtype == BEGL_ND || ndtype == BEGR_ND) && next_ndtype != END_ND {
                    let v = self.nodemap[nd as usize] as usize;
                    self.end[v] = p;

                    // Renormalize transition probabilities so that sum(t[v]) + end[v] = 1
                    // This matches C's cm_localize() behavior
                    let cnum_v = self.cnum[v] as usize;
                    let t_sum: f32 = self.t[v][..cnum_v].iter().sum();
                    let denom = t_sum + self.end[v];
                    if denom > 0.0 {
                        for y in 0..cnum_v {
                            self.t[v][y] /= denom;
                            // Update tsc as well
                            if self.t[v][y] > 0.0 {
                                self.tsc[v][y] = self.t[v][y].log2();
                            } else {
                                self.tsc[v][y] = IMPOSSIBLE_F32;
                            }
                        }
                    }
                }
            }
        }

        // Convert end probabilities to log2 scores
        for v in 0..self.m as usize {
            if self.end[v] > 0.0 {
                self.endsc[v] = self.end[v].log2();
            } else {
                self.endsc[v] = IMPOSSIBLE_F32;
            }
        }

        // Set local end flag
        self.flags |= CM_LOCAL_END;
    }

    /// Check if beginsc[v] is a valid (non-impossible) score
    #[inline]
    pub fn has_local_begin(&self, v: usize) -> bool {
        self.beginsc[v] > IMPOSSIBLE_F32 + 1.0
    }

    /// Check if endsc[v] is a valid (non-impossible) score
    #[inline]
    pub fn has_local_end(&self, v: usize) -> bool {
        self.endsc[v] > IMPOSSIBLE_F32 + 1.0
    }

    /// Configure CM for glocal (global model, local sequence) alignment mode
    ///
    /// In glocal mode:
    /// - Alignments must use the full model (from ROOT to END)
    /// - Alignments can start/end anywhere in the sequence
    /// - Local begins and ends are DISABLED
    ///
    /// This is the opposite of localize() and is activated by the -g flag.
    pub fn globalize(&mut self) {
        // Clear local begin and end flags
        self.flags &= !CM_LOCAL_BEGIN;
        self.flags &= !CM_LOCAL_END;

        // Restore ROOT_S (state 0) transitions from saved probabilities
        // In glocal mode, alignment must begin via ROOT_S transitions, not begin[] array
        if let Some(ref saved_prob) = self.root_trans {
            let cnum_root = self.cnum[0] as usize;
            for y in 0..cnum_root.min(saved_prob.len()) {
                // Restore probability directly
                self.t[0][y] = saved_prob[y];
                // Convert probability back to score: score = log2(prob)
                if saved_prob[y] > 0.0 {
                    self.tsc[0][y] = saved_prob[y].log2();
                } else {
                    self.tsc[0][y] = IMPOSSIBLE_F32;
                }
            }
        }

        // Set all begin probabilities to 0 (impossible)
        // Only the root (state 0) can begin an alignment
        for v in 0..self.m as usize {
            self.begin[v] = 0.0;
            self.beginsc[v] = IMPOSSIBLE_F32;
        }

        // Set all end probabilities to 0 (impossible)
        // Alignments must reach an END state naturally
        for v in 0..self.m as usize {
            self.end[v] = 0.0;
            self.endsc[v] = IMPOSSIBLE_F32;
        }
    }

    /// Check if CM is configured for glocal mode
    pub fn is_glocal(&self) -> bool {
        // Glocal mode = neither local begin nor local end is set
        (self.flags & CM_LOCAL_BEGIN) == 0 && (self.flags & CM_LOCAL_END) == 0
    }

    /// Calculate optimized emission scores for FastCYKScan
    ///
    /// This function computes oesc (optimized emission scores) which use
    /// extended alphabet indexing (Kp=18) instead of canonical K=4.
    /// For pair emitters, index = a*Kp + b instead of a*K + b.
    ///
    /// This matches C's FCalcOptimizedEmitScores() from cm.c.
    pub fn calc_optimized_emit_scores(&mut self) {
        let k = ALPHABET_SIZE;      // 4 (canonical)
        let kp = ALPHABET_SIZE_P;   // 18 (extended)
        let m = self.m as usize;

        // Allocate oesc for all states
        self.oesc = vec![Vec::new(); m];

        // Precompute weights for ambiguous codes (5..17)
        // These are the fractional contributions of each canonical base
        // Index 5 = N (ACGU), 6 = R (AG), 7 = Y (CU), etc.
        let ambig_weights: [[f32; 4]; 18] = [
            // 0-3: canonical (not used for averaging)
            [1.0, 0.0, 0.0, 0.0], // A
            [0.0, 1.0, 0.0, 0.0], // C
            [0.0, 0.0, 1.0, 0.0], // G
            [0.0, 0.0, 0.0, 1.0], // U
            // 4: gap (IMPOSSIBLE)
            [0.0, 0.0, 0.0, 0.0],
            // 5: N (any) = ACGU
            [0.25, 0.25, 0.25, 0.25],
            // 6: R (purine) = AG
            [0.5, 0.0, 0.5, 0.0],
            // 7: Y (pyrimidine) = CU
            [0.0, 0.5, 0.0, 0.5],
            // 8: M = AC
            [0.5, 0.5, 0.0, 0.0],
            // 9: K = GU
            [0.0, 0.0, 0.5, 0.5],
            // 10: S = CG
            [0.0, 0.5, 0.5, 0.0],
            // 11: W = AU
            [0.5, 0.0, 0.0, 0.5],
            // 12: H = ACU (not G)
            [1.0/3.0, 1.0/3.0, 0.0, 1.0/3.0],
            // 13: B = CGU (not A)
            [0.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
            // 14: V = ACG (not U)
            [1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0],
            // 15: D = AGU (not C)
            [1.0/3.0, 0.0, 1.0/3.0, 1.0/3.0],
            // 16: (unused)
            [0.0, 0.0, 0.0, 0.0],
            // 17: missing (IMPOSSIBLE)
            [0.0, 0.0, 0.0, 0.0],
        ];

        for v in 0..m {
            let st = self.sttype[v] as i32;

            match st {
                // Single emitters: IL, ML, IR, MR
                IL_ST | ML_ST | IR_ST | MR_ST => {
                    self.oesc[v] = vec![IMPOSSIBLE_F32; kp];

                    // Copy canonical scores
                    for a in 0..k {
                        self.oesc[v][a] = self.esc[v][a];
                    }

                    // Position 4 = gap: IMPOSSIBLE (already set)

                    // Positions 5-16: ambiguous codes - compute weighted average
                    for a in 5..(kp - 1) {
                        let weights = &ambig_weights[a];
                        let mut sum = 0.0f32;
                        for c in 0..k {
                            if weights[c] > 0.0 {
                                sum += weights[c] * self.esc[v][c];
                            }
                        }
                        self.oesc[v][a] = sum;
                    }

                    // Position 17 = missing: IMPOSSIBLE (already set)
                }

                // Pair emitter: MP
                MP_ST => {
                    self.oesc[v] = vec![IMPOSSIBLE_F32; kp * kp];

                    // Copy canonical pairs: both a and b in 0..4
                    for a in 0..k {
                        for b in 0..k {
                            // C esc index: a * K + b = a * 4 + b
                            // C oesc index: a * Kp + b = a * 18 + b
                            self.oesc[v][a * kp + b] = self.esc[v][a * k + b];
                        }
                    }

                    // Left ambiguous, right canonical: a in 5..17, b in 0..4
                    for a in 5..(kp - 1) {
                        let left_weights = &ambig_weights[a];
                        for b in 0..k {
                            let mut sum = 0.0f32;
                            for c in 0..k {
                                if left_weights[c] > 0.0 {
                                    sum += left_weights[c] * self.esc[v][c * k + b];
                                }
                            }
                            self.oesc[v][a * kp + b] = sum;
                        }
                    }

                    // Left canonical, right ambiguous: a in 0..4, b in 5..17
                    for a in 0..k {
                        for b in 5..(kp - 1) {
                            let right_weights = &ambig_weights[b];
                            let mut sum = 0.0f32;
                            for c in 0..k {
                                if right_weights[c] > 0.0 {
                                    sum += right_weights[c] * self.esc[v][a * k + c];
                                }
                            }
                            self.oesc[v][a * kp + b] = sum;
                        }
                    }

                    // Both ambiguous: a,b in 5..17
                    for a in 5..(kp - 1) {
                        let left_weights = &ambig_weights[a];
                        for b in 5..(kp - 1) {
                            let right_weights = &ambig_weights[b];
                            let mut sum = 0.0f32;
                            for la in 0..k {
                                if left_weights[la] > 0.0 {
                                    for rb in 0..k {
                                        if right_weights[rb] > 0.0 {
                                            sum += left_weights[la] * right_weights[rb]
                                                * self.esc[v][la * k + rb];
                                        }
                                    }
                                }
                            }
                            self.oesc[v][a * kp + b] = sum;
                        }
                    }

                    // Gap (4) and missing (17) positions remain IMPOSSIBLE
                }

                // Non-emitters: empty oesc
                _ => {
                    // oesc[v] stays empty for non-emitting states
                }
            }
        }
    }

    /// Check if oesc has been computed
    pub fn has_oesc(&self) -> bool {
        !self.oesc.is_empty()
    }

    // =========================================================================
    // Build-time helpers (cmbuild). Additive ports of cm.c functions used by
    // the CM construction pipeline (cm_modelmaker.rs / bin/cmbuild.rs).
    // =========================================================================

    /// C: cm.c:CMZero(). Zero all counts/scores that the builder accumulates
    /// into. (CM::new already zeroes t/e; this mirrors the C call made right
    /// after cm_from_guide().)
    pub fn cm_zero(&mut self) {
        let m = self.m as usize;
        for v in 0..m {
            for x in &mut self.e[v] { *x = 0.0; }
            for x in &mut self.t[v] { *x = 0.0; }
            for x in &mut self.tsc[v] { *x = 0.0; }
            for x in &mut self.esc[v] { *x = 0.0; }
        }
        for v in 0..m {
            self.begin[v] = 0.0;
            self.end[v] = 0.0;
            self.beginsc[v] = 0.0;
            self.endsc[v] = 0.0;
        }
    }

    /// C: cm.c:CMSetNullModel(). Copy null[0..K] and renormalize.
    pub fn cm_set_null_model(&mut self, null: &[f32; ALPHABET_SIZE]) {
        for x in 0..ALPHABET_SIZE { self.null[x] = null[x]; }
        let sum: f32 = self.null.iter().sum();
        if sum != 0.0 { for x in &mut self.null { *x /= sum; } }
        else { for x in &mut self.null { *x = 1.0 / ALPHABET_SIZE as f32; } }
    }

    /// C: cm.c:CMRenormalize(). Renormalize all probability distributions.
    /// (Non-local builder path only — no local begin/end.)
    pub fn cm_renormalize(&mut self) {
        // esl_vec_FNorm(cm->null, K)
        fnorm(&mut self.null, ALPHABET_SIZE);
        for v in 0..self.m as usize {
            if self.cnum[v] > 0 && self.sttype[v] as i32 != B_ST {
                let cnum = self.cnum[v] as usize;
                fnorm(&mut self.t[v], cnum);
            }
            match self.sttype[v] as i32 {
                ML_ST | MR_ST | IL_ST | IR_ST => fnorm(&mut self.e[v], ALPHABET_SIZE),
                MP_ST => fnorm(&mut self.e[v], ALPHABET_SIZE * ALPHABET_SIZE),
                _ => {}
            }
        }
    }

    /// C: cm.c:CMCountNodetype(). Count nodes of a given ndtype.
    pub fn cm_count_nodetype(&self, ntype: i32) -> i32 {
        let mut count = 0;
        for nd in 0..self.nodes as usize {
            if self.ndtype[nd] as i32 == ntype { count += 1; }
        }
        count
    }

    /// C: cm.c:CalculateStateIndex(). Map (node, unique-state-id) -> state idx.
    pub fn calculate_state_index(&self, node: usize, utype: i32) -> i32 {
        let base = self.nodemap[node];
        match utype {
            ROOT_S => base,
            ROOT_IL => base + 1,
            ROOT_IR => base + 2,
            BEGL_S => base,
            BEGR_S => base,
            BEGR_IL => base + 1,
            MATP_MP => base,
            MATP_ML => base + 1,
            MATP_MR => base + 2,
            MATP_D => base + 3,
            MATP_IL => base + 4,
            MATP_IR => base + 5,
            MATL_ML => base,
            MATL_D => base + 1,
            MATL_IL => base + 2,
            MATR_MR => base,
            MATR_D => base + 1,
            MATR_IR => base + 2,
            END_E => base,
            BIF_B => base,
            _ => panic!("bogus utype {} in calculate_state_index()", utype),
        }
    }

    /// C: cm.c:CMLogoddsify() (partial). Fills tsc/esc/beginsc/endsc and the
    /// optimized emission scores. The marginal l/rmesc arrays used only by the
    /// search DP are not needed here (the CM writer emits scores from cm.t/cm.e
    /// directly, and cm_SetConsensus only reads cm.esc). Faithful for the
    /// consensus-derivation and el_selfsc use in configure_model.
    pub fn cm_logoddsify(&mut self) {
        let k = ALPHABET_SIZE;
        for v in 0..self.m as usize {
            let st = self.sttype[v] as i32;
            if st != B_ST && st != E_ST {
                for x in 0..self.cnum[v] as usize {
                    self.tsc[v][x] = sre_log2(self.t[v][x]);
                }
            }
            if st == MP_ST {
                for x in 0..k {
                    for y in 0..k {
                        self.esc[v][x * k + y] =
                            sre_log2(self.e[v][x * k + y] / (self.null[x] * self.null[y]));
                    }
                }
            } else if matches!(st, ML_ST | MR_ST | IL_ST | IR_ST) {
                for x in 0..k {
                    self.esc[v][x] = sre_log2(self.e[v][x] / self.null[x]);
                }
            }
            self.beginsc[v] = sre_log2(self.begin[v]);
            self.endsc[v] = sre_log2(self.end[v]);
        }
        self.calc_optimized_emit_scores();
    }

    /// C: cm.c:cm_Validate(). Validate emission/transition probability vectors
    /// (each sums to 1 within <tol>) and the stored consensus length.
    /// (Non-local builder path — no local begin/end validation.)
    pub fn cm_validate(&self, tol: f32) -> Result<(), String> {
        if self.m < 1 { return Err("CM has M < 1".to_string()); }
        let k = ALPHABET_SIZE;
        let mut clen = 0i32;
        for v in 0..self.m as usize {
            let st = self.sttype[v] as i32;
            let delta = state_delta(st);
            if delta == 2 {
                fvalidate(&self.e[v], k * k, tol).map_err(|_| format!("e[{}] fails pvector validation", v))?;
            } else if delta > 0 {
                fvalidate(&self.e[v], k, tol).map_err(|_| format!("e[{}] fails pvector validation", v))?;
            }
            if st != B_ST && st != E_ST {
                fvalidate(&self.t[v], self.cnum[v] as usize, tol)
                    .map_err(|_| format!("t[{}] fails pvector validation", v))?;
            }
            match self.stid[v] as i32 {
                MATL_ML => clen += 1,
                MATR_MR => clen += 1,
                MATP_MP => clen += 2,
                _ => {}
            }
        }
        if self.clen != clen {
            return Err(format!(
                "consensus length {} not correctly stored in CM, should be {}",
                self.clen, clen
            ));
        }
        Ok(())
    }
}

/// C: infernal.h StateDelta() — number of residues a state emits.
pub fn state_delta(sttype: i32) -> i32 {
    match sttype {
        MP_ST => 2,
        ML_ST | MR_ST | IL_ST | IR_ST => 1,
        _ => 0,
    }
}

/// C: esl_vec_FSum(vec, n) (esl_vectorops.c) — Kahan-compensated f32 summation.
/// MUST match C bit-for-bit: `esl_vec_FNorm` divides by this exact sum, and a
/// naive left-to-right sum can differ by 1 ULP for some inputs, which then
/// perturbs the normalized probabilities by 1 ULP (observed in the p7-filter
/// temp CM's `cm_renormalize`, cascading into a few emission cells).
fn fsum(vec: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for i in 0..n {
        let y = vec[i] - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// C: esl_vec_FNorm(vec, n) — normalize first n elements to sum 1 (or uniform
/// 1/n if the sum is 0). Uses the Kahan `esl_vec_FSum` for the divisor.
fn fnorm(vec: &mut [f32], n: usize) {
    let sum = fsum(vec, n);
    if sum != 0.0 {
        for x in &mut vec[..n] { *x /= sum; }
    } else {
        for x in &mut vec[..n] { *x = 1.0 / n as f32; }
    }
}

/// C: esl_vec_FValidate — check the first n elements form a probability vector
/// summing to 1 within <tol>.
fn fvalidate(vec: &[f32], n: usize, tol: f32) -> Result<(), ()> {
    let sum = fsum(vec, n); // C esl_vec_FValidate uses esl_vec_FSum (Kahan)
    if (sum - 1.0).abs() > tol { Err(()) } else { Ok(()) }
}

/// C: infernal.h sreLOG2(x) = (x > 0) ? log2(x) : -inf. Used for score display
/// only in build-time consensus derivation.
#[inline]
fn sre_log2(x: f32) -> f32 {
    if x > 0.0 { x.log2() } else { f32::NEG_INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_type_conversion() {
        assert_eq!(state_type_from_str("S"), Some(S_ST as i8));
        assert_eq!(state_type_from_str("MP"), Some(MP_ST as i8));
        assert_eq!(state_type_from_str("E"), Some(E_ST as i8));
        assert_eq!(state_type_from_str("X"), None);

        assert_eq!(state_type_to_str(S_ST as i8), "S");
        assert_eq!(state_type_to_str(MP_ST as i8), "MP");
    }

    #[test]
    fn test_node_type_conversion() {
        assert_eq!(node_type_from_str("ROOT"), Some(ROOT_ND as i8));
        assert_eq!(node_type_from_str("MATP"), Some(MATP_ND as i8));
        assert_eq!(node_type_from_str("END"), Some(END_ND as i8));
        assert_eq!(node_type_from_str("X"), None);

        assert_eq!(node_type_to_str(ROOT_ND as i8), "ROOT");
        assert_eq!(node_type_to_str(MATP_ND as i8), "MATP");
    }

    #[test]
    fn test_state_emission_checks() {
        assert!(state_emits_pair(MP_ST as i8));
        assert!(!state_emits_pair(ML_ST as i8));
        assert!(state_emits_single(ML_ST as i8));
        assert!(state_emits_single(IL_ST as i8));
        assert!(!state_emits_single(S_ST as i8));
        assert!(state_emits_left(MP_ST as i8));
        assert!(state_emits_right(MP_ST as i8));
        assert!(state_emits_left(ML_ST as i8));
        assert!(!state_emits_right(ML_ST as i8));
    }

    #[test]
    fn test_cm_new() {
        let cm = CM::new(227, 60);
        assert_eq!(cm.m, 227);
        assert_eq!(cm.nodes, 60);
        assert_eq!(cm.t.len(), 227);
        assert_eq!(cm.e.len(), 227);
        assert_eq!(cm.ndtype.len(), 60);
        // sttype/stid carry the EL sentinel at index M (cm.c:271-272), so length M+1.
        assert_eq!(cm.sttype.len(), 228);
        assert_eq!(cm.stid.len(), 228);
        assert_eq!(cm.sttype[227] as i32, EL_ST);
        assert_eq!(cm.stid[227] as i32, EL);
    }

    #[test]
    fn test_score_prob_conversion() {
        // Score of 0 means p = null
        let prob = CM::score_to_prob(0.0);
        assert!((prob - 1.0).abs() < 1e-6);

        // Convert 0.25 probability with null = 0.25 -> score = 0
        let score = CM::prob_to_score(0.25, 0.25);
        assert!(score.abs() < 1e-6);

        // Convert probability back
        let score = CM::prob_to_score(0.5, 0.25);
        let prob_back = CM::score_to_prob(score) * 0.25;
        assert!((prob_back - 0.5).abs() < 1e-6);
    }
}
