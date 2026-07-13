//! P7 HMM (HMMER3-style profile) for filtering
//!
//! P7 HMM is a HMMER3-style profile used for fast filtering in Infernal.
//! It is separate from CP9 - while CP9 is derived from CM structure,
//! P7 is an independent HMM profile optimized for rapid sequence scanning.

use std::io::{BufRead, Write};

// HMMER p7 header flags (hmmer.h:109-126). Only the bits that the ASCII writer
// (p7_hmmfile.c:513 p7_hmmfile_WriteASCII) consults are defined here.
pub const P7H_RF: u32 = 1 << 2; //  #RF annotation available
pub const P7H_CS: u32 = 1 << 3; //  #CS annotation available
pub const P7H_STATS: u32 = 1 << 7; //  model has E-value statistics calibrated
pub const P7H_MAP: u32 = 1 << 8; //  alignment map is available
pub const P7H_GA: u32 = 1 << 10; // gathering thresholds available
pub const P7H_TC: u32 = 1 << 11; // trusted cutoffs available
pub const P7H_NC: u32 = 1 << 12; // noise cutoffs available
pub const P7H_COMPO: u32 = 1 << 14; // model-specific residue composition available
pub const P7H_CHKSUM: u32 = 1 << 15; // model has an alignment checksum
pub const P7H_CONS: u32 = 1 << 16; // consensus residue line available
pub const P7H_MMASK: u32 = 1 << 17; // #MM annotation available
// Legacy NULL-convention flags used by the binary writer only (hmmer.h:110,118).
pub const P7H_DESC: u32 = 1 << 1; //  description exists (legacy; xref SRE:J5/114)
pub const P7H_ACC: u32 = 1 << 9; //  accession is available (legacy; xref SRE:J5/114)

// HMMER cutoff array indices (hmmer.h). GA/TC/NC each carry two values.
const P7_GA1: usize = 0;
const P7_GA2: usize = 1;
const P7_TC1: usize = 2;
const P7_TC2: usize = 3;
const P7_NC1: usize = 4;
const P7_NC2: usize = 5;

// Version stamped into the HMMER3/f banner. The p7 filter block embedded in a
// .cm file is the HMMER3/f ASCII format; C's cmconvert links HMMER 3.4 and emits
// "[3.4 | Aug 2023]". The banner is tool-version dependent and is compared by
// prefix only in round-trip tests (mirrors the INFERNAL1/a banner handling).
const HMMER_VERSION: &str = "3.4";
const HMMER_DATE: &str = "Aug 2023";

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

    // --- Header metadata retained so the block can be re-emitted byte-for-byte
    // by `p7_hmmfile_write_ascii` (port of p7_hmmfile.c:513). Not consumed by the
    // MSV/Forward filters. ---
    /// Accession (ACC line), if present. = C `hmm->acc`.
    pub acc: Option<String>,
    /// Description (DESC line), if present. = C `hmm->desc`.
    pub desc: Option<String>,
    /// Maximum sequence length (MAXL line). = C `hmm->max_length`; 0 if absent.
    pub max_length: i32,
    /// Alphabet type string as written on the ALPH line (e.g. "RNA"). = C
    /// `esl_abc_DecodeType(hmm->abc->type)`.
    pub alph: String,
    /// Creation date (DATE line), verbatim. = C `hmm->ctime`.
    pub ctime: Option<String>,
    /// Command log (COM lines), joined by '\n' with the `[n]` prefixes stripped.
    /// = C `hmm->comlog`.
    pub comlog: Option<String>,
    /// Number of training sequences (NSEQ line). = C `hmm->nseq`; 0 if absent.
    pub nseq: i32,
    /// Effective sequence number (EFFN line). = C `hmm->eff_nseq`; <0 if absent.
    pub eff_nseq: f32,
    /// Alignment checksum (CKSUM line). = C `hmm->checksum` (unsigned 32-bit).
    pub checksum: u32,
    /// GA/TC/NC cutoffs, indices [GA1,GA2,TC1,TC2,NC1,NC2]. = C `hmm->cutoff`.
    pub cutoff: [f32; 6],
    /// Per-node alignment map columns (MAP). = C `hmm->map[1..M]`; index 0 unused.
    pub map: Vec<i32>,
    /// Per-node consensus residues (CONS). = C `hmm->consensus[1..M]`.
    pub consensus: Vec<u8>,
    /// Per-node reference (RF) annotation. = C `hmm->rf[1..M]`.
    pub rf: Vec<u8>,
    /// Per-node model-mask (MM) annotation. = C `hmm->mm[1..M]`.
    pub mm: Vec<u8>,
    /// Per-node consensus-structure (CS) annotation. = C `hmm->cs[1..M]`.
    pub cs: Vec<u8>,
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
            acc: None,
            desc: None,
            max_length: 0,
            alph: "RNA".to_string(),
            ctime: None,
            comlog: None,
            nseq: 0,
            // C default is -1 so an absent EFFN is not re-emitted (writer: eff_nseq >= 0).
            eff_nseq: -1.0,
            checksum: 0,
            cutoff: [0.0; 6],
            map: vec![0; (m + 1) as usize],
            consensus: vec![b' '; (m + 1) as usize],
            rf: vec![b' '; (m + 1) as usize],
            mm: vec![b' '; (m + 1) as usize],
            cs: vec![b' '; (m + 1) as usize],
        }
    }

    /// Parse P7 profile from HMMER3 format lines
    ///
    /// Parses the embedded HMMER3/f section from CM files.
    /// Returns None if parsing fails.
    pub fn parse_hmmer3<B: BufRead>(lines: &mut std::io::Lines<B>) -> Option<Self> {
        // Faithful port of the header half of read_asc30hmm (p7_hmmfile.c:1245).
        // We retain every header field so `p7_hmmfile_write_ascii` can re-emit the
        // block byte-for-byte.
        let mut name = String::new();
        let mut m: i32 = 0;
        let mut msv_mu = 0.0f64;
        let mut msv_lambda = 0.0f64;
        let mut vit_mu = 0.0f64;
        let mut vit_lambda = 0.0f64;
        let mut fwd_tau = 0.0f64;
        let mut fwd_lambda = 0.0f64;

        let mut acc: Option<String> = None;
        let mut desc: Option<String> = None;
        let mut max_length: i32 = 0;
        let mut alph = String::new();
        let mut ctime: Option<String> = None;
        let mut comlog: Option<String> = None;
        let mut nseq: i32 = 0;
        let mut eff_nseq: f32 = -1.0;
        let mut checksum: u32 = 0;
        // C default is p7_CUTOFF_UNSET = -99999.0f (hmmer.h:77; p7_hmm.c:104). The
        // binary writer emits all 6 cutoffs unconditionally, so unset entries must
        // carry this sentinel to match p7_hmmfile_WriteBinary byte-for-byte.
        let mut cutoff = [-99999.0f32; 6];
        let mut flags: u32 = 0;

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

            // Remainder of the line after the tag, with internal spacing preserved
            // (esl_fileparser_GetRemainingLine equivalent). Used by DATE/COM.
            let remaining = |tag_len: usize| -> String {
                line[tag_len..].trim_start().to_string()
            };

            match parts[0] {
                // p7_hmmfile.c:1290 NAME
                "NAME" => {
                    if parts.len() >= 2 {
                        name = parts[1].to_string();
                    }
                }
                // p7_hmmfile.c:1300 ACC / :1304 DESC (GetRemainingLine)
                "ACC" => acc = Some(remaining("ACC".len())),
                "DESC" => desc = Some(remaining("DESC".len())),
                // p7_hmmfile.c:1309 LENG
                "LENG" => {
                    if parts.len() >= 2 {
                        m = parts[1].parse().unwrap_or(0);
                    }
                }
                // p7_hmmfile.c:1314 MAXL
                "MAXL" => {
                    if parts.len() >= 2 {
                        max_length = parts[1].parse().unwrap_or(0);
                    }
                }
                // p7_hmmfile.c:1319 ALPH
                "ALPH" => {
                    if parts.len() >= 2 {
                        alph = parts[1].to_string();
                    }
                }
                // p7_hmmfile.c:1330 RF (flag on "yes")
                "RF" => {
                    if parts.get(1).map(|s| s.eq_ignore_ascii_case("yes")) == Some(true) {
                        flags |= P7H_RF;
                    }
                }
                // p7_hmmfile.c:1337 MM
                "MM" => {
                    if parts.get(1).map(|s| s.eq_ignore_ascii_case("yes")) == Some(true) {
                        flags |= P7H_MMASK;
                    }
                }
                // p7_hmmfile.c:1345 CONS
                "CONS" => {
                    if parts.get(1).map(|s| s.eq_ignore_ascii_case("yes")) == Some(true) {
                        flags |= P7H_CONS;
                    }
                }
                // p7_hmmfile.c:1352 CS
                "CS" => {
                    if parts.get(1).map(|s| s.eq_ignore_ascii_case("yes")) == Some(true) {
                        flags |= P7H_CS;
                    }
                }
                // p7_hmmfile.c:1359 MAP
                "MAP" => {
                    if parts.get(1).map(|s| s.eq_ignore_ascii_case("yes")) == Some(true) {
                        flags |= P7H_MAP;
                    }
                }
                // p7_hmmfile.c:1366 DATE (GetRemainingLine)
                "DATE" => ctime = Some(remaining("DATE".len())),
                // p7_hmmfile.c:1371 COM: skip the "[n]" token, keep the rest; append.
                "COM" => {
                    // after tag: e.g. "[1] cmbuild ...". Drop the first token.
                    let after = line["COM".len()..].trim_start();
                    let cmd = match after.find(char::is_whitespace) {
                        Some(idx) => after[idx..].trim_start().to_string(),
                        None => String::new(),
                    };
                    comlog = Some(match comlog {
                        None => cmd,
                        Some(mut prev) => {
                            prev.push('\n');
                            prev.push_str(&cmd);
                            prev
                        }
                    });
                }
                // p7_hmmfile.c:1383 NSEQ
                "NSEQ" => {
                    if parts.len() >= 2 {
                        nseq = parts[1].parse().unwrap_or(0);
                    }
                }
                // p7_hmmfile.c:1388 EFFN
                "EFFN" => {
                    if parts.len() >= 2 {
                        eff_nseq = parts[1].parse().unwrap_or(-1.0);
                    }
                }
                // p7_hmmfile.c:1393 CKSUM (+ p7H_CHKSUM flag)
                "CKSUM" => {
                    if parts.len() >= 2 {
                        checksum = parts[1].parse().unwrap_or(0);
                        flags |= P7H_CHKSUM;
                    }
                }
                // p7_hmmfile.c:546-554 GA/TC/NC. For RNA/DNA a single value follows.
                "GA" => {
                    if let Some(v) = parts.get(1).and_then(|s| s.parse::<f32>().ok()) {
                        cutoff[P7_GA1] = v;
                        cutoff[P7_GA2] =
                            parts.get(2).and_then(|s| s.parse::<f32>().ok()).unwrap_or(v);
                        flags |= P7H_GA;
                    }
                }
                "TC" => {
                    if let Some(v) = parts.get(1).and_then(|s| s.parse::<f32>().ok()) {
                        cutoff[P7_TC1] = v;
                        cutoff[P7_TC2] =
                            parts.get(2).and_then(|s| s.parse::<f32>().ok()).unwrap_or(v);
                        flags |= P7H_TC;
                    }
                }
                "NC" => {
                    if let Some(v) = parts.get(1).and_then(|s| s.parse::<f32>().ok()) {
                        cutoff[P7_NC1] = v;
                        cutoff[P7_NC2] =
                            parts.get(2).and_then(|s| s.parse::<f32>().ok()).unwrap_or(v);
                        flags |= P7H_NC;
                    }
                }
                "STATS" => {
                    // STATS LOCAL MSV -8.9335 0.71867 (p7_hmmfile.c:1399)
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
                        flags |= P7H_STATS;
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
        p7.acc = acc;
        p7.desc = desc;
        p7.max_length = max_length;
        if !alph.is_empty() {
            p7.alph = alph;
        }
        p7.ctime = ctime;
        p7.comlog = comlog;
        p7.nseq = nseq;
        p7.eff_nseq = eff_nseq;
        p7.checksum = checksum;
        p7.cutoff = cutoff;
        p7.flags = flags;
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
                p7.flags |= P7H_COMPO;
                for i in 0..4 {
                    if let Ok(score) = parts[i + 1].parse::<f32>() {
                        p7.compo[i] = (-score).exp();
                    }
                }
            }
        }

        // Node-0 insert emissions (p7_hmmfile.c:1542 with k=0). C reads and stores
        // hmm->ins[0][x] = expf(-score); p7_hmmfile_WriteBinary emits ins[0..M], so
        // we must retain these rather than leave the P7Profile::new default.
        if let Some(Ok(line)) = lines.next() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                for i in 0..4 {
                    if parts[i] == "*" {
                        p7.ins[0][i] = 0.0;
                    } else if let Ok(score) = parts[i].parse::<f32>() {
                        p7.ins[0][i] = (-score).exp();
                    }
                }
            }
        }

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
                // First value is node index, then 4 emission scores (log-odds).
                // p7_hmmfile.c:1516  hmm->mat[k][x] = (*tok1=='*' ? 0 : expf(-x))
                for i in 0..4 {
                    if parts[i + 1] == "*" {
                        p7.mat[k][i] = 0.0;
                    } else if let Ok(score) = parts[i + 1].parse::<f32>() {
                        p7.mat[k][i] = (-score).exp();
                    }
                }
                // Trailing annotation columns (K=4). In a HMMER3/f file the writer
                // always emits map, consensus, RF, MM and CS columns (p7_hmmfile.c
                // :590-613), so positions are fixed. Store each conditioned on its
                // flag, mirroring the reader (p7_hmmfile.c:1519-1536).
                const K: usize = 4;
                if p7.flags & P7H_MAP != 0 {
                    if let Some(v) = parts.get(K + 1).and_then(|s| s.parse::<i32>().ok()) {
                        p7.map[k] = v;
                    }
                }
                if p7.flags & P7H_CONS != 0 {
                    if let Some(t) = parts.get(K + 2) {
                        p7.consensus[k] = t.as_bytes()[0];
                    }
                }
                if p7.flags & P7H_RF != 0 {
                    if let Some(t) = parts.get(K + 3) {
                        p7.rf[k] = t.as_bytes()[0];
                    }
                }
                if p7.flags & P7H_MMASK != 0 {
                    if let Some(t) = parts.get(K + 4) {
                        p7.mm[k] = t.as_bytes()[0];
                    }
                }
                if p7.flags & P7H_CS != 0 {
                    if let Some(t) = parts.get(K + 5) {
                        p7.cs[k] = t.as_bytes()[0];
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
                // p7_hmmfile.c:1542  hmm->ins[k][x] = (*tok1=='*' ? 0 : expf(-x))
                for i in 0..4 {
                    if parts[i] == "*" {
                        p7.ins[k][i] = 0.0;
                    } else if let Ok(score) = parts[i].parse::<f32>() {
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

/// Print a probability in a fixed field, HMMER-style.
///
/// Faithful port of `printprob` (p7_hmmfile.c:2091). Probabilities are written
/// as `-ln(p)` to 5 decimals in a width-`fieldwidth` field, with `*` for p==0
/// and `0.00000` for p==1. `-ln(p)` is computed at f32 precision (C `logf`) and
/// promoted to f64 for formatting (C promotes `float` to `double` for `%f`).
fn printprob<W: Write>(w: &mut W, fieldwidth: usize, p: f32) -> std::io::Result<()> {
    if p == 0.0 {
        // C: fprintf(fp, " %*s", fieldwidth, "*")
        write!(w, " {:>width$}", "*", width = fieldwidth)
    } else if p == 1.0 {
        // C: fprintf(fp, " %*.5f", fieldwidth, 0.0)
        write!(w, " {:>width$.5}", 0.0f64, width = fieldwidth)
    } else {
        // C: fprintf(fp, " %*.5f", fieldwidth, -logf(p))
        let v = -(p.ln()); // f32, matching C logf
        write!(w, " {:>width$.5}", v as f64, width = fieldwidth)
    }
}

/// Break a multi-line record into prefixed, numbered lines.
///
/// Faithful port of `multiline` (p7_hmmfile.c:2008), used for the COM log. Given
/// `pfx="COM  "` and `s="cmd1\ncmd2"` prints `COM   [1] cmd1` / `COM   [2] cmd2`.
fn multiline<W: Write>(w: &mut W, pfx: &str, s: &str) -> std::io::Result<()> {
    let mut nline = 1;
    let mut rest = s;
    loop {
        match rest.find('\n') {
            Some(idx) => {
                write!(w, "{} [{}] {}\n", pfx, nline, &rest[..idx])?;
                nline += 1;
                rest = &rest[idx + 1..];
                if rest.is_empty() {
                    break;
                }
            }
            None => {
                write!(w, "{} [{}] {}\n", pfx, nline, rest)?;
                break;
            }
        }
    }
    Ok(())
}

/// Symbols for the ALPH-line header and the `mm=='m'` consensus special case.
/// Mirrors easel's canonical nucleic alphabets (esl_alphabet.c:183 for RNA).
fn abc_symbols(alph: &str) -> ([char; 4], char) {
    // Returns (canonical K=4 symbols, tolower(sym[Kp-3]) "any" residue).
    match alph {
        "DNA" => (['A', 'C', 'G', 'T'], 'n'),
        _ => (['A', 'C', 'G', 'U'], 'n'), // RNA (default)
    }
}

/// Write a HMMER3/f ASCII save file for the p7 filter HMM `hmm`.
///
/// Faithful port of `p7_hmmfile_WriteASCII(fp, -1, hmm)` (p7_hmmfile.c:513) with
/// `format = p7_HMMFILE_3f` (the default). Reproduces the block byte-for-byte as
/// emitted by C `cmconvert -a` (except the tool-version banner line). This is the
/// same block appended after the CM `//` by cm_file.c:756-758.
pub fn p7_hmmfile_write_ascii<W: Write>(w: &mut W, hmm: &P7Profile) -> std::io::Result<()> {
    let (syms, any_sym) = abc_symbols(&hmm.alph);

    // Banner (p7_hmmfile.c:521, format==p7_HMMFILE_3f).
    write!(w, "HMMER3/f [{} | {}]\n", HMMER_VERSION, HMMER_DATE)?;

    // Header block (p7_hmmfile.c:529-544).
    write!(w, "NAME  {}\n", hmm.name)?; // :529
    if let Some(ref acc) = hmm.acc {
        write!(w, "ACC   {}\n", acc)?; // :530
    }
    if let Some(ref desc) = hmm.desc {
        write!(w, "DESC  {}\n", desc)?; // :531
    }
    write!(w, "LENG  {}\n", hmm.m)?; // :532
    if hmm.max_length > 0 {
        write!(w, "MAXL  {}\n", hmm.max_length)?; // :533 (format>=3c)
    }
    write!(w, "ALPH  {}\n", hmm.alph)?; // :534
    write!(w, "RF    {}\n", yesno(hmm.flags & P7H_RF))?; // :535
    write!(w, "MM    {}\n", yesno(hmm.flags & P7H_MMASK))?; // :536 (format>=3f)
    write!(w, "CONS  {}\n", yesno(hmm.flags & P7H_CONS))?; // :537 (format>=3e)
    write!(w, "CS    {}\n", yesno(hmm.flags & P7H_CS))?; // :538
    write!(w, "MAP   {}\n", yesno(hmm.flags & P7H_MAP))?; // :539
    if let Some(ref ctime) = hmm.ctime {
        write!(w, "DATE  {}\n", ctime)?; // :540
    }
    if let Some(ref comlog) = hmm.comlog {
        multiline(w, "COM  ", comlog)?; // :541
    }
    if hmm.nseq > 0 {
        write!(w, "NSEQ  {}\n", hmm.nseq)?; // :542
    }
    if hmm.eff_nseq >= 0.0 {
        // :543  fprintf(fp, "EFFN  %f\n", eff_nseq) — %f is 6 decimals; the value
        // is stored as a C float, so format through f32 to match its rounding.
        write!(w, "EFFN  {:.6}\n", hmm.eff_nseq as f64)?;
    }
    if hmm.flags & P7H_CHKSUM != 0 {
        write!(w, "CKSUM {}\n", hmm.checksum)?; // :544
    }

    // GA/TC/NC cutoffs (p7_hmmfile.c:546-554). RNA/DNA emit a single value each.
    if hmm.alph == "RNA" || hmm.alph == "DNA" {
        if hmm.flags & P7H_GA != 0 {
            write!(w, "GA    {:.2}\n", hmm.cutoff[P7_GA1])?;
        }
        if hmm.flags & P7H_TC != 0 {
            write!(w, "TC    {:.2}\n", hmm.cutoff[P7_TC1])?;
        }
        if hmm.flags & P7H_NC != 0 {
            write!(w, "NC    {:.2}\n", hmm.cutoff[P7_NC1])?;
        }
    } else {
        if hmm.flags & P7H_GA != 0 {
            write!(w, "GA    {:.2} {:.2}\n", hmm.cutoff[P7_GA1], hmm.cutoff[P7_GA2])?;
        }
        if hmm.flags & P7H_TC != 0 {
            write!(w, "TC    {:.2} {:.2}\n", hmm.cutoff[P7_TC1], hmm.cutoff[P7_TC2])?;
        }
        if hmm.flags & P7H_NC != 0 {
            write!(w, "NC    {:.2} {:.2}\n", hmm.cutoff[P7_NC1], hmm.cutoff[P7_NC2])?;
        }
    }

    // STATS lines (p7_hmmfile.c:555-565), default (non-3a) format. evparam values
    // are stored as C floats; format through f32 to match `%8.4f`/`%8.5f` rounding.
    if hmm.flags & P7H_STATS != 0 {
        write!(
            w,
            "STATS LOCAL MSV      {:8.4} {:8.5}\n",
            hmm.evparam.lmmu as f32 as f64,
            hmm.evparam.lmlambda as f32 as f64
        )?;
        write!(
            w,
            "STATS LOCAL VITERBI  {:8.4} {:8.5}\n",
            hmm.evparam.lvmu as f32 as f64,
            hmm.evparam.lvlambda as f32 as f64
        )?;
        write!(
            w,
            "STATS LOCAL FORWARD  {:8.4} {:8.5}\n",
            hmm.evparam.lftau as f32 as f64,
            hmm.evparam.lflambda as f32 as f64
        )?;
    }

    // HMM column header (p7_hmmfile.c:567-572).
    write!(w, "HMM     ")?; // "HMM" + 5 spaces
    for &c in syms.iter() {
        write!(w, "     {}   ", c)?; // C: "     %c   "
    }
    write!(w, "\n")?;
    write!(
        w,
        "        {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
        "m->m", "m->i", "m->d", "i->m", "i->i", "d->m", "d->d"
    )?;

    // COMPO line (p7_hmmfile.c:573-578).
    if hmm.flags & P7H_COMPO != 0 {
        write!(w, "  COMPO ")?;
        for x in 0..4 {
            printprob(w, 8, hmm.compo[x])?;
        }
        write!(w, "\n")?;
    }

    // Node 0: insert emissions, then B-> transitions (p7_hmmfile.c:580-589).
    write!(w, "        ")?;
    for x in 0..4 {
        printprob(w, 8, hmm.ins[0][x])?;
    }
    write!(w, "\n")?;
    write!(w, "        ")?;
    for x in 0..7 {
        printprob(w, 8, hmm.trans[0][x])?;
    }
    write!(w, "\n")?;

    // Main model nodes 1..M (p7_hmmfile.c:590-624).
    for k in 1..=hmm.m as usize {
        // Line 1: node index, match emissions, then annotation columns.
        write!(w, " {:6} ", k)?; // C: " %6d "
        for x in 0..4 {
            printprob(w, 8, hmm.mat[k][x])?;
        }
        // MAP (:596-597)
        if hmm.flags & P7H_MAP != 0 {
            write!(w, " {:6}", hmm.map[k])?;
        } else {
            write!(w, " {:>6}", "-")?;
        }
        // Consensus (:599-607), format>=3e.
        let cons: char = if (hmm.flags & P7H_MMASK != 0) && hmm.mm[k] == b'm' {
            any_sym
        } else if hmm.flags & P7H_CONS != 0 {
            hmm.consensus[k] as char
        } else {
            '-'
        };
        write!(w, " {}", cons)?;
        // RF (:610-611)
        let rf: char = if hmm.flags & P7H_RF != 0 {
            hmm.rf[k] as char
        } else {
            '-'
        };
        write!(w, " {}", rf)?;
        // MM (:612), format>=3f.
        let mm: char = if hmm.flags & P7H_MMASK != 0 {
            hmm.mm[k] as char
        } else {
            '-'
        };
        write!(w, " {}", mm)?;
        // CS (:613)
        let cs: char = if hmm.flags & P7H_CS != 0 {
            hmm.cs[k] as char
        } else {
            '-'
        };
        write!(w, " {}\n", cs)?;

        // Line 2: insert emissions (:615-618).
        write!(w, "        ")?;
        for x in 0..4 {
            printprob(w, 8, hmm.ins[k][x])?;
        }
        // Line 3: transitions (:619-623).
        write!(w, "\n        ")?;
        for x in 0..7 {
            printprob(w, 8, hmm.trans[k][x])?;
        }
        write!(w, "\n")?;
    }
    write!(w, "//\n")?; // p7_hmmfile.c:625
    Ok(())
}

// HMMER3/f binary magic number (p7_hmmfile.c:52). v3f = "hmma" + 0x80808080.
const V3F_MAGIC: u32 = 0xe8ededba;

/// C: esl_abc encoding of the alphabet type (esl_alphabet.h:15-20). The p7
/// binary header stores abc->type as an int.
fn abc_type_code(alph: &str) -> i32 {
    match alph {
        "RNA" => 1,   // eslRNA
        "DNA" => 2,   // eslDNA
        "amino" | "AMINO" | "protein" => 3, // eslAMINO
        _ => 1,
    }
}

/// Faithful port of write_bin_string (p7_hmmfile.c:2138): write an int for the
/// string length (strlen+1, including the trailing '\0'), then the string with
/// its '\0'. For a NULL/None string, write a single 0 length.
fn write_bin_string<W: Write>(w: &mut W, s: Option<&str>) -> std::io::Result<()> {
    match s {
        Some(s) => {
            let bytes = s.as_bytes();
            let len = (bytes.len() + 1) as i32;
            w.write_all(&len.to_le_bytes())?;
            w.write_all(bytes)?;
            w.write_all(&[0u8])?; // trailing '\0'
        }
        None => {
            w.write_all(&0i32.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Write an annotation line (rf/mm/consensus/cs) in the HMMER binary layout: a
/// char[M+2] array where index 0 is a spacer ' ', indices 1..M carry the data,
/// and index M+1 is the terminating '\0' (p7_hmmfile.c:3135-3139). Our in-memory
/// arrays are length M+1 (indices 0..M), so we pad the trailing '\0' here.
fn write_bin_annot<W: Write>(w: &mut W, v: &[u8], m: usize) -> std::io::Result<()> {
    let mut buf = vec![b' '; m + 2];
    for i in 1..=m {
        buf[i] = if i < v.len() { v[i] } else { b' ' };
    }
    buf[0] = b' ';
    buf[m + 1] = 0u8;
    w.write_all(&buf)
}

/// Function: p7_hmmfile_write_binary
/// Faithful byte-for-byte port of p7_hmmfile_WriteBinary (p7_hmmfile.c:981) for
/// the default HMMER3/f (p7_HMMFILE_3f) format. Writes the p7 filter HMM block
/// that trails a binary CM (cm_file.c:899 calls p7_hmmfile_WriteBinary(fp,-1,fp7)).
/// All scalars are native little-endian (x86-64).
pub fn p7_hmmfile_write_binary<W: Write>(w: &mut W, hmm: &P7Profile) -> std::io::Result<()> {
    let k = 4usize; // abc->K for RNA/DNA
    let m = hmm.m as usize;

    // p7_hmmfile.c:1005-1006: NULL-convention fixup of the ACC/DESC flag bits.
    let mut flags = hmm.flags;
    if hmm.desc.is_none() { flags &= !P7H_DESC; } else { flags |= P7H_DESC; }
    if hmm.acc.is_none()  { flags &= !P7H_ACC;  } else { flags |= P7H_ACC; }

    // magic (v3f) — p7_hmmfile.c:1009
    w.write_all(&V3F_MAGIC.to_le_bytes())?;
    // flags, M, abc->type (int) — p7_hmmfile.c:1019-1021
    w.write_all(&(flags as i32).to_le_bytes())?;
    w.write_all(&hmm.m.to_le_bytes())?;
    w.write_all(&abc_type_code(&hmm.alph).to_le_bytes())?;

    // Core model probabilities — p7_hmmfile.c:1025-1030.
    for kk in 1..=m {
        for x in 0..k { w.write_all(&hmm.mat[kk][x].to_le_bytes())?; }
    }
    for kk in 0..=m {
        for x in 0..k { w.write_all(&hmm.ins[kk][x].to_le_bytes())?; }
    }
    for kk in 0..=m {
        for x in 0..7 { w.write_all(&hmm.trans[kk][x].to_le_bytes())?; }
    }

    // Annotation section — p7_hmmfile.c:1034-1048.
    write_bin_string(w, Some(&hmm.name))?;
    if flags & P7H_ACC != 0 { write_bin_string(w, hmm.acc.as_deref())?; }
    if flags & P7H_DESC != 0 { write_bin_string(w, hmm.desc.as_deref())?; }
    if flags & P7H_RF != 0 { write_bin_annot(w, &hmm.rf, m)?; }
    if flags & P7H_MMASK != 0 { write_bin_annot(w, &hmm.mm, m)?; }
    if flags & P7H_CONS != 0 { write_bin_annot(w, &hmm.consensus, m)?; }
    if flags & P7H_CS != 0 { write_bin_annot(w, &hmm.cs, m)?; }
    // P7H_CA (surface accessibility) is never present in CM filter HMMs and has
    // no field in P7Profile; C would write hmm->ca here (p7_hmmfile.c:1041).
    write_bin_string(w, hmm.comlog.as_deref())?;
    w.write_all(&hmm.nseq.to_le_bytes())?;
    w.write_all(&hmm.eff_nseq.to_le_bytes())?;
    w.write_all(&hmm.max_length.to_le_bytes())?; // format >= 3c
    write_bin_string(w, hmm.ctime.as_deref())?;
    if flags & P7H_MAP != 0 {
        for kk in 0..=m { w.write_all(&hmm.map[kk].to_le_bytes())?; }
    }
    w.write_all(&hmm.checksum.to_le_bytes())?;

    // E-value parameters (p7_NEVPARAM=6 floats, non-3a) — p7_hmmfile.c:1061.
    // Order: MMU, MLAMBDA, VMU, VLAMBDA, FTAU, FLAMBDA (hmmer.h:72).
    let ev = [
        hmm.evparam.lmmu, hmm.evparam.lmlambda,
        hmm.evparam.lvmu, hmm.evparam.lvlambda,
        hmm.evparam.lftau, hmm.evparam.lflambda,
    ];
    for v in ev { w.write_all(&(v as f32).to_le_bytes())?; }
    // Pfam score cutoffs (p7_NCUTOFFS=6 floats, unconditional) — p7_hmmfile.c:1063.
    for x in 0..6 { w.write_all(&hmm.cutoff[x].to_le_bytes())?; }
    // Model composition — p7_hmmfile.c:1064.
    if flags & P7H_COMPO != 0 {
        for x in 0..k { w.write_all(&hmm.compo[x].to_le_bytes())?; }
    }
    Ok(())
}

// ===================== Binary p7 HMM reader =====================
// Faithful inverse of `p7_hmmfile_write_binary` (above) / C's
// p7_hmmfile_WriteBinary (p7_hmmfile.c:981). Reads the HMMER3/f (p7_HMMFILE_3f)
// binary block that trails a binary CM. Every field is consumed in the exact
// order/width/endianness the writer emits, so a write→read round trip is
// bit-exact and a C-written block (cmpress .i1m / `cmconvert -b`) reads back
// identically. See C read_bin30hmm (p7_hmmfile.c:1560-ish) for the reference.

use std::io::Read;

#[inline]
fn rd_i32<R: Read>(r: &mut R) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
#[inline]
fn rd_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
#[inline]
fn rd_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

/// Faithful port of read_bin_string (p7_hmmfile.c:2170): read an int length
/// (including trailing '\0'); if >0 read that many bytes (dropping the '\0') and
/// return the string; if 0 return None.
fn read_bin_string<R: Read>(r: &mut R) -> std::io::Result<Option<String>> {
    let len = rd_i32(r)?;
    if len > 0 {
        let mut buf = vec![0u8; len as usize];
        r.read_exact(&mut buf)?;
        // strip the trailing '\0'
        if buf.last() == Some(&0) {
            buf.pop();
        }
        Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
    } else {
        Ok(None)
    }
}

/// Read a char[M+2] annotation array (p7_hmmfile.c layout: index 0 spacer, 1..M
/// data, M+1 '\0'). Returns a Vec<u8> of length M+1 (indices 0..=M), matching
/// P7Profile's in-memory convention (data at 1..=M).
fn read_bin_annot<R: Read>(r: &mut R, m: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; m + 2];
    r.read_exact(&mut buf)?;
    Ok(buf[0..=m].to_vec())
}

/// Function: p7_hmmfile_read_binary
/// Read one HMMER3/f binary p7 HMM from `r`. Inverse of
/// `p7_hmmfile_write_binary`. On a bad magic returns an error whose kind is
/// `UnexpectedEof` if the stream is empty (caller can treat as EOF) or
/// `InvalidData` otherwise.
pub fn p7_hmmfile_read_binary<R: Read>(r: &mut R) -> std::io::Result<P7Profile> {
    // magic (v3f) — p7_hmmfile.c:1009 inverse.
    let magic = rd_u32(r)?;
    if magic != V3F_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad p7 HMM magic 0x{:08x} (expected 0x{:08x})", magic, V3F_MAGIC),
        ));
    }
    let flags = rd_i32(r)? as u32;
    let m_i = rd_i32(r)?;
    let _abc_type = rd_i32(r)?;
    let m = m_i as usize;
    let k = 4usize;

    let mut hmm = P7Profile::new(m_i);
    hmm.flags = flags;

    // Core model probabilities — p7_hmmfile.c:1025-1030 inverse.
    for kk in 1..=m {
        for x in 0..k {
            hmm.mat[kk][x] = rd_f32(r)?;
        }
    }
    for kk in 0..=m {
        for x in 0..k {
            hmm.ins[kk][x] = rd_f32(r)?;
        }
    }
    for kk in 0..=m {
        for x in 0..7 {
            hmm.trans[kk][x] = rd_f32(r)?;
        }
    }

    // Annotation section — p7_hmmfile.c:1034-1048 inverse.
    hmm.name = read_bin_string(r)?.unwrap_or_default();
    if flags & P7H_ACC != 0 {
        hmm.acc = read_bin_string(r)?;
    }
    if flags & P7H_DESC != 0 {
        hmm.desc = read_bin_string(r)?;
    }
    if flags & P7H_RF != 0 {
        hmm.rf = read_bin_annot(r, m)?;
    }
    if flags & P7H_MMASK != 0 {
        hmm.mm = read_bin_annot(r, m)?;
    }
    if flags & P7H_CONS != 0 {
        hmm.consensus = read_bin_annot(r, m)?;
    }
    if flags & P7H_CS != 0 {
        hmm.cs = read_bin_annot(r, m)?;
    }
    // P7H_CA (surface accessibility) never present in CM filter HMMs.
    hmm.comlog = read_bin_string(r)?;
    hmm.nseq = rd_i32(r)?;
    hmm.eff_nseq = rd_f32(r)?;
    hmm.max_length = rd_i32(r)?; // format >= 3c
    hmm.ctime = read_bin_string(r)?;
    if flags & P7H_MAP != 0 {
        for kk in 0..=m {
            hmm.map[kk] = rd_i32(r)?;
        }
    }
    hmm.checksum = rd_u32(r)?;

    // E-value parameters (6 floats): MMU, MLAMBDA, VMU, VLAMBDA, FTAU, FLAMBDA.
    hmm.evparam.lmmu = rd_f32(r)? as f64;
    hmm.evparam.lmlambda = rd_f32(r)? as f64;
    hmm.evparam.lvmu = rd_f32(r)? as f64;
    hmm.evparam.lvlambda = rd_f32(r)? as f64;
    hmm.evparam.lftau = rd_f32(r)? as f64;
    hmm.evparam.lflambda = rd_f32(r)? as f64;
    // Pfam score cutoffs (6 floats, unconditional).
    for x in 0..6 {
        hmm.cutoff[x] = rd_f32(r)?;
    }
    // Model composition.
    if flags & P7H_COMPO != 0 {
        for x in 0..k {
            hmm.compo[x] = rd_f32(r)?;
        }
    }
    Ok(hmm)
}

#[inline]
fn yesno(flag: u32) -> &'static str {
    if flag != 0 {
        "yes"
    } else {
        "no"
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
