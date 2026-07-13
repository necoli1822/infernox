//! CM file parser - 1:1 port from cm_file.c
//!
//! Parses Covariance Model files in Infernal 1.0.x and 1.1.x formats.

use crate::cm::{node_type_from_str, state_type_from_str, CM, CM_MAXCONNECT, PAIR_EMIT_SIZE, ALPHABET_SIZE};
use crate::constants::*;
use crate::easel::error::{InfernalError, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// CM file format version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CMFileFormat {
    /// Infernal 1.0.x format
    V1_0,
    /// Infernal 1.1.x format (binary or ASCII)
    V1_1,
}

/// Read a CM from a file in Infernal 1.0.x or 1.1.x format
///
/// # Arguments
/// * `path` - Path to the CM file
///
/// # Returns
/// * `Result<CM>` - The parsed CM or an error
pub fn cm_file_read<P: AsRef<Path>>(path: P) -> Result<CM> {
    let file = File::open(path).map_err(|_| InfernalError::Sys)?;
    let reader = BufReader::new(file);
    cm_file_read_from_reader(reader)
}

/// Read a CM from a file WITHOUT configuring local mode. This matches C's
/// `CMFileRead`, which leaves the CM globally configured (local mode is applied
/// later in `cm_Configure`, *after* the CP9 HMM is built). Use this when you need
/// the global CM — e.g. CP9 construction (`build_cp9_hmm` runs on the global CM).
pub fn cm_file_read_global<P: AsRef<Path>>(path: P) -> Result<CM> {
    let file = File::open(path).map_err(|_| InfernalError::Sys)?;
    let reader = BufReader::new(file);
    cm_file_read_from_reader_opt(reader, false)
}

/// Read a CM from a BufRead source (localizes, preserving legacy behavior).
pub fn cm_file_read_from_reader<R: BufRead>(reader: R) -> Result<CM> {
    cm_file_read_from_reader_opt(reader, true)
}

/// Read a CM from a BufRead source. `do_localize` selects whether to configure
/// local mode after reading (C's CMFileRead does NOT; cm_Configure does later).
pub fn cm_file_read_from_reader_opt<R: BufRead>(reader: R, do_localize: bool) -> Result<CM> {
    let mut lines = reader.lines();

    // Parse header
    let first_line = lines.next().ok_or(InfernalError::Eof)?.map_err(|_| InfernalError::Sys)?;
    let format = parse_format_line(&first_line)?;

    // Initialize temporary storage for header values
    let mut name = String::new();
    let mut acc: Option<String> = None;
    let mut m: i32 = 0;
    let mut nodes: i32 = 0;
    let mut clen: i32 = 0;
    let mut w: i32 = 0;
    let mut el_selfsc: f32 = 0.0;
    let mut ga: f32 = 0.0;
    let mut tc: f32 = 0.0;
    let mut nc: f32 = 0.0;
    let mut null = [0.0f32; ALPHABET_SIZE];
    let mut flags: u32 = 0;
    // E-value parameters for glocal CYK (ECMGC)
    let mut egc_lambda = 0.693;
    let mut egc_mu = 20.0;
    let mut egc_dbsize = 1_000_000.0;
    let mut egc_nrandhits: i32 = 0;
    let mut has_egc = false;
    // E-value parameters for glocal Inside (ECMGI)
    let mut egi_lambda = 0.693;
    let mut egi_mu = 20.0;
    let mut egi_dbsize = 1_000_000.0;
    let mut egi_nrandhits: i32 = 0;
    let mut has_egi = false;
    // E-value parameters for local Inside (ECMLI) - preferred for local search mode
    let mut eli_lambda = 0.693;
    let mut eli_mu = 20.0;
    let mut eli_dbsize = 1_000_000.0;
    let mut eli_nrandhits: i32 = 0;
    let mut has_eli = false;
    // E-value parameters for local CYK (ECMLC) - used by F6 CYK filter P-value
    let mut elc_lambda = 0.693;
    let mut elc_mu = 20.0;
    let mut elc_dbsize = 1_000_000.0;
    let mut elc_nrandhits: i32 = 0;
    let mut has_elc = false;
    // v1.1 specific parameters
    let mut pbegin: f32 = 0.05;
    let mut pend: f32 = 0.05;
    let mut w_beta: f64 = 1e-7;
    let mut qdb_beta1: f64 = 1e-7;
    let mut qdb_beta2: f64 = 1e-15;
    let mut n2_omega: f64 = 0.03125;
    let mut n3_omega: f64 = 0.03125;
    let mut efp7gf_tau: f64 = 0.0;
    let mut efp7gf_lambda: f64 = 0.0;
    // Header scalars the CM ASCII writer must reproduce (cm_file.c:1560-1182 region).
    let mut desc: Option<String> = None;      // DESC (cm_file.c:56)
    let mut ctime: Option<String> = None;     // DATE (cm_file.c:110)
    let mut comlog: Option<String> = None;    // COM, joined with '\n' (cm_file.c:115)
    let mut nseq: i32 = 0;                     // NSEQ (cm_file.c:168)
    let mut eff_nseq: f32 = 0.0;              // EFFN (cm_file.c:173)
    let mut checksum: u32 = 0;                 // CKSUM (cm_file.c:178)
    // C: cm->expA[EXP_NMODES], indexed [GC,GI,LC,LI]. Filled by ECMxx lines; the
    // writer emits ECMLC/GC/LI/GI from here when all four are present.
    let mut expa = [
        crate::evalue::ExpParams::default(),
        crate::evalue::ExpParams::default(),
        crate::evalue::ExpParams::default(),
        crate::evalue::ExpParams::default(),
    ];
    let mut exp_read_mask: u32 = 0; // bit per mode; 0b1111 => CMH_EXPTAIL_STATS
    // v1.0 only: PART line with >1 partition makes all E- lines invalid (they are
    // thrown away). cm_file.c read_asc_1p0_cm:2528.
    let mut evalues_invalid = false;

    // Parse header lines until MODEL: (v1.0) or CM (v1.1)
    for line_result in lines.by_ref() {
        let raw_line = line_result.map_err(|_| InfernalError::Sys)?;
        let line = raw_line.trim();

        // Model section marker differs by version
        if line.starts_with("MODEL:") || line.starts_with("CM") {
            break;
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
            "STATES" => {
                if parts.len() >= 2 {
                    m = parts[1].parse().map_err(|_| InfernalError::Format)?;
                }
            }
            "NODES" => {
                if parts.len() >= 2 {
                    nodes = parts[1].parse().map_err(|_| InfernalError::Format)?;
                }
            }
            "CLEN" => {
                if parts.len() >= 2 {
                    clen = parts[1].parse().map_err(|_| InfernalError::Format)?;
                }
            }
            "ELSELF" => {
                if parts.len() >= 2 {
                    el_selfsc = parts[1].parse().map_err(|_| InfernalError::Format)?;
                    flags |= crate::cm::CM_ELSELF;
                }
            }
            "GA" => {
                if parts.len() >= 2 {
                    ga = parts[1].parse().map_err(|_| InfernalError::Format)?;
                    flags |= crate::cm::CM_GA;
                }
            }
            "TC" => {
                if parts.len() >= 2 {
                    tc = parts[1].parse().map_err(|_| InfernalError::Format)?;
                    flags |= crate::cm::CM_TC;
                }
            }
            "NC" => {
                if parts.len() >= 2 {
                    nc = parts[1].parse().map_err(|_| InfernalError::Format)?;
                    flags |= crate::cm::CM_NC;
                }
            }
            "NULL" => {
                // Parse null model: NULL 0.000 0.000 0.000 0.000 (scores, not probs)
                // In 1.0 format, these are log-odds scores. All 0.0 means uniform.
                for i in 0..ALPHABET_SIZE.min(parts.len() - 1) {
                    let score: f32 = parts[i + 1].parse().map_err(|_| InfernalError::Format)?;
                    // Convert from score back to probability
                    // Score = 0 means probability = null (uniform = 0.25)
                    null[i] = 0.25 * (2.0_f32).powf(score);
                }
            }
            "WBETA" => {
                // W beta parameter (v1.0 format)
                if parts.len() >= 2 {
                    w_beta = parts[1].parse().unwrap_or(1e-7);
                }
                flags |= crate::cm::CM_W;
            }
            "W" => {
                // Window size (v1.1)
                if parts.len() >= 2 {
                    w = parts[1].parse().unwrap_or(0);
                    flags |= crate::cm::CM_W;
                }
            }
            "PBEGIN" => {
                if parts.len() >= 2 {
                    pbegin = parts[1].parse().unwrap_or(0.05);
                }
            }
            "PEND" => {
                if parts.len() >= 2 {
                    pend = parts[1].parse().unwrap_or(0.05);
                }
            }
            "QDBBETA1" => {
                if parts.len() >= 2 {
                    qdb_beta1 = parts[1].parse().unwrap_or(1e-7);
                }
            }
            "QDBBETA2" => {
                if parts.len() >= 2 {
                    qdb_beta2 = parts[1].parse().unwrap_or(1e-15);
                }
            }
            "N2OMEGA" => {
                if parts.len() >= 2 {
                    n2_omega = parts[1].parse().unwrap_or(0.03125);
                }
            }
            "N3OMEGA" => {
                if parts.len() >= 2 {
                    n3_omega = parts[1].parse().unwrap_or(0.03125);
                }
            }
            "EFP7GF" => {
                // EFP7GF tau lambda (cm_file.c:211). Presence implies a filter p7
                // HMM is/was set; C sets CMH_FP7 via cm_SetFilterHMM. We flag it here
                // so the writer emits the EFP7GF line (cm_file.c:645).
                if parts.len() >= 3 {
                    efp7gf_tau = parts[1].parse().unwrap_or(0.0);
                    efp7gf_lambda = parts[2].parse().unwrap_or(0.0);
                    flags |= crate::cm::CM_FP7;
                }
            }
            // v1.0 exp-tail lines (cm_file.c read_asc_1p0_cm:2540-2596). Format:
            //   E-xx <partition> <lambda> <mu_extrap> <mu_orig> <dbsize> <nrandhits> <tailp>
            // The four CM modes (LC/GC/LI/GI) go into expA[]; the four CP9 modes
            // (LV/GV/LF/GF) are ignored (`continue`). If PART reported >1 partition,
            // C throws away all E- params (evalues_are_invalid).
            "E-LC" | "E-GC" | "E-LI" | "E-GI" => {
                if !evalues_invalid && parts.len() >= 8 {
                    // C indices GC=0,GI=1,LC=2,LI=3 (infernal.h:516-519).
                    let exp_mode = match parts[0] {
                        "E-GC" => 0usize,
                        "E-GI" => 1,
                        "E-LC" => 2,
                        "E-LI" => 3,
                        _ => unreachable!(),
                    };
                    if let (Ok(lambda), Ok(mu_extrap), Ok(mu_orig), Ok(dbsize)) = (
                        parts[2].parse::<f64>(), // lambda
                        parts[3].parse::<f64>(), // mu_extrap  (ExpParams.mu)
                        parts[4].parse::<f64>(), // mu_orig
                        parts[5].parse::<f64>(), // dbsize (read as double, written as long)
                    ) {
                        let nrandhits: i32 = parts[6].parse().unwrap_or(0);
                        let tailp: f64 = parts[7].parse().unwrap_or(0.0);
                        expa[exp_mode] = crate::evalue::ExpParams {
                            lambda,
                            mu: mu_extrap,
                            dbsize,
                            nrandhits,
                            mu_orig,
                            tailp,
                        };
                        exp_read_mask |= 1 << exp_mode;
                        match parts[0] {
                            "E-GC" => { egc_lambda = lambda; egc_mu = mu_extrap; egc_dbsize = dbsize; egc_nrandhits = nrandhits; has_egc = true; }
                            "E-GI" => { egi_lambda = lambda; egi_mu = mu_extrap; egi_dbsize = dbsize; egi_nrandhits = nrandhits; has_egi = true; }
                            "E-LC" => { elc_lambda = lambda; elc_mu = mu_extrap; elc_dbsize = dbsize; elc_nrandhits = nrandhits; has_elc = true; }
                            "E-LI" => { eli_lambda = lambda; eli_mu = mu_extrap; eli_dbsize = dbsize; eli_nrandhits = nrandhits; has_eli = true; }
                            _ => {}
                        }
                    }
                }
            }
            // v1.0 CP9 exp-tail modes — irrelevant in the current format, skipped
            // (cm_file.c:2547-2550, `continue`).
            "E-LV" | "E-GV" | "E-LF" | "E-GF" => {}
            // v1.0 partitions (cm_file.c:2522-2538). Only 1 partition is supported;
            // more than 1 invalidates all E-value params.
            "PART" => {
                if parts.len() >= 2 {
                    if let Ok(np) = parts[1].parse::<i32>() {
                        if np != 1 {
                            evalues_invalid = true;
                        }
                    }
                }
            }
            // v1.0 build/calibrate command log (cm_file.c:2475-2504). BCOM then CCOM
            // are concatenated into cm->comlog joined by '\n' (in file order).
            "BCOM" | "CCOM" => {
                let rest = raw_line.trim().splitn(2, char::is_whitespace).nth(1);
                if let Some(r) = rest {
                    let cmd = r.trim_start();
                    match comlog {
                        None => comlog = Some(cmd.to_string()),
                        Some(ref mut s) => {
                            s.push('\n');
                            s.push_str(cmd);
                        }
                    }
                }
            }
            // v1.0 build date (cm_file.c:2487-2492). BDATE -> cm->ctime. CDATE is
            // read but discarded (cm_file.c:2505-2510).
            "BDATE" => {
                let rest = raw_line.trim().splitn(2, char::is_whitespace).nth(1);
                if let Some(r) = rest {
                    ctime = Some(r.trim_start().to_string());
                }
            }
            "CDATE" => {}
            // v1.0 effective sequence number (cm_file.c:2452-2456).
            "EFFNSEQ" => {
                if parts.len() >= 2 {
                    eff_nseq = parts[1].parse().unwrap_or(0.0);
                }
            }
            "ECMLC" | "ECMGC" | "ECMLI" | "ECMGI" => {
                // v1.1 E-value parameters (cm_file.c:219). 6 tokens follow the tag:
                // lambda mu_extrap mu_orig dbsize nrandhits tailp.
                // We use ECMLI for local Inside (default), ECMGI for glocal Inside, ECMGC for CYK
                if parts.len() >= 7 {
                    if let (Ok(lambda), Ok(mu), Ok(mu_orig), Ok(dbsize)) = (
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                        parts[4].parse::<f64>(),
                    ) {
                        let nrandhits: i32 = parts[5].parse().unwrap_or(0);
                        let tailp: f64 = parts[6].parse().unwrap_or(0.0);
                        // Store verbatim into expA[EXP_mode] (C indices GC=0,GI=1,LC=2,LI=3).
                        let exp_mode = match parts[0] {
                            "ECMGC" => 0usize,
                            "ECMGI" => 1,
                            "ECMLC" => 2,
                            "ECMLI" => 3,
                            _ => unreachable!(),
                        };
                        expa[exp_mode] = crate::evalue::ExpParams {
                            lambda,
                            mu,
                            dbsize,
                            nrandhits,
                            mu_orig,
                            tailp,
                        };
                        exp_read_mask |= 1 << exp_mode;

                        match parts[0] {
                            "ECMGC" => {
                                egc_lambda = lambda;
                                egc_mu = mu;
                                egc_dbsize = dbsize;
                                egc_nrandhits = nrandhits;
                                has_egc = true;
                            }
                            "ECMGI" => {
                                egi_lambda = lambda;
                                egi_mu = mu;
                                egi_dbsize = dbsize;
                                egi_nrandhits = nrandhits;
                                has_egi = true;
                            }
                            "ECMLI" => {
                                eli_lambda = lambda;
                                eli_mu = mu;
                                eli_dbsize = dbsize;
                                eli_nrandhits = nrandhits;
                                has_eli = true;
                            }
                            "ECMLC" => {
                                elc_lambda = lambda;
                                elc_mu = mu;
                                elc_dbsize = dbsize;
                                elc_nrandhits = nrandhits;
                                has_elc = true;
                            }
                            _ => {}
                        }
                    }
                }
            }
            "ACC" => {
                // CM accession (tblout `query accession` column).
                if parts.len() >= 2 {
                    acc = Some(parts[1].to_string());
                }
            }
            "DESC" => {
                // C: GetRemainingLine after the tag (cm_file.c:56). Preserve the
                // full remaining text (internal spaces intact); trim only leading ws.
                let rest = raw_line.trim().splitn(2, char::is_whitespace).nth(1);
                if let Some(r) = rest {
                    desc = Some(r.trim_start().to_string());
                }
            }
            "DATE" => {
                // C: GetRemainingLine after the tag (cm_file.c:110).
                let rest = raw_line.trim().splitn(2, char::is_whitespace).nth(1);
                if let Some(r) = rest {
                    ctime = Some(r.trim_start().to_string());
                }
            }
            "COM" => {
                // C: skip the "[n]" token, then GetRemainingLine; append lines with a
                // '\n' separator into a single comlog string (cm_file.c:115-124).
                // Skip the "COM" tag then the "[n]" token, preserving internal
                // spacing in the remaining command (splitn splits once each).
                let after_tag = raw_line
                    .trim()
                    .splitn(2, char::is_whitespace)
                    .nth(1)
                    .unwrap_or("")
                    .trim_start();
                let cmd_opt = after_tag.splitn(2, char::is_whitespace).nth(1);
                if let Some(cmd) = cmd_opt {
                    let cmd = cmd.trim_start();
                    match comlog {
                        None => comlog = Some(cmd.to_string()),
                        Some(ref mut s) => {
                            s.push('\n');
                            s.push_str(cmd);
                        }
                    }
                }
            }
            "RF" => {
                // C: "yes" sets CMH_RF (cm_file.c:92).
                if parts.len() >= 2 && parts[1].eq_ignore_ascii_case("yes") {
                    flags |= crate::cm::CM_RF;
                }
            }
            "CONS" => {
                // C: "yes" sets CMH_CONS (cm_file.c:98).
                if parts.len() >= 2 && parts[1].eq_ignore_ascii_case("yes") {
                    flags |= crate::cm::CM_CONS;
                }
            }
            "MAP" => {
                // C: "yes" sets CMH_MAP (cm_file.c:104).
                if parts.len() >= 2 && parts[1].eq_ignore_ascii_case("yes") {
                    flags |= crate::cm::CM_MAP;
                }
            }
            "NSEQ" => {
                // C: cm->nseq = atoi(tok1) (cm_file.c:168).
                if parts.len() >= 2 {
                    nseq = parts[1].parse().unwrap_or(0);
                }
            }
            "EFFN" => {
                // C: cm->eff_nseq = atof(tok1) (cm_file.c:173).
                if parts.len() >= 2 {
                    eff_nseq = parts[1].parse().unwrap_or(0.0);
                }
            }
            "CKSUM" => {
                // C: cm->checksum = atoll(tok1); sets CMH_CHKSUM (cm_file.c:178).
                if parts.len() >= 2 {
                    checksum = parts[1].parse().unwrap_or(0);
                    flags |= crate::cm::CM_CHKSUM;
                }
            }
            "ALPH" => {
                // Skip: RNA hardcoded (informational only for the writer).
            }
            _ => {
                // Skip unknown header fields
            }
        }
    }

    // Create CM with parsed dimensions
    let mut cm = CM::new(m, nodes);
    cm.name = name;
    cm.acc = acc;
    cm.desc = desc;
    cm.ctime = ctime;
    cm.comlog = comlog;
    cm.nseq = nseq;
    cm.eff_nseq = eff_nseq;
    cm.checksum = checksum;
    // C: CMH_EXPTAIL_STATS set iff all four ECMxx lines present (cm_file.c:268).
    if exp_read_mask == 0b1111 {
        cm.exp_by_mode = expa;
        flags |= crate::cm::CM_EXPTAIL_STATS;
    }
    cm.m = m;
    cm.nodes = nodes;
    cm.clen = clen;
    cm.w = if w > 0 { w } else { clen + 20 };  // Default: clen + some buffer
    cm.el_selfsc = el_selfsc;
    cm.flags = flags;
    cm.ga = ga;
    cm.tc = tc;
    cm.nc = nc;
    cm.null = null;
    cm.pbegin = pbegin;
    cm.pend = pend;
    cm.w_beta = w_beta;
    cm.qdb_beta1 = qdb_beta1;
    cm.qdb_beta2 = qdb_beta2;
    cm.n2_omega = n2_omega;
    cm.n3_omega = n3_omega;
    cm.efp7gf_tau = efp7gf_tau;
    cm.efp7gf_lambda = efp7gf_lambda;

    // If null was all zeros in the file (uniform log-odds), set to uniform probabilities
    if cm.null.iter().all(|&x| x == 0.0) {
        cm.null = [0.25; ALPHABET_SIZE];
    }

    // Parse states
    let mut current_node: i32 = -1;
    let mut current_node_type: i8 = 0;
    let mut node_first_state_seen = vec![false; cm.nodes as usize];
    // Per-node CONS/RF display chars captured from node header lines (the tokens
    // after `]`: map_l map_r cons_l cons_r rf_l rf_r). Placed into cm.consensus/cm.rf
    // (1-based, by emit-map lpos/rpos) after the model is read. b'\0' == absent.
    let nnodes = cm.nodes as usize;
    let mut node_consl = vec![0u8; nnodes];
    let mut node_consr = vec![0u8; nnodes];
    let mut node_rfl = vec![0u8; nnodes];
    let mut node_rfr = vec![0u8; nnodes];
    // Per-node MAP ints (tail[0], tail[1] on the node header line): map_l, map_r.
    // -1 == absent ('-'). Placed into cm.map by emit-map position after the read.
    let mut node_mapl = vec![-1i32; nnodes];
    let mut node_mapr = vec![-1i32; nnodes];

    for line_result in lines.by_ref() {
        let line = line_result.map_err(|_| InfernalError::Sys)?;
        let line = line.trim();

        // End of model marker
        if line == "//" {
            break;
        }

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Check for node header: [ ROOT    0 ]
        // Note: v1.1 format may have trailing characters like "[ ROOT    0 ]      -      - - - - -"
        if let Some(bracket_start) = line.find('[') {
            if let Some(bracket_end) = line.find(']') {
                if bracket_end > bracket_start {
                    let inner = &line[bracket_start+1..bracket_end].trim();
                    let parts: Vec<&str> = inner.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Some(nd_type) = node_type_from_str(parts[0]) {
                            current_node_type = nd_type;
                            current_node = parts[1].parse().unwrap_or(-1);
                            if current_node >= 0 && (current_node as usize) < cm.ndtype.len() {
                                cm.ndtype[current_node as usize] = current_node_type;
                                // Capture CONS/RF display chars from the trailing tokens
                                // after `]`: [map_l map_r cons_l cons_r rf_l rf_r].
                                let tail = line[bracket_end + 1..].split_whitespace()
                                    .collect::<Vec<&str>>();
                                if tail.len() >= 6 {
                                    let nd = current_node as usize;
                                    let ch = |s: &str| -> u8 {
                                        let b = s.as_bytes();
                                        if b.len() == 1 { b[0] } else { b'\0' }
                                    };
                                    // tail = [map_l map_r cons_l cons_r rf_l rf_r]
                                    let mapint = |s: &str| -> i32 {
                                        if s == "-" { -1 } else { s.parse().unwrap_or(-1) }
                                    };
                                    node_mapl[nd] = mapint(tail[0]);
                                    node_mapr[nd] = mapint(tail[1]);
                                    node_consl[nd] = ch(tail[2]);
                                    node_consr[nd] = ch(tail[3]);
                                    node_rfl[nd] = ch(tail[4]);
                                    node_rfr[nd] = ch(tail[5]);
                                }
                            }
                        }
                    }
                    continue;
                }
            }
        }

        // Parse state line
        let parts: Vec<&str> = line.split_whitespace().collect();

        // v1.0 requires 6+ cols, v1.1 requires 10+ cols (4 extra QDB columns)
        let min_cols = if format == CMFileFormat::V1_1 { 10 } else { 6 };
        if parts.len() < min_cols {
            continue;
        }

        // State type
        let st_type = match state_type_from_str(parts[0]) {
            Some(t) => t,
            None => continue,
        };

        // State index
        let v: i32 = parts[1].parse().map_err(|_| InfernalError::Format)?;
        let v_usize = v as usize;

        if v_usize >= cm.sttype.len() {
            continue;
        }

        // Columns 2 and 3 vary by format:
        // Columns 2 and 3 are plast and pnum (parent-info), stored directly in the
        // .cm file. C reads them verbatim (cm_file.c:1984); recomputing pnum from
        // cfirst/cnum mis-handles B states (cnum is the right-child index, not a
        // count), so we parse them straight from the file like C does.
        let plast_col: i32 = parts[2].parse().map_err(|_| InfernalError::Format)?;
        let pnum_col: i32 = parts[3].parse().map_err(|_| InfernalError::Format)?;

        // cfirst and cnum are at different positions based on format
        let (cfirst, cnum, data_col) = if format == CMFileFormat::V1_1 {
            // v1.1: ST V ? ? CFIRST CNUM DMIN_D DMAX_D DMIN_S DMAX_S [TSC...] [ESC...]
            let cfirst: i32 = parts[4].parse().map_err(|_| InfernalError::Format)?;
            let cnum: i32 = parts[5].parse().map_err(|_| InfernalError::Format)?;
            // QDB band columns, in file order dmin2, dmin1, dmax1, dmax2
            // (cm_file.c reader:437-455; writer:723-724). Retained for the writer.
            cm.dmin2[v_usize] = parts[6].parse().unwrap_or(0);
            cm.dmin1[v_usize] = parts[7].parse().unwrap_or(0);
            cm.dmax1[v_usize] = parts[8].parse().unwrap_or(0);
            cm.dmax2[v_usize] = parts[9].parse().unwrap_or(0);
            // Transitions/emissions start at column 10
            (cfirst, cnum, 10)
        } else {
            // v1.0: ST V ? ? CFIRST CNUM [TSC...] [ESC...]
            let cfirst: i32 = parts[4].parse().map_err(|_| InfernalError::Format)?;
            let cnum: i32 = parts[5].parse().map_err(|_| InfernalError::Format)?;
            // Transitions/emissions start at column 6
            (cfirst, cnum, 6)
        };

        // Set state info - node index comes from the node header parsing
        cm.sttype[v_usize] = st_type;
        cm.ndidx[v_usize] = current_node;
        cm.cfirst[v_usize] = cfirst;
        cm.cnum[v_usize] = cnum;
        cm.plast[v_usize] = plast_col;
        cm.pnum[v_usize] = pnum_col;

        // Set nodemap for the first state of each node (track first state seen per node)
        if current_node >= 0 && (current_node as usize) < cm.nodemap.len() {
            let nd_idx = current_node as usize;
            if !node_first_state_seen[nd_idx] {
                cm.nodemap[nd_idx] = v;
                node_first_state_seen[nd_idx] = true;
            }
        }

        // Parse transitions (starting at data_col which is 6 for v1.0, 10 for v1.1)
        let cnum_usize = cnum.max(0) as usize;
        let mut col = data_col;

        // Transitions
        for k in 0..cnum_usize.min(CM_MAXCONNECT) {
            if col < parts.len() {
                if parts[col] == "*" {
                    // Impossible transition
                    cm.tsc[v_usize][k] = IMPOSSIBLE_F32;
                    cm.t[v_usize][k] = 0.0;
                } else {
                    let tsc: f32 = parts[col].parse().unwrap_or(IMPOSSIBLE_F32);
                    cm.tsc[v_usize][k] = tsc;
                    // C: cm->t[v][x] = ascii2prob(tok, 1.) (cm_file.c:2021)
                    cm.t[v_usize][k] = ascii2prob(parts[col], 1.0);
                }
                col += 1;
            }
        }

        // Parse emissions (after transitions)
        let emits_pair = st_type as i32 == MP_ST;
        let emits_single = matches!(st_type as i32, ML_ST | MR_ST | IL_ST | IR_ST);

        let emit_count = if emits_pair {
            PAIR_EMIT_SIZE
        } else if emits_single {
            ALPHABET_SIZE
        } else {
            0
        };

        // Emissions. C: singles use ascii2prob(tok, null[x]); pairs use
        // ascii2prob(tok, null[x]*null[y]) with x=idx/K, y=idx%K (cm_file.c:2025-2039).
        let k_abc = ALPHABET_SIZE;
        for idx in 0..emit_count {
            if col < parts.len() {
                let esc: f32 = parts[col].parse().unwrap_or(IMPOSSIBLE_F32);
                cm.esc[v_usize][idx] = esc;
                let null_emit = if emits_pair {
                    cm.null[idx / k_abc] * cm.null[idx % k_abc]
                } else {
                    cm.null[idx]
                };
                cm.e[v_usize][idx] = ascii2prob(parts[col], null_emit);
                col += 1;
            }
        }

        // Handle B states (bifurcation) - they store left and right child info
        // In CM 1.0 format, for B state:
        // Column 4 (cfirst position) = left child (BEGL_S state)
        // Column 5 (cnum position) = right child (BEGR_S state)
        // Note: We preserve cfirst and cnum as stored in the file for compatibility,
        // but also set lchild/rchild which is how the DP algorithm accesses them.
        if st_type as i32 == B_ST {
            cm.lchild[v_usize] = cfirst;  // cfirst is actually left child for B_ST
            cm.rchild[v_usize] = cnum;    // cnum is actually right child for B_ST
            // Don't modify cfirst/cnum - keep them as parsed for golden test compatibility
        }
    }

    // Build cm.consensus / cm.rf (1-based [1..clen]) from the per-node CONS/RF
    // display chars captured above, placing each into its emit-map consensus
    // position. This mirrors C cm->consensus/cm->rf, read by the alidisplay
    // model/RF lines when CMH_CONS/CMH_RF are set.
    if let Some(emap) = crate::cm_emitmap::create_emit_map(&cm) {
        let clen = emap.clen as usize;
        // C keys these on the CMH_CONS / CMH_RF / CMH_MAP flags (cm_file.c:2074-2096),
        // not on a heuristic; build the arrays whenever the flag is set.
        let has_cons = (cm.flags & crate::cm::CM_CONS) != 0;
        let has_rf = (cm.flags & crate::cm::CM_RF) != 0;
        let has_map = (cm.flags & crate::cm::CM_MAP) != 0;
        if has_cons {
            cm.consensus = vec![b' '; clen + 2];
        }
        if has_rf {
            cm.rf = vec![b' '; clen + 2];
        }
        if has_map {
            // C: cm->map[0]=0, indexed 1..clen (cm_file.c:2090).
            cm.map = vec![0i32; clen + 2];
        }
        for nd in 0..nnodes {
            let ndt = cm.ndtype[nd] as i32;
            let lp = emap.lpos[nd] as usize;
            let rp = emap.rpos[nd] as usize;
            if ndt == MATP_ND || ndt == MATL_ND {
                if has_cons && node_consl[nd] != 0 {
                    cm.consensus[lp] = node_consl[nd];
                }
                if has_rf && node_rfl[nd] != 0 {
                    cm.rf[lp] = node_rfl[nd];
                }
                if has_map {
                    cm.map[lp] = node_mapl[nd];
                }
            }
            if ndt == MATP_ND || ndt == MATR_ND {
                if has_cons && node_consr[nd] != 0 {
                    cm.consensus[rp] = node_consr[nd];
                }
                if has_rf && node_rfr[nd] != 0 {
                    cm.rf[rp] = node_rfr[nd];
                }
                if has_map {
                    cm.map[rp] = node_mapr[nd];
                }
            }
        }
    }

    // plast/pnum are parsed directly from the .cm state lines above (matching C's
    // cm_file.c, which reads them verbatim). We do NOT recompute them.

    // Calculate W from the model (max hit length) only if not already set from file
    if (cm.flags & crate::cm::CM_W) == 0 {
        cm.w = calculate_w(&cm);
        if cm.w > 0 {
            cm.flags |= crate::cm::CM_W;
        }
    }

    // Set stid for each state (position within its node)
    build_stid(&mut cm);

    // v1.0 only: after reading the model, C read_asc_1p0_cm removes the sole
    // source of CM ambiguities by detaching insert states one before an END_E
    // (cm_find_and_detach_dual_inserts(cm, FALSE, TRUE)) and then renormalizes
    // (CMRenormalize). The v1.1 reader path does neither — those files are
    // already detached+normalized. (cm_file.c:2726-2740.)
    if format == CMFileFormat::V1_0 {
        crate::cm_modelmaker::cm_find_and_detach_dual_inserts(&mut cm, false, true);
        cm.cm_renormalize();
    }

    // Set E-value parameters if available
    // Store both local and glocal parameters for use based on search mode
    // Default exp_params to glocal Inside (ECMGI) for compatibility, fallback to glocal CYK (ECMGC)
    if has_egi {
        cm.exp_params = crate::evalue::ExpParams {
            lambda: egi_lambda,
            mu: egi_mu,
            dbsize: egi_dbsize,
            nrandhits: egi_nrandhits,
            ..Default::default()
        };
        cm.flags |= crate::cm::CM_EXP;
    } else if has_egc {
        cm.exp_params = crate::evalue::ExpParams {
            lambda: egc_lambda,
            mu: egc_mu,
            dbsize: egc_dbsize,
            nrandhits: egc_nrandhits,
            ..Default::default()
        };
        cm.flags |= crate::cm::CM_EXP;
    }

    // Store local Inside parameters separately (ECMLI) - used for default local search mode
    if has_eli {
        cm.exp_params_local = crate::evalue::ExpParams {
            lambda: eli_lambda,
            mu: eli_mu,
            dbsize: eli_dbsize,
            nrandhits: eli_nrandhits,
            ..Default::default()
        };
        cm.flags |= crate::cm::CM_EXP_LOCAL;
    }

    // Store local CYK parameters (ECMLC) - used by the F6 CYK filter P-value
    if has_elc {
        cm.exp_params_local_cyk = crate::evalue::ExpParams {
            lambda: elc_lambda,
            mu: elc_mu,
            dbsize: elc_dbsize,
            nrandhits: elc_nrandhits,
            ..Default::default()
        };
    }

    // Store global CYK parameters (ECMGC) - used by the nohmm+global CYK filter
    // P-value (fcyk_cm_exp_mode = EXP_CM_GC). The default reader overwrites
    // cm.exp_params with ECMGI, dropping ECMGC; keep it separately here.
    if has_egc {
        cm.exp_params_global_cyk = crate::evalue::ExpParams {
            lambda: egc_lambda,
            mu: egc_mu,
            dbsize: egc_dbsize,
            nrandhits: egc_nrandhits,
            ..Default::default()
        };
    }

    // Renormalize all probability distributions, exactly as C's CMFileRead does
    // (cm_file.c:2065 -> CMRenormalize). The stored ASCII log-odds scores are
    // rounded, so the round-tripped probabilities don't sum to 1.0; C renormalizes
    // t (skipping B states), single/pair emissions, and the null model.
    cm_renormalize(&mut cm);

    // Configure local alignment using pbegin/pend from the CM file
    // This is equivalent to C Infernal's ConfigCM/cm_localize function
    // Only do this for v1.1 format CMs that explicitly specify localization parameters.
    // C's CMFileRead does NOT localize (cm_Configure does, AFTER building the CP9);
    // callers needing the global CM use cm_file_read_global (do_localize=false).
    if do_localize && format == CMFileFormat::V1_1 {
        cm.localize(cm.pbegin, cm.pend);
    }

    // Try to parse embedded HMMER3 profile (comes after CM data)
    // The HMMER3 section starts with "HMMER3/f" or similar
    loop {
        match lines.next() {
            Some(Ok(line)) => {
                let line = line.trim();
                if line.starts_with("HMMER3") {
                    // Found HMMER3 section - parse it
                    if let Some(mut p7) = crate::p7_hmm::P7Profile::parse_hmmer3(&mut lines) {
                        // Glocal Forward mu/lambda live in the CM-level EFP7GF line
                        // (the HMMER3 STATS lines only carry LOCAL params, so
                        // parse_hmmer3 leaves gfmu/gflambda at 0). C uses these as
                        // p7_evparam[CM_p7_GFMU]/[CM_p7_GFLAMBDA] for the F4 glocal
                        // Forward P-value. EFP7GF = "mu lambda".
                        p7.evparam.gfmu = efp7gf_tau;
                        p7.evparam.gflambda = efp7gf_lambda;
                        cm.p7 = Some(p7);
                    }
                    break;
                }
                if line == "//" {
                    // Another end marker - no more data
                    break;
                }
            }
            _ => break,
        }
    }

    Ok(cm)
}

/// Parse the format line to determine CM file version
fn parse_format_line(line: &str) -> Result<CMFileFormat> {
    let line = line.trim();

    // v1.1 format: INFERNAL1/a [1.1 | April 2012] or INFERNAL1/a [1.1.5 | Sep 2023]
    if line.starts_with("INFERNAL1/a") {
        return Ok(CMFileFormat::V1_1);
    }

    // v1.0 format: INFERNAL-1 [1.0.2]
    if line.starts_with("INFERNAL-1") || line.starts_with("INFERNAL1") {
        // Check version
        if line.contains("1.0") {
            return Ok(CMFileFormat::V1_0);
        } else if line.contains("1.1") {
            return Ok(CMFileFormat::V1_1);
        }
        // Default to 1.0 for INFERNAL-1 prefix
        return Ok(CMFileFormat::V1_0);
    }
    Err(InfernalError::Format)
}

/// C: ascii2prob (cm_file.c:3352). Convert a saved log-odds score string back to
/// a probability: `(*s=='*') ? 0. : exp(atof(s)/1.44269504)*null`. Parse is f64
/// (atof), exp is f64, the truncated constant 1.44269504 (≈1/ln2) is used verbatim,
/// and the double product is finally rounded to f32 (cm->t / cm->e are float).
#[inline]
fn ascii2prob(s: &str, null: f32) -> f32 {
    if s == "*" {
        0.0
    } else {
        let x: f64 = s.parse().unwrap_or(0.0);
        ((x / 1.44269504f64).exp() * null as f64) as f32
    }
}

/// C: esl_vec_FSum (Kahan summation, esl_vectorops.c) over the first `n` entries.
#[inline]
fn esl_vec_fsum_f32(vec: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for &vi in vec {
        let y = vi - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// C: esl_vec_FNorm (esl_vectorops.c:1152) over the first `n` entries in place.
#[inline]
fn esl_vec_fnorm_f32(vec: &mut [f32]) {
    let sum = esl_vec_fsum_f32(vec);
    let n = vec.len();
    if sum != 0.0 {
        for x in vec.iter_mut() {
            *x /= sum;
        }
    } else {
        for x in vec.iter_mut() {
            *x = 1.0 / n as f32;
        }
    }
}

/// C: cm_Exponentiate (cm.c:2181). Exponentiate every emission and transition
/// probability of the CM by `z`, then renormalize. Only valid in global mode
/// (C `cm_Fail`s if CMH_LOCAL_BEGIN|END is set); cmemit calls this before any
/// local configuration, so the CM is always global here.
///
/// ```c
/// for(v = 0; v < cm->M; v++) {
///   if (cm->sttype[v] != B_st && cm->sttype[v] != E_st)
///     for (x = 0; x < cm->cnum[v]; x++) cm->t[v][x] = pow(cm->t[v][x], z);
///   if (cm->sttype[v] == MP_st)
///     for (x=0;x<K;x++) for (y=0;y<K;y++) cm->e[v][x*K+y] = pow(cm->e[v][x*K+y], z);
///   if (ML|MR|IL|IR)
///     for (x=0;x<K;x++) cm->e[v][x] = pow(cm->e[v][x], z);
/// }
/// CMRenormalize(cm);
/// ```
/// `pow` is C `double pow(double,double)`; the CM probabilities are `f32`, so we
/// compute in `f64` and store back as `f32` exactly as C's `float = pow(float,double)`.
pub fn cm_exponentiate(cm: &mut CM, z: f64) {
    let k = ALPHABET_SIZE;
    for v in 0..cm.m as usize {
        let sttype = cm.sttype[v] as i32;
        if sttype != B_ST && sttype != E_ST {
            let cnum = cm.cnum[v] as usize;
            for x in 0..cnum {
                cm.t[v][x] = (cm.t[v][x] as f64).powf(z) as f32;
            }
        }
        if sttype == MP_ST {
            for x in 0..k * k {
                cm.e[v][x] = (cm.e[v][x] as f64).powf(z) as f32;
            }
        }
        if matches!(sttype, ML_ST | MR_ST | IL_ST | IR_ST) {
            for x in 0..k {
                cm.e[v][x] = (cm.e[v][x] as f64).powf(z) as f32;
            }
        }
    }
    cm_renormalize(cm);
    // C: cm->flags &= ~CMH_BITS (invalidates log-odds scores). Our emit path
    // reads probabilities (t/e/begin/end), not the cached scores, so nothing
    // downstream depends on the (now-stale) score arrays for cmemit.
}

/// C: CMRenormalize (cm.c:344). Renormalize all probability distributions in the
/// CM after reading. Called by CMFileRead (cm_file.c:2065).
pub(crate) fn cm_renormalize(cm: &mut CM) {
    let k = ALPHABET_SIZE;
    esl_vec_fnorm_f32(&mut cm.null[..k]);
    for v in 0..cm.m as usize {
        let sttype = cm.sttype[v] as i32;
        if cm.cnum[v] > 0 && sttype != B_ST {
            let cnum = cm.cnum[v] as usize;
            esl_vec_fnorm_f32(&mut cm.t[v][..cnum]);
        }
        if matches!(sttype, ML_ST | MR_ST | IL_ST | IR_ST) {
            esl_vec_fnorm_f32(&mut cm.e[v][..k]);
        }
        if sttype == MP_ST {
            esl_vec_fnorm_f32(&mut cm.e[v][..k * k]);
        }
    }
}

/// Build parent info (plast, pnum) from child info (cfirst, cnum).
/// NOTE: superseded by direct parsing from the .cm file (C reads plast/pnum
/// verbatim). Kept for reference; recomputing here mis-handles B states.
#[allow(dead_code)]
fn build_parent_info(cm: &mut CM) {
    let m = cm.m as usize;

    // Initialize parent info
    for v in 0..m {
        cm.plast[v] = -1;
        cm.pnum[v] = 0;
    }

    // For each state v, update parent info for its children
    for v in 0..m {
        let cfirst = cm.cfirst[v];
        let cnum = cm.cnum[v];

        if cfirst < 0 || cnum <= 0 {
            continue;
        }

        for k in 0..cnum {
            let child = cfirst + k;
            if child >= 0 && (child as usize) < m {
                let child_usize = child as usize;
                cm.plast[child_usize] = v as i32;
                cm.pnum[child_usize] += 1;
            }
        }
    }

    // Also handle B states specially - their children have them as parents
    for v in 0..m {
        if cm.sttype[v] as i32 == B_ST {
            let lchild = cm.lchild[v];
            let rchild = cm.rchild[v];

            if lchild >= 0 && (lchild as usize) < m {
                // Left child's parent is this B state
                cm.plast[lchild as usize] = v as i32;
                cm.pnum[lchild as usize] = 1;
            }
            if rchild >= 0 && (rchild as usize) < m {
                // Right child's parent is this B state
                cm.plast[rchild as usize] = v as i32;
                cm.pnum[rchild as usize] = 1;
            }
        }
    }
}

/// Build stid (state position within node)
fn build_stid(cm: &mut CM) {
    let m = cm.m as usize;
    let nodes = cm.nodes as usize;

    // For each node, assign stid to its states
    for nd in 0..nodes {
        let first_state = cm.nodemap[nd];
        if first_state < 0 {
            continue;
        }

        // Find all states belonging to this node and assign unique state IDs
        let ndtype = cm.ndtype[nd] as i32;
        for v in 0..m {
            if cm.ndidx[v] == nd as i32 {
                let sttype = cm.sttype[v] as i32;
                cm.stid[v] = derive_unique_state_code(ndtype, sttype) as i8;
            }
        }
    }
}

/// Derive unique state code from node type and state type
/// This matches C Infernal's DeriveUniqueStateCode function
fn derive_unique_state_code(ndtype: i32, sttype: i32) -> i32 {
    use crate::constants::*;

    match ndtype {
        BIF_ND => match sttype {
            B_ST => BIF_B,
            _ => -1,
        },
        MATP_ND => match sttype {
            D_ST => MATP_D,
            MP_ST => MATP_MP,
            ML_ST => MATP_ML,
            MR_ST => MATP_MR,
            IL_ST => MATP_IL,
            IR_ST => MATP_IR,
            _ => -1,
        },
        MATL_ND => match sttype {
            D_ST => MATL_D,
            ML_ST => MATL_ML,
            IL_ST => MATL_IL,
            _ => -1,
        },
        MATR_ND => match sttype {
            D_ST => MATR_D,
            MR_ST => MATR_MR,
            IR_ST => MATR_IR,
            _ => -1,
        },
        BEGL_ND => match sttype {
            S_ST => BEGL_S,
            _ => -1,
        },
        BEGR_ND => match sttype {
            S_ST => BEGR_S,
            IL_ST => BEGR_IL,
            _ => -1,
        },
        ROOT_ND => match sttype {
            S_ST => ROOT_S,
            IL_ST => ROOT_IL,
            IR_ST => ROOT_IR,
            _ => -1,
        },
        END_ND => match sttype {
            E_ST => END_E,
            _ => -1,
        },
        _ => -1,
    }
}

/// Calculate W (max hit length) from the model structure
fn calculate_w(cm: &CM) -> i32 {
    // Simple heuristic: W = 2.5 * clen for typical tRNA-like structures
    // A more accurate calculation would use QDB or dynamic programming
    // For now, use a reasonable estimate based on clen
    let clen = cm.clen;

    // tRNA models typically have W around 2.5-3x consensus length
    // The exact W depends on the model's insert state probabilities
    let w = ((clen as f64) * 2.6).ceil() as i32;

    // Cap at reasonable maximum
    w.min(500).max(clen)
}

// =============================================================================
// CM ASCII writer — faithful port of cm_file_WriteASCII (cm_file.c:602-754).
//
// Scope: writes the CM section only — the `INFERNAL1/a [...]` banner through the
// terminating `//`. The trailing HMMER3/f p7 filter block (cm_file.c:758,
// p7_hmmfile_WriteASCII) is intentionally OUT OF SCOPE and not emitted here.
// =============================================================================

// Banner is legitimately tool-version dependent (C uses INFERNAL_VERSION /
// INFERNAL_DATE). infernox has no such constant; use the crate version. The
// round-trip test compares line 1 only by the `INFERNAL1/a [` prefix.
// C hardcodes its release version/date as compile-time constants (configure.ac
// PACKAGE_VERSION="1.1.5", RELEASEDATE="Sep 2023"). Byte-identical drop-in output
// requires emitting exactly those strings in the INFERNAL1/a banner, so mirror the
// C constant values rather than the infernox crate version. Matches the cmfetch
// writer's already-chosen convention.
const INFERNAL_VERSION: &str = "1.1.5";
const INFERNAL_DATE: &str = "Sep 2023";

/// C: prob2ascii (cm_file.c:3338). Format a probability for the ASCII save file:
/// "*" if p==0, else `%.3f` of `sreLOG2(p/null)` where `sreLOG2(x)=log(x)*1.44269504`.
/// `p/null` is a float division (both are f32), matching C exactly; the log is f64.
fn prob2ascii(p: f32, null: f32) -> String {
    if p == 0.0 {
        return "*".to_string();
    }
    let x = ((p / null) as f64).ln() * 1.44269504_f64;
    format!("{:.3}", x)
}

/// Strip trailing zeros (and a trailing decimal point) from a fixed/`%f`-style
/// number, mimicking C `printf` `%g`'s removal of insignificant trailing zeros.
fn strip_trailing_zeros(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        let t = t.trim_end_matches('.');
        t.to_string()
    } else {
        s.to_string()
    }
}

/// Faithful C `printf` `%g` (no width) with the given precision (default 6).
/// Chooses `%e` style when exp < -4 or exp >= precision, else `%f` style, then
/// strips trailing zeros. The exponent is written with a sign and >= 2 digits,
/// matching C (e.g. `1e-07`, `1.52588e-05`).
fn c_printf_g(val: f64, precision: usize) -> String {
    let prec = if precision == 0 { 1 } else { precision };
    if val == 0.0 {
        return "0".to_string();
    }
    let neg = val.is_sign_negative();
    let a = val.abs();
    // Scientific form with prec-1 fractional digits gives the rounded exponent.
    let sci = format!("{:.*e}", prec - 1, a); // e.g. "1.52588e-5"
    let (mant, exp_str) = sci.split_once('e').unwrap();
    let exp: i32 = exp_str.parse().unwrap();
    let s = if exp < -4 || exp >= prec as i32 {
        // %e style: strip trailing zeros in mantissa; sign + >=2 exponent digits.
        let m = strip_trailing_zeros(mant);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", m, sign, exp.abs())
    } else {
        // %f style with prec-1-exp fractional digits, then strip trailing zeros.
        let fprec = (prec as i32 - 1 - exp).max(0) as usize;
        let f = format!("{:.*}", fprec, a);
        strip_trailing_zeros(&f)
    };
    if neg {
        format!("-{}", s)
    } else {
        s
    }
}

/// C: multiline (cm_file.c:3596). Break `s` on '\n' and print each line prefixed
/// with `pfx` and a 1-based `[n]` counter: `"<pfx> [<n>] <line>\n"`.
fn multiline<W: std::io::Write>(w: &mut W, pfx: &str, s: &str) -> std::io::Result<()> {
    // C iterates on strchr('\n'); an equivalent split that preserves the exact
    // number of emitted lines (including a trailing empty line only if the string
    // does NOT end in '\n', matching C's do/while termination on *sptr=='\0').
    let mut nline = 1;
    let mut rest = s;
    loop {
        match rest.find('\n') {
            Some(idx) => {
                write!(w, "{} [{}] {}\n", pfx, nline, &rest[..idx])?;
                nline += 1;
                rest = &rest[idx + 1..];
                // C loop condition: continue only while *sptr != '\0'
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

/// Write a covariance model `cm` in Infernal 1.1 ASCII save-file format (CM
/// section only). Faithful port of `cm_file_WriteASCII` (cm_file.c:602), stopping
/// before the p7 filter block (cm_file.c:758), which is out of scope.
pub fn cm_file_write_ascii<W: std::io::Write>(w: &mut W, cm: &CM) -> std::io::Result<()> {
    use crate::cm::{node_type_to_str, state_type_to_str};
    let k = ALPHABET_SIZE;

    // C: fprintf(fp, "INFERNAL1/a [%s | %s]\n", INFERNAL_VERSION, INFERNAL_DATE);
    write!(w, "INFERNAL1/a [{} | {}]\n", INFERNAL_VERSION, INFERNAL_DATE)?;

    // C: cm_file.c:611-661 header block.
    write!(w, "NAME     {}\n", cm.name)?;
    if let Some(ref acc) = cm.acc {
        write!(w, "ACC      {}\n", acc)?;
    }
    if let Some(ref desc) = cm.desc {
        write!(w, "DESC     {}\n", desc)?;
    }
    write!(w, "STATES   {}\n", cm.m)?;
    write!(w, "NODES    {}\n", cm.nodes)?;
    write!(w, "CLEN     {}\n", cm.clen)?;
    write!(w, "W        {}\n", cm.w)?;
    // C: esl_abc_DecodeType(cm->abc->type) — the model is RNA.
    write!(w, "ALPH     {}\n", "RNA")?;
    write!(w, "RF       {}\n", if cm.flags & crate::cm::CM_RF != 0 { "yes" } else { "no" })?;
    write!(w, "CONS     {}\n", if cm.flags & crate::cm::CM_CONS != 0 { "yes" } else { "no" })?;
    write!(w, "MAP      {}\n", if cm.flags & crate::cm::CM_MAP != 0 { "yes" } else { "no" })?;
    if let Some(ref ctime) = cm.ctime {
        write!(w, "DATE     {}\n", ctime)?;
    }
    if let Some(ref comlog) = cm.comlog {
        multiline(w, "COM     ", comlog)?;
    }
    write!(w, "PBEGIN   {}\n", c_printf_g(cm.pbegin as f64, 6))?;
    write!(w, "PEND     {}\n", c_printf_g(cm.pend as f64, 6))?;
    write!(w, "WBETA    {}\n", c_printf_g(cm.w_beta, 6))?;
    write!(w, "QDBBETA1 {}\n", c_printf_g(cm.qdb_beta1, 6))?;
    write!(w, "QDBBETA2 {}\n", c_printf_g(cm.qdb_beta2, 6))?;
    write!(w, "N2OMEGA  {:>6}\n", c_printf_g(cm.n2_omega, 6))?;
    write!(w, "N3OMEGA  {:>6}\n", c_printf_g(cm.n3_omega, 6))?;
    write!(w, "ELSELF   {:.8}\n", cm.el_selfsc as f64)?;
    write!(w, "NSEQ     {}\n", cm.nseq)?;
    write!(w, "EFFN     {:.6}\n", cm.eff_nseq as f64)?;
    if cm.flags & crate::cm::CM_CHKSUM != 0 {
        write!(w, "CKSUM    {}\n", cm.checksum)?;
    }
    // NULL line: "NULL    " then "%6s " for each residue, prob2ascii(null[x], 1/K).
    write!(w, "NULL    ")?;
    for x in 0..k {
        write!(w, "{:>6} ", prob2ascii(cm.null[x], 1.0 / (k as f32)))?;
    }
    write!(w, "\n")?;
    if cm.flags & crate::cm::CM_GA != 0 {
        write!(w, "GA       {:.2}\n", cm.ga as f64)?;
    }
    if cm.flags & crate::cm::CM_TC != 0 {
        write!(w, "TC       {:.2}\n", cm.tc as f64)?;
    }
    if cm.flags & crate::cm::CM_NC != 0 {
        write!(w, "NC       {:.2}\n", cm.nc as f64)?;
    }
    if cm.flags & crate::cm::CM_FP7 != 0 {
        // C: fp7_evparam[CM_p7_GFMU], [CM_p7_GFLAMBDA] (cm_file.c:645).
        write!(w, "EFP7GF   {:.4} {:.5}\n", cm.efp7gf_tau, cm.efp7gf_lambda)?;
    }
    if cm.flags & crate::cm::CM_EXPTAIL_STATS != 0 {
        // C order: ECMLC, ECMGC, ECMLI, ECMGI; expA indices LC=2,GC=0,LI=3,GI=1.
        for &(tag, idx) in &[("ECMLC", 2usize), ("ECMGC", 0), ("ECMLI", 3), ("ECMGI", 1)] {
            let e = &cm.exp_by_mode[idx];
            write!(
                w,
                "{}    {:.5}  {:>10.5}  {:>10.5}  {:>10}  {:>10}  {:.6}\n",
                tag,
                e.lambda,
                e.mu,
                e.mu_orig,
                (e.dbsize + 0.5) as i64,
                e.nrandhits,
                e.tailp
            )?;
        }
    }

    // main model section
    write!(w, "CM\n")?;

    let emap = crate::cm_emitmap::create_emit_map(cm)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "unable to create emit map"))?;

    // Helper: read a per-consensus-position display char, defaulting to '-'.
    let cons_ch = |pos: i32| -> char {
        let p = pos as usize;
        if p < cm.consensus.len() && cm.consensus[p] != 0 {
            cm.consensus[p] as char
        } else {
            '-'
        }
    };
    let rf_ch = |pos: i32| -> char {
        let p = pos as usize;
        if p < cm.rf.len() && cm.rf[p] != 0 {
            cm.rf[p] as char
        } else {
            '-'
        }
    };

    for v in 0..cm.m as usize {
        let nd = cm.ndidx[v] as usize;

        // Node line (only at the first state of each node).
        if cm.nodemap[nd] as usize == v {
            let ndt = cm.ndtype[nd] as i32;
            write!(w, "{:>45}[ {:<4} {:>4} ]", "", node_type_to_str(cm.ndtype[nd]), nd)?;

            let lp = emap.lpos[nd];
            let rp = emap.rpos[nd];

            // MAP annotation (optional).
            if cm.flags & crate::cm::CM_MAP != 0 {
                if ndt == MATP_ND {
                    write!(w, " {:>6} {:>6}", cm.map[lp as usize], cm.map[rp as usize])?;
                } else if ndt == MATL_ND {
                    write!(w, " {:>6} {:>6}", cm.map[lp as usize], "-")?;
                } else if ndt == MATR_ND {
                    write!(w, " {:>6} {:>6}", "-", cm.map[rp as usize])?;
                } else {
                    write!(w, " {:>6} {:>6}", "-", "-")?;
                }
            } else {
                write!(w, " {:>6} {:>6}", "-", "-")?;
            }
            // Consensus sequence (mandatory).
            if ndt == MATP_ND {
                write!(w, " {} {}", cons_ch(lp), cons_ch(rp))?;
            } else if ndt == MATL_ND {
                write!(w, " {} {}", cons_ch(lp), '-')?;
            } else if ndt == MATR_ND {
                write!(w, " {} {}", '-', cons_ch(rp))?;
            } else {
                write!(w, " {} {}", '-', '-')?;
            }
            // RF annotation (optional).
            if cm.flags & crate::cm::CM_RF != 0 {
                if ndt == MATP_ND {
                    write!(w, " {} {}", rf_ch(lp), rf_ch(rp))?;
                } else if ndt == MATL_ND {
                    write!(w, " {} {}", rf_ch(lp), '-')?;
                } else if ndt == MATR_ND {
                    write!(w, " {} {}", '-', rf_ch(rp))?;
                } else {
                    write!(w, " {} {}", '-', '-')?;
                }
            } else {
                write!(w, " {} {}", '-', '-')?;
            }
            write!(w, "\n")?;
        }

        // State line: type, v, plast, pnum, cfirst, cnum, dmin2, dmin1, dmax1, dmax2.
        write!(
            w,
            "    {:>2} {:>5} {:>5} {:>1} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} ",
            state_type_to_str(cm.sttype[v]),
            v,
            cm.plast[v],
            cm.pnum[v],
            cm.cfirst[v],
            cm.cnum[v],
            cm.dmin2[v],
            cm.dmin1[v],
            cm.dmax1[v],
            cm.dmax2[v]
        )?;

        // Transitions (B states have none; pad to 6 columns of "%7s ").
        let mut x = 0usize;
        if cm.sttype[v] as i32 != B_ST {
            let cnum = cm.cnum[v] as usize;
            while x < cnum {
                write!(w, "{:>7} ", prob2ascii(cm.t[v][x], 1.0))?;
                x += 1;
            }
        }
        while x < 6 {
            write!(w, "{:>7} ", "")?;
            x += 1;
        }

        // Emissions.
        if cm.sttype[v] as i32 == MP_ST {
            for a in 0..k {
                for b in 0..k {
                    write!(
                        w,
                        "{:>6} ",
                        prob2ascii(cm.e[v][a * k + b], cm.null[a] * cm.null[b])
                    )?;
                }
            }
        } else if matches!(cm.sttype[v] as i32, ML_ST | MR_ST | IL_ST | IR_ST) {
            for a in 0..k {
                write!(w, "{:>6} ", prob2ascii(cm.e[v][a], cm.null[a]))?;
            }
        }
        write!(w, "\n")?;
    }
    write!(w, "//\n")?;

    // Print additional p7 hmm filter if any (cm_file.c:756-758):
    //   if (cm->flags & CMH_FP7 && cm->fp7 != NULL)
    //       p7_hmmfile_WriteASCII(fp, -1, cm->fp7);
    if let Some(ref p7) = cm.p7 {
        crate::p7_hmm::p7_hmmfile_write_ascii(w, p7)?;
    }

    Ok(())
}

/// C: cm_file_Write1p0ASCII (cm_file.c:3469) — write a CM in the backward
/// compatible Infernal v0.7→v1.0.2 ASCII format (`cmconvert -1`). This is a
/// stripped format: no W/PBEGIN/QDB/E-value/p7-filter blocks — only the header
/// fields the 1.0 parser understands, the NULL line, and the MODEL section
/// (probabilities rendered as log2-odds via prob2ascii).
///
/// Documented-variable line: the `INFERNAL-1 [converted from <version>]` banner
/// (tool-version dependent, exactly like the modern `INFERNAL1/a` banner).
pub fn cm_file_write_1p0_ascii<W: std::io::Write>(w: &mut W, cm: &CM) -> std::io::Result<()> {
    use crate::cm::{node_type_to_str, state_type_to_str};
    // B_ST/MP_ST/ML_ST/MR_ST/IL_ST/IR_ST come from `use crate::constants::*` (top).
    let k = ALPHABET_SIZE;

    // cm_file.c:3475  "INFERNAL-1 [converted from %s]\n", INFERNAL_VERSION.
    write!(w, "INFERNAL-1 [converted from {}]\n", INFERNAL_VERSION)?;

    write!(w, "NAME     {}\n", cm.name)?; // cm_file.c:3477
    if let Some(ref acc) = cm.acc {
        write!(w, "ACC      {}\n", acc)?; // :3478
    }
    if let Some(ref desc) = cm.desc {
        write!(w, "DESC     {}\n", desc)?; // :3479
    }
    // Rfam cutoffs (cm_file.c:3481-3483), "%.2f".
    if cm.flags & crate::cm::CM_GA != 0 {
        write!(w, "GA       {:.2}\n", cm.ga as f64)?;
    }
    if cm.flags & crate::cm::CM_TC != 0 {
        write!(w, "TC       {:.2}\n", cm.tc as f64)?;
    }
    if cm.flags & crate::cm::CM_NC != 0 {
        write!(w, "NC       {:.2}\n", cm.nc as f64)?;
    }
    write!(w, "STATES   {}\n", cm.m)?; // :3484
    write!(w, "NODES    {}\n", cm.nodes)?; // :3485
    // ALPHABET = cm->abc->type; infernox CMs are always eslRNA (== 1). :3486
    write!(w, "ALPHABET 1\n")?;
    write!(w, "ELSELF   {:.8}\n", cm.el_selfsc as f64)?; // :3487
    write!(w, "WBETA    {}\n", c_printf_g(cm.w_beta, 6))?; // :3488  "%g"
    write!(w, "NSEQ     {}\n", cm.nseq)?; // :3489
    write!(w, "EFFNSEQ  {:.3}\n", cm.eff_nseq as f64)?; // :3490
    write!(w, "CLEN     {}\n", cm.clen)?; // :3491

    // BCOM: cm->comlog up to the first '\n' (cm_file.c:3502-3508). BDATE: ctime.
    if let Some(ref comlog) = cm.comlog {
        let first = comlog.split('\n').next().unwrap_or("");
        write!(w, "BCOM     {}\n", first)?;
    }
    if let Some(ref ctime) = cm.ctime {
        write!(w, "BDATE    {}\n", ctime)?; // :3510
    }

    // NULL line (cm_file.c:3512-3515): "NULL    " + "%6s " prob2ascii(null[x],1/K).
    write!(w, "NULL    ")?;
    for x in 0..k {
        write!(w, "{:>6} ", prob2ascii(cm.null[x], 1.0 / (k as f32)))?;
    }
    write!(w, "\n")?;

    // Main model section (cm_file.c:3522-3556). No E-value stats in 1.0 output.
    write!(w, "MODEL:\n")?;
    for v in 0..cm.m as usize {
        let nd = cm.ndidx[v] as usize;

        // Node line at the first state of each node: "\t\t\t\t[ %-4s %4d ]".
        if cm.nodemap[nd] as usize == v {
            write!(w, "\t\t\t\t[ {:<4} {:>4} ]\n", node_type_to_str(cm.ndtype[nd]), nd)?;
        }

        // State line: "    %2s %5d %5d %1d %5d %5d " then transitions.
        write!(
            w,
            "    {:>2} {:>5} {:>5} {:>1} {:>5} {:>5} ",
            state_type_to_str(cm.sttype[v]),
            v,
            cm.plast[v],
            cm.pnum[v],
            cm.cfirst[v],
            cm.cnum[v]
        )?;
        // Transitions (B states have none); pad to 6 columns of "%7s ".
        let mut x = 0usize;
        if cm.sttype[v] as i32 != B_ST {
            let cnum = cm.cnum[v] as usize;
            while x < cnum {
                write!(w, "{:>7} ", prob2ascii(cm.t[v][x], 1.0))?;
                x += 1;
            }
        }
        while x < 6 {
            write!(w, "{:>7} ", "")?;
            x += 1;
        }
        // Emission line.
        if cm.sttype[v] as i32 == MP_ST {
            for a in 0..k {
                for b in 0..k {
                    write!(w, "{:>6} ", prob2ascii(cm.e[v][a * k + b], cm.null[a] * cm.null[b]))?;
                }
            }
        } else if matches!(cm.sttype[v] as i32, ML_ST | MR_ST | IL_ST | IR_ST) {
            for a in 0..k {
                write!(w, "{:>6} ", prob2ascii(cm.e[v][a], cm.null[a]))?;
            }
        }
        write!(w, "\n")?;
    }
    write!(w, "//\n")?;
    Ok(())
}

// v1.1 binary CM magic number (cm_file.c:47). "cm02" + 0x80808080.
const V1A_MAGIC: u32 = 0xe3edb0b2;

/// Faithful port of write_bin_string (cm_file.c:920): write an int for the
/// string length (strlen+1, including the trailing '\0'), then the string with
/// its '\0'. For a NULL/None string, write a single 0 length.
fn write_bin_string<W: std::io::Write>(w: &mut W, s: Option<&str>) -> std::io::Result<()> {
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

/// Translate infernox's internal unique-state-code (constants.rs) to C's
/// UNIQUESTATES encoding (infernal.h:223-243) for the binary `stid` array.
/// infernox swaps the values of BIF_B and END_E relative to C (C: END_E=18,
/// BIF_B=19; infernox: BIF_B=18, END_E=19); every other code (0..17, EL=20) is
/// identical. The binary file stores C's encoding, so swap those two on write.
fn stid_to_c(stid: i8) -> u8 {
    match stid as i32 {
        18 => 19, // infernox BIF_B -> C BIF_B(19)
        19 => 18, // infernox END_E -> C END_E(18)
        other => other as u8,
    }
}

/// Write a CM annotation array (rf/consensus) in the binary layout: char[clen+2]
/// with index 0 a spacer ' ', indices 1..clen the data, and index clen+1 the
/// terminating '\0' (cm_file.c:2074-2088). The in-memory array is length clen+2,
/// but its clen+1 sentinel may hold a spacer rather than '\0', so we set it here.
fn write_bin_annot<W: std::io::Write>(w: &mut W, v: &[u8], clen: usize) -> std::io::Result<()> {
    let mut buf = vec![b' '; clen + 2];
    for i in 1..=clen {
        buf[i] = if i < v.len() { v[i] } else { b' ' };
    }
    buf[0] = b' ';
    buf[clen + 1] = 0u8;
    w.write_all(&buf)
}

/// Function: cm_file_write_binary
/// Faithful byte-for-byte port of cm_file_WriteBinary (cm_file.c:787) for the
/// CM_FILE_1a format. Writes the CM in the v1.1 native little-endian (x86-64)
/// binary format, then the trailing p7 filter HMM. Every field is emitted in the
/// exact order/width/endianness of the C fwrite() calls.
pub fn cm_file_write_binary<W: std::io::Write>(w: &mut W, cm: &CM) -> std::io::Result<()> {
    // Buffer to a Vec then flush: identical bytes, and lets us reuse the
    // offset-returning core (needed by cmpress for om->offs[p7_MOFFSET]).
    let mut buf: Vec<u8> = Vec::new();
    cm_file_write_binary_vec(&mut buf, cm)?;
    w.write_all(&buf)
}

/// Like [`cm_file_write_binary`] but appends to a `Vec<u8>` and returns the
/// absolute offset within `buf` at which the trailing p7 filter HMM block begins
/// (or `buf.len()` if no fp7 is written). This is exactly C cm_file_WriteBinary's
/// `*opt_fp7_offset` (`ftello(fp)` just before `p7_hmmfile_WriteBinary`), used by
/// cmpress to fill `om->offs[p7_MOFFSET]` (cmpress.c:114).
pub fn cm_file_write_binary_vec(buf: &mut Vec<u8>, cm: &CM) -> std::io::Result<u64> {
    use crate::cm::*;
    use std::io::Write as _;
    let w = buf; // keep the body's `w.write_all(...)` calls unchanged
    let k = ALPHABET_SIZE; // abc->K = 4
    let m = cm.m as usize;
    let nodes = cm.nodes as usize;
    let clen = cm.clen as usize;

    // Translate infernox's internal flag layout (cm.rs) to C's CMH_* bit layout
    // (infernal.h:1926-1943), because the binary file stores the raw flags int.
    // cmconvert reads then writes without configuring, so only these header-derived
    // bits are ever set (verified against `cmconvert -b`: 0x11388 for the tRNA
    // fixtures = CMH_RF|CMH_CHKSUM|CMH_MAP|CMH_CONS|CMH_EXPTAIL_STATS|CMH_FP7).
    let mut cf: i32 = 0;
    if cm.acc.is_some()                 { cf |= 1 << 1; }  // CMH_ACC
    if cm.desc.is_some()                { cf |= 1 << 2; }  // CMH_DESC
    if cm.flags & CM_RF != 0            { cf |= 1 << 3; }  // CMH_RF
    if cm.flags & CM_GA != 0            { cf |= 1 << 4; }  // CMH_GA
    if cm.flags & CM_TC != 0            { cf |= 1 << 5; }  // CMH_TC
    if cm.flags & CM_NC != 0            { cf |= 1 << 6; }  // CMH_NC
    if cm.flags & CM_CHKSUM != 0        { cf |= 1 << 7; }  // CMH_CHKSUM
    if cm.flags & CM_MAP != 0           { cf |= 1 << 8; }  // CMH_MAP
    if cm.flags & CM_CONS != 0          { cf |= 1 << 9; }  // CMH_CONS
    if cm.flags & CM_EXPTAIL_STATS != 0 { cf |= 1 << 12; } // CMH_EXPTAIL_STATS
    if cm.flags & CM_FP7 != 0           { cf |= 1 << 16; } // CMH_FP7

    // Magic — cm_file.c:797.
    w.write_all(&V1A_MAGIC.to_le_bytes())?;
    // flags, M, nodes, clen, abc->type — cm_file.c:802-806.
    w.write_all(&cf.to_le_bytes())?;
    w.write_all(&cm.m.to_le_bytes())?;
    w.write_all(&cm.nodes.to_le_bytes())?;
    w.write_all(&cm.clen.to_le_bytes())?;
    w.write_all(&1i32.to_le_bytes())?; // abc->type = eslRNA (esl_alphabet.h:16)

    // Main model section — cm_file.c:811-823.
    for v in 0..m { w.write_all(&[cm.sttype[v] as u8])?; }        // sttype char[M]
    for v in 0..m { w.write_all(&cm.ndidx[v].to_le_bytes())?; }   // ndidx  int[M]
    for v in 0..m { w.write_all(&[stid_to_c(cm.stid[v])])?; }     // stid   char[M]
    for v in 0..m { w.write_all(&cm.cfirst[v].to_le_bytes())?; }  // cfirst int[M]
    for v in 0..m { w.write_all(&cm.cnum[v].to_le_bytes())?; }    // cnum   int[M]
    for v in 0..m { w.write_all(&cm.plast[v].to_le_bytes())?; }   // plast  int[M]
    for v in 0..m { w.write_all(&cm.pnum[v].to_le_bytes())?; }    // pnum   int[M]
    for nd in 0..nodes { w.write_all(&cm.nodemap[nd].to_le_bytes())?; } // nodemap int[nodes]
    for nd in 0..nodes { w.write_all(&[cm.ndtype[nd] as u8])?; }        // ndtype  char[nodes]
    for v in 0..m { w.write_all(&cm.dmin1[v].to_le_bytes())?; }   // qdbinfo->dmin1 int[M]
    for v in 0..m { w.write_all(&cm.dmax1[v].to_le_bytes())?; }   // qdbinfo->dmax1 int[M]
    for v in 0..m { w.write_all(&cm.dmin2[v].to_le_bytes())?; }   // qdbinfo->dmin2 int[M]
    for v in 0..m { w.write_all(&cm.dmax2[v].to_le_bytes())?; }   // qdbinfo->dmax2 int[M]

    // Per-state transitions/emissions — cm_file.c:825-828. Every state writes
    // MAXCONNECT (6) transition floats and K*K (16) emission floats (unused
    // entries are 0.0, matching C's contiguous e[0]/t[0] allocations).
    for v in 0..m {
        for x in 0..CM_MAXCONNECT { w.write_all(&cm.t[v][x].to_le_bytes())?; }
        for x in 0..(k * k) { w.write_all(&cm.e[v][x].to_le_bytes())?; }
    }

    // Annotation section — cm_file.c:832-841.
    write_bin_string(w, Some(&cm.name))?;
    if cm.acc.is_some()  { write_bin_string(w, cm.acc.as_deref())?; }
    if cm.desc.is_some() { write_bin_string(w, cm.desc.as_deref())?; }
    if cm.flags & CM_RF != 0   { write_bin_annot(w, &cm.rf, clen)?; }        // rf char[clen+2]
    if cm.flags & CM_CONS != 0 { write_bin_annot(w, &cm.consensus, clen)?; } // consensus char[clen+2]
    if cm.flags & CM_MAP != 0 {
        for i in 0..=clen { w.write_all(&cm.map[i].to_le_bytes())?; }        // map int[clen+1]
    }
    w.write_all(&cm.w.to_le_bytes())?; // W int

    write_bin_string(w, cm.ctime.as_deref())?;
    write_bin_string(w, cm.comlog.as_deref())?;

    // cm_file.c:843-854.
    w.write_all(&cm.pbegin.to_le_bytes())?;                 // pbegin  float
    w.write_all(&cm.pend.to_le_bytes())?;                   // pend    float
    w.write_all(&cm.w_beta.to_le_bytes())?;                 // beta_W  double
    w.write_all(&cm.qdb_beta1.to_le_bytes())?;              // qdbinfo->beta1 double
    w.write_all(&cm.qdb_beta2.to_le_bytes())?;              // qdbinfo->beta2 double
    w.write_all(&(cm.n2_omega as f32).to_le_bytes())?;      // null2_omega float
    w.write_all(&(cm.n3_omega as f32).to_le_bytes())?;      // null3_omega float
    w.write_all(&cm.el_selfsc.to_le_bytes())?;              // el_selfsc float
    w.write_all(&cm.nseq.to_le_bytes())?;                   // nseq int
    w.write_all(&cm.eff_nseq.to_le_bytes())?;               // eff_nseq float
    w.write_all(&cm.checksum.to_le_bytes())?;               // checksum uint32
    for x in 0..k { w.write_all(&cm.null[x].to_le_bytes())?; } // null float[K]

    // Rfam cutoffs — cm_file.c:858-860.
    if cm.flags & CM_GA != 0 { w.write_all(&cm.ga.to_le_bytes())?; }
    if cm.flags & CM_TC != 0 { w.write_all(&cm.tc.to_le_bytes())?; }
    if cm.flags & CM_NC != 0 { w.write_all(&cm.nc.to_le_bytes())?; }

    // E-value parameters — cm_file.c:864-885.
    if cm.flags & CM_FP7 != 0 {
        w.write_all(&(cm.efp7gf_tau as f32).to_le_bytes())?;    // fp7_evparam[CM_p7_GFMU]
        w.write_all(&(cm.efp7gf_lambda as f32).to_le_bytes())?; // fp7_evparam[CM_p7_GFLAMBDA]
    }
    if cm.flags & CM_EXPTAIL_STATS != 0 {
        // C iterates expA[z] for z in 0..EXP_NMODES; exp_by_mode uses the same
        // index order (GC=0, GI=1, LC=2, LI=3).
        for z in 0..4 {
            let e = &cm.exp_by_mode[z];
            let dbsize_long = (e.dbsize + 0.5) as i64; // (long)(dbsize + 0.5)
            w.write_all(&e.lambda.to_le_bytes())?;      // double
            w.write_all(&e.mu.to_le_bytes())?;          // mu_extrap double
            w.write_all(&e.mu_orig.to_le_bytes())?;     // double
            w.write_all(&dbsize_long.to_le_bytes())?;   // long (8 bytes on x86-64)
            w.write_all(&e.nrandhits.to_le_bytes())?;   // int
            w.write_all(&e.tailp.to_le_bytes())?;       // double
        }
    }

    // Trailing p7 filter HMM — cm_file.c:892-900: p7_hmmfile_WriteBinary(fp,-1,fp7).
    // fp7_offset = ftello(fp) captured just before the p7 block (cm_file.c:897).
    let mut fp7_offset = w.len() as u64;
    if cm.flags & CM_FP7 != 0 {
        if let Some(ref p7) = cm.p7 {
            fp7_offset = w.len() as u64;
            crate::p7_hmm::p7_hmmfile_write_binary(w, p7)?;
        }
    }
    Ok(fp7_offset)
}

// ===================== Binary CM reader =====================
// Faithful inverse of `cm_file_write_binary` (above) / C read_bin_1p1_cm
// (cm_file.c:2124). Reads one CM in the v1.1 native little-endian (x86-64)
// binary format from `r`, then its trailing p7 filter HMM. Every field is
// consumed in the exact order/width the writer emits, so write→read round trips
// bit-exact and a C-written binary CM (cmpress `.i1m`, `cmconvert -b`) reads
// back identically to the ASCII-read CM (same probabilities → same scores).

use std::io::Read as _BinRead;

#[inline]
fn brd_i32<R: std::io::Read>(r: &mut R) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
#[inline]
fn brd_u32<R: std::io::Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
#[inline]
fn brd_i64<R: std::io::Read>(r: &mut R) -> std::io::Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}
#[inline]
fn brd_f32<R: std::io::Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
#[inline]
fn brd_f64<R: std::io::Read>(r: &mut R) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
#[inline]
fn brd_i8<R: std::io::Read>(r: &mut R) -> std::io::Result<i8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0] as i8)
}

/// Faithful port of read_bin_string (cm_file.c:3311): read an int length
/// (including trailing '\0'); if >0 read that many bytes (dropping the '\0'); if
/// 0 return None.
fn brd_bin_string<R: std::io::Read>(r: &mut R) -> std::io::Result<Option<String>> {
    let len = brd_i32(r)?;
    if len > 0 {
        let mut buf = vec![0u8; len as usize];
        r.read_exact(&mut buf)?;
        if buf.last() == Some(&0) {
            buf.pop();
        }
        Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
    } else {
        Ok(None)
    }
}

/// Invert the ASCII/binary writer's `stid_to_c` remap: infernox swaps BIF_B(18)
/// and END_E(19) relative to C's UNIQUESTATES encoding (C: END_E=18, BIF_B=19).
/// The file stores C's encoding; translate back to infernox's on read.
fn c_to_stid(c: u8) -> i8 {
    match c as i32 {
        18 => 19, // C END_E(18) -> infernox END_E(19)
        19 => 18, // C BIF_B(19) -> infernox BIF_B(18)
        other => other as i8,
    }
}

// C's CMH_* header flag bits (infernal.h:1926-1943), as stored in the binary
// flags int. Mirror image of the write path's `cf` construction.
const CMH_ACC_C: i32 = 1 << 1;
const CMH_DESC_C: i32 = 1 << 2;
const CMH_RF_C: i32 = 1 << 3;
const CMH_GA_C: i32 = 1 << 4;
const CMH_TC_C: i32 = 1 << 5;
const CMH_NC_C: i32 = 1 << 6;
const CMH_CHKSUM_C: i32 = 1 << 7;
const CMH_MAP_C: i32 = 1 << 8;
const CMH_CONS_C: i32 = 1 << 9;
const CMH_EXPTAIL_STATS_C: i32 = 1 << 12;
const CMH_FP7_C: i32 = 1 << 16;

/// Function: cm_file_read_binary
/// Read ONE CM (v1.1 binary format, CM_FILE_1a) from `r`, including its trailing
/// p7 filter HMM. Faithful inverse of `cm_file_write_binary`; equivalent to C's
/// read_bin_1p1_cm with read_fp7=TRUE. The returned CM is in GLOBAL config (not
/// localized) — exactly what `cm_file_read_global` yields and what
/// `FaithfulSearcher::new` expects. Callers reading a `.i1m` database call this
/// once per CM (check the stream is not exhausted first).
pub fn cm_file_read_binary<R: std::io::Read>(r: &mut R) -> std::io::Result<CM> {
    use crate::cm::*;
    use crate::constants::B_ST;

    // Magic — cm_file.c:2151 inverse.
    let magic = brd_u32(r)?;
    if magic != V1A_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad CM magic 0x{:08x} (expected 0x{:08x})", magic, V1A_MAGIC),
        ));
    }

    // Sizes — cm_file.c:2165-2181.
    let cflags = brd_i32(r)?;
    let m_i = brd_i32(r)?;
    let nodes_i = brd_i32(r)?;
    let clen_i = brd_i32(r)?;
    let _abc_type = brd_i32(r)?;
    let m = m_i as usize;
    let nodes = nodes_i as usize;
    let clen = clen_i as usize;
    let k = ALPHABET_SIZE;

    let mut cm = CM::new(m_i, nodes_i);
    cm.m = m_i;
    cm.nodes = nodes_i;
    cm.clen = clen_i;

    // Core model architecture — cm_file.c:2196-2207 inverse.
    for v in 0..m {
        cm.sttype[v] = brd_i8(r)?;
    }
    for v in 0..m {
        cm.ndidx[v] = brd_i32(r)?;
    }
    for v in 0..m {
        cm.stid[v] = c_to_stid({
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            b[0]
        });
    }
    for v in 0..m {
        cm.cfirst[v] = brd_i32(r)?;
    }
    for v in 0..m {
        cm.cnum[v] = brd_i32(r)?;
    }
    for v in 0..m {
        cm.plast[v] = brd_i32(r)?;
    }
    for v in 0..m {
        cm.pnum[v] = brd_i32(r)?;
    }
    for nd in 0..nodes {
        cm.nodemap[nd] = brd_i32(r)?;
    }
    for nd in 0..nodes {
        cm.ndtype[nd] = brd_i8(r)?;
    }
    for v in 0..m {
        cm.dmin1[v] = brd_i32(r)?;
    }
    for v in 0..m {
        cm.dmax1[v] = brd_i32(r)?;
    }
    for v in 0..m {
        cm.dmin2[v] = brd_i32(r)?;
    }
    for v in 0..m {
        cm.dmax2[v] = brd_i32(r)?;
    }

    // For B states, mirror the ASCII reader: cfirst/cnum are the left/right child.
    for v in 0..m {
        if cm.sttype[v] as i32 == B_ST {
            cm.lchild[v] = cm.cfirst[v];
            cm.rchild[v] = cm.cnum[v];
        }
    }

    // Per-state transitions/emissions — cm_file.c:2214-2217 inverse. Probabilities
    // are exact (binary), so no renormalization (matching C read_bin_1p1_cm).
    for v in 0..m {
        for x in 0..CM_MAXCONNECT {
            cm.t[v][x] = brd_f32(r)?;
        }
        for x in 0..(k * k) {
            cm.e[v][x] = brd_f32(r)?;
        }
    }

    // Annotation section — cm_file.c:2223-2229 inverse.
    cm.name = brd_bin_string(r)?.unwrap_or_default();
    if cflags & CMH_ACC_C != 0 {
        cm.acc = brd_bin_string(r)?;
    }
    if cflags & CMH_DESC_C != 0 {
        cm.desc = brd_bin_string(r)?;
    }
    if cflags & CMH_RF_C != 0 {
        let mut buf = vec![0u8; clen + 2];
        r.read_exact(&mut buf)?;
        cm.rf = buf;
    }
    if cflags & CMH_CONS_C != 0 {
        let mut buf = vec![0u8; clen + 2];
        r.read_exact(&mut buf)?;
        cm.consensus = buf;
    }
    if cflags & CMH_MAP_C != 0 {
        let mut mp = vec![0i32; clen + 2];
        for i in 0..=clen {
            mp[i] = brd_i32(r)?;
        }
        cm.map = mp;
    }
    cm.w = brd_i32(r)?;

    cm.ctime = brd_bin_string(r)?;
    cm.comlog = brd_bin_string(r)?;

    // cm_file.c:2233-2244 inverse.
    cm.pbegin = brd_f32(r)?;
    cm.pend = brd_f32(r)?;
    cm.w_beta = brd_f64(r)?;
    cm.qdb_beta1 = brd_f64(r)?;
    cm.qdb_beta2 = brd_f64(r)?;
    cm.n2_omega = brd_f32(r)? as f64;
    cm.n3_omega = brd_f32(r)? as f64;
    cm.el_selfsc = brd_f32(r)?;
    cm.nseq = brd_i32(r)?;
    cm.eff_nseq = brd_f32(r)?;
    cm.checksum = brd_u32(r)?;
    for x in 0..k {
        cm.null[x] = brd_f32(r)?;
    }

    // Rfam cutoffs — cm_file.c:2247-2249 inverse.
    if cflags & CMH_GA_C != 0 {
        cm.ga = brd_f32(r)?;
    }
    if cflags & CMH_TC_C != 0 {
        cm.tc = brd_f32(r)?;
    }
    if cflags & CMH_NC_C != 0 {
        cm.nc = brd_f32(r)?;
    }

    // E-value parameters — cm_file.c:2252-2255 inverse.
    let mut tmp_fp7_gfmu = 0.0f32;
    let mut tmp_fp7_gflambda = 0.0f32;
    if cflags & CMH_FP7_C != 0 {
        tmp_fp7_gfmu = brd_f32(r)?;
        tmp_fp7_gflambda = brd_f32(r)?;
    }
    cm.efp7gf_tau = tmp_fp7_gfmu as f64;
    cm.efp7gf_lambda = tmp_fp7_gflambda as f64;

    if cflags & CMH_EXPTAIL_STATS_C != 0 {
        // expA[z] for z in 0..EXP_NMODES; index order [GC=0, GI=1, LC=2, LI=3].
        let mut expa = [
            crate::evalue::ExpParams::default(),
            crate::evalue::ExpParams::default(),
            crate::evalue::ExpParams::default(),
            crate::evalue::ExpParams::default(),
        ];
        for z in 0..4 {
            let lambda = brd_f64(r)?;
            let mu_extrap = brd_f64(r)?;
            let mu_orig = brd_f64(r)?;
            let dbsize_long = brd_i64(r)?;
            let nrandhits = brd_i32(r)?;
            let tailp = brd_f64(r)?;
            expa[z] = crate::evalue::ExpParams {
                lambda,
                mu: mu_extrap,
                dbsize: dbsize_long as f64,
                nrandhits,
                mu_orig,
                tailp,
            };
        }
        cm.exp_by_mode = expa.clone();
        cm.exp_params_global_cyk = expa[0].clone(); // ECMGC
        cm.exp_params = expa[1].clone(); // ECMGI (default exp_params, as ASCII reader)
        cm.exp_params_local_cyk = expa[2].clone(); // ECMLC
        cm.exp_params_local = expa[3].clone(); // ECMLI
        cm.flags |= CM_EXPTAIL_STATS | CM_EXP | CM_EXP_LOCAL;
    }

    // Reconstruct infernox internal header flags from the C flag bits.
    if cflags & CMH_RF_C != 0 {
        cm.flags |= CM_RF;
    }
    if cflags & CMH_CONS_C != 0 {
        cm.flags |= CM_CONS;
    }
    if cflags & CMH_MAP_C != 0 {
        cm.flags |= CM_MAP;
    }
    if cflags & CMH_CHKSUM_C != 0 {
        cm.flags |= CM_CHKSUM;
    }
    if cflags & CMH_GA_C != 0 {
        cm.flags |= CM_GA;
    }
    if cflags & CMH_TC_C != 0 {
        cm.flags |= CM_TC;
    }
    if cflags & CMH_NC_C != 0 {
        cm.flags |= CM_NC;
    }
    if cflags & CMH_FP7_C != 0 {
        cm.flags |= CM_FP7;
    }
    // W is always present in the binary format.
    cm.flags |= CM_W;

    // Trailing p7 filter HMM — cm_file.c:2258-2273 inverse.
    if cflags & CMH_FP7_C != 0 {
        let mut p7 = crate::p7_hmm::p7_hmmfile_read_binary(r)?;
        // C cm_SetFilterHMM: the CM-level fp7 gfmu/gflambda drive the glocal
        // Forward P-value; the searcher reads them from p7.evparam.
        p7.evparam.gfmu = tmp_fp7_gfmu as f64;
        p7.evparam.gflambda = tmp_fp7_gflambda as f64;
        cm.p7 = Some(p7);
    }

    Ok(cm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_format_line() {
        assert_eq!(parse_format_line("INFERNAL-1 [1.0.2]").unwrap(), CMFileFormat::V1_0);
        assert_eq!(parse_format_line("INFERNAL1/a [1.1.4]").unwrap(), CMFileFormat::V1_1);
        assert!(parse_format_line("HMMER3").is_err());
    }

    #[test]
    fn test_minimal_cm_parse() {
        let cm_text = r#"INFERNAL-1 [1.0.2]
NAME     test
STATES   3
NODES    2
CLEN     2
NULL     0.000  0.000  0.000  0.000
MODEL:
                                [ ROOT    0 ]
     S     0    -1 0     1     2  -1.000  -2.000
    IL     1     1 2     1     1  -1.500                          0.000  0.000  0.000  0.000
                                [ END     1 ]
     E     2     1 1    -1     0
//
"#;
        let reader = Cursor::new(cm_text);
        let cm = cm_file_read_from_reader(reader).unwrap();

        assert_eq!(cm.name, "test");
        assert_eq!(cm.m, 3);
        assert_eq!(cm.nodes, 2);
        assert_eq!(cm.clen, 2);
    }

    #[test]
    fn test_v1_1_cm_parse() {
        let cm_text = r#"INFERNAL1/a [1.1.5 | Sep 2023]
NAME     test-v1.1
ACC      RF00005
DESC     tRNA
STATES   3
NODES    2
CLEN     2
W        10
ALPH     RNA
RF       no
CONS     yes
NULL     0.000  0.000  0.000  0.000
ECMLC    0.1
ECMGC    0.2
CM
                                [ ROOT    0 ]
     S     0    -1 0     1     2    0    5    0    10  -1.000  -2.000
    IL     1     1 2     1     1    0    3    0     8  -1.500                          0.000  0.000  0.000  0.000
                                [ END     1 ]
     E     2     1 1    -1     0    0    0    0     0
//
"#;
        let reader = Cursor::new(cm_text);
        let cm = cm_file_read_from_reader(reader).unwrap();

        assert_eq!(cm.name, "test-v1.1");
        assert_eq!(cm.m, 3);
        assert_eq!(cm.nodes, 2);
        assert_eq!(cm.clen, 2);
    }

    /// Byte-for-byte round-trip of the CM section against C `cmconvert -a`.
    /// Reads the fixture with the infernox reader (global, non-localized — matching
    /// what C's CMFileRead + cmconvert write), writes it via cm_file_write_ascii,
    /// and compares against `cmconvert -a` output truncated at the first `//`
    /// (the p7 HMM block is out of scope). Line 1 (version banner) is compared by
    /// its `INFERNAL1/a [` prefix only, since the version string is tool-dependent.
    #[test]
    fn test_write_ascii_roundtrip_vs_cmconvert() {
        for fixture in [
            "/mnt/DAS/sunju/programme/trnascan-rs/data/models/TRNAinf-euk-SeC.cm",
            "/mnt/DAS/sunju/programme/trnascan-rs/data/models/TRNAinf-euk.cm",
        ] {
            check_roundtrip_against_cmconvert(fixture);
        }
    }

    fn check_roundtrip_against_cmconvert(fixture: &str) {
        use std::process::Command;

        let cmconvert = "/mnt/DAS/sunju/programme/bactars/infernal/original/src/cmconvert";

        if !std::path::Path::new(fixture).exists() || !std::path::Path::new(cmconvert).exists() {
            eprintln!("skipping {}: fixture or cmconvert binary not found", fixture);
            return;
        }

        // Read the fixture (global; do not localize) and write via our writer.
        let file = File::open(fixture).expect("open fixture");
        let cm = cm_file_read_from_reader_opt(BufReader::new(file), false).expect("read fixture");
        let mut out: Vec<u8> = Vec::new();
        cm_file_write_ascii(&mut out, &cm).expect("write ascii");
        let ours = String::from_utf8(out).expect("utf8");

        // Golden: the FULL cmconvert -a output, including the trailing HMMER3/f p7
        // filter block. No truncation — we now emit the p7 block too.
        let golden_full = Command::new(cmconvert)
            .arg("-a")
            .arg(fixture)
            .output()
            .expect("run cmconvert");
        assert!(golden_full.status.success(), "cmconvert failed");
        let golden_str = String::from_utf8(golden_full.stdout).expect("cmconvert utf8");
        let golden_lines: Vec<&str> = golden_str
            .lines()
            // cmconvert logs its own invocation as an extra COM line (e.g. "COM
            // [3] ... cmconvert ...") that is not part of the model we read; drop
            // it so the comparison reflects model content, not the tool's self-log.
            .filter(|line| !(line.starts_with("COM") && line.contains("cmconvert")))
            .collect();

        let our_lines: Vec<&str> = ours.lines().collect();

        assert_eq!(
            our_lines.len(),
            golden_lines.len(),
            "line count differs for {}: ours={} golden={}",
            fixture,
            our_lines.len(),
            golden_lines.len()
        );

        for (i, (o, g)) in our_lines.iter().zip(golden_lines.iter()).enumerate() {
            // Both the INFERNAL1/a and the embedded HMMER3/f banners carry a
            // tool-version/date stamp; compare those two lines by prefix only.
            if o.starts_with("INFERNAL1/a [") && g.starts_with("INFERNAL1/a [") {
                continue;
            }
            if o.starts_with("HMMER3/f [") && g.starts_with("HMMER3/f [") {
                continue;
            }
            assert_eq!(
                o, g,
                "first differing line at line {} (1-based) for {}:\n  ours:   {:?}\n  golden: {:?}",
                i + 1,
                fixture,
                o,
                g
            );
        }
    }

    /// Byte-for-byte parity of the binary CM format against C `cmconvert -b`.
    /// Reads the fixture with the infernox reader (global, non-localized — matching
    /// what C's CMFileRead + cmconvert operate on), writes it via
    /// `cm_file_write_binary`, and compares to the `cmconvert -b` golden bytes.
    ///
    /// The only invocation-dependent field is the CM comlog: cmconvert appends its
    /// own command line to cm->comlog (cm_AppendComlog, cm.c:2777) before writing,
    /// joined to the existing comlog with a '\n'. We replicate that exact append so
    /// the whole file — including the trailing p7 filter block — is byte-identical.
    #[test]
    fn test_write_binary_roundtrip_vs_cmconvert() {
        for fixture in [
            "/mnt/DAS/sunju/programme/trnascan-rs/data/models/TRNAinf-euk-SeC.cm",
            "/mnt/DAS/sunju/programme/trnascan-rs/data/models/TRNAinf-euk.cm",
        ] {
            check_binary_roundtrip_against_cmconvert(fixture);
        }
    }

    fn check_binary_roundtrip_against_cmconvert(fixture: &str) {
        use std::process::Command;

        let cmconvert = "/mnt/DAS/sunju/programme/bactars/infernal/original/src/cmconvert";

        if !std::path::Path::new(fixture).exists() || !std::path::Path::new(cmconvert).exists() {
            eprintln!("skipping {}: fixture or cmconvert binary not found", fixture);
            return;
        }

        // Read the fixture (global; do not localize).
        let file = File::open(fixture).expect("open fixture");
        let mut cm =
            cm_file_read_from_reader_opt(BufReader::new(file), false).expect("read fixture");

        // Replicate cm_AppendComlog(cm, argc, argv, FALSE, 0): argv is exactly the
        // command we launch below, [cmconvert, "-b", fixture], space-joined and
        // appended after a '\n' to the existing comlog.
        let appended = format!("{} -b {}", cmconvert, fixture);
        cm.comlog = Some(match cm.comlog.take() {
            Some(c) => format!("{}\n{}", c, appended),
            None => appended,
        });

        let mut ours: Vec<u8> = Vec::new();
        cm_file_write_binary(&mut ours, &cm).expect("write binary");

        let golden = Command::new(cmconvert)
            .arg("-b")
            .arg(fixture)
            .output()
            .expect("run cmconvert -b");
        assert!(golden.status.success(), "cmconvert -b failed");
        let golden = golden.stdout;

        if ours != golden {
            let first_diff = ours
                .iter()
                .zip(golden.iter())
                .position(|(a, b)| a != b);
            panic!(
                "binary output differs for {}: our_len={} golden_len={} first_diff_offset={:?}\n  ours[..diff+8]={:02x?}\n  gold[..diff+8]={:02x?}",
                fixture,
                ours.len(),
                golden.len(),
                first_diff,
                first_diff.map(|d| &ours[d..(d + 8).min(ours.len())]),
                first_diff.map(|d| &golden[d..(d + 8).min(golden.len())]),
            );
        }
    }

    /// Round-trip the binary CM reader: ASCII-read (global) → write_binary →
    /// read_binary, and assert the recovered CM equals the original in every
    /// search-relevant field (probabilities, null, exp-tail params, W, p7 filter).
    /// This is the inverse-correctness gate for `cm_file_read_binary`.
    #[test]
    fn test_read_binary_roundtrip() {
        for fixture in [
            "/mnt/DAS/sunju/programme/trnascan-rs/data/models/TRNAinf-euk-SeC.cm",
            "/mnt/DAS/sunju/programme/trnascan-rs/data/models/TRNAinf-euk.cm",
        ] {
            if !std::path::Path::new(fixture).exists() {
                eprintln!("skipping {}: fixture not found", fixture);
                continue;
            }
            let file = File::open(fixture).expect("open fixture");
            let a = cm_file_read_from_reader_opt(BufReader::new(file), false).expect("ascii read");

            let mut bytes: Vec<u8> = Vec::new();
            cm_file_write_binary(&mut bytes, &a).expect("write binary");

            let mut cur = Cursor::new(&bytes[..]);
            let b = cm_file_read_binary(&mut cur).expect("read binary");
            // The reader must consume EXACTLY the bytes the writer produced.
            assert_eq!(cur.position() as usize, bytes.len(), "trailing bytes for {}", fixture);

            assert_eq!(a.m, b.m, "M {}", fixture);
            assert_eq!(a.nodes, b.nodes, "nodes {}", fixture);
            assert_eq!(a.clen, b.clen, "clen {}", fixture);
            assert_eq!(a.w, b.w, "W {}", fixture);
            assert_eq!(a.name, b.name, "name {}", fixture);
            assert_eq!(a.sttype, b.sttype, "sttype {}", fixture);
            assert_eq!(a.stid, b.stid, "stid {}", fixture);
            assert_eq!(a.ndidx, b.ndidx, "ndidx {}", fixture);
            assert_eq!(a.cfirst, b.cfirst, "cfirst {}", fixture);
            assert_eq!(a.cnum, b.cnum, "cnum {}", fixture);
            assert_eq!(a.nodemap, b.nodemap, "nodemap {}", fixture);
            assert_eq!(a.ndtype, b.ndtype, "ndtype {}", fixture);
            assert_eq!(a.null, b.null, "null {}", fixture);
            // Probabilities: bit-exact (binary stores the exact floats).
            assert_eq!(a.t, b.t, "t {}", fixture);
            assert_eq!(a.e, b.e, "e {}", fixture);
            // Exp-tail params (drive E-values).
            assert_eq!(a.exp_params.lambda, b.exp_params.lambda, "GI lambda {}", fixture);
            assert_eq!(a.exp_params.mu, b.exp_params.mu, "GI mu {}", fixture);
            assert_eq!(a.exp_params_local.lambda, b.exp_params_local.lambda, "LI lambda {}", fixture);
            assert_eq!(a.exp_params_local.dbsize, b.exp_params_local.dbsize, "LI dbsize {}", fixture);
            assert_eq!(a.exp_params_local.nrandhits, b.exp_params_local.nrandhits, "LI nrandhits {}", fixture);
            // p7 filter HMM present + core dimensions/emissions preserved.
            let pa = a.p7.as_ref().expect("orig p7");
            let pb = b.p7.as_ref().expect("roundtrip p7");
            assert_eq!(pa.m, pb.m, "p7 M {}", fixture);
            assert_eq!(pa.mat, pb.mat, "p7 mat {}", fixture);
            assert_eq!(pa.trans, pb.trans, "p7 trans {}", fixture);
            // gfmu/gflambda are stored as f32 in the binary format (as in C's
            // fp7_evparam), so compare at f32 precision — the round-trip through
            // f32 is exactly what C cmscan sees when it reads a pressed .i1m.
            assert_eq!(pa.evparam.gfmu as f32, pb.evparam.gfmu as f32, "p7 gfmu {}", fixture);
            assert_eq!(pa.evparam.gflambda as f32, pb.evparam.gflambda as f32, "p7 gflambda {}", fixture);
        }
    }

    /// Read a C-`cmconvert -b` binary CM and assert it equals the infernox
    /// ASCII-read CM (probabilities, exp params, p7). Proves the reader inverts
    /// C's writer, not merely our own.
    #[test]
    fn test_read_binary_vs_cmconvert() {
        use std::process::Command;
        let cmconvert = "/mnt/DAS/sunju/programme/bactars/infernal/original/src/cmconvert";
        let fixture = "/mnt/DAS/sunju/programme/trnascan-rs/data/models/TRNAinf-euk.cm";
        if !std::path::Path::new(fixture).exists() || !std::path::Path::new(cmconvert).exists() {
            eprintln!("skipping test_read_binary_vs_cmconvert: fixture/cmconvert missing");
            return;
        }
        let file = File::open(fixture).expect("open fixture");
        let a = cm_file_read_from_reader_opt(BufReader::new(file), false).expect("ascii read");

        let golden = Command::new(cmconvert).arg("-b").arg(fixture).output().expect("cmconvert -b");
        assert!(golden.status.success());
        let mut cur = Cursor::new(&golden.stdout[..]);
        let b = cm_file_read_binary(&mut cur).expect("read cmconvert binary");

        assert_eq!(a.m, b.m);
        assert_eq!(a.clen, b.clen);
        assert_eq!(a.w, b.w);
        // Probabilities agree bit-for-bit: cmconvert wrote the same renormalized
        // probs our ASCII reader holds (already proven by the write test).
        assert_eq!(a.t, b.t, "t vs cmconvert");
        assert_eq!(a.e, b.e, "e vs cmconvert");
        assert_eq!(a.exp_params_local.lambda, b.exp_params_local.lambda);
        assert_eq!(a.p7.as_ref().unwrap().mat, b.p7.as_ref().unwrap().mat, "p7 mat vs cmconvert");
    }
}
