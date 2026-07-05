//! CM file parser - 1:1 port from cm_file.c
//!
//! Parses Covariance Model files in Infernal 1.0.x and 1.1.x formats.

use crate::cm::{node_type_from_str, state_type_from_str, CM, CM_MAXCONNECT, PAIR_EMIT_SIZE, ALPHABET_SIZE};
use crate::constants::*;
use easel::error::{InfernalError, Result};
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

    // Parse header lines until MODEL: (v1.0) or CM (v1.1)
    for line_result in lines.by_ref() {
        let line = line_result.map_err(|_| InfernalError::Sys)?;
        let line = line.trim();

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
                // EFP7GF tau lambda
                if parts.len() >= 3 {
                    efp7gf_tau = parts[1].parse().unwrap_or(0.0);
                    efp7gf_lambda = parts[2].parse().unwrap_or(0.0);
                }
            }
            "E-GC" | "E-LC" => {
                // E-value parameters: type partition lambda mu tau dbsize nrandhits tailp
                // We use E-GC (global CYK) as default for cmsearch
                if parts.len() >= 6 && parts[0] == "E-GC" {
                    // parts[1] = partition (0)
                    // parts[2] = lambda
                    // parts[3] = mu
                    // parts[4] = tau (threshold)
                    // parts[5] = dbsize
                    if let (Ok(lambda), Ok(mu), Ok(_tau), Ok(dbsize)) = (
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                        parts[4].parse::<f64>(),
                        parts[5].parse::<f64>(),
                    ) {
                        egc_lambda = lambda;
                        egc_mu = mu;
                        egc_dbsize = dbsize;
                        has_egc = true;
                    }
                }
            }
            "ECMLC" | "ECMGC" | "ECMLI" | "ECMGI" => {
                // v1.1 E-value parameters
                // Format: ECMXX lambda mu_extrap mu_orig dbsize nrandhits tailp
                // We use ECMLI for local Inside (default), ECMGI for glocal Inside, ECMGC for CYK
                if parts.len() >= 6 {
                    if let (Ok(lambda), Ok(mu), Ok(_mu_orig), Ok(dbsize)) = (
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                        parts[4].parse::<f64>(),
                    ) {
                        let nrandhits: i32 = if parts.len() >= 6 {
                            parts[5].parse().unwrap_or(0)
                        } else {
                            0
                        };

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
            "DESC" | "ALPH" | "RF" | "CONS" | "MAP" | "DATE" | "COM" => {
                // Skip these v1.1 headers (informational only)
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

    #[cfg(debug_assertions)]
    eprintln!("DEBUG null model: A={:.6}, C={:.6}, G={:.6}, U={:.6}",
        cm.null[0], cm.null[1], cm.null[2], cm.null[3]);

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
            // Skip QDB columns: DMIN_D(6), DMAX_D(7), DMIN_S(8), DMAX_S(9)
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
    if let Some(emap) = crate::legacy::cm_emitmap::create_emit_map(&cm) {
        let clen = emap.clen as usize;
        let has_cons = node_consl.iter().chain(node_consr.iter()).any(|&c| c != 0 && c != b'-');
        let has_rf = node_rfl.iter().chain(node_rfr.iter()).any(|&c| c != 0 && c != b'-');
        if has_cons {
            cm.consensus = vec![b' '; clen + 2];
        }
        if has_rf {
            cm.rf = vec![b' '; clen + 2];
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
            }
            if ndt == MATP_ND || ndt == MATR_ND {
                if has_cons && node_consr[nd] != 0 {
                    cm.consensus[rp] = node_consr[nd];
                }
                if has_rf && node_rfr[nd] != 0 {
                    cm.rf[rp] = node_rfr[nd];
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

    // Set E-value parameters if available
    // Store both local and glocal parameters for use based on search mode
    // Default exp_params to glocal Inside (ECMGI) for compatibility, fallback to glocal CYK (ECMGC)
    if has_egi {
        cm.exp_params = crate::evalue::ExpParams {
            lambda: egi_lambda,
            mu: egi_mu,
            dbsize: egi_dbsize,
            nrandhits: egi_nrandhits,
        };
        cm.flags |= crate::cm::CM_EXP;
    } else if has_egc {
        cm.exp_params = crate::evalue::ExpParams {
            lambda: egc_lambda,
            mu: egc_mu,
            dbsize: egc_dbsize,
            nrandhits: egc_nrandhits,
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

/// C: CMRenormalize (cm.c:344). Renormalize all probability distributions in the
/// CM after reading. Called by CMFileRead (cm_file.c:2065).
fn cm_renormalize(cm: &mut CM) {
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
}
