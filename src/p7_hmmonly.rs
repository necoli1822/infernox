// SPDX-License-Identifier: BSD-3-Clause
//! p7_hmmonly — the full `cmsearch --hmmonly` final-stage logic, self-contained
//! (STEP 4 PART A). NONE of this touches humanout's files (cm_pipeline.rs /
//! cm_alidisplay.rs / cm_tophits.rs); it produces plain result structs that PART B
//! will map onto the pipeline's CM_HIT / CM_ALIDISPLAY types + dispatch.
//!
//! Faithful ports of:
//!   * `cm_pipeline.c:pli_final_stage_hmmonly` (:3844) — per-window
//!     ForwardParser/BackwardParser + LOCAL p7 domain definition (this crate's
//!     `p7_domaindef_local`) + the nhmmer score adjustment (:3986) + exp-tail
//!     P-value + hit creation + overlap-rerun recursion.
//!   * `p7_alidisplay.c:p7_alidisplay_Create` — trace → P7_ALIDISPLAY strings.
//!   * `cm_alidisplay.c:cm_alidisplay_CreateFromP7` (:659) — P7_ALIDISPLAY → the
//!     CM_ALIDISPLAY fields (mdl column = "hmm", HMM coords).
//!   * `cm_pipeline.c` NewModel `do_hmmonly_cur` gating + the F1/F2/F3_hmmonly
//!     filter-threshold setup + `pli_hmmonly_pass_statistics`.

use crate::p7_domaindef::p7_domaindef_local;
use crate::p7_fwdback::ForwardFilter;
use crate::p7_hmm::{P7Profile, P7H_CS, P7H_RF};

// Trace state codes (p7T_*), matching p7_domaindef.
const ST_M: u8 = 3;
const ST_D: u8 = 4;
const ST_I: u8 = 5;
const ST_E: u8 = 2;
const ST_B: u8 = 7;

const LOG2: f32 = std::f32::consts::LN_2; // eslCONST_LOG2

/// RNA symbol alphabet (esl_abc order for eslRNA, Kp=18):
/// 0..3 = A C G U, 4 = gap, 5..15 = degeneracies, 16 = '*', 17 = '~'.
const SYM: &[u8; 18] = b"ACGU-RYMKSWHBVDN*~";

/// esl_abc digitize for RNA (uppercase/lowercase canonical + T→U). Returns the
/// residue code, or 15 ('N', any) for anything unrecognized.
fn digitize(c: u8) -> u8 {
    match c {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'U' | b'u' | b'T' | b't' => 3,
        b'-' | b'.' | b'_' => 4,
        _ => {
            for (i, &s) in SYM.iter().enumerate() {
                if s == c.to_ascii_uppercase() {
                    return i as u8;
                }
            }
            15
        }
    }
}

// ===========================================================================
// P7_ALIDISPLAY built from an OA trace.
// ===========================================================================

/// The subset of P7_ALIDISPLAY fields we build and that `cm_alidisplay_create_from_p7`
/// consumes. Strings are the aligned display lines (no trailing '\0').
#[derive(Clone, Debug)]
pub struct P7AliDisplay {
    pub rfline: Option<String>,
    pub csline: Option<String>,
    pub model: String,
    pub mline: String,
    pub aseq: String,
    pub ppline: Option<String>,
    pub n: usize,     // aligned length (# display columns)
    pub hmmfrom: i64, // model coord of first M
    pub hmmto: i64,   // model coord of last M
    pub m: i64,       // model length (om->M)
    pub sqfrom: i64,  // seq coord of first M
    pub sqto: i64,    // seq coord of last M
    pub l: i64,       // sq->n
}

/// C `p7_alidisplay_EncodePostProb` (p7_alidisplay.c:1083):
///   `return (p + 0.05 >= 1.0) ? '*' : (char) ((p + 0.05) * 10.0) + '0';`
/// C's `p` is a `float` but `0.05`/`1.0`/`10.0` are `double`, so the `+ 0.05`, the
/// `>= 1.0` compare, and the `* 10.0` all run in DOUBLE (the float is promoted).
/// Doing it in f32 mis-rounds the knife-edge (e.g. p=0.95f → C '9' via double vs
/// f32 '*', because 0.95f+0.05f rounds up to exactly 1.0f).
fn encode_postprob(p: f32) -> u8 {
    let p = p as f64;
    if p + 0.05 >= 1.0 {
        b'*'
    } else {
        (((p + 0.05) * 10.0) as u8) + b'0'
    }
}

/// C `p7_alidisplay_Create` (which=0, first domain). Builds the P7_ALIDISPLAY from
/// the OA `tr` (forward-order (state, k, i, pp), seq coords window-local). `p7` is
/// the filter HMM (consensus/rf/cs/emissions). `ff` supplies emission odds for the
/// `+`/match mline decision. Returns None on a corrupt (M-less) trace.
pub fn build_p7_alidisplay(
    tr: &[(u8, i32, i32, f32)],
    p7: &P7Profile,
    ff: &ForwardFilter,
    sq_dsq: &[u8],
    sq_n: i64,
) -> Option<P7AliDisplay> {
    // z1 = first M after the first B; z2 = last M before the following E.
    let n = tr.len();
    let mut z = 0usize;
    while z < n && tr[z].0 != ST_B {
        z += 1;
    }
    // first M at/after z
    let mut z1 = z;
    while z1 < n && tr[z1].0 != ST_M {
        z1 += 1;
    }
    if z1 == n {
        return None;
    }
    // find the E after z1
    let mut ze = z1;
    while ze < n && tr[ze].0 != ST_E {
        ze += 1;
    }
    // last M at/before ze
    let mut z2 = ze.min(n - 1) as i64;
    while z2 >= 0 && tr[z2 as usize].0 != ST_M {
        z2 -= 1;
    }
    if z2 < z1 as i64 {
        return None;
    }
    let z2 = z2 as usize;

    let has_rf = (p7.flags & P7H_RF) != 0 && !p7.rf.is_empty();
    let has_cs = (p7.flags & P7H_CS) != 0 && !p7.cs.is_empty();

    let mut rfline = if has_rf { Some(String::new()) } else { None };
    let mut csline = if has_cs { Some(String::new()) } else { None };
    let mut model = String::new();
    let mut mline = String::new();
    let mut aseq = String::new();
    let mut ppline = String::new(); // trace always carries pp

    for z in z1..=z2 {
        let (st, k, i, pp) = tr[z];
        let k = k as usize;
        let iu = i as usize;
        let x = sq_dsq[iu] as usize;
        // optional rf/cs
        if let Some(s) = rfline.as_mut() {
            s.push(if st == ST_I { '.' } else { p7.rf[k] as char });
        }
        if let Some(s) = csline.as_mut() {
            s.push(if st == ST_I { '.' } else { p7.cs[k] as char });
        }
        // pp
        ppline.push(if st == ST_D { '.' } else { encode_postprob(pp) as char });
        // model / mline / aseq
        match st {
            ST_M => {
                let cons = p7.consensus[k];
                model.push(cons as char);
                if x == digitize(cons) as usize {
                    mline.push(cons as char);
                } else if ff.rfv(k, x) > 1.0 {
                    mline.push('+');
                } else {
                    mline.push(' ');
                }
                aseq.push((SYM[x.min(17)] as char).to_ascii_uppercase());
            }
            ST_I => {
                model.push('.');
                mline.push(' ');
                aseq.push((SYM[x.min(17)] as char).to_ascii_lowercase());
            }
            ST_D => {
                model.push(p7.consensus[k] as char);
                mline.push(' ');
                aseq.push('-');
            }
            _ => return None,
        }
    }

    Some(P7AliDisplay {
        rfline,
        csline,
        model,
        mline,
        aseq,
        ppline: Some(ppline),
        n: z2 - z1 + 1,
        hmmfrom: tr[z1].1 as i64,
        hmmto: tr[z2].1 as i64,
        m: p7.m as i64,
        sqfrom: tr[z1].2 as i64,
        sqto: tr[z2].2 as i64,
        l: sq_n,
    })
}

// ===========================================================================
// CM_ALIDISPLAY built from a P7_ALIDISPLAY (cm_alidisplay.c:659).
// ===========================================================================

/// The CM_ALIDISPLAY fields produced by `cm_alidisplay_CreateFromP7`. Mirrors the
/// C struct; PART B maps this onto humanout's `CmAlidisplay`.
#[derive(Clone, Debug)]
pub struct CmAliDisplayP7 {
    pub sc: f32,
    pub avgpp: f32,
    pub hmmonly: bool,
    pub n: usize,     // display columns (== p7ad.N)
    pub n_el: usize,  // N + 5'/3' skipped
    pub clen: i32,
    pub sqfrom: i64,
    pub sqto: i64,
    pub cfrom_emit: i64,
    pub cto_emit: i64,
    pub cfrom_span: i64,
    pub cto_span: i64,
    pub gc: f32,
    // display lines (None → not present)
    pub rfline: Option<String>,
    pub csline: String,
    pub model: String,
    pub mline: String,
    pub aseq: String,
    pub ppline: Option<String>,
    pub aseq_el: String,
    pub rfline_el: String,
    pub ppline_el: Option<String>,
    pub cmname: String,
    pub cmacc: String,
    pub cmdesc: String,
    pub sqname: String,
    pub sqacc: String,
    pub sqdesc: String,
}

/// C `cm_alidisplay_CreateFromP7` (cm_alidisplay.c:659). `p7sc`/`p7pp` = hit
/// score / avgpp. Builds the CM display fields (mdl column = "hmm"): copies the
/// p7 display lines, prepends/appends the 5'/3' skipped match positions into the
/// `_el` lines, and computes the GC fraction over the hit residues.
#[allow(clippy::too_many_arguments)]
pub fn cm_alidisplay_create_from_p7(
    cm_name: &str,
    cm_acc: &str,
    cm_desc: &str,
    cm_clen: i32,
    sq_name: &str,
    sq_acc: &str,
    sq_desc: &str,
    sq_dsq: &[u8],
    p7sc: f32,
    p7pp: f32,
    p7ad: &P7AliDisplay,
) -> CmAliDisplayP7 {
    let len = p7ad.n;
    let n5p_skipped = (p7ad.hmmfrom - 1) as usize;
    let n3p_skipped = (p7ad.m - p7ad.hmmto) as usize;
    let len_el = len + n5p_skipped + n3p_skipped;

    // GC over the hit residues [sqfrom..=sqto]: (#C + #G) / len. Canonical dsq.
    let mut cg = 0.0f32;
    for x in p7ad.sqfrom..=p7ad.sqto {
        let code = sq_dsq[x as usize];
        if code == 1 || code == 2 {
            cg += 1.0;
        }
    }
    let gc = cg / (p7ad.sqto - p7ad.sqfrom + 1) as f32;

    // _el lines: aseq_el = n5p '-' + aseq + n3p '-'; rfline_el = n5p 'x' + model +
    // n3p 'x' (uses p7ad.model, NOT rfline); ppline_el = n5p '-' + ppline + n3p '-'.
    let mut aseq_el = String::with_capacity(len_el);
    aseq_el.push_str(&"-".repeat(n5p_skipped));
    aseq_el.push_str(&p7ad.aseq);
    aseq_el.push_str(&"-".repeat(n3p_skipped));

    let mut rfline_el = String::with_capacity(len_el);
    rfline_el.push_str(&"x".repeat(n5p_skipped));
    rfline_el.push_str(&p7ad.model);
    rfline_el.push_str(&"x".repeat(n3p_skipped));

    let ppline_el = p7ad.ppline.as_ref().map(|pp| {
        let mut s = String::with_capacity(len_el);
        s.push_str(&"-".repeat(n5p_skipped));
        s.push_str(pp);
        s.push_str(&"-".repeat(n3p_skipped));
        s
    });

    CmAliDisplayP7 {
        sc: p7sc,
        avgpp: p7pp,
        hmmonly: true,
        n: len,
        n_el: len_el,
        clen: cm_clen,
        sqfrom: p7ad.sqfrom,
        sqto: p7ad.sqto,
        cfrom_emit: p7ad.hmmfrom,
        cto_emit: p7ad.hmmto,
        cfrom_span: p7ad.hmmfrom,
        cto_span: p7ad.hmmto,
        gc,
        rfline: p7ad.rfline.clone(),
        csline: p7ad.csline.clone().unwrap_or_default(),
        model: p7ad.model.clone(),
        mline: p7ad.mline.clone(),
        aseq: p7ad.aseq.clone(),
        ppline: p7ad.ppline.clone(),
        aseq_el,
        rfline_el,
        ppline_el,
        cmname: cm_name.to_string(),
        cmacc: cm_acc.to_string(),
        cmdesc: cm_desc.to_string(),
        sqname: sq_name.to_string(),
        sqacc: sq_acc.to_string(),
        sqdesc: sq_desc.to_string(),
    }
}

// ===========================================================================
// The hmmonly final stage.
// ===========================================================================

/// Scalar pipeline parameters the hmmonly stage needs (set by PART B from the
/// CM_PIPELINE + CM p7 filter). `max_length` = om->max_length (loc_window_length);
/// `omega` = bg->omega (default 1/256); `lftau`/`lflambda` = the CM's p7 local
/// Forward exp-tail params (CM_p7_LFTAU/LFLAMBDA).
#[derive(Clone, Debug)]
pub struct HmmonlyParams {
    pub t: f32,          // pli->T reporting bit threshold
    pub max_length: i32, // om->max_length
    pub omega: f32,      // bg->omega
    pub lftau: f32,      // p7_evparam[CM_p7_LFTAU]
    pub lflambda: f32,   // p7_evparam[CM_p7_LFLAMBDA]
    pub do_null2: bool,  // pli->do_null2_hmmonly
    pub search_mode: bool, // CM_SEARCH_SEQS (name from sq) vs SCAN (name from cm)
}

/// One hmmonly hit (the CM_HIT fields the stage sets). PART B copies these into
/// the pipeline's CM_HIT + builds the tblout row.
#[derive(Clone, Debug)]
pub struct HmmonlyHit {
    pub start: i64,
    pub stop: i64,
    pub score: f32,
    pub pvalue: f64,
    pub bias: f32,
    pub hmmonly: bool,
    pub glocal: bool,
    pub name: String,
    pub acc: String,
    pub desc: String,
    pub ad: CmAliDisplayP7,
}

/// C `esl_exp_surv`: P(X >= x) = exp(-lambda (x - mu)) for x >= mu, else 1.
fn esl_exp_surv(x: f64, mu: f64, lambda: f64) -> f64 {
    let y = x - mu;
    if y >= 0.0 {
        (-lambda * y).exp()
    } else {
        1.0
    }
}

/// C `p7_FLogsum(0, v)` = log(1 + e^v) computed via the (exact here) identity;
/// HMMER uses a lookup table but for the null2 dombias the exact form is within
/// the table's tolerance. `p7_FLogsum(a,b)=log(e^a+e^b)`.
fn p7_flogsum0(v: f32) -> f32 {
    // log(e^0 + e^v) = log(1 + e^v)
    if v > 0.0 {
        v + (1.0 + (-v).exp()).ln()
    } else {
        (1.0 + v.exp()).ln()
    }
}

/// C `cm_pipeline.c:pli_final_stage_hmmonly` (:3844), long_target=FALSE. Runs the
/// per-window domain definition + scoring + hit creation over `windows` (sq
/// coords, 1-based inclusive). `ff`/`p7` are the odds profile + filter HMM.
/// Returns the surviving hits (dom_score >= T). Handles the overlap-rerun recursion.
#[allow(clippy::too_many_arguments)]
pub fn run_hmmonly_stage(
    ff: &ForwardFilter,
    p7: &P7Profile,
    cm_name: &str,
    cm_acc: &str,
    cm_desc: &str,
    cm_clen: i32,
    sq_name: &str,
    sq_acc: &str,
    sq_desc: &str,
    sq_dsq: &[u8],
    sq_n: i64,
    windows: &[(i64, i64)],
    params: &HmmonlyParams,
) -> Vec<HmmonlyHit> {
    let mut hits = Vec::new();
    if sq_n == 0 || windows.is_empty() {
        return hits;
    }
    let loc_window_length = params.max_length as f32;
    // nullsc2 (nhmmer-style), same for all windows.
    let nullsc2 = loc_window_length * (loc_window_length / (loc_window_length + 1.0)).ln()
        + (1.0 / (loc_window_length + 1.0)).ln();

    let mut rerun: Vec<(i64, i64)> = Vec::new();

    for &(ws, we) in windows {
        let wlen = (we - ws + 1) as usize;
        // Build window dsq (1-indexed, sentinels): pos p (1..wlen) = sq_dsq[ws+p-1].
        let mut wdsq = vec![255u8; wlen + 2];
        for p in 1..=wlen {
            wdsq[p] = sq_dsq[(ws as usize) + p - 1];
        }
        wdsq[0] = 255;
        wdsq[wlen + 1] = 255;

        let dd = p7_domaindef_local(ff, &wdsq, wlen, params.do_null2);
        if dd.nregions == 0 || dd.nenvelopes == 0 {
            continue;
        }

        let mut prv_sqto: i64 = 0;
        for (d, dom) in dd.domains.iter().enumerate() {
            // Build the alidisplay first (for sqfrom/sqto == iali/jali).
            let p7ad = match build_p7_alidisplay(&dom.tr, p7, ff, &wdsq, wlen as i64) {
                Some(a) => a,
                None => continue,
            };
            let ad_sqfrom = p7ad.sqfrom;
            let ad_sqto = p7ad.sqto;

            // Overlap check with previous hit → schedule a rerun window, skip.
            if d > 0 && ad_sqfrom <= prv_sqto {
                let cur_rerun_ws = prv_sqto + ws - 1 + 1;
                let cur_rerun_we = ad_sqto + ws - 1;
                if cur_rerun_ws <= cur_rerun_we {
                    rerun.push((cur_rerun_ws, cur_rerun_we));
                }
                prv_sqto = ad_sqto;
                continue;
            }

            // nhmmer score adjustment (:3986).
            let env_len = (dom.jenv - dom.ienv + 1) as f32;
            let ali_len = (ad_sqto - ad_sqfrom + 1) as f32;
            let window_len = wlen as f32;
            let mut bitscore = dom.envsc;
            bitscore -= 2.0 * (2.0 / (window_len + 2.0)).ln()
                + (env_len - ali_len) * (window_len / (window_len + 2.0)).ln();
            bitscore += 2.0 * (2.0 / (loc_window_length + 2.0)).ln();
            bitscore += (loc_window_length.max(env_len) - ali_len)
                * (loc_window_length / (loc_window_length + 2.0)).ln();

            let dom_bias = if params.do_null2 {
                p7_flogsum0(params.omega.ln() + dom.domcorrection)
            } else {
                0.0
            };
            let dom_score = (bitscore - (nullsc2 + dom_bias)) / LOG2;
            prv_sqto = ad_sqto;

            if dom_score >= params.t {
                let start = ad_sqfrom + ws - 1;
                let stop = ad_sqto + ws - 1;
                let pvalue = esl_exp_surv(dom_score as f64, params.lftau as f64, params.lflambda as f64);
                let avgpp = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);

                // GC is computed over the window dsq at window-local coords, which
                // indexes the same residues as C's full-seq dsq at sq coords (C
                // shifts ad->sqfrom/sqto to sq coords before CreateFromP7; we build
                // with window-local p7ad for the GC, then set sq coords below).
                let mut ad = cm_alidisplay_create_from_p7(
                    cm_name, cm_acc, cm_desc, cm_clen, sq_name, sq_acc, sq_desc, &wdsq,
                    dom_score, avgpp, &p7ad,
                );
                ad.sqfrom = start;
                ad.sqto = stop;

                let (name, acc, desc) = if params.search_mode {
                    (sq_name.to_string(), sq_acc.to_string(), sq_desc.to_string())
                } else {
                    (cm_name.to_string(), cm_acc.to_string(), cm_desc.to_string())
                };

                hits.push(HmmonlyHit {
                    start,
                    stop,
                    score: dom_score,
                    pvalue,
                    bias: dom_bias,
                    hmmonly: true,
                    glocal: false,
                    name,
                    acc,
                    desc,
                    ad,
                });
            }
        }
    }

    // Overlap reruns (recursive, exactly as C does).
    if !rerun.is_empty() {
        let mut more = run_hmmonly_stage(
            ff, p7, cm_name, cm_acc, cm_desc, cm_clen, sq_name, sq_acc, sq_desc, sq_dsq, sq_n,
            &rerun, params,
        );
        hits.append(&mut more);
    }

    hits
}

// ===========================================================================
// NewModel gating + filter thresholds + pass statistics.
// ===========================================================================

/// C `cm_pli_NewModel` do_hmmonly_cur gating (cm_pipeline.c:1028-1030):
///   never    → FALSE; always || cm has 0 basepairs → TRUE; else → FALSE.
pub fn newmodel_do_hmmonly_cur(
    do_hmmonly_never: bool,
    do_glocal_cm_cur: bool,
    do_hmmonly_always: bool,
    cm_nbp: i32,
) -> bool {
    if do_hmmonly_never || do_glocal_cm_cur {
        false
    } else {
        do_hmmonly_always || cm_nbp == 0
    }
}

/// C `cm_pipeline.c:613-631` hmmonly filter thresholds + null2/bias toggles.
#[derive(Clone, Debug)]
pub struct HmmonlyFilterCfg {
    pub do_max: bool,
    pub do_bias: bool,
    pub do_null2: bool,
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
}

/// C `cm_pipeline.c:613-631`. `--hmmonly` sets always; `hmm_f1/2/3` are the
/// --hmmF1/2/3 opts (default 0.02/0.02/0.0002... capped at 1). `--hmmmax` flips
/// to max mode (do_max, no bias, F1=0.3, F2=F3=1.0). --hmmnonull2 / --hmmnobias
/// clear the respective toggle.
pub fn hmmonly_filter_cfg(
    hmm_f1: f32,
    hmm_f2: f32,
    hmm_f3: f32,
    do_hmmmax: bool,
    do_hmmnonull2: bool,
    do_hmmnobias: bool,
) -> HmmonlyFilterCfg {
    let mut cfg = HmmonlyFilterCfg {
        do_max: false,
        do_bias: true,
        do_null2: true,
        f1: hmm_f1.min(1.0),
        f2: hmm_f2.min(1.0),
        f3: hmm_f3.min(1.0),
    };
    if do_hmmmax {
        cfg.do_max = true;
        cfg.do_bias = false;
        cfg.f1 = 0.3;
        cfg.f2 = 1.0;
        cfg.f3 = 1.0;
    }
    if do_hmmnonull2 {
        cfg.do_null2 = false;
    }
    if do_hmmnobias {
        cfg.do_bias = false;
    }
    cfg
}

/// Counters + config for the "Internal HMM-only pipeline statistics summary"
/// block. Fields mirror the CM_PLI_ACCT[PLI_PASS_HMM_ONLY_ANY] + pli fields the
/// C printer reads.
#[derive(Clone, Debug, Default)]
pub struct HmmonlyPassStats {
    pub do_hmmonly_always: bool,
    pub search_mode: bool, // CM_SEARCH_SEQS vs CM_SCAN_MODELS
    pub nmodels: i64,      // pli->nmodels (for match_cm_spacing)
    pub nmodels_hmmonly: i64,
    pub nnodes_hmmonly: i64,
    pub nseqs: i64,
    pub nres_searched: i64, // nres_top + nres_bot
    pub do_bias: bool,
    pub do_max: bool,
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
    pub n_past_msv: i64,
    pub pos_past_msv: i64,
    pub n_past_msvbias: i64,
    pub pos_past_msvbias: i64,
    pub n_past_vit: i64,
    pub pos_past_vit: i64,
    pub n_past_fwd: i64,
    pub pos_past_fwd: i64,
    pub n_output: i64,
    pub pos_output: i64,
}

/// C printf `%.<prec>g` (default-style general float). Chooses %e vs %f by
/// exponent, then strips trailing zeros and a trailing '.'.
fn fmt_g(x: f64, prec: usize) -> String {
    let p = if prec == 0 { 1 } else { prec };
    if x == 0.0 {
        return "0".to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    if exp >= -4 && exp < p as i32 {
        let dec = (p as i32 - 1 - exp).max(0) as usize;
        let s = format!("{:.*}", dec, x);
        strip_g(&s)
    } else {
        let s = format!("{:.*e}", p - 1, x); // Rust: "1.234e2"
        // Convert Rust exponent (e2) to C style (e+02).
        reformat_exp(&s)
    }
}
fn strip_g(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        t.trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}
fn reformat_exp(s: &str) -> String {
    if let Some(epos) = s.find(['e', 'E']) {
        let (mant, exp) = s.split_at(epos);
        let mant = strip_g(mant);
        let exp_num: i32 = exp[1..].parse().unwrap_or(0);
        format!("{}e{}{:02}", mant, if exp_num < 0 { "-" } else { "+" }, exp_num.abs())
    } else {
        s.to_string()
    }
}
fn ratio(pos: i64, nres: i64) -> f64 {
    if nres == 0 { 0.0 } else { pos as f64 / nres as f64 }
}

/// C `pli_hmmonly_pass_statistics` (cm_pipeline.c). Returns the full formatted
/// summary block (header + query/target + SSV/MSV-bias/Viterbi/Forward filter
/// lines + total reported), byte-for-byte with the C `fprintf`s.
pub fn pli_hmmonly_pass_statistics(s: &HmmonlyPassStats) -> String {
    let mut o = String::new();
    let mcs = s.nmodels > 0; // match_cm_spacing
    let nres = s.nres_searched;

    if s.do_hmmonly_always {
        o.push_str("Internal HMM-only pipeline statistics summary: (--hmmonly used)\n");
        o.push_str("---------------------------------------------------------------\n");
    } else {
        o.push_str("Internal HMM-only pipeline statistics summary: (run for model(s) with zero basepairs)\n");
        o.push_str("--------------------------------------------------------------------------------------\n");
    }

    if s.search_mode {
        o.push_str(&format!(
            "Query model(s):                            {}{:15}  ({} consensus positions)\n",
            if mcs { "       " } else { "" }, s.nmodels_hmmonly, s.nnodes_hmmonly
        ));
        o.push_str(&format!(
            "Target sequences:                          {}{:15}  ({} residues searched)\n",
            if mcs { "        " } else { "" }, s.nseqs, nres
        ));
    } else {
        let per = if s.nmodels_hmmonly != 0 { nres / s.nmodels_hmmonly } else { 0 };
        o.push_str(&format!(
            "Query sequence(s):                         {}{:15}  ({} residues searched)\n",
            if mcs { "        " } else { "" }, s.nseqs, per
        ));
        o.push_str(&format!(
            "Target model(s):                           {}{:15}  ({} consensus positions)\n",
            if mcs { "        " } else { "" }, s.nmodels_hmmonly, s.nnodes_hmmonly
        ));
    }

    let (sp2, sp1, sp5) = (
        if mcs { "  " } else { "" },
        if mcs { " " } else { "" },
        if mcs { "     " } else { "" },
    );
    o.push_str(&format!(
        "Windows {}passing {}local HMM SSV      {}filter: {:15}  ({}); expected ({})\n",
        sp2, sp1, sp5, s.n_past_msv, fmt_g(ratio(s.pos_past_msv, nres), 4), fmt_g(s.f1 as f64, 4)
    ));
    if s.do_bias {
        o.push_str(&format!(
            "Windows {}passing {}local HMM MSV {}bias filter: {:15}  ({}); expected ({})\n",
            sp2, sp1, sp5, s.n_past_msvbias, fmt_g(ratio(s.pos_past_msvbias, nres), 4), fmt_g(s.f1 as f64, 4)
        ));
    } else {
        o.push_str(&format!(
            "Windows {}passing {}local HMM MSV {}bias filter: {:15}  (off)\n",
            sp2, sp1, sp5, ""
        ));
    }
    if !s.do_max {
        o.push_str(&format!(
            "Windows {}passing {}local HMM Viterbi  {}filter: {:15}  ({}); expected ({})\n",
            sp2, sp1, sp5, s.n_past_vit, fmt_g(ratio(s.pos_past_vit, nres), 4), fmt_g(s.f2 as f64, 4)
        ));
    } else {
        o.push_str(&format!(
            "Windows {}passing {}local HMM Viterbi  {}filter: {:15}  (off)\n",
            sp2, sp1, sp5, ""
        ));
    }
    if !s.do_max {
        o.push_str(&format!(
            "Windows {}passing {}local HMM Forward  {}filter: {:15}  ({}); expected ({})\n",
            sp2, sp1, sp5, s.n_past_fwd, fmt_g(ratio(s.pos_past_fwd, nres), 4), fmt_g(s.f3 as f64, 4)
        ));
    } else {
        o.push_str(&format!(
            "Windows {}passing {}local HMM Forward  {}filter: {:15}  (off)\n",
            sp2, sp1, sp5, ""
        ));
    }
    o.push_str(&format!(
        "Total HMM hits reported:                   {}{:15}  ({})\n",
        if mcs { "        " } else { "" }, s.n_output, fmt_g(ratio(s.pos_output, nres), 4)
    ));
    o
}

/// Convenience: run the whole stage on a single window (the common case).
pub fn run_hmmonly_one_window(
    ff: &ForwardFilter,
    p7: &P7Profile,
    cm_name: &str,
    cm_clen: i32,
    sq_name: &str,
    sq_dsq: &[u8],
    sq_n: i64,
    ws: i64,
    we: i64,
    params: &HmmonlyParams,
) -> Vec<HmmonlyHit> {
    run_hmmonly_stage(
        ff, p7, cm_name, "", "", cm_clen, sq_name, "", "", sq_dsq, sq_n, &[(ws, we)], params,
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::p7_fwdback::build_forward_filter;
    use crate::p7_hmm::{P7Profile, P7H_CONS};

    fn synthetic_p7(m: usize, seed: u64) -> P7Profile {
        let mut s = seed;
        let mut rng = || {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut p7 = P7Profile::new(m as i32);
        for k in 1..=m {
            let mut e = [0.0f32; 4];
            let mut sum = 0.0f32;
            for x in 0..4 { e[x] = (0.05 + rng()) as f32; sum += e[x]; }
            for x in 0..4 { e[x] /= sum; }
            p7.mat[k] = e;
        }
        for k in 0..=m {
            if k == m { p7.trans[k] = [1.0, 0.0, 0.0, 0.75, 0.25, 0.72, 0.28]; }
            else { p7.trans[k] = [0.80, 0.12, 0.08, 0.75, 0.25, 0.72, 0.28]; }
        }
        // consensus = argmax match residue per node; enable CONS flag.
        p7.consensus = vec![0u8; m + 1];
        for k in 1..=m {
            let mut best = 0usize;
            for x in 1..4 { if p7.mat[k][x] > p7.mat[k][best] { best = x; } }
            p7.consensus[k] = SYM[best];
        }
        p7.flags |= P7H_CONS;
        p7
    }

    fn rand_dsq(l: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut v = vec![0u8; l + 2];
        v[0] = 255; v[l + 1] = 255;
        for i in 1..=l { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = (s % 4) as u8; }
        v
    }

    // Build a window, run the stage, and check the display + hit fields are sane.
    #[test]
    fn hmmonly_stage_produces_sane_hits() {
        let m = 40usize;
        let p7 = synthetic_p7(m, 0x5AA5);
        let ff = build_forward_filter(&p7);
        let l = 160usize;
        let dsq = rand_dsq(l, 0x1234);
        let params = HmmonlyParams {
            t: -1000.0, // very permissive so we exercise hit creation
            max_length: 200,
            omega: 1.0 / 256.0,
            lftau: 3.0,
            lflambda: 0.7,
            do_null2: true,
            search_mode: true,
        };
        let hits = run_hmmonly_one_window(&ff, &p7, "modelX", m as i32, "seqY", &dsq, l as i64, 1, l as i64, &params);
        for h in &hits {
            assert!(h.start >= 1 && h.stop <= l as i64 && h.start <= h.stop);
            assert!(h.score.is_finite());
            assert!(h.pvalue >= 0.0 && h.pvalue <= 1.0);
            assert!(h.ad.gc >= 0.0 && h.ad.gc <= 1.0);
            // display lines all equal length = N.
            assert_eq!(h.ad.model.chars().count(), h.ad.n);
            assert_eq!(h.ad.mline.chars().count(), h.ad.n);
            assert_eq!(h.ad.aseq.chars().count(), h.ad.n);
            if let Some(pp) = &h.ad.ppline { assert_eq!(pp.chars().count(), h.ad.n); }
            // _el lines length = N_el.
            assert_eq!(h.ad.aseq_el.chars().count(), h.ad.n_el);
            assert_eq!(h.ad.rfline_el.chars().count(), h.ad.n_el);
            // HMM coords within [1, clen].
            assert!(h.ad.cfrom_emit >= 1 && h.ad.cto_emit <= m as i64);
            assert!(h.ad.hmmonly);
        }
    }

    // encode_postprob matches C's mapping.
    #[test]
    fn postprob_encoding() {
        assert_eq!(encode_postprob(1.0), b'*');
        assert_eq!(encode_postprob(0.99), b'*');
        assert_eq!(encode_postprob(0.0), b'0');
        assert_eq!(encode_postprob(0.5), b'5'); // (0.55)*10 = 5.5 → 5 → '5'
        assert_eq!(encode_postprob(0.94), b'9'); // 0.94+0.05=0.99 → 9.9 → 9 → '9'
        // C uses DOUBLE arithmetic: 0.95f promotes to 0.94999998..., +0.05=0.99999998
        // < 1.0 → (char)(9.9999) → '9'. (In f32, 0.95f+0.05f rounds up to 1.0 → '*',
        // which diverged from C — verified with a standalone C snippet.)
        assert_eq!(encode_postprob(0.95), b'9');
        assert_eq!(encode_postprob(0.951), b'*'); // 0.951+0.05 >= 1.0 → '*'
    }

    // NewModel gating truth table.
    #[test]
    fn gating_logic() {
        assert!(!newmodel_do_hmmonly_cur(true, false, true, 0)); // never wins
        assert!(!newmodel_do_hmmonly_cur(false, true, true, 0)); // glocal wins
        assert!(newmodel_do_hmmonly_cur(false, false, true, 5)); // always
        assert!(newmodel_do_hmmonly_cur(false, false, false, 0)); // 0 bp
        assert!(!newmodel_do_hmmonly_cur(false, false, false, 3)); // has bp, not forced
    }

    // filter cfg: default vs --hmmmax.
    #[test]
    fn filter_cfg_max() {
        let d = hmmonly_filter_cfg(0.02, 0.02, 0.0002, false, false, false);
        assert!(!d.do_max && d.do_bias && d.do_null2);
        let mx = hmmonly_filter_cfg(0.02, 0.02, 0.0002, true, false, false);
        assert!(mx.do_max && !mx.do_bias && mx.f1 == 0.3 && mx.f2 == 1.0 && mx.f3 == 1.0);
        let nn = hmmonly_filter_cfg(0.02, 0.02, 0.0002, false, true, true);
        assert!(!nn.do_null2 && !nn.do_bias);
    }

    // %.4g formatting vs known C printf outputs.
    #[test]
    fn fmt_g_matches_c() {
        assert_eq!(fmt_g(0.0, 4), "0");
        assert_eq!(fmt_g(0.02, 4), "0.02");
        assert_eq!(fmt_g(0.0002, 4), "0.0002");
        assert_eq!(fmt_g(0.00002, 4), "2e-05");
        assert_eq!(fmt_g(1.0, 4), "1");
        assert_eq!(fmt_g(0.5, 4), "0.5");
        assert_eq!(fmt_g(12345.0, 4), "1.234e+04"); // 4 sig digits, exp form
        assert_eq!(fmt_g(0.3, 4), "0.3");
    }

    // Statistics block renders the expected line structure.
    #[test]
    fn pass_statistics_structure() {
        let s = HmmonlyPassStats {
            do_hmmonly_always: true,
            search_mode: true,
            nmodels: 0,
            nmodels_hmmonly: 1,
            nnodes_hmmonly: 72,
            nseqs: 1,
            nres_searched: 4600,
            do_bias: true,
            do_max: false,
            f1: 0.02, f2: 0.02, f3: 0.0002,
            n_past_msv: 3, pos_past_msv: 600,
            n_past_msvbias: 2, pos_past_msvbias: 400,
            n_past_vit: 2, pos_past_vit: 400,
            n_past_fwd: 1, pos_past_fwd: 200,
            n_output: 1, pos_output: 100,
        };
        let out = pli_hmmonly_pass_statistics(&s);
        assert!(out.contains("Internal HMM-only pipeline statistics summary: (--hmmonly used)\n"));
        assert!(out.contains("Query model(s):"));
        assert!(out.contains("Target sequences:"));
        assert!(out.contains("passing local HMM SSV      filter:"));
        assert!(out.contains("passing local HMM MSV bias filter:"));
        assert!(out.contains("passing local HMM Viterbi  filter:"));
        assert!(out.contains("passing local HMM Forward  filter:"));
        assert!(out.contains("Total HMM hits reported:"));
        assert!(out.contains("; expected (0.02)"));
        // no trailing garbage; each line ends with newline.
        assert!(out.ends_with('\n'));
    }

    // Real filter HMM end-to-end (gated).
    #[test]
    fn real_cm_hmmonly_stage() {
        let path = match std::env::var("INFERNOX_TEST_CM") { Ok(p) => p, Err(_) => return };
        let cm = crate::cm_file::cm_file_read(&path).expect("read cm");
        let p7 = cm.p7.as_ref().expect("p7 filter");
        let ff = build_forward_filter(p7);
        let l = 220usize;
        let dsq = rand_dsq(l, 0xC0FFEE);
        let params = HmmonlyParams {
            t: -1000.0, max_length: 200, omega: 1.0 / 256.0,
            lftau: p7.evparam.lftau as f32, lflambda: p7.evparam.lflambda as f32,
            do_null2: true, search_mode: true,
        };
        let hits = run_hmmonly_one_window(&ff, p7, &p7.name, p7.m, "seq", &dsq, l as i64, 1, l as i64, &params);
        for h in &hits {
            assert!(h.score.is_finite());
            assert_eq!(h.ad.model.chars().count(), h.ad.n);
        }
        eprintln!("real-cm hmmonly OK: m={} nhits={}", p7.m, hits.len());
    }
}
