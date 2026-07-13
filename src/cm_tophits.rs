//! cm_tophits — tabular-output helpers (port of C `cm_tophits.c` tabular section).
//! Currently: the `--tblout` footer written by cmsearch/cmscan.

/// Faithful port of `cm_tophits_TabularTail` (cm_tophits.c:2927). C emits, after the
/// hit rows:
/// ```text
/// #
/// # Program:         <progname>
/// # Version:         <INFERNAL_VERSION> (<INFERNAL_DATE>)
/// # Pipeline mode:   <SEARCH|SCAN>
/// # Query file:      <qfile>
/// # Target file:     <tfile>
/// # Option settings: <esl_opt_SpoofCmdline>
/// # Current dir:     <cwd>
/// # Date:            <ctime_r>
/// # [ok]
/// ```
/// Stable lines (#, # Program, # Pipeline mode, # [ok]) are byte-identical to C.
/// `# Version` carries the writing tool's own version (tool-identity dependent, like the
/// CM banner — same INFERNAL_VERSION/INFERNAL_DATE convention as cm_file.rs). The
/// Query/Target/Option settings/Current dir/Date lines are invocation-variable.
pub fn tabular_tail(progname: &str, modestamp: &str, qfile: &str, tfile: &str, argv: &[String]) -> String {
    // C prints its compile-time release version/date here ("1.1.5 (Sep 2023)"); a
    // byte-identical drop-in must emit exactly that, so mirror C's constants (same
    // convention as cm_file.rs INFERNAL_VERSION/INFERNAL_DATE).
    let ver = "1.1.5";
    let date_const = "Sep 2023";
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "[unknown]".to_string());
    let mut s = String::new();
    s.push_str("#\n");
    s.push_str(&format!("# Program:         {}\n", progname));
    s.push_str(&format!("# Version:         {} ({})\n", ver, date_const));
    s.push_str(&format!("# Pipeline mode:   {}\n", modestamp));
    s.push_str(&format!("# Query file:      {}\n", qfile));
    s.push_str(&format!("# Target file:     {}\n", tfile));
    s.push_str(&format!("# Option settings: {}\n", argv.join(" ")));
    s.push_str(&format!("# Current dir:     {}\n", cwd));
    s.push_str(&format!("# Date:            {}\n", ctime_now()));
    s.push_str("# [ok]\n");
    s
}

// ===========================================================================
// Human-readable stdout: the "Hit scores" table (cm_tophits.c:cm_tophits_Targets)
// plus the per-query header block printed by cmsearch.c:602-604. This is the
// DEFAULT (no --tblout) output. See cm_tophits_targets() below.
// ===========================================================================

/// One reported hit's fields needed for the "Hit scores" table and the alignment
/// header, mirroring the `CM_HIT`/`CM_ALIDISPLAY` fields C reads in
/// `cm_tophits_Targets` (cm_tophits.c:1534-1549). `name`/`acc`/`desc` are the
/// *target* (search: sequence; scan: model). `trunc` is `cm_alidisplay_TruncString`.
pub struct HitRow<'a> {
    pub name: &'a str,
    pub acc: &'a str,
    pub desc: &'a str,
    pub evalue: f64,
    pub score: f32,
    pub bias: f32,
    pub start: i64,
    pub stop: i64,
    pub in_rc: bool,
    pub hmmonly: bool,
    pub trunc: &'a str,
    pub gc: f64,
    pub included: bool,
}

/// C `integer_textwidth` (cm_tophits.c:580 / cm_alidisplay.c:1074): number of
/// decimal digits in |n| (plus 1 for a leading '-' if n<0). Note w=0 for n=0.
pub fn integer_textwidth(mut n: i64) -> i32 {
    let mut w = if n < 0 { 1 } else { 0 };
    while n != 0 {
        n /= 10;
        w += 1;
    }
    w
}

/// C `%.2g` core (printf, 2 significant figures), used for the E-value column.
/// Byte-identical to the tblout `fmt_evalue` in bin/cmsearch.rs (same algorithm).
pub fn fmt_g2(e: f64) -> String {
    if e == 0.0 {
        return "0".to_string();
    }
    let p: i32 = 2;
    let s = format!("{:.*e}", (p - 1) as usize, e);
    let parts: Vec<&str> = s.splitn(2, 'e').collect();
    let mant = parts[0].to_string();
    let ex = parts[1].parse::<i32>().unwrap_or(0);
    if ex < -4 || ex >= p {
        let mant = strip_zeros(&mant);
        let sign = if ex < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", mant, sign, ex.abs())
    } else {
        let dec = (p - 1 - ex).max(0) as usize;
        strip_zeros(&format!("{:.*}", dec, e))
    }
}

fn strip_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// Faithful port of `cm_tophits_Targets` (cm_tophits.c:1472): the "Hit scores"
/// table. `mode_scan` = (pli->mode == CM_SCAN_MODELS); `show_accessions` = --acc;
/// `textw` = the wrap width (0 for --notextw). `hits` are the REPORTED hits, in
/// output order (already thresholded/sorted); `nreported == hits.len()`. The
/// `included` flag on each row is `CM_HIT_IS_INCLUDED`.
pub fn cm_tophits_targets(
    hits: &[HitRow],
    mode_scan: bool,
    show_accessions: bool,
    textw: i32,
) -> String {
    let nreported = hits.len();
    // namew: C uses max(mode==SEQS?8:9, GetMaxShownLength/GetMaxNameLength).
    let base_namew = if mode_scan { 9 } else { 8 };
    let max_name = hits
        .iter()
        .map(|h| {
            if show_accessions && !h.acc.is_empty() {
                h.acc.len()
            } else {
                h.name.len()
            }
        })
        .max()
        .unwrap_or(0);
    let namew = base_namew.max(max_name);
    // posw = max(6, GetMaxPositionLength) where PositionLength = digits of start/stop.
    let max_pos = hits
        .iter()
        .map(|h| integer_textwidth(h.start).max(integer_textwidth(h.stop)) as usize)
        .max()
        .unwrap_or(0);
    let posw = 6usize.max(max_pos);
    // descw (only used with textw>0). 31 magic chars, per C comment.
    let descw: i32 = if textw > 0 {
        32.max(textw - namew as i32 - 2 * posw as i32 - 31)
    } else {
        0
    };
    let rankw = 4.max(integer_textwidth(nreported as i64) + 2) as usize;

    let dashes = |n: usize| "-".repeat(n);
    let namestr = dashes(namew);
    let posstr = dashes(posw);
    let rankstr = dashes(rankw);

    let mut s = String::new();
    s.push_str("Hit scores:\n");
    // Header line 1 (cm_tophits.c:1511).
    s.push_str(&format!(
        " {:>rankw$} {:1} {:>9} {:>6} {:>5}  {:<namew$} {:>posw$} {:>posw$} {:1} {:>3} {:>5} {:>4}  {}\n",
        "rank", "", "E-value", " score", " bias",
        if mode_scan { "modelname" } else { "sequence" },
        "start", "end", "", "mdl", "trunc", "gc", "description",
        rankw = rankw, namew = namew, posw = posw,
    ));
    // Header line 2 (dashes) (cm_tophits.c:1512).
    s.push_str(&format!(
        " {:>rankw$} {:1} {:>9} {:>6} {:>5}  {:<namew$} {:>posw$} {:>posw$} {:1} {:>3} {:>5} {:>4}  {}\n",
        rankstr, "", "---------", "------", "-----",
        namestr, posstr, posstr, "", "---", "-----", "----", "-----------",
        rankw = rankw, namew = namew, posw = posw,
    ));

    let mut have_printed_incthresh = false;
    for (i, h) in hits.iter().enumerate() {
        if !h.included && !have_printed_incthresh {
            s.push_str(" ------ inclusion threshold ------\n");
            have_printed_incthresh = true;
        }
        let showname = if show_accessions && !h.acc.is_empty() {
            h.acc
        } else {
            h.name
        };
        let cur_rank = format!("({})", i + 1);
        // C 1534: leading " %*s %c %9.2g %6.1f %5.1f  %-*s %*ld %*ld %c %3s %5s %4.2f  "
        s.push_str(&format!(
            " {:>rankw$} {} {:>9} {:>6.1} {:>5.1}  {:<namew$} {:>posw$} {:>posw$} {} {:>3} {:>5} {:>4.2}  ",
            cur_rank,
            if h.included { '!' } else { '?' },
            fmt_g2(h.evalue),
            h.score,
            h.bias,
            showname,
            h.start,
            h.stop,
            if h.in_rc { '-' } else { '+' },
            if h.hmmonly { "hmm" } else { "cm" },
            h.trunc,
            h.gc,
            rankw = rankw, namew = namew, posw = posw,
        ));
        let desc = if h.desc.is_empty() { "-" } else { h.desc };
        if textw > 0 {
            // %-.*s : truncate to descw, left-justified (no min width).
            let dw = descw as usize;
            let truncated: String = desc.chars().take(dw).collect();
            s.push_str(&truncated);
            s.push('\n');
        } else {
            s.push_str(desc);
            s.push('\n');
        }
    }
    if nreported == 0 {
        s.push_str("\n   [No hits detected that satisfy reporting thresholds]\n");
    }
    s
}

// ===========================================================================
// Human-readable stdout: the "Hit alignments" section
// (cm_tophits.c:cm_tophits_HitAlignments) — the ">> name  desc" header, the
// per-hit rank/E/score/bias/mdl-from-to/seq-from-to/acc/trunc/gc line, and the
// aligned block (cm_alidisplay_Print). DEFAULT (no --noali) output.
// ===========================================================================

use crate::cm_alidisplay::{cm_alidisplay_print, CmAliDisplay};

/// One reported hit + its alidisplay, for [`cm_tophits_hit_alignments`]. Mode-symmetric:
/// `cmname`/`cmacc` are the MODEL (labels the model line + is the `>>` target in SCAN
/// mode); `sqname`/`sqacc` are the SEQUENCE (labels the seq line + is the `>>` target in
/// SEARCH mode); `tdesc` is the target's description; `target_is_model` selects which is
/// the `>>` target (TRUE for cmscan, FALSE for cmsearch). `src_l` = full source-sequence
/// length (C `hit->srcL`); `clen` = model consensus length (C `ad->clen`).
pub struct AliHit<'a> {
    pub ad: &'a CmAliDisplay,
    pub cmname: &'a str,
    pub cmacc: &'a str,
    pub sqname: &'a str,
    pub sqacc: &'a str,
    pub tdesc: &'a str,
    pub target_is_model: bool,
    pub evalue: f64,
    pub score: f32,
    pub bias: f32,
    pub hmmonly: bool,
    pub start: i64,
    pub stop: i64,
    pub in_rc: bool,
    pub src_l: i64,
    pub clen: i32,
    pub included: bool,
}

/// Faithful port of `cm_tophits_HitAlignments` (cm_tophits.c:1688), non-verbose
/// (`be_verbose == FALSE`). Mode-symmetric via [`AliHit::target_is_model`] (cmscan)
/// vs FALSE (cmsearch). `nreported` is the total reported-hit count (for the rank
/// column width). Renders the "Hit alignments:" header and one block per hit.
pub fn cm_tophits_hit_alignments(
    hits: &[AliHit],
    show_accessions: bool,
    textw: i32,
    nreported: usize,
) -> String {
    let rankw = 4.max(integer_textwidth(nreported as i64) + 2) as usize;
    let rankstr = "-".repeat(rankw);

    let mut s = String::new();
    s.push_str("Hit alignments:\n");

    for (i, h) in hits.iter().enumerate() {
        let ad = h.ad;
        // The ">>" target: model for SCAN, sequence for SEARCH.
        let (tname, tacc) = if h.target_is_model {
            (h.cmname, h.cmacc)
        } else {
            (h.sqname, h.sqacc)
        };
        let (showname, namew): (&str, usize) = if show_accessions && !tacc.is_empty() {
            (tacc, tacc.len())
        } else {
            (tname, tname.len())
        };
        // ">> name  desc" header
        if textw > 0 {
            let descw = 32.max(textw - namew as i32 - 5) as usize;
            let d: String = h.tdesc.chars().take(descw).collect();
            s.push_str(&format!(">> {}  {}\n", showname, d));
        } else {
            s.push_str(&format!(">> {}  {}\n", showname, h.tdesc));
        }

        // header line 1 (labels)
        s.push_str(&format!(
            " {:>rankw$} {:1} {:>9} {:>6} {:>5} {:<3} {:>8} {:>8} {:2} {:>11} {:>11} {:1} {:2}",
            "rank", "", "E-value", "score", "bias", "mdl", "mdl from", "mdl to", "",
            "seq from", "seq to", "", "",
            rankw = rankw,
        ));
        if ad.ppline.is_some() {
            s.push_str(&format!(" {:>4} {:>5} {:>4}", "acc", "trunc", "gc"));
        } else {
            s.push_str(&format!(" {:>6} {:>5} {:>4}", "cyksc", "trunc", "gc"));
        }
        s.push('\n');
        // header line 2 (dashes)
        s.push_str(&format!(
            " {:>rankw$} {:1} {:>9} {:>6} {:>5} {:<3} {:>8} {:>8} {:2} {:>11} {:>11} {:1} {:2}",
            rankstr, "", "---------", "------", "-----", "---", "--------", "--------", "",
            "-----------", "-----------", "", "",
            rankw = rankw,
        ));
        if ad.ppline.is_some() {
            s.push_str(&format!(" {:>4} {:>5} {:>4}", "----", "-----", "----"));
        } else {
            s.push_str(&format!(" {:>6} {:>5} {:>4}", "------", "-----", "----"));
        }
        s.push('\n');

        // local-end / truncation markers on the info line (C:1761-1778)
        let is5p = ad.cfrom_emit != ad.cfrom_span;
        let is3p = ad.cto_emit != ad.cto_span;
        let (lmod, lseq);
        if is5p {
            lmod = '~';
            lseq = '~';
        } else {
            lmod = if ad.cfrom_emit == 1 { '[' } else { '.' };
            lseq = if h.in_rc {
                if h.start == h.src_l { '[' } else { '.' }
            } else if h.start == 1 {
                '['
            } else {
                '.'
            };
        }
        let (rmod, rseq);
        if is3p {
            rmod = '~';
            rseq = '~';
        } else {
            rmod = if ad.cto_emit == h.clen { ']' } else { '.' };
            rseq = if h.in_rc {
                if h.stop == 1 { ']' } else { '.' }
            } else if h.stop == h.src_l {
                ']'
            } else {
                '.'
            };
        }

        let cur_rank = format!("({})", i + 1);
        s.push_str(&format!(
            " {:>rankw$} {} {:>9} {:>6.1} {:>5.1} {:>3} {:>8} {:>8} {}{} {:>11} {:>11} {} {}{}",
            cur_rank,
            if h.included { '!' } else { '?' },
            fmt_g2(h.evalue),
            h.score,
            h.bias,
            if h.hmmonly { "hmm" } else { "cm" },
            ad.cfrom_emit,
            ad.cto_emit,
            lmod,
            rmod,
            h.start,
            h.stop,
            if h.in_rc { '-' } else { '+' },
            lseq,
            rseq,
            rankw = rankw,
        ));
        let truncstr = ali_trunc_string(ad, h.hmmonly);
        if ad.ppline.is_some() {
            s.push_str(&format!(" {:>4.2} {:>5} {:>4.2}", ad.avgpp, truncstr, ad.gc));
        } else {
            s.push_str(&format!(" {:>6.1} {:>5} {:>4.2}", ad.sc, truncstr, ad.gc));
        }
        s.push_str("\n\n");

        // the aligned block: model line is always the CM, seq line always the sequence.
        s.push_str(&cm_alidisplay_print(
            ad, h.start, h.stop, h.cmname, h.cmacc, h.sqname, h.sqacc,
            40, textw, show_accessions,
        ));
        s.push('\n');
    }
    if nreported == 0 {
        s.push_str("\n   [No hits detected that satisfy reporting thresholds]\n");
    }
    s
}

/// C `cm_alidisplay_TruncString` (cm_alidisplay.c:1317).
fn ali_trunc_string(ad: &CmAliDisplay, hmmonly: bool) -> &'static str {
    if hmmonly {
        "-"
    } else {
        let is5 = ad.cfrom_emit != ad.cfrom_span;
        let is3 = ad.cto_emit != ad.cto_span;
        if is5 && is3 {
            "5'&3'"
        } else if is5 {
            "5'"
        } else if is3 {
            "3'"
        } else {
            "no"
        }
    }
}

// ===========================================================================
// Shared "Internal CM pipeline statistics summary" (cm_pipeline.c:cm_pli_Statistics
// + pli_pass_statistics, non-verbose CM_SUMMED). Used by BOTH cmsearch (mode_scan
// = false) and cmscan (mode_scan = true). The two modes differ only in the top
// three lines (SEARCH: Query model(s)/Target sequences/re-searched; SCAN: Query
// sequence(s)/re-searched avg-per-model/Target model(s)) — the filter lines and
// Total line are identical.
// ===========================================================================

use crate::cm_search::{PassAcctSnapshot, PliStats};

/// printf `%.*g` with `p` significant figures (shared helper).
pub fn fmt_g_prec(x: f64, p: i32) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let s = format!("{:.*e}", (p - 1) as usize, x);
    let parts: Vec<&str> = s.splitn(2, 'e').collect();
    let mant = parts[0];
    let ex = parts[1].parse::<i32>().unwrap_or(0);
    if ex < -4 || ex >= p {
        let m = if mant.contains('.') {
            mant.trim_end_matches('0').trim_end_matches('.')
        } else {
            mant
        };
        let sign = if ex < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", m, sign, ex.abs())
    } else {
        let dec = (p - 1 - ex).max(0) as usize;
        strip_zeros(&format!("{:.*}", dec, x))
    }
}

/// Faithful port of `cm_pli_Statistics`/`pli_pass_statistics` for the non-verbose
/// `PLI_PASS_CM_SUMMED` block. `pli` carries nmodels/nseqs/nnodes + the F*/do_* fields
/// (for cmscan, set nmodels = DB size, nseqs = 1, nnodes = sum of clen). `summed` is
/// the CM_SUMMED accounting (for cmscan, summed across all models of this query).
/// `std_nres` = STD-pass residues (nres_top+nres_bot of PLI_PASS_STD_ANY, summed over
/// models for scan). Emits through the "Total CM hits reported" line + a blank line.
pub fn format_pli_statistics(
    pli: &PliStats,
    summed: &PassAcctSnapshot,
    std_nres: u64,
    n_output_trunc: u64,
    pos_output_trunc: u64,
    mode_scan: bool,
) -> String {
    let nres_searched = summed.nres_top + summed.nres_bot; // CM_SUMMED total
    let nres_researched = nres_searched - std_nres; // truncated-pass residues
    let ratio = |pos: u64| -> String {
        if nres_searched == 0 {
            "0".to_string()
        } else {
            fmt_g_prec(pos as f64 / nres_searched as f64, 4)
        }
    };
    let mut s = String::new();
    s.push_str("Internal CM pipeline statistics summary:\n");
    s.push_str("----------------------------------------\n");
    if mode_scan {
        let nmod = pli.nmodels.max(1);
        // Query sequence(s): residues searched are per-model averaged (integer division).
        s.push_str(&format!(
            "Query sequence(s):                                 {:>15}  ({} residues searched)\n",
            pli.nseqs,
            (nres_searched - nres_researched) / nmod as u64
        ));
        // Re-searched line (CM_SUMMED): count = nseqs when truncation ran, residues are
        // a per-model average (%.1f). C --onlytrunc (no STD pass) uses the distinct
        // "searched for truncated hits" wording with count = nseqs unconditionally
        // (cm_pipeline.c:89-91).
        if pli.do_trunc_only {
            s.push_str(&format!(
                "Query sequences searched for truncated hits:      {:>15}  ({:.1} residues searched, avg per model)\n",
                pli.nseqs,
                nres_researched as f64 / nmod as f64
            ));
        } else {
            s.push_str(&format!(
                "Query sequences re-searched for truncated hits:    {:>15}  ({:.1} residues re-searched, avg per model)\n",
                if pli.do_trunc_ends { pli.nseqs } else { 0 },
                nres_researched as f64 / nmod as f64
            ));
        }
        s.push_str(&format!(
            "Target model(s):                                   {:>15}  ({} consensus positions)\n",
            pli.nmodels, pli.nnodes
        ));
    } else {
        s.push_str(&format!(
            "Query model(s):                                    {:>15}  ({} consensus positions)\n",
            pli.nmodels, pli.nnodes
        ));
        s.push_str(&format!(
            "Target sequences:                                  {:>15}  ({} residues searched)\n",
            pli.nseqs,
            nres_searched - nres_researched
        ));
        if pli.do_trunc_only {
            // C --onlytrunc (no STD pass): "searched" wording, count = nseqs always
            // (cm_pipeline.c:59-62).
            s.push_str(&format!(
                "Target sequences searched for truncated hits:      {:>15}  ({} residues searched)\n",
                pli.nseqs,
                nres_researched
            ));
        } else {
            s.push_str(&format!(
                "Target sequences re-searched for truncated hits:   {:>15}  ({} residues re-searched)\n",
                if pli.do_trunc_ends { pli.nseqs } else { 0 },
                nres_researched
            ));
        }
    }
    // Filter lines (identical in both modes). Each prints count + ratio + expected(F*),
    // or "(off)". Bias lines that are off are OMITTED (no else) — off by default here.
    let line = |label: &str, on: bool, n: u64, pos: u64, f: f64| -> String {
        if on {
            format!("{}{:>15}  ({}); expected ({})\n", label, n, ratio(pos), fmt_g_prec(f, 4))
        } else {
            format!("{}{:>15}  (off)\n", label, "")
        }
    };
    s.push_str(&line("Windows   passing  local HMM SSV           filter: ", pli.do_msv, summed.n_past_msv, summed.pos_past_msv, pli.f1));
    // C cm_pipeline.c:1925-1932: the MSV composition-bias line is inserted ONLY when
    // do_msvbias (off by default → omitted entirely, not rendered "(off)").
    if pli.do_msvbias {
        s.push_str(&line("Windows   passing  local HMM MSV      bias filter: ", true, summed.n_past_msvbias, summed.pos_past_msvbias, pli.f1b));
    }
    s.push_str(&line("Windows   passing  local HMM Viterbi       filter: ", pli.do_vit, summed.n_past_vit, summed.pos_past_vit, pli.f2));
    s.push_str(&line("Windows   passing  local HMM Viterbi  bias filter: ", pli.do_vitbias, summed.n_past_vitbias, summed.pos_past_vitbias, pli.f2b));
    s.push_str(&line("Windows   passing  local HMM Forward       filter: ", pli.do_fwd, summed.n_past_fwd, summed.pos_past_fwd, pli.f3));
    s.push_str(&line("Windows   passing  local HMM Forward  bias filter: ", pli.do_fwdbias, summed.n_past_fwdbias, summed.pos_past_fwdbias, pli.f3b));
    s.push_str(&line("Windows   passing glocal HMM Forward       filter: ", pli.do_gfwd, summed.n_past_gfwd, summed.pos_past_gfwd, pli.f4));
    s.push_str(&line("Windows   passing glocal HMM Forward  bias filter: ", pli.do_gfwdbias, summed.n_past_gfwdbias, summed.pos_past_gfwdbias, pli.f4b));
    s.push_str(&line("Envelopes passing glocal HMM envelope defn filter: ", pli.do_edef, summed.n_past_edef, summed.pos_past_edef, pli.f5));
    // C cm_pipeline.c:2010-2018: the envelope composition-bias line is inserted ONLY
    // when do_edefbias (off by default → omitted, not rendered "(off)").
    if pli.do_edefbias {
        s.push_str(&line("Envelopes passing glocal HMM envelope bias filter: ", true, summed.n_past_edefbias, summed.pos_past_edefbias, pli.f5b));
    }
    let cmlabel = if pli.do_glocal_cm { "glocal" } else { "local" };
    if pli.do_fcyk {
        s.push_str(&format!(
            "Envelopes passing {:>6} CM  CYK           filter: {:>15}  ({}); expected ({})\n",
            cmlabel, summed.n_past_cyk, ratio(summed.pos_past_cyk), fmt_g_prec(pli.f6, 4)
        ));
    } else {
        s.push_str(&format!(
            "Envelopes passing {:>6} CM  CYK           filter: {:>15}  (off)\n",
            cmlabel, ""
        ));
    }
    let total_pos = summed.pos_output + pos_output_trunc;
    s.push_str(&format!(
        "Total CM hits reported:                            {:>15}  ({}); includes {} truncated hit(s)\n",
        summed.n_output, ratio(total_pos), n_output_trunc
    ));
    s.push('\n');
    s
}

/// ctime(3)-style timestamp "Www Mmm DD HH:MM:SS YYYY" (day space-padded to width 2),
/// matching C's `ctime_r`. Self-contained (Hinnant civil-from-days). The Date line is
/// invocation-variable (cosmetic for parity), but a faithful drop-in emits a real date.
pub fn ctime_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let (h, mi, se) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let dow = (((days % 7) + 4) % 7) as usize; // 1970-01-01 was a Thursday; 0=Sun
    let dows = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    // civil_from_days (Howard Hinnant)
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y0 = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y0 + 1 } else { y0 };
    let mons = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!("{} {} {:2} {:02}:{:02}:{:02} {}", dows[dow], mons[m as usize], d, h, mi, se, y)
}
