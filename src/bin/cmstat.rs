// SPDX-License-Identifier: BSD-3-Clause
// infernox-cmstat — display summary statistics for a CM or CM database.
//
// Faithful Rust port of Infernal 1.1.5 `cmstat.c` (src/cmstat.c). Byte-parity
// verified against the C `cmstat` binary for the DEFAULT output mode and the
// E-value / bit-score conversion modes (-E, -P, -T, --cut_ga/--cut_tc/--cut_nc),
// plus -Z and --key, for BOTH the CM (use_cm==TRUE) and the filter-HMM
// (use_cm==FALSE) reports. The filter-HMM report is selected by --hmmonly, or
// by default when the model has 0 basepairs; see output_stats_hmm(). The
// porting-comment convention embeds the C `file:function:line` next to each
// transcription.

use std::io::Cursor;

use infernox::cm::{CM, CM_EXPTAIL_STATS, CM_GA, CM_NC, CM_TC};
use infernox::constants::{B_ST, MATL_ML, MATP_MP, MATP_ND, MATR_MR, MP_ST};
use infernox::cm_file::cm_file_read_from_reader_opt;
use infernox::cp9::{
    cm_create_transition_map, cm_expected_state_occupancy, cp9_build_and_configure_global,
    cp9_map_cm2hmm, create_emit_map, CP9,
};

// cmstat.c:32-39 — output mode enum.
const OUTMODE_DEFAULT: i32 = 0;
const OUTMODE_BITSCORES_E: i32 = 1;
const OUTMODE_BITSCORES_P: i32 = 2;
const OUTMODE_EVALUES: i32 = 3;
const OUTMODE_GA: i32 = 4;
const OUTMODE_NC: i32 = 5;
const OUTMODE_TC: i32 = 6;

// cmstat.c:57-58
const USAGE: &str = "Usage: cmstat [-options] <cmfile>";
const BANNER: &str = "display summary statistics for CMs";

// -----------------------------------------------------------------------------
// Faithful C `printf` "%g" (no width), precision 6 by default (cm_file.c:1114
// c_printf_g). Chooses "%e" style when exp < -4 or exp >= precision, else "%f"
// style, then strips trailing zeros; exponent gets a sign and >= 2 digits.
fn c_printf_g(val: f64, precision: usize) -> String {
    let prec = if precision == 0 { 1 } else { precision };
    if val == 0.0 {
        return "0".to_string();
    }
    let neg = val.is_sign_negative();
    let a = val.abs();
    let sci = format!("{:.*e}", prec - 1, a);
    let (mant, exp_str) = sci.split_once('e').unwrap();
    let exp: i32 = exp_str.parse().unwrap();
    let s = if exp < -4 || exp >= prec as i32 {
        let m = strip_trailing_zeros(mant);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", m, sign, exp.abs())
    } else {
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

fn strip_trailing_zeros(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        let t = t.trim_end_matches('.');
        t.to_string()
    } else {
        s.to_string()
    }
}

// C `%13g`: right-justify the "%g" string to width 13.
fn g13(val: f64) -> String {
    format!("{:>13}", c_printf_g(val, 6))
}

// -----------------------------------------------------------------------------
// esl_vec_FRelEntropy (esl_vectorops.c): sum_i p[i] * log2(p[i]/q[i]) over p[i]>0,
// accumulated in a `float`. Returns +inf if any p[i]>0 while q[i]==0.
fn esl_vec_f_rel_entropy(p: &[f32], q: &[f32], n: usize) -> f32 {
    let mut kl: f32 = 0.0;
    for i in 0..n {
        if p[i] > 0.0 {
            if q[i] == 0.0 {
                return f32::INFINITY;
            }
            let ratio: f32 = p[i] / q[i];
            let term = (p[i] as f64) * (ratio as f64).log2();
            kl = (kl as f64 + term) as f32;
        }
    }
    kl
}

// eweight.c:378 cm_MeanMatchRelativeEntropy(): mean relative entropy per match
// state emission distribution, in bits, divided by cm->clen.
fn cm_mean_match_relative_entropy(cm: &CM) -> f64 {
    let k = cm.null.len(); // cm->abc->K (=4 for RNA)
    // eweight.c:395-398 — pair_null[i*K+j] = cm->null[i] * cm->null[j]
    let mut pair_null = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            pair_null[i * k + j] = cm.null[i] * cm.null[j];
        }
    }
    let mut kl = 0.0f64;
    // eweight.c:400-434
    for v in 0..(cm.m as usize) {
        let stid = cm.stid[v] as i32;
        if stid == MATP_MP {
            // eweight.c:402
            kl += esl_vec_f_rel_entropy(&cm.e[v], &pair_null, k * k) as f64;
        } else if stid == MATL_ML || stid == MATR_MR {
            // eweight.c:429
            kl += esl_vec_f_rel_entropy(&cm.e[v], &cm.null, k) as f64;
        }
    }
    // eweight.c:442
    kl / (cm.clen as f64)
}

// eweight.c:669 cp9_MeanMatchRelativeEntropy(): mean rel entropy per CP9 match
// state, divided by cp9->M.
fn cp9_mean_match_relative_entropy(cp9: &CP9) -> f64 {
    let k = cp9.null.len();
    let mut kl = 0.0f64;
    // eweight.c:674-676 — for k=1..=M
    for kpos in 1..=(cp9.m as usize) {
        kl += esl_vec_f_rel_entropy(&cp9.mat[kpos], &cp9.null, k) as f64;
    }
    kl / (cp9.m as f64)
}

// cm.c CMCountStatetype: number of states with sttype == <type>.
fn cm_count_statetype(cm: &CM, sttype: i32) -> i32 {
    (0..(cm.m as usize)).filter(|&v| cm.sttype[v] as i32 == sttype).count() as i32
}

// cm.c CMCountNodetype: number of nodes with ndtype == <type>.
fn cm_count_nodetype(cm: &CM, ndtype: i32) -> i32 {
    (0..(cm.nodes as usize)).filter(|&n| cm.ndtype[n] as i32 == ndtype).count() as i32
}

// -----------------------------------------------------------------------------
// stats.c:E2ScoreGivenExpInfo — sc = mu_extrap + log(E/cur_eff_dbsize)/(-lambda),
// where cur_eff_dbsize was set by UpdateExpsForDBSize (stats.c) to
// (Z / exp->dbsize) * nrandhits.
fn e2score(exp: &infernox::evalue::ExpParams, e: f64, z: f64) -> f32 {
    let cur_eff_dbsize = (z / exp.dbsize) * (exp.nrandhits as f64);
    (exp.mu + (e / cur_eff_dbsize).ln() / (-exp.lambda)) as f32
}

// stats.c:P2ScoreGivenExpInfo — sc = mu_extrap + log(P)/(-lambda).
fn p2score(exp: &infernox::evalue::ExpParams, p: f64) -> f32 {
    (exp.mu + p.ln() / (-exp.lambda)) as f32
}

// stats.c:Score2E — esl_exp_surv(x, mu, lambda) * eff_dbsize, with eff_dbsize =
// cur_eff_dbsize (from UpdateExpsForDBSize). esl_exp_surv(x,mu,lambda) = 1.0 for
// x < mu, else exp(-lambda*(x-mu)).
fn score2e(exp: &infernox::evalue::ExpParams, x: f32, z: f64) -> f32 {
    let cur_eff_dbsize = (z / exp.dbsize) * (exp.nrandhits as f64);
    let xd = x as f64;
    let surv = if xd < exp.mu {
        1.0
    } else {
        (-exp.lambda * (xd - exp.mu)).exp()
    };
    (surv * cur_eff_dbsize) as f32
}

// exp mode indices into cm.exp_by_mode (infernal.h:516-519): [GC,GI,LC,LI].
const EXP_CM_GC: usize = 0;
const EXP_CM_GI: usize = 1;
const EXP_CM_LC: usize = 2;
const EXP_CM_LI: usize = 3;

// -----------------------------------------------------------------------------

struct Opts {
    e: Option<f64>,
    p: Option<f64>,
    t: Option<f64>,
    z_mb: f64,
    cut_ga: bool,
    cut_nc: bool,
    cut_tc: bool,
    key: Option<String>,
    hmmonly: bool,
    nohmmonly: bool,
    cmfile: Option<String>,
}

// cmstat.c:82-101 — the process_commandline ERROR path. ProcessCmdline/
// VerifyConfig failures print `Failed to parse command line: <errbuf>`; the
// `esl_opt_ArgNumber != 1` path prints `Incorrect number of command line
// arguments.` (NO prefix). Both then print esl_usage + the "-h" pointer to
// STDOUT and exit(1). The program name is fixed to "cmstat" to match esl_usage's
// basename byte output; the final "do <argv0> -h" line is path-dependent
// (normalized out of byte-diffs).
fn err_block(first_line: &str) -> ! {
    let argv0 = std::env::args().next().unwrap_or_else(|| "cmstat".to_string());
    // USAGE already carries the "Usage: cmstat [-options] <cmfile>" text.
    println!("{}", first_line);
    println!("{}", USAGE);
    println!("\nTo see more help on available options, do {} -h\n", argv0);
    std::process::exit(1);
}

// esl_getopts value/parse/require/incompat errors all route through the ERROR
// block with the "Failed to parse command line: " prefix.
fn cmdline_failure(msg: &str) -> ! {
    err_block(&format!("Failed to parse command line: {}", msg));
}

// Truncate to 24 chars to mirror esl_getopts's `%.24s` field width in its
// error format strings (esl_getopts.c:1698, 775).
fn t24(s: &str) -> String {
    s.chars().take(24).collect()
}

// esl_getopts type+range check for an eslARG_REAL option with range "x>0"
// (verify_type_and_range, esl_getopts.c:1670-1699). Range check runs during
// esl_opt_ProcessCmdline as each option is set, in command-line order. Returns
// the parsed value; on a non-real or out-of-range arg, fails exactly like C.
fn get_real_pos(flag: &str, raw: Option<&String>) -> f64 {
    let s = match raw {
        Some(s) => s,
        // esl_getopts: a missing linked arg -> "Option %.24s requires an argument".
        None => cmdline_failure(&format!("Option {} requires an argument.", t24(flag))),
    };
    // esl_str_IsReal check first, then verify_real_range("x>0").
    match s.parse::<f64>() {
        Ok(v) if v > 0.0 => v,
        Ok(_) => cmdline_failure(&format!(
            "Option {} takes real-valued arg in range x>0; got {} on cmdline",
            t24(flag),
            t24(s)
        )),
        Err(_) => cmdline_failure(&format!(
            "Option {} takes real-valued arg; got {} on cmdline",
            t24(flag),
            t24(s)
        )),
    }
}

fn print_banner() {
    // cm.c:2289 cm_banner()
    println!("# {} :: {}", "cmstat", BANNER);
    println!("# INFERNAL {} ({})", "1.1.5", "Sep 2023");
    println!("# Copyright (C) 2023 Howard Hughes Medical Institute.");
    println!("# Freely distributed under the BSD open source license.");
    println!("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
}

fn main() {
    // C esl_getopts accepts attached short-option values (`-E50` == `-E 50`); expand
    // them for the match parser. cmstat's value-taking short options: -E,-P,-T,-Z.
    let argv: Vec<String> =
        infernox::search_cli::expand_short_opts(&std::env::args().collect::<Vec<_>>(), &['E', 'P', 'T', 'Z']);

    let mut opts = Opts {
        e: None,
        p: None,
        t: None,
        z_mb: 10.0, // cmstat.c:47 -Z default "10"
        cut_ga: false,
        cut_nc: false,
        cut_tc: false,
        key: None,
        hmmonly: false,
        nohmmonly: false,
        cmfile: None,
    };

    // C esl_getopts stops option processing at the first non-option token (a bare
    // "-", "--", or a token not starting with '-'); everything after is a
    // positional (esl_getopts.c:1428-1440). `done_opts` mirrors that.
    let mut positionals: Vec<String> = Vec::new();
    let mut done_opts = false;
    let mut i = 1;
    while i < argv.len() {
        if !done_opts && argv[i] == "--" {
            done_opts = true;
            i += 1;
            continue;
        }
        if done_opts || !(argv[i].starts_with('-') && argv[i] != "-") {
            positionals.push(argv[i].clone());
            done_opts = true;
            i += 1;
            continue;
        }
        match argv[i].as_str() {
            "-h" => {
                // cmstat.c:88-96 — cm_banner + esl_usage + "\nOptions:" +
                // esl_opt_DisplayHelp(stdout, go, 0, 2, 80). Reproduce the exact
                // option listing esl_opt_DisplayHelp emits for cmstat's table.
                print_banner();
                println!("{}", USAGE);
                println!("\nOptions:");
                println!("  -h          : show brief help on version and usage");
                println!("  -E <x>      : print bit scores that correspond to E-value threshold of <x>");
                println!("  -P <x>      : print bit scores that correspond to E-value threshold of <x>");
                println!("  -T <x>      : print E-values that correspond to bit score threshold of <x>");
                println!("  -Z <x>      : set database size in *Mb* to <x> for E-value calculations  [10]");
                println!("  --cut_ga    : print E-values that correspond to GA bit score thresholds");
                println!("  --cut_nc    : print E-values that correspond to NC bit score thresholds");
                println!("  --cut_tc    : print E-values that correspond to TC bit score thresholds");
                println!("  --key <s>   : only print statistics for CM with name or accession <s>");
                println!("  --hmmonly   : print filter HMM bit scores/E-values, not CM ones");
                println!("  --nohmmonly : print CM bit scores/E-values, even for models with 0 basepairs");
                std::process::exit(0);
            }
            "-E" => {
                opts.e = Some(get_real_pos("-E", argv.get(i + 1)));
                i += 2;
            }
            "-P" => {
                opts.p = Some(get_real_pos("-P", argv.get(i + 1)));
                i += 2;
            }
            "-T" => {
                opts.t = Some(get_real_pos("-T", argv.get(i + 1)));
                i += 2;
            }
            "-Z" => {
                opts.z_mb = get_real_pos("-Z", argv.get(i + 1));
                i += 2;
            }
            "--cut_ga" => {
                opts.cut_ga = true;
                i += 1;
            }
            "--cut_nc" => {
                opts.cut_nc = true;
                i += 1;
            }
            "--cut_tc" => {
                opts.cut_tc = true;
                i += 1;
            }
            "--key" => {
                // Long-option missing arg -> trailing period (process_longopt).
                let v = argv
                    .get(i + 1)
                    .unwrap_or_else(|| cmdline_failure("Option --key requires an argument."));
                opts.key = Some(v.clone());
                i += 2;
            }
            "--hmmonly" => {
                opts.hmmonly = true;
                i += 1;
            }
            "--nohmmonly" => {
                opts.nohmmonly = true;
                i += 1;
            }
            // C esl_getopts: unrecognized option.
            other => {
                cmdline_failure(&format!("No such option \"{other}\"."));
            }
        }
    }

    // cmstat.c OUTOPTS "-E,-P,-T,--cut_ga,--cut_nc,--cut_tc": each of these carries
    // incomp=OUTOPTS, so any two set together fail esl_opt_VerifyConfig
    // (esl_getopts.c:756-780). C reports the FIRST set option in table order, and
    // the incompat list is the OUTOPTS constant truncated to 24 chars (`%.24s`).
    const OUTOPTS_T24: &str = "-E,-P,-T,--cut_ga,--cut_"; // OUTOPTS truncated to 24
    let outopts: [(&str, bool); 6] = [
        ("-E", opts.e.is_some()),
        ("-P", opts.p.is_some()),
        ("-T", opts.t.is_some()),
        ("--cut_ga", opts.cut_ga),
        ("--cut_nc", opts.cut_nc),
        ("--cut_tc", opts.cut_tc),
    ];
    if outopts.iter().filter(|(_, set)| *set).count() >= 2 {
        let first = outopts.iter().find(|(_, set)| *set).map(|(n, _)| *n).unwrap();
        cmdline_failure(&format!(
            "Option {} is incompatible with option(s) {}",
            t24(first),
            OUTOPTS_T24
        ));
    }

    // cmstat.c:98-101 — exactly 1 non-option argument required
    // (esl_opt_ArgNumber != 1). Arg-count errors carry NO "Failed to parse..."
    // prefix (they route through the ERROR block directly).
    if positionals.len() != 1 {
        err_block("Incorrect number of command line arguments.");
    }
    opts.cmfile = Some(positionals[0].clone());
    let cmfile = opts.cmfile.clone().unwrap();

    // cmstat.c:114 — banner always printed.
    print_banner();

    // cmstat.c:116-121 — open (and validate the format of) the CM file. This
    // happens AFTER the banner but BEFORE the column headers, so a bad-format
    // file (e.g. old "cove"/INFERNAL-1.0) yields banner-only stdout + exit 1.
    let text = std::fs::read_to_string(&cmfile).unwrap_or_else(|e| {
        eprintln!(
            "File existence/permissions problem in trying to open CM file {}.\n{}",
            cmfile, e
        );
        std::process::exit(1);
    });
    // cm_file_Open validates the magic. infernox reads INFERNAL1/a (and INFERNAL-1)
    // format lines; anything else is a format error (cm_file.c parse_format_line).
    let first = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    if !first.starts_with("INFERNAL") {
        let tag: String = first.chars().next().map(|c| (c as u32).to_string()).unwrap_or_default();
        eprintln!(
            "File format problem in trying to open CM file {}.\nFormat tag is '{}': unrecognized or not supported.",
            cmfile, tag
        );
        std::process::exit(1);
    }

    // cmstat.c:125-131 — determine output mode (precedence order).
    let output_mode = if opts.e.is_some() {
        OUTMODE_BITSCORES_E
    } else if opts.p.is_some() {
        OUTMODE_BITSCORES_P
    } else if opts.t.is_some() {
        OUTMODE_EVALUES
    } else if opts.cut_ga {
        OUTMODE_GA
    } else if opts.cut_tc {
        OUTMODE_TC
    } else if opts.cut_nc {
        OUTMODE_NC
    } else {
        OUTMODE_DEFAULT
    };

    let do_hmmonly = opts.hmmonly;

    // cmstat.c:135-178 — column headers.
    print_headers(&opts, output_mode, do_hmmonly);

    // Read CMs. C uses cm_file_Read one at a time; we split the file on the
    // "INFERNAL1" record boundary and read each record independently.
    let records = split_records(&text);
    if records.is_empty() {
        eprintln!("File format problem in trying to open CM file {}.", cmfile);
        std::process::exit(1);
    }

    if opts.key.is_none() {
        // cmstat.c:183-195 — read all, print each.
        let mut ncm = 0;
        for rec in &records {
            let cm = read_cm_record(rec, &cmfile);
            ncm += 1;
            output_stats(&opts, &cm, ncm, output_mode, do_hmmonly);
        }
    } else {
        // cmstat.c:196-224 — --key: print stats for a single matching CM.
        let key = opts.key.as_ref().unwrap();
        let mut found = false;
        for rec in &records {
            let cm = read_cm_record(rec, &cmfile);
            let acc_match = cm.acc.as_deref().map(|a| a == key).unwrap_or(false);
            if cm.name == *key || acc_match {
                output_stats(&opts, &cm, 1, output_mode, do_hmmonly);
                found = true;
                break;
            }
        }
        if !found {
            eprintln!("CM {} not found in file {}", key, cmfile);
            std::process::exit(1);
        }
    }

    // cmstat.c:225
    println!("#");
}

// Split a (possibly multi-model) CM file into per-record text chunks. Each
// record begins with a line starting with "INFERNAL" (the file magic).
fn split_records(text: &str) -> Vec<String> {
    let mut records: Vec<String> = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        if line.starts_with("INFERNAL") && !cur.is_empty() {
            records.push(std::mem::take(&mut cur));
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if cur.trim().len() > 0 {
        records.push(cur);
    }
    records
}

fn read_cm_record(rec: &str, cmfile: &str) -> CM {
    // C's cm_file_Read leaves the CM globally configured (do_localize=false).
    let reader = Cursor::new(rec.as_bytes());
    cm_file_read_from_reader_opt(reader, false).unwrap_or_else(|e| {
        eprintln!("read failed, CM file {} may be truncated? ({:?})", cmfile, e);
        std::process::exit(1);
    })
}

fn print_headers(opts: &Opts, output_mode: i32, do_hmmonly: bool) {
    if output_mode == OUTMODE_DEFAULT {
        // cmstat.c:136-139
        println!(
            "# {:<4}  {:<20}  {:<9}  {:>8}  {:>8}  {:>5}  {:>5}  {:>4}  {:>4}  {:>5}  {:>12}",
            "", "", "", "", "", "", "", "", "", "", "rel entropy"
        );
        println!(
            "# {:<4}  {:<20}  {:<9}  {:>8}  {:>8}  {:>5}  {:>5}  {:>4}  {:>4}  {:>5}  {:>12}",
            "", "", "", "", "", "", "", "", "", "", "------------"
        );
        println!(
            "# {:<4}  {:<20}  {:<9}  {:>8}  {:>8}  {:>5}  {:>5}  {:>4}  {:>4}  {:>5}  {:>5}  {:>5}",
            "idx", "name", "accession", "nseq", "eff_nseq", "clen", "W", "bps", "bifs", "model",
            "cm", "hmm"
        );
        println!(
            "# {:<4}  {:<20}  {:<9}  {:>8}  {:>8}  {:>5}  {:>5}  {:>4}  {:>4}  {:>5}  {:>5}  {:>5}",
            "----",
            "--------------------",
            "---------",
            "--------",
            "--------",
            "-----",
            "-----",
            "----",
            "----",
            "-----",
            "-----",
            "-----"
        );
    } else if output_mode == OUTMODE_BITSCORES_E {
        // cmstat.c:143-147
        println!(
            "# Printing cmsearch bit scores corresponding to E-value of {} in a database of size {:.6} Mb",
            c_printf_g(opts.e.unwrap(), 6),
            opts.z_mb
        );
        println!("#");
        print_bitscore_col_header(do_hmmonly);
    } else if output_mode == OUTMODE_BITSCORES_P {
        // cmstat.c:150-154
        println!(
            "# Printing bit scores corresponding to P-value of {}",
            c_printf_g(opts.p.unwrap(), 6)
        );
        println!("#");
        print_bitscore_col_header(do_hmmonly);
    } else if output_mode == OUTMODE_EVALUES {
        // cmstat.c:157-161 — note the TWO spaces after the %.2f bit score.
        println!(
            "# Printing cmsearch E-values corresponding to a bit score of {:.2}  in a database of size {:.6} Mb",
            opts.t.unwrap(),
            opts.z_mb
        );
        println!("#");
        print_bitscore_col_header(do_hmmonly);
    } else {
        // cmstat.c:164-176 — GA/NC/TC.
        let label = if output_mode == OUTMODE_GA {
            "GA"
        } else if output_mode == OUTMODE_NC {
            "NC"
        } else {
            "TC"
        };
        println!(
            "# Printing cmsearch E-values corresponding to {} bit score thresholds in a database of size {:.6} Mb",
            label, opts.z_mb
        );
        println!("#");
        let (c1, c2, c3, c4) = if do_hmmonly {
            ("local-forward", "local-viterbi", "glocal-forwrd", "glocal-vitrbi")
        } else {
            ("local-inside", "local-cyk", "glocal-inside", "glocal-cyk")
        };
        println!(
            "# {:<4}  {:<20}  {:<9}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}  {:>5}",
            "idx", "name", "accession", "bit-score", c1, c2, c3, c4, "model"
        );
        println!(
            "# {:<4}  {:<20}  {:<9}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}  {:>5}",
            "----",
            "--------------------",
            "---------",
            "-------------",
            "-------------",
            "-------------",
            "-------------",
            "-------------",
            "-----"
        );
    }
}

fn print_bitscore_col_header(do_hmmonly: bool) {
    // cmstat.c:145-147 / 152-154 / 159-161
    let (c1, c2, c3, c4) = if do_hmmonly {
        ("local-forward", "local-viterbi", "glocal-forwrd", "glocal-vitrbi")
    } else {
        ("local-inside", "local-cyk", "glocal-inside", "glocal-cyk")
    };
    println!(
        "# {:<4}  {:<20}  {:<9}  {:>13}  {:>13}  {:>13}  {:>13}  {:>5}",
        "idx", "name", "accession", c1, c2, c3, c4, "model"
    );
    println!(
        "# {:<4}  {:<20}  {:<9}  {:>13}  {:>13}  {:>13}  {:>13}  {:>5}",
        "----",
        "--------------------",
        "---------",
        "-------------",
        "-------------",
        "-------------",
        "-------------",
        "-----"
    );
}

// p7_MeanMatchRelativeEntropy (modelstats.c:80): mean rel entropy per p7 match
// state, in bits, over the background bg->f, divided by hmm->M. For a non-amino
// (RNA/DNA) alphabet p7_bg_Create sets bg->f = 1/K uniform (p7_bg.c:70), so
// q = [0.25;4] here. mat[k] runs k=1..=M (index 0 unused).
fn p7_mean_match_relative_entropy(p7: &infernox::p7_hmm::P7Profile) -> f64 {
    let k = 4usize; // hmm->abc->K
    let q = [0.25f32; 4]; // bg->f, uniform for RNA
    let mut kl = 0.0f64;
    // modelstats.c:93-94
    for kpos in 1..=(p7.m as usize) {
        kl += esl_vec_f_rel_entropy(&p7.mat[kpos], &q, k) as f64;
    }
    // modelstats.c:95
    kl / (p7.m as f64)
}

// stats.c:346 cm_p7_E2Score — bit score for a given E-value under a p7 tail.
// return mu + ((log(E/(Z / (float) hitlen))) / (-1 * lambda)). mu/lambda are the
// (float) fp7_evparam entries; cast through f32 to match C storage width.
fn cm_p7_e2score(e: f64, z: f64, hitlen: i32, mu: f32, lambda: f32) -> f32 {
    (mu as f64 + ((e / (z / (hitlen as f32) as f64)).ln()) / (-1.0 * lambda as f64)) as f32
}

// stats.c:360 cm_p7_P2Score — bit score for a given P-value: mu + log(P)/(-lambda).
fn cm_p7_p2score(p: f64, mu: f32, lambda: f32) -> f32 {
    (mu as f64 + p.ln() / (-1.0 * lambda as f64)) as f32
}

// stats.c:330 Score2E for the p7 filter HMM path — esl_exp_surv(x,mu,lambda) *
// eff_dbsize, with eff_dbsize = Z / (float) max_length (cmstat.c:349). mu/lambda
// are the (float) fp7_evparam entries.
fn cm_p7_score2e(x: f32, mu: f32, lambda: f32, eff_dbsize: f64) -> f32 {
    let xd = x as f64;
    let mud = mu as f64;
    let lam = lambda as f64;
    let surv = if xd < mud {
        1.0
    } else {
        (-lam * (xd - mud)).exp()
    };
    (surv * eff_dbsize) as f32
}

// cmstat.c:236 output_stats()
fn output_stats(opts: &Opts, cm: &CM, ncm: i32, output_mode: i32, _do_hmmonly: bool) {
    // cmstat.c:254
    let z = opts.z_mb * 1_000_000.0;

    // cmstat.c:256-258 — use_cm selection.
    let use_cm = if opts.nohmmonly {
        true
    } else if opts.hmmonly {
        false
    } else {
        cm_count_nodetype(cm, MATP_ND) > 0
    };

    if !use_cm {
        output_stats_hmm(opts, cm, ncm, output_mode, z);
        return;
    }

    // cmstat.c:260-268 — E-value modes require exp-tail stats.
    if output_mode != OUTMODE_DEFAULT && (cm.flags & CM_EXPTAIL_STATS) == 0 {
        let which = match output_mode {
            OUTMODE_BITSCORES_E => "-E",
            OUTMODE_EVALUES => "-T",
            OUTMODE_GA => "--cut_ga",
            OUTMODE_TC => "--cut_tc",
            OUTMODE_NC => "--cut_nc",
            OUTMODE_BITSCORES_P => "-P",
            _ => "",
        };
        eprintln!(
            "{} requires E-value statistics (from cmcalibrate), model number {} has none.",
            which, ncm
        );
        std::process::exit(1);
    }

    let acc = cm.acc.as_deref().unwrap_or("-");

    if output_mode == OUTMODE_DEFAULT {
        // cmstat.c:270-293
        // build the cp9 HMM, just to get HMM RE (build_cp9_hmm, do_local=FALSE).
        let emap = create_emit_map(cm);
        let map = cp9_map_cm2hmm(cm);
        let psi = cm_expected_state_occupancy(cm);
        let tmap = cm_create_transition_map();
        let cp9 = cp9_build_and_configure_global(cm, &emap, &map, &psi, &tmap);

        // cmstat.c:277-287
        print!(
            "{:>6}  {:<20}  {:<9}  {:>8}  {:>8.2}  {:>5}  {:>5}  {:>4}  {:>4}  {:>5}",
            ncm,
            cm.name,
            acc,
            cm.nseq,
            cm.eff_nseq as f64,
            cm.clen,
            cm.w,
            cm_count_statetype(cm, MP_ST),
            cm_count_statetype(cm, B_ST),
            "cm"
        );
        // cmstat.c:289
        let cm_re = cm_mean_match_relative_entropy(cm);
        let hmm_re = cp9_mean_match_relative_entropy(&cp9);
        println!("  {:>5.3}  {:>5.3}", cm_re, hmm_re);
        return;
    }

    if output_mode == OUTMODE_BITSCORES_E {
        // cmstat.c:295-313
        let e = opts.e.unwrap();
        let lins = e2score(&cm.exp_by_mode[EXP_CM_LI], e, z);
        let lcyk = e2score(&cm.exp_by_mode[EXP_CM_LC], e, z);
        let gins = e2score(&cm.exp_by_mode[EXP_CM_GI], e, z);
        let gcyk = e2score(&cm.exp_by_mode[EXP_CM_GC], e, z);
        print!("{:>6}  {:<20}  {:<9}", ncm, cm.name, acc);
        println!(
            "  {:>13.2}  {:>13.2}  {:>13.2}  {:>13.2}  {:>5}",
            lins as f64, lcyk as f64, gins as f64, gcyk as f64, "cm"
        );
        return;
    }

    if output_mode == OUTMODE_BITSCORES_P {
        // cmstat.c:314-333
        let p = opts.p.unwrap();
        let lins = p2score(&cm.exp_by_mode[EXP_CM_LI], p);
        let lcyk = p2score(&cm.exp_by_mode[EXP_CM_LC], p);
        let gins = p2score(&cm.exp_by_mode[EXP_CM_GI], p);
        let gcyk = p2score(&cm.exp_by_mode[EXP_CM_GC], p);
        print!("{:>6}  {:<20}  {:<9}", ncm, cm.name, acc);
        println!(
            "  {:>13.2}  {:>13.2}  {:>13.2}  {:>13.2}  {:>5}",
            lins as f64, lcyk as f64, gins as f64, gcyk as f64, "cm"
        );
        return;
    }

    // cmstat.c:335-380 — EVALUES / GA / NC / TC.
    // cmstat.c:337-340 — determine T.
    let (t, t_present) = match output_mode {
        OUTMODE_EVALUES => (opts.t.unwrap() as f32, true),
        // cmstat.c:338-340 — note C guards all three on CMH_GA (a bug preserved
        // faithfully); T is set to the corresponding cutoff when CMH_GA is set.
        OUTMODE_GA => (if (cm.flags & CM_GA) != 0 { cm.ga } else { 0.0 }, (cm.flags & CM_GA) != 0),
        OUTMODE_NC => (if (cm.flags & CM_GA) != 0 { cm.nc } else { 0.0 }, (cm.flags & CM_NC) != 0),
        OUTMODE_TC => (if (cm.flags & CM_GA) != 0 { cm.tc } else { 0.0 }, (cm.flags & CM_TC) != 0),
        _ => (0.0, false),
    };

    let lins = score2e(&cm.exp_by_mode[EXP_CM_LI], t, z);
    let lcyk = score2e(&cm.exp_by_mode[EXP_CM_LC], t, z);
    let gins = score2e(&cm.exp_by_mode[EXP_CM_GI], t, z);
    let gcyk = score2e(&cm.exp_by_mode[EXP_CM_GC], t, z);

    if output_mode == OUTMODE_EVALUES {
        // cmstat.c:351-358
        print!("{:>6}  {:<20}  {:<9}", ncm, cm.name, acc);
        println!(
            "  {}  {}  {}  {}  {:>5}",
            g13(lins as f64),
            g13(lcyk as f64),
            g13(gins as f64),
            g13(gcyk as f64),
            "cm"
        );
        return;
    }

    // cmstat.c:359-380 — GA / NC / TC.
    if !t_present {
        // cmstat.c:360-371 — cutoff not present for this CM.
        println!(
            "{:>6}  {:<20}  {:<9}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}  {:>5}",
            ncm, cm.name, acc, "<not-set>", "-", "-", "-", "-", "cm"
        );
    } else {
        // cmstat.c:372-378
        print!("{:>6}  {:<20}  {:<9}  {:>13.2}", ncm, cm.name, acc, t as f64);
        println!(
            "  {}  {}  {}  {}  {:>5}",
            g13(lins as f64),
            g13(lcyk as f64),
            g13(gins as f64),
            g13(gcyk as f64),
            "cm"
        );
    }
}

// cmstat.c output_stats() — the use_cm==FALSE (filter-HMM) branches. Reports
// statistics of the CM's embedded p7 filter HMM (cm->fp7 / cm->fp7_evparam)
// rather than the CM itself. Reached via --hmmonly, or (by default) when the
// model has 0 basepairs. cm->fp7_evparam entries map to the p7's own evparam
// (cm_SetFilterHMM, cm_p7_modelmaker.c:439-446): LFTAU/LFLAMBDA/LVMU/LVLAMBDA
// come from the p7 STATS LOCAL lines; GFMU/GFLAMBDA from the CM's EFP7GF line
// (the reader stores these into p7.evparam.gfmu/gflambda).
fn output_stats_hmm(opts: &Opts, cm: &CM, ncm: i32, output_mode: i32, z: f64) {
    let p7 = cm
        .p7
        .as_ref()
        .expect("--hmmonly / 0-basepair model requires a p7 filter HMM in the CM");

    let acc = cm.acc.as_deref().unwrap_or("-");

    // fp7_evparam entries (stored as C float; cast through f32 to match width).
    let lftau = p7.evparam.lftau as f32;
    let lflambda = p7.evparam.lflambda as f32;
    let lvmu = p7.evparam.lvmu as f32;
    let lvlambda = p7.evparam.lvlambda as f32;
    let gfmu = p7.evparam.gfmu as f32;
    let gflambda = p7.evparam.gflambda as f32;

    if output_mode == OUTMODE_DEFAULT {
        // cmstat.c:277-293 — use_cm==FALSE branch.
        print!(
            "{:>6}  {:<20}  {:<9}  {:>8}  {:>8.2}  {:>5}  {:>5}  {:>4}  {:>4}  {:>5}",
            ncm,
            cm.name,
            acc,
            cm.nseq,
            p7.eff_nseq as f64, // cm->fp7->eff_nseq
            cm.clen,
            cm.w,
            0, // bps
            0, // bifs
            "hmm"
        );
        // cmstat.c:292
        let hmm_re = p7_mean_match_relative_entropy(p7);
        println!("  {:>5}  {:>5.3}", "-", hmm_re);
        return;
    }

    if output_mode == OUTMODE_BITSCORES_E {
        // cmstat.c:305-312
        let lfwd = cm_p7_e2score(opts.e.unwrap(), z, p7.max_length, lftau, lflambda);
        print!("{:>6}  {:<20}  {:<9}", ncm, cm.name, acc);
        println!(
            "  {:>13.2}  {:>13}  {:>13}  {:>13}  {:>5}",
            lfwd as f64, "-", "-", "-", "hmm"
        );
        return;
    }

    if output_mode == OUTMODE_BITSCORES_P {
        // cmstat.c:323-332
        let p = opts.p.unwrap();
        let lfwd = cm_p7_p2score(p, lftau, lflambda);
        let lvit = cm_p7_p2score(p, lvmu, lvlambda);
        let gfwd = cm_p7_p2score(p, gfmu, gflambda);
        print!("{:>6}  {:<20}  {:<9}", ncm, cm.name, acc);
        println!(
            "  {:>13.2}  {:>13.2}  {:>13.2}  {:>13}  {:>5}",
            lfwd as f64, lvit as f64, gfwd as f64, "-", "hmm"
        );
        return;
    }

    // cmstat.c:335-380 — EVALUES / GA / NC / TC.
    // cmstat.c:336 — UpdateExpsForDBSize() is called UNCONDITIONALLY (before the
    // use_cm split), so even the filter-HMM path dies here if the CM lacks
    // exp-tail stats (stats.c:543). The earlier -E/-T/--cut_* guard at
    // cmstat.c:260-268 only fires for use_cm==TRUE, so this is the only place an
    // uncalibrated 0-basepair model is caught in these modes.
    if (cm.flags & CM_EXPTAIL_STATS) == 0 {
        eprintln!(
            "\nError: model {}: UpdateExpsForDBSize(), cm does not have Exp stats\nYou may need to run cmcalibrate.",
            cm.name
        );
        std::process::exit(1);
    }
    // cmstat.c:337-340 — determine T (same CMH_GA-guarded logic as the CM path).
    let (t, t_present) = match output_mode {
        OUTMODE_EVALUES => (opts.t.unwrap() as f32, true),
        OUTMODE_GA => (if (cm.flags & CM_GA) != 0 { cm.ga } else { 0.0 }, (cm.flags & CM_GA) != 0),
        OUTMODE_NC => (if (cm.flags & CM_GA) != 0 { cm.nc } else { 0.0 }, (cm.flags & CM_NC) != 0),
        OUTMODE_TC => (if (cm.flags & CM_GA) != 0 { cm.tc } else { 0.0 }, (cm.flags & CM_TC) != 0),
        _ => (0.0, false),
    };

    // cmstat.c:349 — lfwd = Score2E(T, LFTAU, LFLAMBDA, Z / (float) max_length).
    let eff_dbsize = z / (p7.max_length as f32) as f64;
    let lfwd = cm_p7_score2e(t, lftau, lflambda, eff_dbsize);

    if output_mode == OUTMODE_EVALUES {
        // cmstat.c:356-357
        print!("{:>6}  {:<20}  {:<9}", ncm, cm.name, acc);
        println!(
            "  {}  {:>13}  {:>13}  {:>13}  {:>5}",
            g13(lfwd as f64),
            "-",
            "-",
            "-",
            "hmm"
        );
        return;
    }

    // cmstat.c:359-380 — GA / NC / TC.
    if !t_present {
        // cmstat.c:360-371 — cutoff not present for this CM.
        println!(
            "{:>6}  {:<20}  {:<9}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}  {:>5}",
            ncm, cm.name, acc, "<not-set>", "-", "-", "-", "-", "hmm"
        );
    } else {
        // cmstat.c:372-378
        print!("{:>6}  {:<20}  {:<9}  {:>13.2}", ncm, cm.name, acc, t as f64);
        println!(
            "  {}  {:>13}  {:>13}  {:>13}  {:>5}",
            g13(lfwd as f64),
            "-",
            "-",
            "-",
            "hmm"
        );
    }
}
