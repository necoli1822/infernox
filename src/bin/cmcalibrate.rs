// SPDX-License-Identifier: BSD-3-Clause
// infernox-cmcalibrate — faithful port of Infernal 1.1.5 `cmcalibrate`.
//
// Fits exponential-tail E-value parameters (ECMGC/ECMGI/ECMLC/ECMLI) for each CM
// in the file by scoring random genomic sequences, and rewrites the CM file in
// place. See infernox::cm_calibrate for the engine (genomic HMM, Mersenne-Twister
// RNG, CYK/Inside scan, histogram tail fit).
//
// The p7 MSV/bias filter parameters (EFP7GF) are NOT calibrated here: as in C
// Infernal 1.1.5, those come from cmbuild, and cmcalibrate leaves them untouched.
//
// Stdout report is a faithful drop-in for C cmcalibrate: cm_banner + output_header
// (cmcalibrate.c:1668) + print_calibration_column_headings (cmcalibrate.c:2099) +
// per-CM predicted/actual-time rows (print_forecasted_time:2127) + print_total_time
// (2150, if >1 CM) + print_summary (2172) + "# CPU time:" + "[ok]" footer (:561).
// The predicted/actual running-time columns are wall-clock (esl_stopwatch in C;
// std::time::Instant here) — non-deterministic, they normalize out in verification.

use infernox::cm::CM;
use infernox::cm_calibrate::{
    apply_calibration, calibrate_cm, estimate_calibration_seconds, CalibrateConfig, CalibrateResult,
};
use infernox::cm_file::{cm_file_read_from_reader_opt, cm_file_write_ascii};
use std::collections::HashSet;
use std::io::{Cursor, Write};
use std::process::exit;
use std::time::Instant;

/// C `cmcalibrate.c` process_commandline `-h` path (esl_usage + grouped
/// esl_opt_DisplayHelp for docgroups 1-6): the full banner + usage + every
/// option group, verbatim. Byte-identical to C's `-h`. (The dev options listed
/// here — --nforecast/--memreq/--split/--hfile/… — are advertised for help
/// parity; only the subset the engine implements is accepted by the parser.)
fn full_help() -> ! {
    print!(
        "# cmcalibrate :: fit exponential tails for CM E-values\n\
# INFERNAL 1.1.5 (Sep 2023)\n\
# Copyright (C) 2023 Howard Hughes Medical Institute.\n\
# Freely distributed under the BSD open source license.\n\
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\
Usage: cmcalibrate [-options] <cmfile>\n\
\n\
Basic options:\n  \
-h     : show brief help on version and usage\n  \
-L <x> : set random seq length to search in Mb to <x>  [1.6]  (0.01<=x<=160.)\n\
\n\
Options for predicting running time and memory requirements:\n  \
--forecast      : don't do calibration, predict running time and exit\n  \
--nforecast <n> : w/--forecast, predict time with <n> processors (maybe for MPI)\n  \
--memreq        : don't do calibration, print required memory and exit\n  \
--noforecast    : do calibration, but skip running time prediction\n\
\n\
Options controlling exponential tail fits:\n  \
--gtailn <n> : fit the top <n> hits/Mb in histogram for glocal modes [df: 250]\n  \
--ltailn <n> : fit the top <n> hits/Mb in histogram for  local modes [df: 750]\n  \
--tailp <x>  : set fraction of histogram tail to fit to exp tail to <x>\n\
\n\
Optional output files:\n  \
--hfile <f>  : save fitted score histogram(s) to file <f>\n  \
--sfile <f>  : save survival plot to file <f>\n  \
--qqfile <f> : save Q-Q plot for score histograms to file <f>\n  \
--ffile <f>  : save lambdas for different tail fit probs to file <f>\n  \
--xfile <f>  : save scores in fit tail to file <f>\n\
\n\
Options controlling split, partition and merge modes:\n  \
--split     : prepare partitioned calibration\n  \
--cfile <f> : with --split, save file with commands for each partition to <f>\n  \
--cbash     : with --split, output commands as a bash for loop script\n  \
--proot <s> : with --split or --merge, root for partition output files is <s>\n  \
--part <n>  : this is partition number <n> (1..<n2> from --ptot <n2>)  (n>0)\n  \
--ptot <n>  : total number of partitions is <n>  (n>0)\n  \
--pfile <f> : with --part, save scores to file <f>\n  \
--merge     : merge scores from multiple partitions for calibration\n\
\n\
Other options:\n  \
--seed <n>  : set RNG seed to <n> (if 0: one-time arbitrary seed)\n  \
--beta <x>  : set tail loss prob for query dependent banding (QDB) to <x>\n  \
--nonbanded : do not use QDB\n  \
--nonull3   : turn OFF the NULL3 post hoc additional null model\n  \
--random    : use GC content of random null background model of CM\n  \
--gc <f>    : use GC content distribution from file <f>\n  \
--cpu <n>   : number of parallel CPU workers to use for multithreads\n"
    );
    exit(0);
}

fn help() -> ! {
    full_help();
}

/// C `cmcalibrate.c` process_commandline ERROR: block (cmcalibrate.c:1657-1663).
/// Every esl_getopts command-line error routes here: the offending first line
/// (`Failed to parse command line: <errbuf>` or `Incorrect number of command line
/// arguments.`), then esl_usage + `puts("\nwhere basic options are:")` +
/// esl_opt_DisplayHelp(group 1) + `"\nTo see more help on available options, do
/// <argv0> -h\n\n"`, all to STDOUT, exit(1). Program name hardcoded to
/// esl_usage's basename "cmcalibrate"; the "do <argv0> -h" line is path-dependent.
fn cmdline_fail(first_line: &str) -> ! {
    let argv0 = std::env::args().next().unwrap_or_else(|| "cmcalibrate".to_string());
    print!(
        "{first_line}\n\
Usage: cmcalibrate [-options] <cmfile>\n\
\n\
where basic options are:\n  \
-h     : show brief help on version and usage\n  \
-L <x> : set random seq length to search in Mb to <x>  [1.6]  (0.01<=x<=160.)\n\
\n\
To see more help on available options, do {argv0} -h\n\n"
    );
    exit(1);
}

/// esl_getopts value/parse/require/incompat errors prefixed with the standard
/// "Failed to parse command line: " string.
fn cmdline_failure(msg: &str) -> ! {
    cmdline_fail(&format!("Failed to parse command line: {msg}"));
}

/// Truncate to 24 characters, mirroring esl_getopts' `%.24s` field width used in
/// its error format strings (esl_getopts.c:1681-1698,1521).
fn t24(s: &str) -> String {
    s.chars().scan(0usize, |n, c| {
        *n += c.len_utf8();
        if *n <= 24 { Some(c) } else { None }
    }).collect()
}

/// esl_getopts missing-arg: short options carry no trailing period; long ones do.
fn require_arg(next: Option<&String>, flag: &str) -> String {
    match next {
        Some(v) => v.clone(),
        None => {
            let dot = if flag.starts_with("--") { "." } else { "" };
            cmdline_failure(&format!("Option {flag} requires an argument{dot}"))
        }
    }
}

fn real_in_range(v: f64, range: &str) -> bool {
    match range {
        "0.01<=x<=160." => v >= 0.01 && v <= 160.0,
        "x>0" => v > 0.0,
        "0.0<x<0.6" => v > 0.0 && v < 0.6,
        _ => true,
    }
}
fn int_in_range(v: i64, range: &str) -> bool {
    match range {
        "n>=0" => v >= 0,
        "n>=100" => v >= 100,
        "n>0" => v > 0,
        _ => true,
    }
}

/// eslARG_REAL with range check. Type and range variants both capitalize "Option".
fn parse_real_range(next: Option<&String>, flag: &str, range: &str) -> f64 {
    let s = require_arg(next, flag);
    let v = s.parse::<f64>().unwrap_or_else(|_| {
        cmdline_failure(&format!("Option {flag} takes real-valued arg; got {} on cmdline", t24(&s)))
    });
    if !real_in_range(v, range) {
        cmdline_failure(&format!(
            "Option {flag} takes real-valued arg in range {range}; got {} on cmdline",
            t24(&s)
        ));
    }
    v
}
/// eslARG_INT with range check. NOTE the integer *range* variant lowercases
/// "option" (esl_getopts.c:1686); the integer *type* variant capitalizes it.
fn parse_int_range(next: Option<&String>, flag: &str, range: &str) -> i64 {
    let s = require_arg(next, flag);
    let v = s.parse::<i64>().unwrap_or_else(|_| {
        cmdline_failure(&format!("Option {flag} takes integer arg; got {} on cmdline", t24(&s)))
    });
    if !int_in_range(v, range) {
        cmdline_failure(&format!(
            "option {flag} takes integer arg in range {range}; got {} on cmdline",
            t24(&s)
        ));
    }
    v
}

/// C `cm_banner()` (cm.c:2289): the 5-line program banner. The program name is
/// hardcoded to C's "cmcalibrate" (not argv[0] basename) to byte-match C, exactly
/// as the other infernox tool banners (cmstat/cmbuild) do.
fn banner() {
    println!("# cmcalibrate :: fit exponential tails for CM E-values");
    println!("# INFERNAL 1.1.5 (Sep 2023)");
    println!("# Copyright (C) 2023 Howard Hughes Medical Institute.");
    println!("# Freely distributed under the BSD open source license.");
    println!("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
}

/// C `FormatTimeString(buf, n, sec, FALSE)` (display.c:1268): "%02d:%02d:%02d".
fn format_time(sec: f64) -> String {
    let h = (sec / 3600.0) as i64;
    let m = (sec / 60.0) as i64 - h * 60;
    let s = sec as i64 - h * 3600 - m * 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

/// C `%g` formatting (matches esl/printf %g: 6 sig figs, trailing zeros stripped).
fn g_fmt(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    if exp < -4 || exp >= 6 {
        let s = format!("{:.5e}", x);
        let (mant, e) = s.split_once('e').unwrap();
        let mant = strip_trailing_zeros(mant);
        let ei: i32 = e.parse().unwrap();
        format!("{}e{}{:02}", mant, if ei < 0 { "-" } else { "+" }, ei.abs())
    } else {
        let prec = (5 - exp).max(0) as usize;
        let s = format!("{:.*}", prec, x);
        strip_trailing_zeros(&s)
    }
}

fn strip_trailing_zeros(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        t.trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// Split a (possibly multi-model) CM file into per-record text chunks (mirrors
/// cmstat's split_records). Each record begins with a line starting "INFERNAL".
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

fn main() {
    // C esl_getopts accepts attached short-opt values (`-L1.0` == `-L 1.0`); expand
    // for the match parser. cmcalibrate's only value-taking short option is -L.
    let args: Vec<String> =
        infernox::search_cli::expand_short_opts(&std::env::args().collect::<Vec<_>>(), &['L']);
    let mut cfg = CalibrateConfig::default();
    let mut forecast = false;
    let mut memreq = false;
    let mut cpu_opt: Option<i32> = None;
    // Raw command-line token given to --seed, kept verbatim so we can mirror
    // esl_opt_IsDefault (esl_getopts.c:887), which strcmp's the value string
    // against the option's declared default "181" (cmcalibrate.c:106).
    let mut seed_raw: Option<String> = None;
    // Track which options were given on the command line (esl_opt_IsUsed) so
    // output_header echoes exactly the lines C does.
    let mut used: HashSet<&'static str> = HashSet::new();

    // C esl_getopts stops option processing at the first non-option token (bare
    // "-", "--", or a token not starting with '-'); the rest are positional
    // arguments (esl_getopts.c:1428-1440). `done_opts` mirrors that.
    let mut positionals: Vec<String> = Vec::new();
    let mut done_opts = false;
    let mut i = 1;
    while i < args.len() {
        let a = args[i].clone();
        if !done_opts && a == "--" {
            done_opts = true;
            i += 1;
            continue;
        }
        if done_opts || !(a.starts_with('-') && a != "-") {
            positionals.push(a);
            done_opts = true;
            i += 1;
            continue;
        }
        match a.as_str() {
            "-h" => help(),
            "-L" => { cfg.l_mb = parse_real_range(args.get(i + 1), "-L", "0.01<=x<=160."); used.insert("-L"); i += 1; }
            "--seed" => { seed_raw = args.get(i + 1).cloned(); cfg.seed = parse_int_range(args.get(i + 1), "--seed", "n>=0") as u32; used.insert("--seed"); i += 1; }
            "--beta" => { cfg.beta = parse_real_range(args.get(i + 1), "--beta", "x>0"); used.insert("--beta"); i += 1; }
            "--nonbanded" => { cfg.nonbanded = true; used.insert("--nonbanded"); }
            "--gtailn" => { cfg.gtailn = parse_int_range(args.get(i + 1), "--gtailn", "n>=100") as i32; used.insert("--gtailn"); i += 1; }
            "--ltailn" => { cfg.ltailn = parse_int_range(args.get(i + 1), "--ltailn", "n>=100") as i32; used.insert("--ltailn"); i += 1; }
            "--tailp" => { cfg.tailp = Some(parse_real_range(args.get(i + 1), "--tailp", "0.0<x<0.6") as f32); used.insert("--tailp"); i += 1; }
            "--nonull3" => { cfg.do_null3 = false; used.insert("--nonull3"); }
            "--cpu" => { cpu_opt = Some(parse_int_range(args.get(i + 1), "--cpu", "n>=0") as i32); used.insert("--cpu"); i += 1; }
            "--forecast" => { forecast = true; used.insert("--forecast"); }
            // cmcalibrate.c:84 declares --memreq. It is recognized here so the
            // esl_opt_VerifyConfig incompatibility check below (--cpu vs CPUOPTS)
            // reports the same "--forecast,--memreq" message C does for
            // `--memreq --cpu`. (The --memreq memory-report mode itself is not
            // yet ported; it is out of scope for this CLI-parity change.)
            "--memreq" => { memreq = true; used.insert("--memreq"); }
            // C esl_getopts: unrecognized option.
            s => cmdline_failure(&format!("No such option \"{s}\".")),
        }
        i += 1;
    }

    // cmcalibrate.c:1632 — esl_opt_VerifyConfig runs BEFORE the ArgNumber check
    // (line 1653). --cpu declares CPUOPTS = "--forecast,--memreq" (cmcalibrate.c:73,
    // 113) so esl_getopts.c:775 fails with the fixed message
    // "Option %.24s is incompatible with option(s) %.24s" listing the WHOLE
    // incompat string (not just the offending option). "--cpu" (5) and
    // "--forecast,--memreq" (18) both fit under the %.24s field, so no truncation.
    if used.contains("--cpu") && (used.contains("--forecast") || used.contains("--memreq")) {
        cmdline_failure(&format!(
            "Option {} is incompatible with option(s) {}",
            t24("--cpu"),
            t24("--forecast,--memreq")
        ));
    }

    // cmcalibrate.c:1653 — exactly 1 non-option argument (esl_opt_ArgNumber != 1).
    if positionals.len() != 1 {
        cmdline_fail("Incorrect number of command line arguments.");
    }
    let cmfile = positionals.into_iter().next().unwrap();

    // Plain --memreq (no --cpu, so it survived the incompat check above): the
    // memory-report mode (cmcalibrate.c print_required_memory / print_required_
    // memory_tail) is not yet ported. --memreq is recognized solely for esl
    // VerifyConfig incompatibility parity, so refuse here rather than fall
    // through and silently run a full calibration.
    if memreq {
        eprintln!("Error: cmcalibrate --memreq mode is not yet implemented in this port.");
        exit(1);
    }

    // Worker-thread count (cmcalibrate.c:628-633):
    //   relevant_ncpus = min(--cpu value, esl_threads_GetCPUCount()).
    // --cpu default = CMNCPU ("4", config.h:35), overridable by env INFERNAL_NCPU.
    let ncores = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1);
    let cpu_default: i32 = std::env::var("INFERNAL_NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let cpu_val = cpu_opt.unwrap_or(cpu_default);
    let relevant_ncpus = cpu_val.min(ncores);

    // ---- banner + output_header (cmcalibrate.c:1670-1717) ----
    banner();
    println!("# CM file:                                     {}", cmfile);
    if used.contains("-L") {
        println!("# total sequence length to search per mode:    {} Mb", g_fmt(cfg.l_mb));
    }
    if used.contains("--forecast") {
        println!("# forecast mode (no calibration):              on");
    }
    if used.contains("--gtailn") {
        println!("# number of hits/Mb to fit (glocal):           {}", cfg.gtailn);
    }
    if used.contains("--ltailn") {
        println!("# number of hits/Mb to fit (local):            {}", cfg.ltailn);
    }
    if used.contains("--tailp") {
        println!("# fraction of histogram tail to fit:           {}", g_fmt(cfg.tailp.unwrap() as f64));
    }
    // C: `if (esl_opt_IsUsed(go,"--seed"))` (cmcalibrate.c:1697). esl_opt_IsUsed
    // (esl_getopts.c:935) returns FALSE when esl_opt_IsDefault is TRUE, and
    // IsDefault (esl_getopts.c:887) strcmp's the value string against defval
    // "181" — so `--seed 181` (matches default) prints NOTHING, while
    // `--seed 0`/`--seed 5` (non-default) print the line.
    if used.contains("--seed") && seed_raw.as_deref() != Some("181") {
        if cfg.seed == 0 {
            println!("# random number seed:                          one-time arbitrary");
        } else {
            println!("# random number seed set to:                   {}", cfg.seed);
        }
    }
    if used.contains("--beta") {
        println!("# tail loss probability for QDBs:              {}", g_fmt(cfg.beta));
    }
    if used.contains("--nonbanded") {
        println!("# query dependent bands (QDBs):                off");
    }
    if used.contains("--nonull3") {
        println!("# null3 bias corrections:                      off");
    }
    if !forecast {
        // C: "# number of worker threads:  %d%s" (ncpus, " [--cpu]" if --cpu used).
        println!(
            "# number of worker threads:                    {}{}",
            relevant_ncpus,
            if used.contains("--cpu") { " [--cpu]" } else { "" }
        );
    }
    println!("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");

    // ---- read all CMs in the file ----
    let text = match std::fs::read_to_string(&cmfile) {
        Ok(t) => t,
        Err(_) => {
            eprintln!(
                "\nError: File existence/permissions problem in trying to open CM file {}.",
                cmfile
            );
            exit(1);
        }
    };
    let records = split_records(&text);
    if records.is_empty() {
        eprintln!("\nError: CM file {} appears to be empty or malformed.", cmfile);
        exit(1);
    }

    // ---- print_calibration_column_headings (cmcalibrate.c:2099) ----
    println!("#");
    if forecast {
        println!(
            "# Forecasting running time for CM calibration(s) on {} cpus:",
            relevant_ncpus
        );
        println!("#");
        println!("# {:<20}  {:>12}", "", " predicted");
        println!("# {:<20}  {:>12}", "", "running time");
        println!("# {:<20}  {:>12}", "model name", "(hr:min:sec)");
        println!("# {:<20}  {:>12}", "--------------------", "------------");
    } else {
        println!("# Calibrating CM(s):");
        println!("#");
        println!("# {:<20}  {:<12}  {:<42}  {:<12}", "", " predicted", "", "   actual");
        println!(
            "# {:<20}  {:<12}  {:<42}  {:<12}",
            "", "running time", "            percent complete", "running time"
        );
        println!(
            "# {:<20}  {:>12}  {:>42}  {:>12}",
            "model name", "(hr:min:sec)", "[........25........50........75..........]", "(hr:min:sec)"
        );
        println!(
            "# {:<20}  {:>12}  {:>42}  {:>12}",
            "--------------------", "------------", "------------------------------------------", "------------"
        );
    }

    let cmd = args.join(" ");
    let mut total_psec = 0.0f64;
    let mut total_asec = 0.0f64;
    let mut names: Vec<String> = Vec::new();
    let mut results: Vec<CalibrateResult> = Vec::new();
    let mut cms: Vec<CM> = Vec::new();

    for rec in &records {
        let reader = Cursor::new(rec.as_bytes());
        let cm = match cm_file_read_from_reader_opt(reader, false) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("\nError: read failed, CM file {} may be truncated? ({:?})", cmfile, e);
                exit(1);
            }
        };

        // predicted running time (forecast_time micro-benchmark; normalizes out).
        let psec = estimate_calibration_seconds(&cm, &cfg, relevant_ncpus);
        total_psec += psec;

        if forecast {
            // print_forecasted_time (--forecast branch): "  %-20s  %12s\n".
            println!("  {:<20}  {:>12}", cm.name, format_time(psec));
            continue;
        }

        // print_forecasted_time (normal): "  %-20s  %12s" then "  [".
        print!("  {:<20}  {:>12}  [", cm.name, format_time(psec));
        let _ = std::io::stdout().flush();

        // run the calibration (timed for the "actual running time" column).
        let t0 = Instant::now();
        let res = match calibrate_cm(&cm, &cfg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("\nError: calibration failed: {}", e);
                exit(1);
            }
        };
        let asec = t0.elapsed().as_secs_f64();
        total_asec += asec;

        // progress bar: 40 '=' (20 glocal + 20 local), then "]  %12s\n".
        print!("{}", "=".repeat(40));
        println!("]  {:>12}", format_time(asec));
        let _ = std::io::stdout().flush();

        names.push(cm.name.clone());
        results.push(res);
        cms.push(cm);
    }

    // print_total_time (cmcalibrate.c:2150), only if >1 CM.
    if records.len() > 1 {
        if forecast {
            println!("# {:>20}  {:>12}", "--------------------", "------------");
            println!("# {:<20}  {:>12}", "all models", format_time(total_psec));
        } else {
            println!(
                "# {:>20}  {:>12}  {:>42}  {:>12}",
                "--------------------", "------------", "------------------------------------------", "------------"
            );
            println!(
                "# {:<20}  {:>12}  {:>42}  {:>12}",
                "all models", format_time(total_psec), "", format_time(total_asec)
            );
        }
    }

    if forecast {
        // footer (cmcalibrate.c:561-565); CPU-time line normalizes out.
        println!("#");
        println!("# CPU time: 0.00u 0.00s 00:00:00.00 Elapsed: 00:00:00.00");
        println!("[ok]");
        return;
    }

    // ---- print_summary (cmcalibrate.c:2172): exp-tail fit mu/lambda/nrandhits ----
    println!("#");
    println!("# Calibration summary statistics:");
    println!("#");
    println!(
        "# {:>20}  {:<31}  {:<31}  {:<31}",
        "", "    exponential tail fit mu", "  exponential tail fit lambda", "     total number of hits"
    );
    println!(
        "# {:>20}  {:>31}  {:>31}  {:>31}",
        "",
        "-".repeat(31),
        "-".repeat(31),
        "-".repeat(31)
    );
    println!(
        "# {:<20}  {:>7} {:>7} {:>7} {:>7}  {:>7} {:>7} {:>7} {:>7}  {:>7} {:>7} {:>7} {:>7}",
        "model name",
        "glc cyk", "glc ins", "loc cyk", "loc ins",
        "glc cyk", "glc ins", "loc cyk", "loc ins",
        "glc cyk", "glc ins", "loc cyk", "loc ins"
    );
    println!(
        "# {:>20}  {:>7} {:>7} {:>7} {:>7}  {:>7} {:>7} {:>7} {:>7}  {:>7} {:>7} {:>7} {:>7}",
        "--------------------",
        "-------", "-------", "-------", "-------",
        "-------", "-------", "-------", "-------",
        "-------", "-------", "-------", "-------"
    );
    // exp indices: 0=GC 1=GI 2=LC 3=LI.
    for (n, res) in names.iter().zip(results.iter()) {
        print!("  {:<20}", n);
        print!(
            "  {:>7.2} {:>7.2} {:>7.2} {:>7.2}",
            res.exp[0].mu_orig, res.exp[1].mu_orig, res.exp[2].mu_orig, res.exp[3].mu_orig
        );
        print!(
            "  {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
            res.exp[0].lambda, res.exp[1].lambda, res.exp[2].lambda, res.exp[3].lambda
        );
        println!(
            "  {:>7} {:>7} {:>7} {:>7}",
            res.exp[0].nrandhits, res.exp[1].nrandhits, res.exp[2].nrandhits, res.exp[3].nrandhits
        );
    }

    // ---- apply calibration + rewrite the CM file in place (tmpfile + rename) ----
    let tmpfile = format!("{}.xxx", cmfile);
    {
        let f = match std::fs::File::create(&tmpfile) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("cannot create temp file {}: {}", tmpfile, e);
                exit(1);
            }
        };
        let mut w = std::io::BufWriter::new(f);
        for (res, cm) in results.iter().zip(cms.iter_mut()) {
            apply_calibration(cm, res);
            // Append this invocation to the CM's command log (C logs its command
            // line; the writer renumbers COM lines). The recorded text differs
            // from C's (different program path), so this COM line is not
            // byte-identical to C — only the behavior (logging) matches.
            cm.comlog = Some(match cm.comlog.take() {
                Some(prev) if !prev.is_empty() => format!("{}\n{}", prev, cmd),
                _ => cmd.clone(),
            });
            if let Err(e) = cm_file_write_ascii(&mut w, cm) {
                eprintln!("error writing CM: {}", e);
                exit(1);
            }
        }
        if let Err(e) = w.flush() {
            eprintln!("error flushing CM: {}", e);
            exit(1);
        }
    }
    if let Err(e) = std::fs::rename(&tmpfile, &cmfile) {
        eprintln!("cannot rename {} to {}: {}", tmpfile, cmfile, e);
        exit(1);
    }

    // footer (cmcalibrate.c:561-565); CPU-time line normalizes out.
    println!("#");
    println!("# CPU time: 0.00u 0.00s 00:00:00.00 Elapsed: 00:00:00.00");
    println!("[ok]");
}
