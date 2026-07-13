// SPDX-License-Identifier: BSD-3-Clause
// infernox-cmemit — faithful Rust port of Infernal 1.1.5 `cmemit`.
//
// Samples sequences from a covariance model. Reproduces C's output
// byte-for-byte for a fixed --seed (the RNG is the FAST/LCG generator that C's
// cmemit uses via esl_randomness_CreateFast). See src/cm_emit.rs.
//
// Implemented modes: -u (default, unaligned FASTA), -c (consensus),
// -a (multiple alignment, Stockholm/Pfam via Parsetrees2Alignment). All modes
// loop over every model in a multi-model .cm (C master() while-loop), sharing one
// RNG across models exactly like C's cfg->r.
//
// Filter-HMM emission (--hmmonly, and the default path for 0-basepair models) is
// implemented via the ported HMMER p7 samplers (src/p7_emit.rs): -u uses
// p7_ProfileEmit (UNILOCAL if -l, else UNIGLOCAL), -c uses p7_emit_FancyConsensus,
// -a uses p7_CoreEmit + p7_tracealign_Seqs. Verified byte-identical to C `cmemit
// --hmmonly` on minifam (3 models incl. the shared-RNG chain) and the 0-bp snR75
// across -u/-c/-a, local, and --u5p/--u3p, for fixed --seed. --nohmmonly forces
// CM emission for 0-bp models.
// Options: -N, --seed, -o, --rna, --dna, --idx, --u5p, --u3p, --outformat, -l,
// --hmmonly, --nohmmonly, -h.
// Deferred (reported, not silently wrong):
// -e (needs the generative genomic HMM), --exp, --tfile, --a5p/--a3p (alignment
// truncation).

use std::io::{self, Write};
use std::process::exit;

use infernox::easel::alphabet::EslAlphabet;
use infernox::easel::msa::EslMsa;
use infernox::easel::msafile::{esl_msafile_write, MsaFormat};
use infernox::cm_modelmaker::{ct2wuss, wuss2ct, wuss_nopseudo};
use infernox::cm::{ALPHABET_SIZE, CM, CM_LOCAL_BEGIN, CM_LOCAL_END};
use infernox::cm_alidisplay::create_cm_consensus;
use infernox::cm_dpalign::parsetrees_to_alignment;
use infernox::cm_calibrate::{create_genomic_hmm, GenomicHmm};
use infernox::cm_emit::{
    emit_parsetree_seq, esl_rnd_dchoose, esl_rnd_fchoose, residues_to_chars,
    sample_genomic_sequence_from_hmm, write_fasta, EslRandomFast,
};
use infernox::cm_file::cm_file_read_from_reader_opt;
use std::io::BufReader;
use infernox::constants::{
    B_ST, D_ST, EL_ST, E_ST, IL_ST, IR_ST, MATP_ND, MAXCONNECT, ML_ST, MP_ST, MR_ST, S_ST,
};
use infernox::parsetree::Parsetree;

/// C: infernal.h:247 TRACE_LEFT_CHILD.
const TRACE_LEFT_CHILD: i32 = 1;
/// C: infernal.h:248 TRACE_RIGHT_CHILD.
const TRACE_RIGHT_CHILD: i32 = 2;


struct Opts {
    cmfile: Option<String>,
    ofile: Option<String>,
    n: i32,
    mode_c: bool, // -c consensus
    mode_a: bool, // -a alignment (deferred)
    local: bool,  // -l (deferred)
    embed: Option<i32>, // -e (deferred)
    u5p: bool,
    u3p: bool,
    seed: u32,
    dna: bool,
    idx: i32,
    hmmonly: bool,
    nohmmonly: bool, // --nohmmonly (always emit from CM, even for 0-bp models)
    outfmt: MsaFormat, // --outformat (w/ -a)
    exp: Option<f64>, // --exp <x> exponentiate CM probs by <x> before emitting
    iid: bool,        // --iid (with -e, generate background as 25% ACGU iid)
    tfile: Option<String>, // --tfile <f> dump parsetrees
    a5p: Option<i32>, // --a5p <n> truncate aln 5' at match col <n> (0=random)
    a3p: Option<i32>, // --a3p <n> truncate aln 3' at match col <n> (0=random)
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            cmfile: None,
            ofile: None,
            n: 10, // C default -N "10"
            mode_c: false,
            mode_a: false,
            local: false,
            embed: None,
            u5p: false,
            u3p: false,
            seed: 0, // C default --seed "0" (arbitrary one-time seed)
            dna: false,
            idx: 1, // C default --idx "1"
            hmmonly: false,
            nohmmonly: false,
            outfmt: MsaFormat::Stockholm, // C default --outformat "Stockholm"
            exp: None,
            iid: false,
            tfile: None,
            a5p: None,
            a3p: None,
        }
    }
}

fn die(msg: &str) -> ! {
    eprintln!("{}", msg);
    exit(1);
}

/// C `cm_Fail()` (errors.c:56): user-error handler. Writes `"\nError: "`, then the
/// message, then `"\n"` to stderr, flushes, and exit(1). cmemit routes emit_unaligned
/// failures through this (cmemit.c:252 `cm_Fail(errbuf)`). Unlike Rust's plain
/// `exit()`, C's exit() flushes stdio, so any sequences already written to stdout are
/// emitted before the abort — callers must flush the stdout writer before calling this.
fn cm_fail(msg: &str) -> ! {
    eprint!("\nError: {}\n", msg);
    let _ = io::stderr().flush();
    exit(1);
}

/// C `cmemit.c:114-119` ProcessCmdline/VerifyConfig failure ERROR block (SHORT):
/// `Failed to parse command line: <errbuf>` + esl_usage + `"\nTo see more help on
/// available options, do <argv0> -h\n\n"`, all to STDOUT, exit(1). Program name
/// hardcoded to esl_usage's basename "cmemit"; the "do <argv0> -h" line embeds the
/// real binary path (path-dependent, normalized out of byte-diffs).
fn cmdline_fail(msg: &str) -> ! {
    let argv0 = std::env::args().next().unwrap_or_else(|| "cmemit".to_string());
    print!(
        "Failed to parse command line: {msg}\n\
Usage: cmemit [-options] <cmfile>\n\
\n\
To see more help on available options, do {argv0} -h\n\n"
    );
    exit(1);
}

/// C `cmemit.c:129-136` `esl_opt_ArgNumber != 1` ERROR block (FULL): the arg-count
/// message + esl_usage + `puts("\n  where basic options are:")` +
/// `esl_opt_DisplayHelp(stdout, go, 1, 2, 80)` (docgroup 1) + `"\nTo see more help
/// on other available options, do <argv0> -h\n\n"`, all to STDOUT, exit(1).
fn argcount_fail() -> ! {
    let argv0 = std::env::args().next().unwrap_or_else(|| "cmemit".to_string());
    print!(
        "Incorrect number of command line arguments.\n\
Usage: cmemit [-options] <cmfile>\n\
\n  \
where basic options are:\n  \
-h     : show brief help on version and usage\n  \
-o <f> : send sequence output to file <f>, not stdout\n  \
-N <n> : generate <n> sequences  [10]\n  \
-u     : write generated sequences as unaligned FASTA  [default]\n  \
-a     : write generated sequences as an alignment\n  \
-c     : generate a single \"consensus\" sequence only\n  \
-e <n> : embed emitted sequences within larger random sequences of length <n>\n  \
-l     : local; emit from a locally configured model [default: global]\n\
\n\
To see more help on other available options, do {argv0} -h\n\n"
    );
    exit(1);
}

/// Truncate to 24 characters, mirroring esl_getopts' `%.24s` field width in its
/// error format strings (esl_getopts.c:1681-1698).
fn t24(s: &str) -> String {
    s.chars().scan(0usize, |n, c| {
        *n += c.len_utf8();
        if *n <= 24 { Some(c) } else { None }
    }).collect()
}

/// esl_getopts missing-arg: short options carry NO trailing period, long options do.
fn require_arg(next: Option<String>, flag: &str) -> String {
    next.unwrap_or_else(|| {
        let dot = if flag.starts_with("--") { "." } else { "" };
        cmdline_fail(&format!("Option {flag} requires an argument{dot}"))
    })
}

// (parse_int superseded by parse_int_range, which enforces esl's range messages.)

/// eslARG_INT with a range check. On a non-integer arg: capital-"Option ... takes
/// integer arg; got X on cmdline". On an out-of-range value: lowercase-"option ...
/// takes integer arg in range <range>; got X on cmdline" (esl_getopts.c:1686).
fn parse_int_range(arg: Option<String>, flag: &str, range: &str) -> i32 {
    let s = require_arg(arg, flag);
    let v = s.parse::<i32>().unwrap_or_else(|_| {
        cmdline_fail(&format!("Option {flag} takes integer arg; got {} on cmdline", t24(&s)))
    });
    let ok = match range {
        "n>0" => v > 0,
        "n>=0" => v >= 0,
        _ => true,
    };
    if !ok {
        cmdline_fail(&format!(
            "option {flag} takes integer arg in range {range}; got {} on cmdline", t24(&s)
        ));
    }
    v
}

/// eslARG_REAL with a range check. Both the type and range variants use capital
/// "Option" (esl_getopts.c:1693,1698).
fn parse_real_range(arg: Option<String>, flag: &str, range: &str) -> f64 {
    let s = require_arg(arg, flag);
    let v = s.parse::<f64>().unwrap_or_else(|_| {
        cmdline_fail(&format!("Option {flag} takes real-valued arg; got {} on cmdline", t24(&s)))
    });
    let ok = match range {
        "x>0" => v > 0.0,
        _ => true,
    };
    if !ok {
        cmdline_fail(&format!(
            "Option {flag} takes real-valued arg in range {range}; got {} on cmdline", t24(&s)
        ));
    }
    v
}

/// C `cmemit.c` process_commandline `-h` path: full banner + usage + grouped
/// esl_opt_DisplayHelp, verbatim (byte-identical to C's `-h`).
fn print_help() {
    print!(
        "# cmemit :: sample sequences from a covariance model\n\
# INFERNAL 1.1.5 (Sep 2023)\n\
# Copyright (C) 2023 Howard Hughes Medical Institute.\n\
# Freely distributed under the BSD open source license.\n\
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\
Usage: cmemit [-options] <cmfile>\n\
\n\
Basic options:\n  \
-h     : show brief help on version and usage\n  \
-o <f> : send sequence output to file <f>, not stdout\n  \
-N <n> : generate <n> sequences  [10]\n  \
-u     : write generated sequences as unaligned FASTA  [default]\n  \
-a     : write generated sequences as an alignment\n  \
-c     : generate a single \"consensus\" sequence only\n  \
-e <n> : embed emitted sequences within larger random sequences of length <n>\n  \
-l     : local; emit from a locally configured model [default: global]\n\
\n\
Options for truncating sequences:\n  \
--u5p     : truncate unaligned sequences 5', choosing a random start posn\n  \
--u3p     : truncate unaligned sequences 3', choosing a random end   posn\n  \
--a5p <n> : truncate aln 5', start at match column <n> (use 0 for random posn)\n  \
--a3p <n> : truncate aln 3', end   at match column <n> (use 0 for random posn)\n\
\n\
Other options:\n  \
--seed <n>      : set RNG seed to <n> [default: one-time arbitrary seed]  [0]\n  \
--iid           : with -e, generate larger sequences as 25% ACGU (iid) \n  \
--rna           : output as RNA sequence data  [default]\n  \
--dna           : output as DNA sequence data\n  \
--idx <n>       : start sequence numbering at <n>  [1]\n  \
--outformat <s> : w/-a output alignment in format <s>  [Stockholm]\n  \
--tfile <f>     : dump parsetrees to file <f>\n  \
--exp <x>       : exponentiate CM probabilities by <x> before emitting\n  \
--hmmonly       : emit from filter HMM, not from CM\n  \
--nohmmonly     : always emit from CM, even for models with 0 basepairs\n\
\n\
Alignment output formats (-a) include: Stockholm, Pfam, AFA (aligned FASTA), A2M, Clustal, PHYLIP\n\n"
    );
}

fn parse_args() -> Opts {
    use std::collections::HashSet;
    let mut o = Opts::default();
    // C esl_getopts accepts attached short-opt values (`-N10` == `-N 10`); expand them
    // for the match parser. cmemit value-taking short options: -o,-N,-e.
    let expanded = infernox::search_cli::expand_short_opts(
        &std::env::args().collect::<Vec<_>>(),
        &['o', 'N', 'e'],
    );
    // `used` = options given on the command line (esl_getopts "set": setby !=
    // DEFAULT && val != NULL). Drives esl_opt_VerifyConfig below.
    let mut used: HashSet<&'static str> = HashSet::new();
    // --outformat is decoded AFTER VerifyConfig/arg-count (as C does, in main),
    // so a require violation (needs -a) is reported before a bad-format error.
    let mut outformat_raw: Option<String> = None;
    let mut positionals: Vec<String> = Vec::new();
    let mut done_opts = false;

    // esl_getopts INT/REAL range checks (verify_type_and_range). NOTE the integer
    // *range* variant lowercases "option" (esl_getopts.c:1686); the integer type,
    // real type, and real range variants all use capital "Option".
    let mut args = expanded.into_iter().skip(1).peekable();
    while let Some(a) = args.next() {
        if !done_opts && a == "--" {
            done_opts = true;
            continue;
        }
        if done_opts || !(a.starts_with('-') && a != "-") {
            positionals.push(a);
            done_opts = true;
            continue;
        }
        match a.as_str() {
            "-h" => {
                print_help();
                exit(0);
            }
            "-o" => { o.ofile = Some(require_arg(args.next(), "-o")); used.insert("-o"); }
            "-N" => { o.n = parse_int_range(args.next(), "-N", "n>0"); used.insert("-N"); }
            "-u" => { used.insert("-u"); } // default
            "-a" => { o.mode_a = true; used.insert("-a"); }
            "-c" => { o.mode_c = true; used.insert("-c"); }
            "-l" => { o.local = true; used.insert("-l"); }
            "-e" => { o.embed = Some(parse_int_range(args.next(), "-e", "n>0")); used.insert("-e"); }
            "--u5p" => { o.u5p = true; used.insert("--u5p"); }
            "--u3p" => { o.u3p = true; used.insert("--u3p"); }
            "--seed" => { o.seed = parse_int_range(args.next(), "--seed", "n>=0") as u32; used.insert("--seed"); }
            "--rna" => { o.dna = false; used.insert("--rna"); }
            "--dna" => { o.dna = true; used.insert("--dna"); }
            "--idx" => { o.idx = parse_int_range(args.next(), "--idx", "n>0"); used.insert("--idx"); }
            "--outformat" => { outformat_raw = Some(require_arg(args.next(), "--outformat")); used.insert("--outformat"); }
            "--hmmonly" => { o.hmmonly = true; used.insert("--hmmonly"); }
            "--nohmmonly" => { o.nohmmonly = true; used.insert("--nohmmonly"); }
            "--iid" => { o.iid = true; used.insert("--iid"); }
            "--exp" => { o.exp = Some(parse_real_range(args.next(), "--exp", "x>0")); used.insert("--exp"); }
            "--tfile" => { o.tfile = Some(require_arg(args.next(), "--tfile")); used.insert("--tfile"); }
            "--a5p" => { o.a5p = Some(parse_int_range(args.next(), "--a5p", "n>=0")); used.insert("--a5p"); }
            "--a3p" => { o.a3p = Some(parse_int_range(args.next(), "--a3p", "n>=0")); used.insert("--a3p"); }
            // C esl_getopts: unrecognized option.
            s => cmdline_fail(&format!("No such option \"{s}\".")),
        }
    }

    // esl_opt_VerifyConfig (esl_getopts.c:719): require loop (table order) then
    // incompat loop (table order). "set" == given on the command line, which for
    // every require/incompat target here (all NULL-default booleans/ints) equals
    // membership in `used`. The message uses the FULL optlist string.
    let req = |set: bool, opt: &str, reqs: &[&str], full: &str| {
        if set {
            for r in reqs {
                if !used.contains(r) {
                    cmdline_fail(&format!(
                        "Option {opt} requires (or has no effect without) option(s) {full}"
                    ));
                }
            }
        }
    };
    req(used.contains("--a5p"), "--a5p", &["--a3p", "-a"], "--a3p,-a");
    req(used.contains("--a3p"), "--a3p", &["--a5p", "-a"], "--a5p,-a");
    req(used.contains("--iid"), "--iid", &["-e"], "-e");
    req(used.contains("--outformat"), "--outformat", &["-a"], "-a");
    let inc = |set: bool, opt: &str, incs: &[&str], full: &str| {
        if set {
            for c in incs {
                if *c != opt && used.contains(c) {
                    cmdline_fail(&format!("Option {opt} is incompatible with option(s) {full}"));
                }
            }
        }
    };
    inc(used.contains("-e"), "-e", &["-a", "-c"], "-a,-c");
    inc(used.contains("--u5p"), "--u5p", &["-a", "-c"], "-a,-c");
    inc(used.contains("--u3p"), "--u3p", &["-a", "-c"], "-a,-c");
    inc(used.contains("--tfile"), "--tfile", &["-c", "-e", "--u5p", "--u3p"], "-c,-e,--u5p,--u3p");
    inc(used.contains("--nohmmonly"), "--nohmmonly", &["--hmmonly"], "--hmmonly");

    // cmemit.c:129-136 — exactly 1 non-option argument (esl_opt_ArgNumber != 1).
    if positionals.len() != 1 {
        argcount_fail();
    }
    o.cmfile = Some(positionals.into_iter().next().unwrap());

    // C main (after process_commandline): decode --outformat now, so a require
    // violation above is reported first. C: outfmt = esl_msafile_EncodeFormat(...);
    // if unknown -> "\nError: <s> is not a recognized output MSA file format\n\n".
    if let Some(f) = outformat_raw {
        o.outfmt = infernox::easel::msafile::esl_msafile_encode_format(&f).unwrap_or_else(|| {
            eprint!("\nError: {} is not a recognized output MSA file format\n\n\n", f);
            exit(1);
        });
    }
    o
}

/// Split a multi-model CM file's text into one text chunk per model record.
/// C `cmemit` loops `while (cm_file_Read(...) == eslOK)` over all records; our
/// per-record reader consumes one reader per model, so we slice the file at each
/// `INFERNAL...` format-magic line (mirrors `cmsearch`'s split_cm_records).
fn split_cm_records(text: &str) -> Vec<String> {
    let mut records: Vec<String> = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        if line.trim_start().starts_with("INFERNAL") && !cur.is_empty() {
            records.push(std::mem::take(&mut cur));
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        records.push(cur);
    }
    records
}

fn main() {
    let o = parse_args();
    // parse_args() guarantees exactly one positional (esl_opt_ArgNumber == 1) and
    // has already run esl_opt_VerifyConfig (require/incompat), so cmfile is present.
    let cmfile = o.cmfile.clone().expect("parse_args guarantees a cmfile");

    // Read the whole CM file and split into per-model records. C's master() loops
    // `while (cm_file_Read(...) == eslOK)` over every model in the file.
    let cmtext = match std::fs::read_to_string(&cmfile) {
        Ok(t) => t,
        Err(e) => die(&format!("Failed to read CM file {}: {}", cmfile, e)),
    };
    let records = split_cm_records(&cmtext);
    if records.is_empty() {
        die(&format!("Failed to read CM from {} -- file corrupt?", cmfile));
    }

    // Open output.
    let mut out: Box<dyn Write> = match &o.ofile {
        Some(path) => match std::fs::File::create(path) {
            Ok(f) => Box::new(io::BufWriter::new(f)),
            Err(e) => die(&format!("Failed to open output file {}: {}", path, e)),
        },
        None => Box::new(io::BufWriter::new(io::stdout())),
    };

    // C: init_cfg() opens the --tfile parsetree-dump file, if requested.
    let mut tfp: Option<Box<dyn Write>> = match &o.tfile {
        Some(path) => match std::fs::File::create(path) {
            Ok(f) => Some(Box::new(io::BufWriter::new(f))),
            Err(e) => die(&format!("Failed to open --tfile output file {}: {}", path, e)),
        },
        None => None,
    };

    // C: cfg->r = esl_randomness_CreateFast(--seed), created ONCE before the model
    // loop and reused across models. So model k's RNG stream continues from the
    // state left by model k-1 — we hoist it here to reproduce that byte-for-byte.
    let mut r = EslRandomFast::new(o.seed);

    // C: cfg->ncm, incremented per model; used to name sequences when cm->name
    // is NULL ("%d-sample%d", ncm). Named models ignore it.
    let mut ncm: i32 = 0;

    for rec in &records {
        // C: cm_file_Read (global; -l local config applied below).
        let mut cm = match cm_file_read_from_reader_opt(BufReader::new(rec.as_bytes()), false) {
            Ok(cm) => cm,
            Err(e) => die(&format!("Failed to read CM from {}: {:?}", cmfile, e)),
        };
        ncm += 1;

        // C: use_cm = (--nohmmonly) ? TRUE : (--hmmonly) ? FALSE : (CMCountNodetype(cm,MATP_nd)>0).
        let has_matp = (0..cm.nodes).any(|nd| cm.ndtype[nd as usize] as i32 == MATP_ND);
        let use_cm = if o.nohmmonly {
            true
        } else if o.hmmonly {
            false
        } else {
            has_matp
        };

        // C: initialize_cm() — exponentiate BEFORE any local configuration
        // (cm_Exponentiate requires global mode). Only the CM path is exercised
        // here (the --hmmonly / 0-bp path would use cm_p7_Exponentiate on cm->fp7,
        // which cmemit's filter-HMM emission is not yet wired to under --exp).
        if let Some(z) = o.exp {
            if use_cm {
                infernox::cm_file::cm_exponentiate(&mut cm, z);
                // C: cm_Exponentiate invalidates CMH_BITS; the subsequent
                // cm_Configure() rebuilds the log-odds scores (CMLogoddsify) from
                // the exponentiated probabilities. -c/-a derive the consensus (and
                // RF casing) from cm->esc, so we must rebuild the scores here to
                // stay byte-identical to C. (-u reads probabilities only, so this
                // is a no-op for it.)
                cm.cm_logoddsify();
            } else {
                die("infernox-cmemit: --exp with filter-HMM (--hmmonly/0-bp) emission \
                     is not yet implemented");
            }
        }

        // For --tfile, ParsetreeScore/ParsetreeDump read the log-odds score arrays
        // (cm->tsc/esc/beginsc/endsc/lmesc/rmesc). C recomputes all of these from
        // the probabilities in cm_Configure()->CMLogoddsify(); our read path keeps
        // the ASCII-rounded file scores (which differ by ~0.01), and leaves the
        // marginals empty. So we logoddsify here (after any --exp, before local
        // config) to match C's dumped scores bit-for-bit. Emission uses
        // probabilities, so this does not affect any non-tfile output.
        if o.tfile.is_some() && use_cm {
            infernox::cp9::cm_logoddsify(&mut cm);
        }

        // C: initialize_cm() — if -l, set CM_CONFIG_LOCAL|HMMLOCAL|HMMEL then
        // cm_Configure() localizes the CM. For emission the salient effect is the
        // local begin/end probability vectors + EL, which cm.localize() sets.
        if o.local && use_cm {
            configure_local(&mut cm);
        }

        if use_cm {
            if o.mode_c {
                emit_consensus(&mut *out, &cm, o.dna, ncm);
            } else if o.mode_a {
                emit_alignment(&mut *out, &mut cm, &o, &mut r, ncm, &mut tfp);
            } else {
                emit_unaligned(&mut *out, &mut cm, &o, &mut r, ncm, &mut tfp);
            }
        } else {
            // Filter-HMM emission path (cm->fp7). For -l the profile is configured
            // UNILOCAL, else UNIGLOCAL (cmemit.c:339). cm->fp7 is read from the CM
            // file's HMMER3/f block.
            if cm.p7.is_none() {
                die("infernox-cmemit: model has no filter p7 HMM for --hmmonly emission");
            }
            if o.tfile.is_some() {
                // C dumps a p7 trace (p7_trace_Dump) here, a different format from
                // ParsetreeDump. Not yet ported; report rather than emit wrong.
                die("infernox-cmemit: --tfile with filter-HMM (--hmmonly/0-bp) emission \
                     is not yet implemented");
            }
            if o.mode_c {
                emit_consensus_hmm(&mut *out, &cm, o.dna, ncm);
            } else if o.mode_a {
                emit_alignment_hmm(&mut *out, &cm, &o, &mut r, ncm);
            } else {
                emit_unaligned_hmm(&mut *out, &cm, &o, &mut r, ncm);
            }
        }
    }

    if let Err(e) = out.flush() {
        die(&format!("Error writing output: {}", e));
    }
    if let Some(tfp) = tfp.as_mut() {
        if let Err(e) = tfp.flush() {
            die(&format!("Error writing parsetree file: {}", e));
        }
    }
}

/// Build a sequence name the way C's cmemit does: "%s-sample%d" when the model is
/// named, else "%d-sample%d" using the 1-based model index (cfg->ncm).
fn sample_name(cm: &CM, idx: i32, ncm: i32) -> String {
    if cm.name.is_empty() {
        format!("{}-sample{}", ncm, idx)
    } else {
        format!("{}-sample{}", cm.name, idx)
    }
}

/// C: cmemit.c:initialize_cm() `-l` branch + cm_Configure() local setup. Sets the
/// CM_CONFIG_LOCAL|HMMLOCAL|HMMEL flags' emission-relevant effect: the local
/// begin/end probability vectors (cm->begin/cm->end) and CM_LOCAL_BEGIN/END flags,
/// using Infernal's defaults pbegin=0.05 (DEFAULT_PBEGIN), pend=0.05 (DEFAULT_PEND).
fn configure_local(cm: &mut CM) {
    // C constants.h: DEFAULT_PBEGIN 0.05, DEFAULT_PEND 0.05 (cm->pbegin/pend).
    cm.localize(0.05, 0.05);
}

/// C: cmemit.c:emit_unaligned() `-e` block. Generates a length-`embedL` background
/// sequence (genomic HMM by default, iid 25% ACGU with `--iid`) and embeds the
/// emitted (possibly truncated) sequence `residues` in it at a position chosen
/// exactly as C. Returns the FASTA name ("<name>/<start>-<end>") and the residue
/// indices of the full embedded sequence.
///
/// RNG draw order (matches C's shared stream): (1) all emission + any --u5p/--u3p
/// Roll draws happened before this call; (2) the background generation draws — for
/// the genomic HMM, 1 DChoose on the start vector then 2 DChoose per residue; for
/// --iid, `embedL` DChoose draws on the uniform vector; (3) the embed-position
/// Roll, drawn only when neither --u5p nor --u3p is set.
fn embed_in_background(
    r: &mut EslRandomFast,
    residues: &[u8],
    name: &str,
    o: &Opts,
    ghmm: Option<&GenomicHmm>,
    fq: &[f64],
) -> Result<(String, Vec<u8>), String> {
    let embedl = o.embed.unwrap() as usize;
    let n = residues.len();
    // C: cmemit.c:396 —
    //   if(sq2print->n > embedL) ESL_FAIL(eslEINCOMPAT, errbuf,
    //     "<n>=%d from -eL <n> too small for emitted seq of length %" PRId64
    //     ", increase <n> and rerun", embedL, sq2print->n);
    // The message text says "-eL" though the option is "-e" (a latent C quirk,
    // reproduced verbatim). ESL_FAIL fills errbuf and returns; the caller
    // (cmemit.c:252) reports it via cm_Fail(errbuf) after C's exit() flushes the
    // sequences already written to stdout. We propagate the errbuf so the caller
    // can flush the stdout writer, then abort identically.
    if n > embedl {
        return Err(format!(
            "<n>={} from -eL <n> too small for emitted seq of length {}, increase <n> and rerun",
            embedl, n
        ));
    }
    // C: generate background of length embedL.
    let mut gsq: Vec<u8> = if o.iid {
        // C: esl_rsq_xIID(r, fq, K, embedL, gsq->dsq) — L DChoose draws on fq.
        (0..embedl).map(|_| esl_rnd_dchoose(r, fq) as u8).collect()
    } else {
        // C: SampleGenomicSequenceFromHMM(r, abc, ..., embedL, &gsq->dsq).
        sample_genomic_sequence_from_hmm(r, ghmm.unwrap(), embedl)
    };
    // C: embed start position (1-based). Contract: at most one of --u5p/--u3p.
    let start: usize = if o.u5p {
        1
    } else if o.u3p {
        embedl - n + 1
    } else {
        // C: esl_rnd_Roll(r, embedL - sq2print->n + 1) + 1.
        (r.roll((embedl - n + 1) as u32) + 1) as usize
    };
    // C: for(x=start; x<start+n; x++) gsq->dsq[x] = sq2print->dsq[x-start+1].
    for (x, &res) in residues.iter().enumerate() {
        gsq[start - 1 + x] = res;
    }
    // C: esl_sq_FormatName(gsq, "%s/%d-%d", sq2print->name, start, start+n-1).
    let newname = format!("{}/{}-{}", name, start, start + n - 1);
    Ok((newname, gsq))
}

/// C: cmemit.c:emit_consensus() (use_cm path). Emits the CM consensus sequence.
/// No RNG is consumed.
fn emit_consensus(out: &mut dyn Write, cm: &CM, dna: bool, ncm: i32) {
    // C: csq name = cm->name + "-cmconsensus"; seq = cm->cmcons->cseq (text).
    let cons = create_cm_consensus(cm);
    let base = if cm.name.is_empty() { ncm.to_string() } else { cm.name.clone() };
    let name = format!("{}-cmconsensus", base);
    // cseq is a text display string; write it verbatim. For DNA output, C's
    // text-mode FASTA write leaves the (already-text) consensus unchanged except
    // that any 'U'/'u' would be an alphabet concern — the consensus string uses
    // the CM (RNA) symbols; map U->T for --dna to match C's output alphabet.
    let seq: Vec<u8> = if dna {
        cons.cseq
            .iter()
            .map(|&c| match c {
                b'U' => b'T',
                b'u' => b't',
                other => other,
            })
            .collect()
    } else {
        cons.cseq.clone()
    };
    if let Err(e) = write_fasta(out, &name, &seq) {
        die(&format!("error writing consensus sequence: {}", e));
    }
}

/// C: cmemit.c:emit_unaligned() (filter-HMM path, `! use_cm`). Samples N
/// sequences from cm->fp7 via p7_ProfileEmit and writes them as unaligned FASTA.
/// The profile is UNILOCAL if `-l`, else UNIGLOCAL. --u5p/--u3p truncation is
/// supported (its Roll draws happen after emission, exactly as C).
fn emit_unaligned_hmm(out: &mut dyn Write, cm: &CM, o: &Opts, r: &mut EslRandomFast, ncm: i32) {
    let fp7 = cm.p7.as_ref().unwrap();
    let offset = o.idx;
    // C: if(-e) build the background generator once, before the loop.
    let ghmm: Option<GenomicHmm> = if o.embed.is_some() && !o.iid {
        Some(create_genomic_hmm(ALPHABET_SIZE))
    } else {
        None
    };
    let fq = [1.0 / ALPHABET_SIZE as f64; ALPHABET_SIZE];
    for i in 0..o.n {
        let name = sample_name(cm, i + offset, ncm);
        // C: p7_ProfileEmit(cfg->r, cm->fp7, gm, bg, esq, p7tr).
        let mut residues = infernox::p7_emit::p7_profile_emit(r, fp7, o.local);
        // C: truncate esq if --u5p / --u3p (Roll draws happen here, after emit).
        if o.u5p || o.u3p {
            let full = residues.len() as u32;
            let mut start = if o.u5p { r.roll(full) + 1 } else { 1 };
            let mut end = if o.u3p { r.roll(full) + 1 } else { full };
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }
            residues = residues[(start - 1) as usize..end as usize].to_vec();
        }
        // C: if(-e) embed the (truncated) sequence in a larger background.
        let (outname, outres) = if o.embed.is_some() {
            match embed_in_background(r, &residues, &name, o, ghmm.as_ref(), &fq) {
                Ok(v) => v,
                // C: cm_Fail(errbuf) after exit() flushes stdout. Flush the seqs
                // already written this run, then abort with cm_Fail's format.
                Err(msg) => { let _ = out.flush(); cm_fail(&msg); }
            }
        } else {
            (name, residues)
        };
        let chars = residues_to_chars(&outres, o.dna);
        if let Err(err) = write_fasta(out, &outname, &chars) {
            die(&format!("Error writing unaligned sequences: {}", err));
        }
    }
}

/// C: cmemit.c:emit_consensus() (filter-HMM path, `! use_cm`). Writes the HMM
/// consensus (p7_emit_FancyConsensus, min_lower=0.0, min_upper=0.5) as FASTA,
/// named "%s-hmmconsensus". No RNG consumed.
fn emit_consensus_hmm(out: &mut dyn Write, cm: &CM, dna: bool, ncm: i32) {
    let fp7 = cm.p7.as_ref().unwrap();
    let cons = infernox::p7_emit::p7_emit_fancy_consensus(fp7, 0.0, 0.5);
    let base = if cm.name.is_empty() { ncm.to_string() } else { cm.name.clone() };
    let name = format!("{}-hmmconsensus", base);
    // FancyConsensus returns text-mode RNA symbols; for --dna map U/u -> T/t
    // exactly as C's text-mode FASTA output alphabet does.
    let seq: Vec<u8> = if dna {
        cons.iter()
            .map(|&c| match c {
                b'U' => b'T',
                b'u' => b't',
                other => other,
            })
            .collect()
    } else {
        cons
    };
    if let Err(e) = write_fasta(out, &name, &seq) {
        die(&format!("error writing consensus sequence: {}", e));
    }
}

/// C: cmemit.c:emit_unaligned() (use_cm path, no -e). Samples N sequences via
/// EmitParsetree and writes them as unaligned FASTA. --u5p/--u3p truncation is
/// supported (and consumes RNG draws in the same order as C).
fn emit_unaligned(
    out: &mut dyn Write,
    cm: &mut CM,
    o: &Opts,
    r: &mut EslRandomFast,
    ncm: i32,
    tfp: &mut Option<Box<dyn Write>>,
) {
    let offset = o.idx;

    // C: if(-e) build the background generator once, before the loop. iid uses a
    // uniform 1/K vector (esl_vec_DSet); else the 5-state genomic HMM.
    let ghmm: Option<GenomicHmm> = if o.embed.is_some() && !o.iid {
        Some(create_genomic_hmm(ALPHABET_SIZE))
    } else {
        None
    };
    let fq = [1.0 / ALPHABET_SIZE as f64; ALPHABET_SIZE];

    for i in 0..o.n {
        // C: snprintf(name, "%s-sample%d", cm->name, i+offset)
        let name = sample_name(cm, i + offset, ncm);

        // C: EmitParsetree(cm, r, name, TRUE, &tr, &esq, &L). When --tfile is set we
        // need the parse tree too; both emitters consume the RNG identically, so the
        // sampled residues are the same either way. --tfile is incompatible with
        // -e/--u5p/--u3p (checked in main), so with a tfile there is no truncation.
        let (tr, mut residues): (Option<Parsetree>, Vec<u8>) = if tfp.is_some() {
            let (tr, dsq) = emit_parsetree(cm, r);
            let res = dsq[1..dsq.len() - 1].to_vec(); // strip sentinels
            (Some(tr), res)
        } else {
            (None, emit_parsetree_seq(cm, r))
        };

        // C: truncate esq if --u5p / --u3p (Roll draws happen here, after emit).
        if o.u5p || o.u3p {
            let full = residues.len() as u32;
            // start/end are 1-based inclusive over the emitted sequence.
            let mut start = if o.u5p { r.roll(full) + 1 } else { 1 };
            let mut end = if o.u3p { r.roll(full) + 1 } else { full };
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }
            let s = (start - 1) as usize;
            let e = end as usize;
            residues = residues[s..e].to_vec();
        }

        // C: if(-e) embed the (truncated) sequence in a larger background.
        let (outname, outres) = if o.embed.is_some() {
            match embed_in_background(r, &residues, &name, o, ghmm.as_ref(), &fq) {
                Ok(v) => v,
                // C: cm_Fail(errbuf) after exit() flushes stdout. Flush the seqs
                // already written this run, then abort with cm_Fail's format.
                Err(msg) => { let _ = out.flush(); cm_fail(&msg); }
            }
        } else {
            (name.clone(), residues)
        };

        let chars = residues_to_chars(&outres, o.dna);
        if let Err(err) = write_fasta(out, &outname, &chars) {
            die(&format!("Error writing unaligned sequences: {}", err));
        }

        // C: output parsetree if nec (sq2print == esq here; no -e/--u5p/--u3p).
        if let (Some(tfp), Some(tr)) = (tfp.as_deref_mut(), tr.as_ref()) {
            // Rebuild the 1-based sentinel-padded dsq from the emitted residues for
            // ParsetreeScore/Dump (matches C's sq2print->dsq).
            let mut dsq: Vec<u8> = Vec::with_capacity(outres.len() + 2);
            dsq.push(255);
            dsq.extend_from_slice(&outres);
            dsq.push(255);
            let (sc, struct_sc) = parsetree_score(cm, tr, &dsq);
            let res: io::Result<()> = (|| {
                writeln!(tfp, "> {}", name)?;
                writeln!(tfp, "  {:>16} {:.2} bits", "SCORE:", sc)?;
                writeln!(tfp, "  {:>16} {:.2} bits", "STRUCTURE SCORE:", struct_sc)?;
                parsetree_dump(tfp, tr, cm, &dsq)?;
                writeln!(tfp, "//")?;
                Ok(())
            })();
            if let Err(e) = res {
                die(&format!("Error writing parsetree file: {}", e));
            }
        }
    }

    // Reference the local/end flags so the (currently global-only) faithfulness
    // note stays wired to the CM state EmitParsetree reads.
    let _ = cm.flags & (CM_LOCAL_BEGIN | CM_LOCAL_END);
}

/// C: cmemit.c:emit_alignment() (filter-HMM path, `! use_cm`). Samples N
/// sequences+traces from cm->fp7 via p7_CoreEmit (always glocal core mode), builds
/// an MSA with p7_tracealign_Seqs, adds an SS_cons string (':' at RF match cols,
/// '.' at insert cols), and writes it. RNG draw order matches C so a fixed seed
/// reproduces C's sampled sequences (hence the same alignment).
fn emit_alignment_hmm(mut out: &mut dyn Write, cm: &CM, o: &Opts, r: &mut EslRandomFast, ncm: i32) {
    let fp7 = cm.p7.as_ref().unwrap();
    let offset = o.idx;
    let nseq = o.n as usize;

    // C: for(i=0;i<nseq;i++) { p7_CoreEmit(r, fp7, sqA[i], p7trA[i]); esl_sq_SetName. }
    let mut names: Vec<String> = Vec::with_capacity(nseq);
    let mut trs: Vec<infernox::p7_emit::P7Trace> = Vec::with_capacity(nseq);
    let mut seqs: Vec<Vec<u8>> = Vec::with_capacity(nseq);
    for i in 0..nseq {
        let (tr, seq) = infernox::p7_emit::p7_core_emit(r, fp7);
        trs.push(tr);
        seqs.push(seq);
        names.push(sample_name(cm, i as i32 + offset, ncm));
    }

    // C: p7_tracealign_Seqs(sqA, p7trA, nseq, fp7->M, p7_ALL_CONSENSUS_COLS, fp7, &msa).
    let mut msa = infernox::p7_emit::p7_tracealign_seqs(&trs, &seqs, &names, fp7.m as usize);

    // C: build an SS_cons string: ':' where RF != '.', else '.'.
    if let Some(ref rf) = msa.rf {
        let ss: String = rf
            .bytes()
            .map(|c| if c == b'.' { '.' } else { ':' })
            .collect();
        msa.ss_cons = Some(ss);
    }

    // C: msa->name = cm->name; desc = "Synthetic sequence alignment generated by
    // cmemit [hmm-mode]" (the " [hmm-mode]" suffix is present because ! use_cm).
    msa.name = Some(cm.name.clone());
    msa.desc = Some("Synthetic sequence alignment generated by cmemit [hmm-mode]".to_string());

    // C: if(do_truncate) truncate_msa(). do_truncate = both --a5p and --a3p set.
    if let (Some(a5p), Some(a3p)) = (o.a5p, o.a3p) {
        truncate_msa(&mut msa, r, a5p, a3p);
    }

    if let Err(e) = esl_msafile_write(&mut out, &msa, o.outfmt) {
        die(&format!("Writing alignment file failed: {}", e));
    }
}

/// C: cmemit.c:emit_alignment() (use_cm path). Samples N parsetrees + sequences
/// via EmitParsetree, then builds a multiple alignment with Parsetrees2Alignment
/// (do_full=TRUE, do_matchonly=FALSE, allow_trunc=FALSE) and writes it in the
/// requested MSA format. The RNG draw order is identical to the -u path, so a
/// fixed seed reproduces C's sampled sequences (and hence the same alignment).
fn emit_alignment(
    mut out: &mut dyn Write,
    cm: &mut CM,
    o: &Opts,
    r: &mut EslRandomFast,
    ncm: i32,
    tfp: &mut Option<Box<dyn Write>>,
) {
    let offset = o.idx;
    let nseq = o.n as usize;

    // Output alphabet: RNA by default; DNA maps residue index 3 (U) -> 'T'.
    // C: Parsetrees2Alignment(cm, ..., cfg->abc_out, sqA, ...).
    let mut abc_out = EslAlphabet::rna();
    if o.dna {
        abc_out.sym[3] = 'T';
    }

    let mut names: Vec<String> = Vec::with_capacity(nseq);
    let mut dsqs: Vec<Vec<u8>> = Vec::with_capacity(nseq);
    let mut trs: Vec<Parsetree> = Vec::with_capacity(nseq);

    // C: for(i=0;i<nseq;i++) { name="%s-sample%d"; EmitParsetree(...); sqA[i]->abc=abc_out; }
    for i in 0..nseq {
        let name = sample_name(cm, i as i32 + offset, ncm);
        let (tr, dsq) = emit_parsetree(cm, r);
        // C: if(cfg->tfp != NULL) dump the parsetree right after emission (no score
        // lines in -a mode). Uses sqA[i]->dsq (the emitted digital sequence).
        if let Some(tfp) = tfp.as_deref_mut() {
            let res: io::Result<()> = (|| {
                writeln!(tfp, "> {}", name)?;
                parsetree_dump(tfp, &tr, cm, &dsq)?;
                writeln!(tfp, "//")?;
                Ok(())
            })();
            if let Err(e) = res {
                die(&format!("Error writing parsetree file: {}", e));
            }
        }
        names.push(name);
        dsqs.push(dsq);
        trs.push(tr);
    }

    // C: cmemit passes postcode == NULL -> do_post = FALSE. No per-seq PP, and
    // the #=GR PP margin is NOT reserved (unlike cmalign, which always sets it).
    let ppstrs: Vec<Option<Vec<u8>>> = vec![None; nseq];
    // C cmemit.c:527: do_full=TRUE, do_matchonly=FALSE, allow_trunc=FALSE; no flush.
    let mut msa =
        parsetrees_to_alignment(cm, &abc_out, &names, &dsqs, &trs, &ppstrs, false, false, false, false);

    // C: esl_strdup(cm->name, ...) -> msa->name; esl_msa_FormatDesc(...) -> msa->desc.
    // use_cm is TRUE here, so the desc has no " [hmm-mode]" suffix.
    msa.name = Some(cm.name.clone());
    msa.desc = Some("Synthetic sequence alignment generated by cmemit".to_string());

    // C: if(do_truncate) truncate_msa(). do_truncate = both --a5p and --a3p set.
    if let (Some(a5p), Some(a3p)) = (o.a5p, o.a3p) {
        truncate_msa(&mut msa, r, a5p, a3p);
    }

    // C: esl_msafile_Write(cfg->ofp, msa, outfmt).
    if let Err(e) = esl_msafile_write(&mut out, &msa, o.outfmt) {
        die(&format!("Writing alignment file failed: {}", e));
    }
}

/// C: cm.c:Statetype(type) — short state-type mnemonic used by ParsetreeDump.
fn statetype(sttype: i32) -> &'static str {
    match sttype {
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
        _ => "?",
    }
}

/// C: cm.c:MarginalMode(mode) — marginal-mode name used by ParsetreeDump. Emitted
/// parsetrees are always TRMODE_J ("Joint").
fn marginal_mode(mode: i8) -> &'static str {
    match mode {
        3 => "Joint",   // TRMODE_J
        2 => "Left",    // TRMODE_L
        1 => "Right",   // TRMODE_R
        0 => "Term",    // TRMODE_T
        _ => "Unkwn",   // TRMODE_UNKNOWN
    }
}

/// C: alphabet.c:DegeneratePairScore(abc, esc, syml, symr) — canonical fast path.
/// Emitted parsetrees only ever carry canonical residues (`syml,symr < K`), so this
/// returns the direct pair score `esc[syml*K+symr]`, matching C's canonical branch.
fn degenerate_pair_score(esc_v: &[f32], syml: usize, symr: usize) -> f32 {
    let k = ALPHABET_SIZE;
    esc_v[syml * k + symr]
}

/// C: cm_parsetree.c:ParsetreeScore() — score an emitted (standard, Joint-mode)
/// parsetree. Returns `(sc, struct_sc)` in bits; cmemit calls this with
/// `do_null2=FALSE`, `emap=NULL`. `dsq` is the 1-based sentinel-padded digital
/// sequence. Global emitted trees never take the local-begin/local-end branches,
/// but they are ported for the `-l` case.
fn parsetree_score(cm: &CM, tr: &Parsetree, dsq: &[u8]) -> (f32, f32) {
    let k = ALPHABET_SIZE;
    let mut sc = 0.0f32;
    let mut struct_sc = 0.0f32;
    for tidx in 0..tr.n as usize {
        let v = tr.state[tidx];
        // C reads tr->mode[tidx] for truncated (L/R) marginal scores; emitted
        // parsetrees are all Joint, so mode is not needed for the score here.
        if v == cm.m {
            continue; // EL, local end
        }
        let vu = v as usize;
        let st = cm.sttype[vu] as i32;
        if st != E_ST && st != B_ST {
            // transition score contribution
            if tr.nxtl[tidx] == -1 {
                // truncated end: no transition contribution (never for emitted)
            } else {
                let y = tr.state[tr.nxtl[tidx] as usize];
                if v == 0 {
                    if (cm.flags & CM_LOCAL_BEGIN) != 0 {
                        sc += cm.beginsc[y as usize];
                    } else {
                        sc += cm.tsc[vu][(y - cm.cfirst[vu]) as usize];
                    }
                } else if y == cm.m {
                    // local end (CMH_LOCAL_END). Emitted trees are Joint mode.
                    let sd = infernox::cm::state_delta(st);
                    sc += cm.endsc[vu]
                        + cm.el_selfsc * (tr.emitr[tidx] - tr.emitl[tidx] + 1 - sd) as f32;
                } else {
                    sc += cm.tsc[vu][(y - cm.cfirst[vu]) as usize];
                }
            }
            // emission score contribution (mode is Joint for emitted trees)
            if st == MP_ST {
                let symi = dsq[tr.emitl[tidx] as usize] as usize;
                let symj = dsq[tr.emitr[tidx] as usize] as usize;
                // canonical residues: sc/struct_sc use esc[symi*K+symj].
                sc += cm.esc[vu][symi * k + symj];
                struct_sc += cm.esc[vu][symi * k + symj];
                let lsc = cm.lmesc[vu][symi];
                let rsc = cm.rmesc[vu][symj];
                struct_sc -= lsc;
                struct_sc -= rsc;
            } else if matches!(st, ML_ST | IL_ST) {
                let symi = dsq[tr.emitl[tidx] as usize] as usize;
                sc += cm.esc[vu][symi];
            } else if matches!(st, MR_ST | IR_ST) {
                let symj = dsq[tr.emitr[tidx] as usize] as usize;
                sc += cm.esc[vu][symj];
            }
        }
    }
    (sc, struct_sc)
}

/// C: cm_parsetree.c:ParsetreeDump(fp, tr, cm, dsq) — dump a parse tree in the
/// human-readable table form used by cmemit --tfile. `is_std/pass_idx/trpenalty`
/// are constants for emitted (standard, untruncated) parsetrees (TRUE/1/0.000).
/// `dsq` is the 1-based sentinel-padded digital sequence.
fn parsetree_dump(out: &mut dyn Write, tr: &Parsetree, cm: &CM, dsq: &[u8]) -> io::Result<()> {
    let k = ALPHABET_SIZE;
    const SYM: &[u8; 4] = b"ACGU"; // cm->abc->sym for RNA (canonical residues)

    writeln!(out, "Parsetree dump")?;
    writeln!(out, "------------------")?;
    writeln!(out, "is_std              = TRUE (alignment is not truncated)")?;
    writeln!(out, "pass_idx            = 1")?;
    writeln!(out, "trpenalty           = 0.000")?;
    writeln!(out, "parsetree:")?;
    writeln!(out)?;
    writeln!(
        out,
        "{:>5} {:>6} {:>6} {:>7} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        " idx ", "emitl", "emitr", "state", " mode", " nxtl", " nxtr", " prv ", " tsc ", " esc "
    )?;
    writeln!(
        out,
        "{:>5} {:>6} {:>6} {:>7} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "-----", "------", "------", "-------", "-----", "-----", "-----", "-----", "-----", "-----"
    )?;

    for x in 0..tr.n as usize {
        let v = tr.state[x];
        let mode = tr.mode[x];
        let el = v == cm.m;
        let st = if el { EL_ST } else { cm.sttype[v as usize] as i32 };

        // syml, symr, esc (only P/L/R states emit; emitted trees are Joint mode).
        let mut syml = b' ';
        let mut symr = b' ';
        let mut esc = 0.0f32;
        if st == MP_ST {
            syml = SYM[dsq[tr.emitl[x] as usize] as usize];
            symr = SYM[dsq[tr.emitr[x] as usize] as usize];
            esc = degenerate_pair_score(
                &cm.esc[v as usize],
                dsq[tr.emitl[x] as usize] as usize,
                dsq[tr.emitr[x] as usize] as usize,
            );
        } else if matches!(st, IL_ST | ML_ST) {
            syml = SYM[dsq[tr.emitl[x] as usize] as usize];
            esc = infernox::cm::abc_favg_score(dsq[tr.emitl[x] as usize] as usize, &cm.esc[v as usize]);
        } else if matches!(st, IR_ST | MR_ST) {
            symr = SYM[dsq[tr.emitr[x] as usize] as usize];
            esc = infernox::cm::abc_favg_score(dsq[tr.emitr[x] as usize] as usize, &cm.esc[v as usize]);
        }

        // tsc: transition score (0 for B, E, EL, or truncated end).
        let mut tsc = 0.0f32;
        if !el && st != B_ST && st != E_ST && tr.nxtl[x] != -1 {
            let y = tr.state[tr.nxtl[x] as usize];
            if v == 0 {
                if (cm.flags & CM_LOCAL_BEGIN) != 0 {
                    tsc = cm.beginsc[y as usize];
                } else {
                    tsc = cm.tsc[v as usize][(y - cm.cfirst[v as usize]) as usize];
                }
            } else if y == cm.m {
                let sd = infernox::cm::state_delta(st);
                tsc = cm.endsc[v as usize]
                    + cm.el_selfsc * (tr.emitr[x] - tr.emitl[x] + 1 - sd) as f32;
            } else {
                tsc = cm.tsc[v as usize][(y - cm.cfirst[v as usize]) as usize];
            }
        }
        let _ = k;

        // C: "%5d %5d%c %5d%c %5d%-2s %5s %5d %5d %5d %5.2f %5.2f\n"
        writeln!(
            out,
            "{:5} {:5}{} {:5}{} {:5}{:<2} {:>5} {:5} {:5} {:5} {:5.2} {:5.2}",
            x,
            tr.emitl[x],
            syml as char,
            tr.emitr[x],
            symr as char,
            tr.state[x],
            statetype(st),
            marginal_mode(mode),
            tr.nxtl[x],
            tr.nxtr[x],
            tr.prv[x],
            tsc,
            esc,
        )?;
    }
    writeln!(
        out,
        "{:>5} {:>6} {:>6} {:>7} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "-----", "------", "------", "-------", "-----", "-----", "-----", "-----", "-----", "-----"
    )?;
    Ok(())
}

/// C: esl_abc_CIsGap for eslRNA/eslDNA — a column-annotation char is a gap iff it
/// is one of '-', '_', '.' (the three inmap==K gap symbols; '~' is missing-data,
/// not a gap). Used to count consensus (non-gap RF) columns in truncate_msa.
fn c_is_gap(c: u8) -> bool {
    c == b'-' || c == b'_' || c == b'.'
}

/// C: esl_msa_ColumnSubset (esl_msa.c) restricted to the annotations cmemit's
/// text-mode MSA carries. Keeps only columns `apos` (0-based) with `useme[apos] !=
/// 0`, rewriting aseq/ss_cons/rf and any present per-column / per-sequence markup.
/// RemoveBrokenBasepairs is a no-op here because truncate_msa has already zeroed
/// every base pair that crosses the kept region before rebuilding ss_cons.
fn esl_msa_column_subset(msa: &mut EslMsa, useme: &[u8]) {
    let keep = |s: &str| -> String {
        s.bytes()
            .enumerate()
            .filter(|&(i, _)| useme.get(i).copied().unwrap_or(0) != 0)
            .map(|(_, c)| c as char)
            .collect()
    };
    for a in msa.aseq.iter_mut() {
        *a = keep(a);
    }
    if let Some(ref s) = msa.ss_cons {
        msa.ss_cons = Some(keep(s));
    }
    if let Some(ref s) = msa.rf {
        msa.rf = Some(keep(s));
    }
    if let Some(ref s) = msa.pp_cons {
        msa.pp_cons = Some(keep(s));
    }
    if let Some(ref s) = msa.sa_cons {
        msa.sa_cons = Some(keep(s));
    }
    if let Some(ref s) = msa.mm {
        msa.mm = Some(keep(s));
    }
    for opt in [&mut msa.ss, &mut msa.sa, &mut msa.pp] {
        if let Some(v) = opt.as_mut() {
            for e in v.iter_mut() {
                if let Some(s) = e.as_mut() {
                    *s = keep(s);
                }
            }
        }
    }
    // New alignment length = number of kept columns.
    msa.alen = msa.aseq.first().map(|a| a.len() as i64).unwrap_or(0);
}

/// C: cmemit.c:truncate_msa() — truncate the alignment outside consensus columns
/// [spos..epos], removing any consensus structure that crosses the boundary.
/// Called only when both --a5p and --a3p are set (C `do_truncate`). `--a5p 0` /
/// `--a3p 0` pick random consensus positions via esl_rnd_Roll (drawn spos-then-epos
/// after all emission, matching C's shared RNG order). The `abc` is the CM alphabet
/// (RNA) for the gap test on RF.
fn truncate_msa(msa: &mut EslMsa, r: &mut EslRandomFast, a5p: i32, a3p: i32) {
    let alen = msa.alen as usize;
    let rf = msa
        .rf
        .as_ref()
        .unwrap_or_else(|| die("cmemit: --a5p/--a3p require an MSA with RF annotation"))
        .as_bytes()
        .to_vec();
    let ss_cons = msa
        .ss_cons
        .as_ref()
        .unwrap_or_else(|| die("cmemit: --a5p/--a3p require an MSA with SS_cons annotation"))
        .as_bytes()
        .to_vec();

    // C: set_spos/set_epos are always TRUE here (do_truncate requires both set).
    let rnd_spos = a5p == 0;
    let rnd_epos = a3p == 0;

    // C: determine clen (# non-gap RF columns).
    let mut clen = 0i32;
    for apos in 0..alen {
        if !c_is_gap(rf[apos]) {
            clen += 1;
        }
    }
    // C: with --a3p <n>, <n> must be <= clen (uses a5p's set flag as the guard).
    if a3p > clen {
        die(&format!(
            "with --a3p <n> option, <n> must be <= consensus length of CM ({}).",
            clen
        ));
    }
    // C: if both set, either both 0 or neither 0.
    if (rnd_spos && !rnd_epos) || (!rnd_spos && rnd_epos) {
        die("with --a5p <n1> and --a3p <n2>, either <n1> and <n2> must be 0, or neither must be 0");
    }

    // C: determine spos then epos. Roll draws (random path) happen in this order.
    let mut spos = if rnd_spos { r.roll(clen as u32) as i32 + 1 } else { a5p };
    let mut epos = if rnd_epos { r.roll(clen as u32) as i32 + 1 } else { a3p };

    // C: ensure spos <= epos.
    if spos > epos {
        if rnd_spos && rnd_epos {
            std::mem::swap(&mut spos, &mut epos);
        } else {
            die("with --a5p <n1> and --a3p <n2>, <n1> must be <= <n2>");
        }
    }

    // C: remove pknots in place, then build ct array from the structure.
    let mut ss = ss_cons.clone();
    wuss_nopseudo(&mut ss);
    let mut ct = match wuss2ct(&ss, alen) {
        Ok(ct) => ct,
        Err(()) => die("cmemit: failed to parse SS_cons structure for truncation"),
    };

    // C: build useme and clean ct. Placement of cc++ mirrors C exactly.
    let mut useme = vec![0u8; alen];
    let mut cc = 0i32;
    for apos in 0..alen {
        if cc < (spos - 1) || cc > epos {
            useme[apos] = 0;
            let p = ct[apos + 1];
            if p != 0 {
                ct[p as usize] = 0;
            }
            ct[apos + 1] = 0;
        } else {
            useme[apos] = 1;
        }
        if !c_is_gap(rf[apos]) {
            cc += 1;
            if cc == (epos + 1) {
                useme[apos] = 0;
                let p = ct[apos + 1];
                if p != 0 {
                    ct[p as usize] = 0;
                }
                ct[apos + 1] = 0;
            }
        }
    }

    // C: rebuild ss_cons from the cleaned ct array.
    let new_ss = match ct2wuss(&ct, alen) {
        Ok(s) => s,
        Err(()) => die("cmemit: failed to rebuild SS_cons after truncation"),
    };
    msa.ss_cons = Some(String::from_utf8_lossy(&new_ss).into_owned());

    // C: esl_msa_ColumnSubset(msa, useme).
    esl_msa_column_subset(msa, &useme);
}

/// C: cm_parsetree.c:InsertTraceNode() (mode = TRMODE_J). Appends a node to the
/// parse tree and fixes up the parent/child linkage exactly as C does. Returns
/// the new node index (tpos). The Parsetree field vectors are 1:1 with C's
/// tr->{emitl,emitr,state,nxtl,nxtr,prv,mode} arrays.
fn insert_trace_node(
    tr: &mut Parsetree,
    y: i32,
    whichway: i32,
    emitl: i32,
    emitr: i32,
    state: i32,
) -> i32 {
    let n = tr.n;
    // a == -1 unless we're inserting into an existing tree (never here).
    let a = if y >= 0 {
        if whichway == TRACE_LEFT_CHILD {
            tr.nxtl[y as usize]
        } else {
            tr.nxtr[y as usize]
        }
    } else {
        -1
    };
    tr.emitl.push(emitl);
    tr.emitr.push(emitr);
    tr.state.push(state);
    tr.mode.push(3); // TRMODE_J
    tr.nxtl.push(a);
    tr.nxtr.push(-1);
    tr.prv.push(y);
    if y >= 0 {
        if whichway == TRACE_LEFT_CHILD {
            tr.nxtl[y as usize] = n;
        } else {
            tr.nxtr[y as usize] = n;
        }
    }
    if a != -1 {
        tr.prv[a as usize] = n;
    }
    tr.n += 1;
    n
}

/// C: cm_parsetree.c:EmitParsetree() — faithful port that builds BOTH the parse
/// tree and the emitted digital sequence (the `-a`/`--tfile` variant of the
/// sequence-only `emit_parsetree_seq`). RNG draws happen in the exact same order
/// as `emit_parsetree_seq`, so both paths reproduce C's stream for a fixed seed.
///
/// Returns `(tr, dsq)` where `dsq` is 1-based and sentinel-padded
/// (`dsq[0] = dsq[N+1] = 255`, `dsq[1..=N]` = canonical residue indices), the
/// format Parsetrees2Alignment expects (`sq->dsq`).
fn emit_parsetree(cm: &CM, r: &mut EslRandomFast) -> (Parsetree, Vec<u8>) {
    let k = ALPHABET_SIZE as i32; // cm->abc->K == 4 for RNA
    let mut tr = Parsetree::new(100); // C: CreateParsetree(100)
    let mut pda: Vec<Frame> = Vec::new();
    let mut gsq: Vec<u8> = Vec::new(); // growing emitted seq (residue indices)
    let mut n: i32 = 0; // C: N, current emitted length
    let mut tmp_tvec = [0f32; (MAXCONNECT as usize) + 1];

    // C init: push root state's info (v=0, tparent=-1, whichway=LEFT).
    pda.push(Frame::State {
        rchar: -1,
        lchar: -1,
        whichway: TRACE_LEFT_CHILD,
        tparent: -1,
        v: 0,
    });

    while let Some(frame) = pda.pop() {
        match frame {
            Frame::Residue { rchar, tpos } => {
                // C: PDA_RESIDUE — emit deferred right char, set tr->emitr[tpos].
                if rchar != -1 {
                    gsq.push(rchar as u8);
                    n += 1;
                }
                tr.emitr[tpos as usize] = n;
            }
            Frame::State {
                rchar,
                lchar,
                whichway,
                tparent,
                v,
            } => {
                // C: PDA_STATE — attach v to tparent; emitl = N+1, emitr deferred.
                let tpos = insert_trace_node(&mut tr, tparent, whichway, n + 1, -1, v);

                if lchar != -1 {
                    gsq.push(lchar as u8);
                    n += 1;
                }

                // Push deferred right-emission marker for v (even if rchar == -1).
                pda.push(Frame::Residue { rchar, tpos });

                if cm.sttype[v as usize] as i32 == B_ST {
                    let y = cm.cfirst[v as usize]; // left child
                    let z = cm.cnum[v as usize]; // right child
                    // Push right start, then left start (left expands first).
                    pda.push(Frame::State {
                        rchar: -1,
                        lchar: -1,
                        whichway: TRACE_RIGHT_CHILD,
                        tparent: tpos,
                        v: z,
                    });
                    pda.push(Frame::State {
                        rchar: -1,
                        lchar: -1,
                        whichway: TRACE_LEFT_CHILD,
                        tparent: tpos,
                        v: y,
                    });
                } else {
                    // Decide next state y (RNG draw order identical to -u path).
                    let y: i32;
                    if v == 0 && (cm.flags & CM_LOCAL_BEGIN) != 0 {
                        y = esl_rnd_fchoose(r, &cm.begin, cm.m as usize) as i32;
                    } else if (cm.flags & CM_LOCAL_END) != 0 {
                        for x in tmp_tvec.iter_mut() {
                            *x = 0.0;
                        }
                        let cn = cm.cnum[v as usize] as usize;
                        tmp_tvec[..cn].copy_from_slice(&cm.t[v as usize][..cn]);
                        tmp_tvec[cn] = cm.end[v as usize];
                        let off = esl_rnd_fchoose(r, &tmp_tvec, cn + 1) as i32;
                        if off == cm.cnum[v as usize] {
                            y = cm.m; // local end (EL)
                        } else {
                            y = cm.cfirst[v as usize] + off;
                        }
                    } else {
                        y = cm.cfirst[v as usize]
                            + esl_rnd_fchoose(r, &cm.t[v as usize], cm.cnum[v as usize] as usize)
                                as i32;
                    }

                    let yst = if y == cm.m {
                        EL_ST
                    } else {
                        cm.sttype[y as usize] as i32
                    };

                    // Sample emission char(s) for y (C switch on cm->sttype[y]).
                    let mut ylchar: i32 = -1;
                    let mut yrchar: i32 = -1;
                    match yst {
                        MP_ST => {
                            let x = esl_rnd_fchoose(r, &cm.e[y as usize], (k * k) as usize) as i32;
                            ylchar = x / k;
                            yrchar = x % k;
                        }
                        ML_ST | IL_ST => {
                            ylchar = esl_rnd_fchoose(r, &cm.e[y as usize], k as usize) as i32;
                        }
                        MR_ST | IR_ST => {
                            yrchar = esl_rnd_fchoose(r, &cm.e[y as usize], k as usize) as i32;
                        }
                        _ => {}
                    }

                    if yst == E_ST {
                        // C: InsertTraceNode(tr, tpos, LEFT, N+1, N, y). Single node.
                        insert_trace_node(&mut tr, tpos, TRACE_LEFT_CHILD, n + 1, n, y);
                    } else if yst == EL_ST {
                        // EL emits on transition; single trace node spanning lpos..N.
                        let lpos = n + 1; // remember before emitting
                        for x in tmp_tvec.iter_mut() {
                            *x = 0.0;
                        }
                        tmp_tvec[0] = (cm.el_selfsc as f64).exp2() as f32; // EL self prob
                        tmp_tvec[1] = 1.0 - tmp_tvec[0]; // prob of implicit END
                        let mut yy = esl_rnd_fchoose(r, &tmp_tvec, 2);
                        while yy == 0 {
                            let lc = esl_rnd_fchoose(r, &cm.null, k as usize);
                            gsq.push(lc as u8);
                            n += 1;
                            yy = esl_rnd_fchoose(r, &tmp_tvec, 2);
                        }
                        // C: InsertTraceNode(tr, tpos, LEFT, lpos, N, cm->M).
                        insert_trace_node(&mut tr, tpos, TRACE_LEFT_CHILD, lpos, n, cm.m);
                    } else {
                        // Defer expansion of y as left child of v.
                        pda.push(Frame::State {
                            rchar: yrchar,
                            lchar: ylchar,
                            whichway: TRACE_LEFT_CHILD,
                            tparent: tpos,
                            v: y,
                        });
                    }
                }
            }
        }
    }

    // Build the 1-based, sentinel-padded digital sequence (C: sq->dsq).
    let mut dsq: Vec<u8> = Vec::with_capacity(gsq.len() + 2);
    dsq.push(255);
    dsq.extend_from_slice(&gsq);
    dsq.push(255);

    (tr, dsq)
}

/// Parse-tree PDA frame for `emit_parsetree`. Mirrors the two C `pda` frame
/// kinds, but keeps the full parse-tree bookkeeping fields (tpos/tparent/
/// whichway) that the sequence-only `Frame` in cm_emit.rs drops.
enum Frame {
    /// C: PDA_RESIDUE marker — deferred right char + the tree node it closes.
    Residue { rchar: i32, tpos: i32 },
    /// C: PDA_STATE marker — a state v to expand, with its decided emission
    /// chars and how to attach it (whichway) to its parent node (tparent).
    State {
        rchar: i32,
        lchar: i32,
        whichway: i32,
        tparent: i32,
        v: i32,
    },
}
