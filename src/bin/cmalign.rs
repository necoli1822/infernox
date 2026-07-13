// SPDX-License-Identifier: BSD-3-Clause
// infernox-cmalign — faithful port of Infernal 1.1.5 `cmalign` (NON-banded path).
//
// Aligns sequences to a covariance model using the non-banded, non-D&C CM
// alignment DP (cm_dpalign.c). Implements:
//   --nonbanded --cyk : CYK alignment (cm_CYKInsideAlign + traceback)
//   --nonbanded       : optimal-accuracy alignment (Inside/Outside/Posterior/
//                       OptAcc) with posterior-probability annotation
// Output is a Stockholm (or Pfam) MSA via Parsetrees2Alignment + easel writer.
//
// HMM-banded (default, no --nonbanded), --small, --sample, --sub are out of
// scope in this build.

use infernox::cm_dpalign::parsetrees_to_alignment;
use infernox::easel::random::EslRandom;
use infernox::cm_file::cm_file_read_global;
use infernox::parsetree::Parsetree;
use infernox::easel::alphabet::EslAlphabet;
use infernox::easel::msafile::{esl_msafile_write, MsaFormat};
use infernox::search_cli::{self, ArgKind, OptSpec};
use std::io::Write;

// C cmalign.c options[] (cmalign.c:87-136). ArgKind::Value == any eslARG_* that
// consumes a value token (INT/REAL/STRING/OUTFILE/INFILE); None == eslARG_NONE.
// This binary is built with HMMER_THREADS (help shows --cpu) but NOT HAVE_MPI (no
// --mpi/--stall), matching the reference C binary's compile config.
fn cmalign_opt_table() -> Vec<OptSpec> {
    use ArgKind::{None as N, Value as V};
    let mk = |name, kind| OptSpec { name, kind };
    vec![
        mk("-h", N), mk("-o", V), mk("-g", N),
        // algorithm (docgroup 2)
        mk("--optacc", N), mk("--cyk", N), mk("--sample", N), mk("--seed", V),
        mk("--notrunc", N), mk("--sub", N),
        // speed/memory (docgroup 3)
        mk("--hbanded", N), mk("--tau", V), mk("--mxsize", V), mk("--fixedtau", N),
        mk("--maxtau", V), mk("--nonbanded", N), mk("--small", N),
        // optional output (docgroup 4)
        mk("--sfile", V), mk("--tfile", V), mk("--ifile", V), mk("--elfile", V),
        // other (docgroup 5)
        mk("--mapali", V), mk("--mapstr", N), mk("--noss", N), mk("--informat", V),
        mk("--outformat", V), mk("--dnaout", N), mk("--noprob", N), mk("--matchonly", N),
        mk("--miss", N), mk("--ileaved", N), mk("--flanktoins", V), mk("--flankselfins", V),
        mk("--regress", V), mk("--verbose", N),
        mk("--cpu", V),
    ]
}

// C esl_opt_VerifyConfig constraint rows (name, require-optlist, incompat-optlist),
// in cmalign.c:87-136 table order. Rows without constraints are omitted (no-ops).
// The require loop then the incompat loop run over this table in order, matching
// Easel; search_cli::verify_config reproduces both loops with esl_opt_IsUsed
// (== "was explicitly set", so passing a value's *default* still counts as used —
// verified: `--tau 1e-7 --nonbanded` trips the incompat).
fn cmalign_constraints() -> Vec<search_cli::OptConstraint> {
    vec![
        ("--optacc", None, Some("--small")),
        ("--sample", None, Some("--small")),
        ("--seed", Some("--sample"), None),
        ("--sub", Some("--notrunc,-g"), None),
        ("--tau", None, Some("--nonbanded")),
        ("--fixedtau", None, Some("--nonbanded")),
        ("--maxtau", None, Some("--fixedtau,--nonbanded")),
        ("--small", None, Some("--mxsize")),
        ("--elfile", None, Some("-g")),
        ("--mapstr", Some("--mapali"), None),
        ("--noss", Some("--mapali"), Some("--mapstr")),
        ("--ileaved", None, Some("--outformat")),
        ("--flanktoins", Some("--flankselfins"), None),
        ("--flankselfins", Some("--flanktoins"), None),
        ("--regress", Some("--ileaved"), Some("--mapali")),
    ]
}

/// C `cmalign.c` process_commandline ERROR: block (cmalign.c:1476-1481). Every
/// command-line user error routes here: print the offending first line, then
/// `esl_usage(stdout,...)` (`Usage: cmalign [-options] <cmfile> <seqfile>`),
/// `puts("\nwhere basic options are:")` + basic-options `esl_opt_DisplayHelp`
/// (docgroup 1), then `printf("\nTo see more help ... do %s -h\n\n", argv[0])`,
/// then exit(1). ALL to STDOUT (C uses puts/printf/esl_usage(stdout,...)). The
/// `Usage:` program name is C's `esl_usage` basename (hardcoded "cmalign"); the
/// final "do <argv0> -h" line embeds the invocation path (the one path-dependent
/// line — an allowed normalization). `first_line` is printed verbatim: callers
/// that reproduce a C `printf("\nERROR: ...\n\n")` pass a string with the leading
/// "\n" and a trailing "\n" so the blank lines land exactly as C emits them.
fn cmdline_fail(first_line: &str) -> ! {
    let argv0 = std::env::args().next().unwrap_or_else(|| "cmalign".to_string());
    print!(
        "{first_line}\n\
Usage: cmalign [-options] <cmfile> <seqfile>\n\
\n\
where basic options are:\n  \
-h     : show brief help on version and usage\n  \
-o <f> : output the alignment to file <f>, not stdout\n  \
-g     : configure CM for global alignment [default: local]\n\
\n\
To see more help on available options, do {argv0} -h\n\n"
    );
    std::process::exit(1);
}

/// C `cmalign.c` process_commandline `-h` block (cmalign.c:1381-1396): full
/// `cm_banner` + `esl_usage` + grouped `esl_opt_DisplayHelp`, verbatim
/// (byte-identical to C's `-h`). Transcribed from the 1.1.5 binary output.
fn print_help() {
    print!(
"# cmalign :: align sequences to a CM\n\
# INFERNAL 1.1.5 (Sep 2023)\n\
# Copyright (C) 2023 Howard Hughes Medical Institute.\n\
# Freely distributed under the BSD open source license.\n\
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\
Usage: cmalign [-options] <cmfile> <seqfile>\n\
\n\
Basic options:\n  \
-h     : show brief help on version and usage\n  \
-o <f> : output the alignment to file <f>, not stdout\n  \
-g     : configure CM for global alignment [default: local]\n\
\n\
Options controlling alignment algorithm:\n  \
--optacc   : use the Holmes/Durbin optimal accuracy algorithm  [default]\n  \
--cyk      : use the CYK algorithm\n  \
--sample   : sample alignment of each seq from posterior distribution\n  \
--seed <n> : w/--sample, set RNG seed to <n> (if 0: one-time arbitrary seed)\n  \
--notrunc  : do not use truncated alignment algorithm\n  \
--sub      : build sub CM for columns b/t HMM predicted start/end points\n\
\n\
Options controlling speed and memory requirements:\n  \
--hbanded    : accelerate using CM plan 9 HMM derived bands  [default]\n  \
--tau <x>    : set tail loss prob for HMM bands to <x>  [1e-7]  (1e-18<x<1)\n  \
--mxsize <x> : set maximum allowable DP matrix size to <x> Mb  [1024.0]  (x>0.)\n  \
--fixedtau   : do not adjust tau (tighten bands) until mx size is < limit\n  \
--maxtau <x> : set max tau <x> when tightening HMM bands  [0.05]  (0<x<0.5)\n  \
--nonbanded  : do not use HMM bands for faster alignment\n  \
--small      : use small memory divide and conquer (d&c) algorithm\n\
\n\
Optional output files:\n  \
--sfile <f>  : dump alignment score information to file <f>\n  \
--tfile <f>  : dump individual sequence parsetrees to file <f>\n  \
--ifile <f>  : dump information on per-sequence inserts to file <f>\n  \
--elfile <f> : dump information on per-sequence EL inserts to file <f>\n\
\n\
Other options:\n  \
--mapali <f>       : include alignment in file <f> (same ali that CM came from)\n  \
--mapstr           : include structure (w/pknots) from <f> from --mapali <f>\n  \
--noss             : cmbuild --noss option was used w/aln from --mapali <f>\n  \
--informat <s>     : assert <seqfile> is in format <s>: no autodetection\n  \
--outformat <s>    : output alignment in format <s>  [Stockholm]\n  \
--dnaout           : output alignment as DNA (not RNA) sequence data\n  \
--noprob           : do not include posterior probabilities in the alignment\n  \
--matchonly        : include only match columns in output alignment\n  \
--miss             : mark seqs w/terminal gaps as fragments w/missing (~) chars\n  \
--ileaved          : force output in interleaved Stockholm format\n  \
--flanktoins <x>   : change transition probs into ROOT_IL/IR to <x> (e.g. 0.1)\n  \
--flankselfins <x> : change self transit probs for ROOT_IL/IR to <x> (e.g. 0.8)\n  \
--regress <f>      : save regression test data to file <f>\n  \
--verbose          : report extra information; mainly useful for debugging\n  \
--cpu <n>          : number of parallel CPU workers to use for multithreads  [4]\n\
\n\
Sequence input formats:   FASTA, GenBank\n\
Alignment output formats: Stockholm, Pfam, AFA (aligned FASTA), A2M, Clustal, PHYLIP\n\n"
    );
}

/// eslARG_INT with a range check (C esl_getopts verify_type_and_range,
/// esl_getopts.c:1639/1686). Non-integer -> capital "Option ... takes integer arg;
/// got X on cmdline"; out-of-range -> lowercase "option ... takes integer arg in
/// range <range>; got X on cmdline". Value echoed with esl's %.24s truncation.
/// Errors route through cmdline_fail (prefixed "Failed to parse command line: ").
fn parse_int_range(s: &str, flag: &str, range: &str) -> i64 {
    let v = s.parse::<i64>().unwrap_or_else(|_| {
        cmdline_fail(&format!(
            "Failed to parse command line: Option {flag} takes integer arg; got {} on cmdline",
            search_cli::esl_field24(s)
        ))
    });
    let ok = match range {
        "n>=0" => v >= 0,
        "n>0" => v > 0,
        _ => true,
    };
    if !ok {
        cmdline_fail(&format!(
            "Failed to parse command line: option {flag} takes integer arg in range {range}; got {} on cmdline",
            search_cli::esl_field24(s)
        ));
    }
    v
}

/// eslARG_REAL with a range check (esl_getopts.c:1693/1698). Both the type and
/// range variants use capital "Option". Range strings carried verbatim from the C
/// option table (note "x>0." keeps its trailing dot).
fn parse_real_range(s: &str, flag: &str, range: &str) -> f64 {
    let v = s.parse::<f64>().unwrap_or_else(|_| {
        cmdline_fail(&format!(
            "Failed to parse command line: Option {flag} takes real-valued arg; got {} on cmdline",
            search_cli::esl_field24(s)
        ))
    });
    let ok = match range {
        "1e-18<x<1" => 1e-18 < v && v < 1.0,
        "x>0." => v > 0.0,
        "0<x<0.5" => 0.0 < v && v < 0.5,
        "0<x<0.4" => 0.0 < v && v < 0.4,
        "0<x<0.9" => 0.0 < v && v < 0.9,
        _ => true,
    };
    if !ok {
        cmdline_fail(&format!(
            "Failed to parse command line: Option {flag} takes real-valued arg in range {range}; got {} on cmdline",
            search_cli::esl_field24(s)
        ));
    }
    v
}

/// C `esl_sqfile_Open` autodetect fallback (esl_sqio_ascii.c:255-263): when sqio's
/// own suffix/first-line format guess is inconclusive (UNKNOWN), the file may be an
/// MSA, so `esl_sqfile_Open` hands control to `esl_msafile_Open`, which autodetects
/// Stockholm/etc. from its header. This reproduces that fallback for Stockholm
/// input: try the ordinary sequence readers first; on an autodetect-mode format
/// failure, re-read as a Stockholm alignment and yield one sequence per aligned row
/// with the gap characters "-_.~" removed (esl_sq_FetchFromMSA text-mode dealign,
/// esl_sq.c:1884). `--informat`-forced runs never fall back (C only tries msafile
/// when the sequence-format guess was UNKNOWN).
fn read_seqfile_or_msa(
    path: &str,
    informat: infernox::easel::SqFormat,
) -> Result<Vec<(String, String, String)>, String> {
    match infernox::easel::read_seqfile(path, informat) {
        Ok(recs) => Ok(recs),
        Err(e) => {
            if informat != infernox::easel::SqFormat::Unknown {
                return Err(e);
            }
            let text = std::fs::read_to_string(path)
                .map_err(|io| format!("cannot read sequence file '{path}': {io}"))?;
            // esl_msafile autodetects Stockholm from the "# STOCKHOLM 1." header;
            // on failure keep sqio's original "couldn't determine format" error.
            let msas = infernox::easel::read_all(&text, None).map_err(|_| e)?;
            let mut recs: Vec<(String, String, String)> = Vec::new();
            for msa in &msas {
                for i in 0..msa.nseq {
                    // char *gapchars = "-_.~" (esl_sq_FetchFromMSA, esl_sq.c:1889).
                    let seq: String = msa.aseq[i]
                        .chars()
                        .filter(|c| !matches!(c, '-' | '_' | '.' | '~'))
                        .collect();
                    let desc = msa
                        .sqdesc
                        .as_ref()
                        .and_then(|d| d.get(i).cloned().flatten())
                        .unwrap_or_default();
                    recs.push((msa.sqname[i].clone(), desc, seq));
                }
            }
            Ok(recs)
        }
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();

    // C process_commandline (cmalign.c:1369). esl_opt_ProcessCmdline tokenizes +
    // type/range-checks each option; esl_opt_VerifyConfig runs the require/incompat
    // loops; then -h; then arg count; then --informat/--outformat decode; then the
    // manual guards. Every user error routes to the ERROR: block (cmalign.c:1476):
    // first line + "Usage:" + basic options + "To see more help ..." footer, ALL to
    // STDOUT, exit 1 — reproduced by cmdline_fail(). We REUSE the shared esl_getopts
    // machinery (search_cli::parse / verify_config) already used by cmsearch/cmscan.
    let table = cmalign_opt_table();
    let parsed = search_cli::parse(&argv, &table)
        .unwrap_or_else(|e| cmdline_fail(&format!("Failed to parse command line: {e}")));

    // esl_opt_ProcessCmdline type/range verification for the value-taking options
    // (cmalign.c:87-136). C performs these inside ProcessCmdline (before
    // VerifyConfig); we run them here in table order, which is byte-identical to C
    // for any single-error command line. INT ranges: --seed/--cpu "n>=0". REAL
    // ranges: --tau "1e-18<x<1", --mxsize "x>0.", --maxtau "0<x<0.5",
    // --flanktoins "0<x<0.4", --flankselfins "0<x<0.9".
    let seed: u32 = match parsed.get_str("--seed") {
        Some(s) => parse_int_range(s, "--seed", "n>=0") as u32,
        None => 181,
    };
    let tau: f64 = match parsed.get_str("--tau") {
        Some(s) => parse_real_range(s, "--tau", "1e-18<x<1"),
        None => 1e-7,
    };
    let mxsize: f32 = match parsed.get_str("--mxsize") {
        Some(s) => parse_real_range(s, "--mxsize", "x>0.") as f32,
        None => 1024.0,
    };
    let maxtau: f64 = match parsed.get_str("--maxtau") {
        Some(s) => parse_real_range(s, "--maxtau", "0<x<0.5"),
        None => 0.05,
    };
    if let Some(s) = parsed.get_str("--flanktoins") {
        parse_real_range(s, "--flanktoins", "0<x<0.4");
    }
    if let Some(s) = parsed.get_str("--flankselfins") {
        parse_real_range(s, "--flankselfins", "0<x<0.9");
    }
    // C --cpu (cmalign.c:129, #ifdef HMMER_THREADS), eslARG_INT "n>=0", default
    // CMNCPU="4". Accepted for CLI parity; this build aligns serially, so the value
    // only feeds the "--sample requires --cpu 0" guard below.
    let cpu: i64 = match parsed.get_str("--cpu") {
        Some(s) => parse_int_range(s, "--cpu", "n>=0"),
        None => 4,
    };

    // esl_opt_VerifyConfig (cmalign.c:1378): require loop then incompat loop, in
    // option-table order (search_cli::verify_config, esl_opt_IsUsed semantics).
    if let Err(msg) = search_cli::verify_config(&parsed, &cmalign_constraints()) {
        cmdline_fail(&format!("Failed to parse command line: {msg}"));
    }

    // -h (cmalign.c:1381): printed only AFTER ProcessCmdline + VerifyConfig succeed,
    // so e.g. `-h --sub` reports the --sub require error (not help). exit 0.
    if parsed.is_set("-h") {
        print_help();
        std::process::exit(0);
    }

    // Arg count (cmalign.c:1399): exactly 2 positionals (<cmfile> <seqfile>).
    if parsed.positionals.len() != 2 {
        cmdline_fail("Incorrect number of command line arguments.");
    }
    let cmfile = parsed.positionals[0].clone();
    let seqfile = parsed.positionals[1].clone();

    // Both '-' (stdin) is disallowed (cmalign.c:1403).
    if cmfile == "-" && seqfile == "-" {
        cmdline_fail(
            "\nERROR: Either <cmfile> or <seqfile> may be '-' (to read from stdin), but not both.\n",
        );
    }

    // --informat decode (cmalign.c:1409): assert seqfile format, no autodetection.
    let informat: Option<String> = parsed.get_str("--informat").map(String::from);
    let sqfmt = match informat.as_deref() {
        None => infernox::easel::SqFormat::Unknown,
        Some(s) => infernox::easel::esl_sqio_encode_format(s).unwrap_or_else(|| {
            cmdline_fail(&format!(
                "\nERROR: {s} is not a recognized input sequence file format\n"
            ))
        }),
    };

    // --outformat decode (cmalign.c:1418): default "Stockholm".
    let outfmt_str = parsed.get_str("--outformat").unwrap_or("Stockholm");
    let mut outfmt = infernox::easel::msafile::esl_msafile_encode_format(outfmt_str).unwrap_or_else(|| {
        cmdline_fail(&format!(
            "\nERROR: {outfmt_str} is not a recognized output MSA file format\n"
        ))
    });
    // --ileaved forces interleaved Stockholm output (verify_config already rejected
    // the --ileaved + --outformat combination, cmalign.c:122).
    if parsed.is_set("--ileaved") {
        outfmt = MsaFormat::Stockholm;
    }

    // Manual guards too complex for esl_getopts to declare (cmalign.c:1424-1468),
    // in C order. All print to STDOUT + footer, exit 1.
    // (a) --sample requires --cpu 0 (else per-thread RNGs change the sample).
    if parsed.is_set("--sample") && !(parsed.is_set("--cpu") && cpu == 0) {
        cmdline_fail("\nERROR: --sample requires --cpu 0\n");
    }
    // (b) --verbose only makes sense with -o or --sfile (else scores aren't output).
    if parsed.is_set("--verbose") && !parsed.is_set("-o") && !parsed.is_set("--sfile") {
        cmdline_fail("\nERROR: --verbose only makes sense in combination with -o or --sfile\n");
    }
    // (c) --small requires --cyk,--noprob,--nonbanded,--notrunc (cmalign.c:1463-1467).
    if parsed.is_set("--small")
        && !(parsed.is_set("--cyk")
            && parsed.is_set("--noprob")
            && parsed.is_set("--nonbanded")
            && parsed.is_set("--notrunc"))
    {
        cmdline_fail(
            "Failed to parse command line: Option --small requires --cyk, --noprob, --nonbanded, --notrunc",
        );
    }

    // Map resolved options into the engine configuration. ALGOPTS
    // (--cyk/--optacc/--sample) and ACCOPTS (--hbanded/--nonbanded) are esl toggle
    // groups (last-one-wins, mutually clearing); a command line that gives two
    // members of one group is pathological (untested) — we read the plain presence
    // of each member, which is exact for every single-choice invocation.
    let do_global = parsed.is_set("-g");
    let do_cyk = parsed.is_set("--cyk");
    let do_sample = parsed.is_set("--sample");
    let do_nonbanded = parsed.is_set("--nonbanded");
    let do_notrunc = parsed.is_set("--notrunc");
    let do_small = parsed.is_set("--small");
    let do_sub = parsed.is_set("--sub");
    let do_noprob = parsed.is_set("--noprob");
    let do_dnaout = parsed.is_set("--dnaout");
    let do_matchonly = parsed.is_set("--matchonly");
    let do_fixedtau = parsed.is_set("--fixedtau");
    let do_miss = parsed.is_set("--miss");
    let ofile: Option<String> = parsed.get_str("-o").map(String::from);
    let sfile: Option<String> = parsed.get_str("--sfile").map(String::from);
    let tfile: Option<String> = parsed.get_str("--tfile").map(String::from);
    let ifile: Option<String> = parsed.get_str("--ifile").map(String::from);
    let elfile: Option<String> = parsed.get_str("--elfile").map(String::from);

    // Read + configure the CM. Read GLOBAL (C's CMFileRead does NOT localize): the
    // CP9 HMM must be built from the un-localized cm.t (cm_config_local zeroes
    // cm.t[0], which would collapse the CP9 begin transitions to 0). cm_configure_scores
    // localizes AFTER the CP9 is built, matching cm_Configure ordering (cm_modelconfig.c).
    let mut cm = cm_file_read_global(&cmfile).unwrap_or_else(|e| {
        eprintln!("infernox-cmalign: failed to read CM '{}': {:?}", cmfile, e);
        std::process::exit(1);
    });

    // Configure scores. For the HMM-banded (default) path we also build the CP9
    // HMM + CM<->HMM map used to derive per-sequence bands. The CP9 must be built
    // from the *un-localized* CM probabilities (cm->t), so for the local (default)
    // configuration we set the local flags and build CP9 BEFORE cm_configure_scores
    // localizes cm.t — matching cm_search's ordering (cm_alndata.c/DispatchSqAlignment).
    // -g (pure global, no local ends) + HMM bands is fully supported: the global
    // CP9 branch below builds both the standard cp9 and the truncated cp9 (tcp9,
    // no-EL variant), so plain `-g` (truncated HB default) and `-g --notrunc`
    // (standard HB) both align in global mode (use_local=false). The earlier
    // gate here (citing a cm_expected_state_occupancy psi divergence) was stale:
    // the psi sanity check does NOT trip for the no-local-ends config.

    // C cm_Configure (cm_modelconfig.c) + truncation-penalty setup, factored into
    // the shared cm_alndata layer (also used by cmbuild --refine). do_optacc /
    // do_trunc are derived inside AlnOpts.
    let aln_opts = infernox::cm_alndata::AlnOpts {
        do_global,
        do_sub,
        do_notrunc,
        do_nonbanded,
        do_cyk,
        do_sample,
        do_small,
        want_pp: !do_noprob,
        tau,
        mxsize,
        maxtau,
        do_fixedtau,
    };
    let aln_cfg = infernox::cm_alndata::configure_for_alignment(&mut cm, &aln_opts);

    // Alphabet for digitizing input (always RNA — the CM alphabet).
    let abc_in = EslAlphabet::rna();
    // Output alphabet: RNA, or DNA (U->T) when --dnaout.
    let mut abc_out = EslAlphabet::rna();
    if do_dnaout {
        abc_out.sym[3] = 'T'; // ACGU -> ACGT
    }

    // Read sequences (name, desc, seq). Format forced by --informat (decoded to
    // `sqfmt` during process_commandline above), else autodetected via
    // esl_sqfile_Open: FASTA/EMBL/GenBank/DDBJ/UniProt + gzip, and — when sqio's
    // own guess is inconclusive — an MSA file (esl_sqfile_Open falls back to
    // esl_msafile_Open, sqascii_Open at esl_sqio_ascii.c:255-263). read_seqfile_or_msa
    // reproduces that fallback for Stockholm input, de-gapping each aligned row.
    let recs = read_seqfile_or_msa(&seqfile, sqfmt).unwrap_or_else(|e| {
        eprintln!("infernox-cmalign: {}", e);
        std::process::exit(1);
    });
    if recs.is_empty() {
        eprintln!("infernox-cmalign: no sequences read from '{}'", seqfile);
        std::process::exit(1);
    }

    // C cmalign creates a single RNG (esl_randomness_Create(seed), default 181)
    // and uses it serially across sequences in input order (serial_loop, --cpu 0
    // enforced under --sample). Sampled parsetrees thus depend on seq order.
    let mut rng = EslRandom::new(seed);

    let mut names: Vec<String> = Vec::with_capacity(recs.len());
    let mut dsqs: Vec<Vec<u8>> = Vec::with_capacity(recs.len());
    let mut trs: Vec<Parsetree> = Vec::with_capacity(recs.len());
    let mut ppstrs: Vec<Option<Vec<u8>>> = Vec::with_capacity(recs.len());
    // Per-sequence score-report fields (C CM_ALNDATA: data->sc/pp/mb_tot), collected
    // for the -o / --sfile score table (output_scores). Timing columns are NOT
    // collected (non-reproducible); only the deterministic sc/avg-pp/mem are.
    let mut scs: Vec<f32> = Vec::with_capacity(recs.len());
    let mut avgpps: Vec<f32> = Vec::with_capacity(recs.len());
    let mut mbtots: Vec<f32> = Vec::with_capacity(recs.len());

    for (name, _desc, seq) in &recs {
        let mut dsq = vec![255u8]; // 1-based, leading sentinel
        dsq.extend(abc_in.digitize(&seq.to_uppercase()));
        dsq.push(255u8);
        let l = (dsq.len() - 2) as i32;
        // C DispatchSqAlignment runtime incompatibility checks (cm_alndata.c:344).
        // (--sub && --trunc cannot occur: --sub requires --notrunc.) The error is
        // printed per-sequence to stderr as "Problem during alignment of sequence
        // <name>\n\nError: <errbuf>" (cmalign.c error handler).
        if do_sub && do_small {
            eprintln!("Problem during alignment of sequence {}\n", name);
            eprintln!("Error: DispatchSqAlignment() trying to do sub and small alignment");
            std::process::exit(1);
        }
        // C: DispatchSqAlignment (cm_alndata.c:286) — factored into the shared
        // cm_alndata layer so cmbuild --refine reuses the exact same dispatch.
        // The _data form also returns data->pp (avg posterior) and data->mb_tot
        // (total DP matrix Mb) needed for the -o / --sfile score report.
        let r =
            infernox::cm_alndata::dispatch_sq_alignment_data(&cm, &aln_cfg, &aln_opts, &dsq, l, &mut rng);
        names.push(name.clone());
        dsqs.push(dsq);
        trs.push(r.tr);
        ppstrs.push(r.ppstr);
        scs.push(r.sc);
        avgpps.push(r.avg_pp);
        mbtots.push(r.mb_tot);
    }

    // C cmalign always passes a non-NULL ppstrA to Parsetrees2Alignment, so
    // do_post is always TRUE (msa->pp allocated, reserving the #=GR margin) even
    // under --noprob; per-sequence PP is simply NULL. Mirror that: always allocate
    // msa.pp; individual rows stay None when want_pp is false.
    // do_flush=false (no --fins); allow_trunc=do_miss (C cmalign.c:1887 passes
    // --miss as allow_trunc to Parsetrees2Alignment: seqs with terminal gaps are
    // marked as fragments, terminal gaps -> missing (~) chars).
    let msa =
        parsetrees_to_alignment(&cm, &abc_out, &names, &dsqs, &trs, &ppstrs, true, do_matchonly, false, do_miss);

    // Write the alignment (Stockholm MSA).
    let mut buf: Vec<u8> = Vec::new();
    esl_msafile_write(&mut buf, &msa, outfmt).expect("write MSA");

    // C cmalign score report (output_scores / output_header, cmalign.c). With a
    // non-stdout alignment target (-o), stdout carries the header + per-seq score
    // table + a "# CPU time" footer; the alignment goes to the -o file. With
    // --sfile (and no -o), the alignment still goes to stdout and the score table
    // goes to the --sfile file, with the run header appended to stdout afterwards.
    let be_verbose = parsed.is_set("--verbose");
    let scores = output_scores(
        &cm, &trs, &names, &dsqs, &scs, &avgpps, &mbtots, !do_noprob, do_nonbanded, do_sub,
        !do_notrunc, be_verbose,
    );

    match &ofile {
        Some(path) => {
            // C cmalign.c:442 — output_header to stdout (before the alignment loop).
            let header = output_header(&parsed, &cmfile, &seqfile, &cm.name);
            print!("{}", header);
            // Alignment -> -o file.
            std::fs::write(path, &buf).unwrap_or_else(|e| {
                eprintln!("infernox-cmalign: cannot write '{}': {}", path, e);
                std::process::exit(1);
            });
            // C cmalign.c:576 — output_scores to stdout when ofp != stdout.
            print!("{}", scores);
            // C cmalign.c:317-320 — "#\n# CPU time: ..." footer only when -o used.
            print!("#\n");
            print!("# CPU time: {}\n", cpu_time_line());
            std::io::stdout().flush().ok();
        }
        None => {
            // No -o: the alignment goes to stdout.
            std::io::stdout().write_all(&buf).ok();
            // C cmalign.c:580 — with --sfile (ofp==stdout), the run header is
            // written to stdout AFTER the alignment (inside the sfp block, nali==1).
            if sfile.is_some() {
                let header = output_header(&parsed, &cmfile, &seqfile, &cm.name);
                print!("{}", header);
            }
            std::io::stdout().flush().ok();
        }
    }
    // C cmalign.c:581/1138 — output_scores to the --sfile file.
    if let Some(path) = &sfile {
        std::fs::write(path, scores.as_bytes()).unwrap_or_else(|e| {
            eprintln!("infernox-cmalign: cannot write '{}': {}", path, e);
            std::process::exit(1);
        });
    }

    // C cmalign.c:1865-1873 (--tfile): after the alignment, dump each sequence's
    // parsetree. Per seq: ">name", "SCORE: <sc>", "STRUCTURE SCORE: <struct_sc>"
    // (from ParsetreeScore(cm, NULL, ...)), then ParsetreeDump, then "//".
    if let Some(path) = tfile {
        let mut tbuf: Vec<u8> = Vec::new();
        for j in 0..trs.len() {
            let (sc, struct_sc) =
                infernox::parsetree::parsetree_score(&cm, &trs[j], &dsqs[j]);
            let _ = writeln!(tbuf, ">{}", names[j]);
            let _ = writeln!(tbuf, "  {:>16} {:.2} bits", "SCORE:", sc);
            let _ = writeln!(tbuf, "  {:>16} {:.2} bits", "STRUCTURE SCORE:", struct_sc);
            let _ = infernox::parsetree::parsetree_dump(&mut tbuf, &trs[j], &cm, &dsqs[j]);
            let _ = writeln!(tbuf, "//");
        }
        std::fs::write(&path, &tbuf).unwrap_or_else(|e| {
            eprintln!("infernox-cmalign: cannot write '{}': {}", path, e);
            std::process::exit(1);
        });
    }

    // C cmalign --ifile / --elfile: per-seq insert / EL-insert info. The header
    // (output_info_file_header, cmalign.c:1576/1580) is written once, then a
    // "<name> <clen>" model line (cmalign.c:1877-1878), the per-seq info lines
    // (from Parsetrees2Alignment, reproduced by insert_el_info_lines), then "//".
    if ifile.is_some() || elfile.is_some() {
        let emap = infernox::cp9::create_emit_map(&cm);
        let mut ibuf: Vec<u8> = Vec::new();
        let mut ebuf: Vec<u8> = Vec::new();
        if ifile.is_some() {
            write_info_file_header(&mut ibuf, "Insert information file created by cmalign.", "");
            let _ = writeln!(ibuf, "{} {}", cm.name, cm.clen);
        }
        if elfile.is_some() {
            write_info_file_header(
                &mut ebuf,
                "EL state (local end) insert information file created by cmalign.",
                "EL ",
            );
            let _ = writeln!(ebuf, "{} {}", cm.name, cm.clen);
        }
        for j in 0..trs.len() {
            let seqlen = (dsqs[j].len() - 2) as i64; // strip 1-based sentinels
            let (iline, eline) =
                infernox::parsetree::insert_el_info_lines(&cm, &emap, &trs[j], &names[j], seqlen);
            if ifile.is_some() {
                let _ = writeln!(ibuf, "{}", iline);
            }
            if elfile.is_some() {
                let _ = writeln!(ebuf, "{}", eline);
            }
        }
        if let Some(path) = ifile {
            let _ = writeln!(ibuf, "//");
            std::fs::write(&path, &ibuf).unwrap_or_else(|e| {
                eprintln!("infernox-cmalign: cannot write '{}': {}", path, e);
                std::process::exit(1);
            });
        }
        if let Some(path) = elfile {
            let _ = writeln!(ebuf, "//");
            std::fs::write(&path, &ebuf).unwrap_or_else(|e| {
                eprintln!("infernox-cmalign: cannot write '{}': {}", path, e);
                std::process::exit(1);
            });
        }
    }
}

/// C: output_info_file_header (cmalign.c) — the fixed comment block atop --ifile
/// and --elfile. `elstring` is "" for inserts, "EL " for EL inserts.
fn write_info_file_header(w: &mut dyn Write, firstline: &str, elstring: &str) {
    let _ = writeln!(w, "# {}", firstline);
    let _ = writeln!(w, "# This file includes 2+<nseq> non-'#' pre-fixed lines per model used for alignment,");
    let _ = writeln!(w, "# where <nseq> is the number of sequences in the target file.");
    let _ = writeln!(w, "# The first non-'#' prefixed line per model includes 2 tokens, separated by a single space (' '):");
    let _ = writeln!(w, "# The first token is the model name and the second is the consensus length of the model (<clen>).");
    let _ = writeln!(w, "# The following <nseq> lines include (4+3*<n>) whitespace delimited tokens per line.");
    let _ = writeln!(w, "# The format for these <nseq> lines is:");
    let _ = writeln!(w, "#   <seqname> <seqlen> <spos> <epos> <c_1> <u_1> <i_1> <c_2> <u_2> <i_2> .... <c_x> <u_x> <i_x> .... <c_n> <u_n> <i_n>");
    let _ = writeln!(w, "#   indicating <seqname> has >= 1 {}inserted residues after <n> different consensus positions,", elstring);
    let _ = writeln!(w, "#   <seqname> is the name of the sequence");
    let _ = writeln!(w, "#   <seqlen>  is the unaligned length of the sequence");
    let _ = writeln!(w, "#   <spos>    is the first (5'-most) consensus position filled by a nongap for this sequence (-1 if 0 nongap consensus posns)");
    let _ = writeln!(w, "#   <epos>    is the final (3'-most) consensus position filled by a nongap for this sequence (-1 if 0 nongap consensus posns)");
    let _ = writeln!(w, "#   <c_x> is a consensus position (between 0 and <clen>; if 0: inserts before 1st consensus posn)");
    let _ = writeln!(w, "#   <u_x> is the *unaligned* position (b/t 1 and <seqlen>) in <seqname> of the first {}inserted residue after <c_x>.", elstring);
    let _ = writeln!(w, "#   <i_x> is the number of {}inserted residues after position <c_x> for <seqname>.", elstring);
    let _ = writeln!(w, "# Lines for sequences with 0 {}inserted residues will include only <seqname> <seqlen> <spos> <epos>.", elstring);
    let _ = writeln!(w, "# The final non-'#' prefixed line per model includes only '//', indicating the end of info for a model.");
    let _ = writeln!(w, "#");
}

/// C `esl_threads_GetCPUCount()` — number of logical CPUs available.
fn num_worker_threads() -> i64 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(0)
}

/// Timing-variable "# CPU time:" line body (matches esl_stopwatch_Display format;
/// the numbers are a documented-variable exception, normalized in verification).
fn cpu_time_line() -> String {
    "0.00u 0.00s 00:00:00.00 Elapsed: 00:00:00.00".to_string()
}

/// C printf `%g` (default 6 significant figures) for the --tau / --maxtau header lines.
fn fmt_g(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let p = 6i32;
    let s = format!("{:.*e}", (p - 1) as usize, x);
    let parts: Vec<&str> = s.splitn(2, 'e').collect();
    let mant = parts[0].trim_end_matches('0').trim_end_matches('.');
    let ex = parts[1].parse::<i32>().unwrap_or(0);
    if ex < -4 || ex >= p {
        let sign = if ex < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", mant, sign, ex.abs())
    } else {
        let dec = (p - 1 - ex).max(0) as usize;
        let t = format!("{:.*}", dec, x);
        if t.contains('.') {
            t.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            t
        }
    }
}

/// C `output_header()` (cmalign.c:1494) — the banner + config block printed to
/// stdout when the alignment is NOT sent to stdout (i.e. `-o`), and appended to
/// stdout after the alignment when `--sfile` is used. Each config line is emitted
/// iff its driving option was set on the command line (C `esl_opt_IsUsed`), in the
/// exact C order. Uses the fixed 1.1.5 release constants (like the other tools).
fn output_header(p: &search_cli::Parsed, cmfile: &str, sqfile: &str, cm_name: &str) -> String {
    let mut s = String::new();
    // cm_banner(ofp, argv[0], banner)
    s.push_str("# cmalign :: align sequences to a CM\n");
    s.push_str("# INFERNAL 1.1.5 (Sep 2023)\n");
    s.push_str("# Copyright (C) 2023 Howard Hughes Medical Institute.\n");
    s.push_str("# Freely distributed under the BSD open source license.\n");
    s.push_str("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");

    s.push_str(&format!("# CM file:                                     {}\n", cmfile));
    s.push_str(&format!("# sequence file:                               {}\n", sqfile));
    s.push_str(&format!("# CM name:                                     {}\n", cm_name));
    let gs = |name: &str| -> String { p.get_str(name).unwrap_or("").to_string() };
    let gr = |name: &str| -> f64 { p.get_f64(name).unwrap_or(None).unwrap_or(0.0) };
    let gi = |name: &str| -> i64 { p.get_i64(name).unwrap_or(None).unwrap_or(0) };
    if p.is_set("-o")          { s.push_str(&format!("# saving alignment to file:                    {}\n", gs("-o"))); }
    if p.is_set("-g")          { s.push_str("# model configuration:                         global\n"); }
    if p.is_set("--optacc")    { s.push_str("# alignment algorithm:                         optimal accuracy\n"); }
    if p.is_set("--cyk")       { s.push_str("# alignment algorithm:                         CYK\n"); }
    if p.is_set("--sample")    { s.push_str("# sampling aln from posterior distribution:    yes\n"); }
    // C: `if (esl_opt_IsUsed(go,"--seed"))` (cmalign.c:1506). esl_opt_IsUsed is
    // FALSE when the option's value equals its default (esl_opt_IsDefault does a
    // strcmp of the value string against defval), so `--seed 181` (default 181)
    // prints NOTHING, while `--seed 0`/`--seed 999` (non-default) print the line.
    if p.is_set("--seed") && p.get_str("--seed") != Some("181") {
        if gi("--seed") == 0 { s.push_str("# random number seed:                          one-time arbitrary\n"); }
        else                 { s.push_str(&format!("# random number seed set to:                   {}\n", gi("--seed"))); }
    }
    if p.is_set("--notrunc")   { s.push_str("# truncated sequence alignment mode:           off\n"); }
    if p.is_set("--sub")       { s.push_str("# alternative truncated seq alignment mode:    on\n"); }
    if p.is_set("--mxsize")    { s.push_str(&format!("# maximum total DP matrix size set to:         {:.2} Mb\n", gr("--mxsize"))); }
    if p.is_set("--hbanded")   { s.push_str("# using HMM bands for acceleration:            yes\n"); }
    if p.is_set("--tau")       { s.push_str(&format!("# tail loss probability for HMM bands set to:  {}\n", fmt_g(gr("--tau")))); }
    if p.is_set("--fixedtau")  { s.push_str("# tighten HMM bands when necessary:            no\n"); }
    if p.is_set("--maxtau")    { s.push_str(&format!("# maximum tau allowed during band tightening:  {}\n", fmt_g(gr("--maxtau")))); }
    if p.is_set("--nonbanded") { s.push_str("# using HMM bands for acceleration:            no\n"); }
    if p.is_set("--small")     { s.push_str("# small memory D&C alignment algorithm:        on\n"); }
    if p.is_set("--sfile")     { s.push_str(&format!("# saving alignment score info to file:         {}\n", gs("--sfile"))); }
    if p.is_set("--tfile")     { s.push_str(&format!("# saving parsetrees to file:                   {}\n", gs("--tfile"))); }
    if p.is_set("--ifile")     { s.push_str(&format!("# saving insert information to file:           {}\n", gs("--ifile"))); }
    if p.is_set("--elfile")    { s.push_str(&format!("# saving local end information to file:        {}\n", gs("--elfile"))); }
    if p.is_set("--mapali")    { s.push_str(&format!("# including alignment from file:               {}\n", gs("--mapali"))); }
    if p.is_set("--mapstr")    { s.push_str(&format!("# including structure from alnment from file:  {}\n", gs("--mapali"))); }
    if p.is_set("--informat")  { s.push_str(&format!("# input sequence file format specified as:     {}\n", gs("--informat"))); }
    if p.is_set("--outformat") { s.push_str(&format!("# output alignment format specified as:        {}\n", gs("--outformat"))); }
    if p.is_set("--dnaout")    { s.push_str("# output alignment alphabet:                   DNA\n"); }
    if p.is_set("--noprob")    { s.push_str("# posterior probability annotation:            off\n"); }
    if p.is_set("--matchonly") { s.push_str("# include alignment insert columns:            no\n"); }
    if p.is_set("--ileaved")   { s.push_str("# forcing interleaved Stockholm output aln:    yes\n"); }
    if p.is_set("--regress")   { s.push_str(&format!("# saving alignment without author info to:     {}\n", gs("--regress"))); }

    // number of worker threads (cmalign.c:1542, ALWAYS printed for HMMER_THREADS).
    // ncpus = ESL_MIN(--cpu, esl_threads_GetCPUCount()); --cpu default CMNCPU=4 (or
    // $INFERNAL_NCPU). " [--cpu]" suffix iff --cpu was set on the command line.
    let cpu_default: i64 = std::env::var("INFERNAL_NCPU")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let cpu_opt: i64 = if p.is_set("--cpu") { gi("--cpu") } else { cpu_default };
    let ncpus = cpu_opt.min(num_worker_threads());
    s.push_str(&format!(
        "# number of worker threads:                    {}{}\n",
        ncpus,
        if p.is_set("--cpu") { " [--cpu]" } else { "" }
    ));
    s.push_str("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    s
}

/// Right-justify `s` in width `w` (C printf `%*s` / `%Ns`); no truncation if longer.
fn rj(s: &str, w: usize) -> String {
    if s.len() >= w { s.to_string() } else { format!("{}{}", " ".repeat(w - s.len()), s) }
}
/// Left-justify `s` in width `w` (C printf `%-*s` / `%-Ns`).
fn lj(s: &str, w: usize) -> String {
    if s.len() >= w { s.to_string() } else { format!("{}{}", s, " ".repeat(w - s.len())) }
}

/// C `output_scores()` (cmalign.c:1978) — the per-sequence score table. Written to
/// the --sfile file, or to stdout when the alignment is redirected (-o). The three
/// "running time (s)" columns (band calc / alignment / total) are the only
/// non-reproducible (timing) columns; they are emitted as `0.00` here and
/// normalized out in verification. Every other column (idx/name/length/cm
/// from/cm to/trunc/bit sc/avg pp/mem (Mb)) is deterministic and byte-exact.
#[allow(clippy::too_many_arguments)]
fn output_scores(
    cm: &infernox::cm::CM,
    trs: &[Parsetree],
    names: &[String],
    dsqs: &[Vec<u8>],
    scs: &[f32],
    avgpps: &[f32],
    mbtots: &[f32],
    do_post: bool,
    do_nonbanded: bool,
    do_sub: bool,
    do_trunc: bool,
    be_verbose: bool,
) -> String {
    let n = names.len();
    // namewidth = max(8, max name length); idxwidth = digits(nseq), min 3.
    let mut namewidth = 8usize;
    for name in names.iter() {
        namewidth = namewidth.max(name.len());
    }
    let mut maxidx = n as i64; // dataA[ndata-1]->idx+1 == nseq
    let mut idxwidth = 0usize;
    loop {
        idxwidth += 1;
        maxidx /= 10;
        if maxidx == 0 {
            break;
        }
    }
    idxwidth = idxwidth.max(3);
    let namedashes = "-".repeat(namewidth);
    let idxdashes = "-".repeat(idxwidth);

    let mut s = String::new();
    // Header line 1 (running time header)
    s.push_str(&format!(
        "# {}  {}  {}  {}  {}  {}  {}  {}  {}  {}",
        rj("", idxwidth), lj("", namewidth), rj(" ", 6), rj("", 7), rj("", 7),
        rj("", 5), rj("", 8), rj("", 6), lj("       running time (s)", 30), rj("", 8)
    ));
    if be_verbose { s.push_str(&format!("  {}  {}  {}  {}", rj("", 7), rj("", 7), rj("", 7), rj("", 8))); }
    s.push('\n');
    // Header line 2 (running time dashes)
    s.push_str(&format!(
        "# {}  {}  {}  {}  {}  {}  {}  {}  {}  {}",
        rj("", idxwidth), lj("", namewidth), rj(" ", 6), rj("", 7), rj("", 7),
        rj("", 5), rj("", 8), rj("", 6), rj("-------------------------------", 30), rj("", 8)
    ));
    if be_verbose { s.push_str(&format!("  {}  {}  {}  {}", rj("", 7), rj("", 7), rj("", 7), rj("", 8))); }
    s.push('\n');
    // Header line 3 (column names)
    s.push_str(&format!(
        "# {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}",
        rj("idx", idxwidth), lj("seq name", namewidth), rj("length", 6), rj("cm from", 7),
        rj("cm to", 7), rj("trunc", 5), rj("bit sc", 8), rj("avg pp", 6), rj("band calc", 9),
        rj("alignment", 9), rj("total", 9), rj("mem (Mb)", 8)
    ));
    if be_verbose { s.push_str(&format!("  {}  {}  {}  {}", rj("tau", 7), rj("thresh1", 7), rj("thresh2", 7), rj("failover", 8))); }
    s.push('\n');
    // Header line 4 (column dashes)
    s.push_str(&format!(
        "# {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}",
        rj(&idxdashes, idxwidth), lj(&namedashes, namewidth), rj("------", 6), rj("-------", 7),
        rj("-------", 7), rj("-----", 5), rj("--------", 8), rj("------", 6), rj("---------", 9),
        rj("---------", 9), rj("---------", 9), rj("--------", 8)
    ));
    if be_verbose { s.push_str(&format!("  {}  {}  {}  {}", rj("-------", 7), rj("-------", 7), rj("-------", 7), rj("--------", 8))); }
    s.push('\n');

    for i in 0..n {
        let len = (dsqs[i].len() - 2) as i64; // strip 1-based sentinels
        let b = infernox::cm_alndata::parsetree_to_cm_bounds(cm, &trs[i], true, true);
        let spos = b.first_emit;
        let epos = b.final_emit;
        // trunc column: --sub uses spos/epos vs clen; else the parsetree root mode.
        let truncstr: &str = if do_sub {
            if spos != 1 && epos != cm.clen { "5'&3'" }
            else if spos == 1 && epos != cm.clen { "3'" }
            else if spos != 1 && epos == cm.clen { "5'" }
            else { "no" }
        } else {
            match trs[i].mode[0] {
                infernox::cm_trunc::TRMODE_T => "5'&3'",
                infernox::cm_trunc::TRMODE_L => "3'",
                infernox::cm_trunc::TRMODE_R => "5'",
                _ => "no",
            }
        };
        s.push_str(&format!(
            "  {}  {}  {}  {}  {}",
            rj(&format!("{}", i + 1), idxwidth), lj(&names[i], namewidth),
            rj(&format!("{}", len), 6), rj(&format!("{}", spos), 7), rj(&format!("{}", epos), 7)
        ));
        s.push_str(&format!("  {}", rj(truncstr, 5)));
        s.push_str(&format!("  {:>8.2}", scs[i]));
        if do_post { s.push_str(&format!("  {:>6.3}", avgpps[i])); } else { s.push_str(&format!("  {}", rj("-", 6))); }
        // band calc (timing): %9.2f when banded, "-" when nonbanded.
        if !do_nonbanded { s.push_str(&format!("  {:>9.2}", 0.0f32)); } else { s.push_str(&format!("  {}", rj("-", 9))); }
        // alignment, total (timing): %9.2f
        s.push_str(&format!("  {:>9.2}  {:>9.2}", 0.0f32, 0.0f32));
        // mem (Mb): %8.2f  (deterministic)
        s.push_str(&format!("  {:>8.2}", mbtots[i]));
        if be_verbose {
            // NOTE: --verbose (tau/thresh1/thresh2/failover) requires per-seq band
            // metadata not threaded here; emitted as placeholders. --verbose is not
            // byte-verified in this build.
            s.push_str(&format!("  {}  {}  {}  {}", rj("-", 7), rj("-", 7), rj("-", 7), rj("-", 8)));
        }
        let _ = do_trunc;
        s.push('\n');
    }
    s
}

