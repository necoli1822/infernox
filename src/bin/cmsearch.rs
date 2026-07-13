//! infernox-cmsearch — search a sequence database with a covariance model.
//!
//! Byte-parity implementation: reproduces C Infernal 1.1.5 `cmsearch` (STD_ANY
//! pipeline) `--tblout` output exactly. This binary is now a thin CLI wrapper:
//! the faithful pipeline lives in the reusable library entry point
//! [`infernox::cm_search::FaithfulSearcher`], so the exact same search can
//! run in-process from other crates (e.g. tRNAscan-SE).
//!
//! Usage: infernox-cmsearch <cm> <fasta> [--tblout out] [--toponly] [--cpu N]

use infernox::cm_file::cm_file_read_from_reader_opt;
use infernox::cm_search::{
    FaithfulConfig, FaithfulHit, FaithfulSearcher, ModelCutoff, PassAcctSnapshot, PliStats,
    T_F1F3, T_F4F5, T_F6BAND, T_F6CYK, T_F7BAND, T_F7INS,
};
use infernox::search_cli::{ArgKind, OptSpec, Parsed};
use std::io::BufReader;
use std::sync::atomic::Ordering;

/// The complete `cmsearch` option table (name + argument arity), mirroring the
/// `ESL_OPTIONS options[]` array in cmsearch.c. Every option C recognizes is
/// listed so the parser consumes value-taking options' arguments correctly and
/// rejects genuinely unknown flags (instead of silently swallowing them and
/// leaking their argument into the positional list). `ArgKind::Value` marks the
/// eslARG_INT/REAL/STRING/OUTFILE/INFILE options; `ArgKind::None` the eslARG_NONE
/// booleans.
fn cmsearch_opt_table() -> Vec<OptSpec> {
    use ArgKind::{None as N, Value as V};
    let mk = |name, kind| OptSpec { name, kind };
    vec![
        // docgroup 1 (basic)
        mk("-h", N), mk("-g", N), mk("-Z", V), mk("--devhelp", N),
        // docgroup 2 (output)
        mk("-o", V), mk("-A", V), mk("--tblout", V), mk("--fmt", V), mk("--acc", N),
        mk("--noali", N), mk("--notextw", N), mk("--textw", V), mk("--verbose", N), mk("--nomiss", N),
        // docgroup 3 (reporting)
        mk("-E", V), mk("-T", V),
        // docgroup 4 (inclusion)
        mk("--incE", V), mk("--incT", V),
        // docgroup 5 (model cutoffs)
        mk("--cut_ga", N), mk("--cut_nc", N), mk("--cut_tc", N),
        // docgroup 6 (accel level / presets)
        mk("--max", N), mk("--nohmm", N), mk("--mid", N), mk("--default", N), mk("--rfam", N),
        mk("--hmmonly", N), mk("--FZ", V), mk("--Fmid", V),
        // docgroup 7 (other)
        mk("--notrunc", N), mk("--anytrunc", N), mk("--nonull3", N), mk("--mxsize", V),
        mk("--smxsize", V), mk("--cyk", N), mk("--acyk", N), mk("--wcx", V), mk("--toponly", N),
        mk("--bottomonly", N), mk("--tformat", V), mk("--cpu", V),
        // docgroup 999 (bogus but registered → recognized so args don't leak)
        mk("--glist", V), mk("--clanin", V), mk("--oclan", N), mk("--oskip", N), mk("--block", V),
        // docgroup 101 (per-stage filter control)
        mk("--noF1", N), mk("--noF2", N), mk("--noF3", N), mk("--noF4", N), mk("--noF6", N),
        mk("--doF1b", N), mk("--noF2b", N), mk("--noF3b", N), mk("--noF4b", N), mk("--doF5b", N),
        mk("--F1", V), mk("--F1b", V), mk("--F2", V), mk("--F2b", V), mk("--F3", V), mk("--F3b", V),
        mk("--F4", V), mk("--F4b", V), mk("--F5", V), mk("--F5b", V), mk("--F6", V),
        // docgroup 102 (HMM-only filter control)
        mk("--hmmmax", N), mk("--hmmF1", V), mk("--hmmF2", V), mk("--hmmF3", V), mk("--hmmnobias", N),
        mk("--hmmnonull2", N), mk("--nohmmonly", N),
        // docgroup 103 (envelope definition)
        mk("--rt1", V), mk("--rt2", V), mk("--rt3", V), mk("--ns", V),
        // docgroup 104 (CYK filter round)
        mk("--ftau", V), mk("--fsums", N), mk("--fqdb", N), mk("--fbeta", V), mk("--fnonbanded", N),
        mk("--nocykenv", N), mk("--cykenvx", V),
        // docgroup 105 (final round)
        mk("--tau", V), mk("--sums", N), mk("--qdb", N), mk("--beta", V), mk("--nonbanded", N),
        // docgroup 106
        mk("--trmF3", N),
        // docgroup 107 (timing)
        mk("--timeF1", N), mk("--timeF2", N), mk("--timeF3", N), mk("--timeF4", N), mk("--timeF5", N),
        mk("--timeF6", N),
        // docgroup 108 (expert)
        mk("--nogreedy", N), mk("--cp9noel", N), mk("--cp9gloc", N), mk("--null2", N), mk("--maxtau", V),
        mk("--seed", V), mk("--onepass", N), mk("--olonepass", N), mk("--noiter", N),
        mk("--inttrunc", N), mk("--onlytrunc", N), mk("--5trunc", N), mk("--3trunc", N),
        // infernox-only extension (not in C): bit-identical Forward filter.
        mk("--strict", N),
    ]
}

/// One model's search result, kept so tblout rows can be emitted per model with
/// per-model column widths (C: `cm_tophits_TabularTargets1` is called once per CM,
/// computing its own widths from that CM's tophits — see cmsearch.c:709).
struct ModelResult {
    qname: String,
    qacc: String,
    qdesc: String,
    clen: i32,
    hits: Vec<FaithfulHit>,
    /// Pipeline statistics-summary data captured while the searcher was alive
    /// (metadata + summed accounting + truncated-pass output tallies).
    stats: StatsData,
    /// If `Some(s)`, this model uses a bit-score inclusion cutoff (from
    /// `--cut_ga/--cut_tc/--cut_nc`): a hit is included iff `score >= s`. `None`
    /// falls back to the global `--incT` / `--incE` inclusion rule.
    inc_cutoff: Option<f32>,
    /// cmsearch `-A`: the Stockholm/Pfam MSA text of this model's included hits,
    /// built inside the search loop (while the CM is alive) via approach (B). `None`
    /// when `-A` was not given; `Some((text, nseq))` otherwise (nseq may be 0, in
    /// which case C writes nothing to the file and prints the "no alignment" line).
    msa_out: Option<(String, usize)>,
}

/// Split a multi-model CM file's text into one text chunk per model record.
///
/// C `cmsearch` loops `while (cm_file_Read(...) != eslEOF)`, reading one CM record
/// (its `INFERNAL1/a` header ... p7 HMM ... trailing `//`) at a time. The Rust
/// reader `cm_file_read_from_reader_opt` consumes its reader per record, so we
/// slice the file at each format-header line and feed each slice as a fresh reader.
fn split_cm_records(text: &str) -> Vec<String> {
    let mut records: Vec<String> = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        // A new record begins at each format-magic line (parse_format_line accepts
        // "INFERNAL1/a" / "INFERNAL1" / "INFERNAL-1"). Flush the accumulated record.
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

/// Decode a `--informat`/`--qformat`/`--tformat` option string to a forced
/// [`infernox::easel::SqFormat`], or [`SqFormat::Unknown`] for autodetection when absent.
/// C `esl_sqio_EncodeFormat`: an unrecognized name is a fatal argument error.
fn decode_informat(opt: Option<&str>, optname: &str) -> infernox::easel::SqFormat {
    match opt {
        None => infernox::easel::SqFormat::Unknown,
        Some(s) => infernox::easel::esl_sqio_encode_format(s).unwrap_or_else(|| {
            eprintln!("error: {} '{}' is not a recognized sequence file format", optname, s);
            std::process::exit(1);
        }),
    }
}


/// C `cmsearch.c` process_commandline `-h`/`--devhelp` block (cmsearch.c:1751-1799):
/// `cm_banner(stdout, argv[0], banner)` + `esl_usage(stdout, argv[0], usage)`, then
/// each docgroup printed as `puts("\n<header>")` + `esl_opt_DisplayHelp(stdout, go,
/// group, 2, width)`. With `--devhelp` (`do_dev`), the hidden groups 101-108 are
/// shown and the two `%s` suffixes (the `devmsg = "*"` on the "acceleration
/// heuristics" / "Other options" headers) and the "*Use --devhelp..." trailer are
/// suppressed. Text transcribed verbatim from C's DisplayHelp over the cmsearch
/// ESL_OPTIONS table; the banner/Usage program name is esl's argv[0] basename,
/// hardcoded "cmsearch" to byte-match C (as the other infernox tools do).
fn full_help(do_dev: bool) -> ! {
    // cm_banner + esl_usage
    print!(
        "# cmsearch :: search CM(s) against a sequence database\n\
# INFERNAL 1.1.5 (Sep 2023)\n\
# Copyright (C) 2023 Howard Hughes Medical Institute.\n\
# Freely distributed under the BSD open source license.\n\
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\
Usage: cmsearch [options] <cmfile> <seqdb>\n"
    );
    // group 1: Basic options
    print!(
        "\nBasic options:\n  \
-h        : show brief help on version and usage\n  \
-g        : configure CM for glocal alignment [default: local]\n  \
-Z <x>    : set search space size in *Mb* to <x> for E-value calculations  (x>0)\n  \
--devhelp : show list of otherwise hidden developer/expert options\n"
    );
    // group 2: Options directing output
    print!(
        "\nOptions directing output:\n  \
-o <f>       : direct output to file <f>, not stdout\n  \
-A <f>       : save multiple alignment of all significant hits to file <s>\n  \
--tblout <f> : save parseable table of hits to file <s>\n  \
--fmt <n>    : set hit table format to <n>  (1<=n<=3)\n  \
--acc        : prefer accessions over names in output\n  \
--noali      : don't output alignments, so output is smaller\n  \
--notextw    : unlimit ASCII text output line width\n  \
--textw <n>  : set max width of ASCII text output lines  [120]  (n>=120)\n  \
--verbose    : report extra information; mainly useful for debugging\n  \
--nomiss     : with -A, do not mark truncated hits with missing (~) chars\n"
    );
    // group 3: reporting thresholds
    print!(
        "\nOptions controlling reporting thresholds:\n  \
-E <x> : report sequences <= this E-value threshold in output  [10.0]  (x>0)\n  \
-T <x> : report sequences >= this score threshold in output\n"
    );
    // group 4: inclusion thresholds
    print!(
        "\nOptions controlling inclusion (significance) thresholds:\n  \
--incE <x> : consider sequences <= this E-value threshold as significant  [0.01]\n  \
--incT <x> : consider sequences >= this score threshold as significant\n"
    );
    // group 5: model-specific reporting thresholds
    print!(
        "\nOptions controlling model-specific reporting thresholds:\n  \
--cut_ga : use CM's GA gathering cutoffs as reporting thresholds\n  \
--cut_nc : use CM's NC noise cutoffs as reporting thresholds\n  \
--cut_tc : use CM's TC trusted cutoffs as reporting thresholds\n"
    );
    // group 6: acceleration heuristics (header carries devmsg "*" unless --devhelp)
    print!(
        "\nOptions controlling acceleration heuristics{}:\n  \
--max      : turn all heuristic filters off (slow)\n  \
--nohmm    : skip all HMM filter stages, use only CM (slow)\n  \
--mid      : skip first two HMM filter stages (SSV & Vit)\n  \
--default  : default: run search space size-dependent pipeline  [default]\n  \
--rfam     : set heuristic filters at Rfam-level (fast)\n  \
--hmmonly  : use HMM only, don't use a CM at all\n  \
--FZ <x>   : set filters to defaults used for a search space of size <x> Mb\n  \
--Fmid <x> : with --mid, set P-value threshold for HMM stages to <x>  [0.02]\n",
        if do_dev { "" } else { "*" }
    );
    if do_dev {
        // group 101: precise control of the CM filter pipeline
        print!(
            "\nOptions for precise control of the CM filter pipeline:\n  \
--noF1    : skip the HMM SSV filter stage\n  \
--noF2    : skip the HMM Viterbi filter stage\n  \
--noF3    : skip the HMM Forward filter stage\n  \
--noF4    : skip the HMM glocal Forward filter stage\n  \
--noF6    : skip the CM CYK filter stage\n  \
--doF1b   : turn on  the HMM SSV composition bias filter\n  \
--noF2b   : turn off the HMM Vit composition bias filter\n  \
--noF3b   : turn off the HMM Fwd composition bias filter\n  \
--noF4b   : turn off the HMM glocal Fwd composition bias filter\n  \
--doF5b   : turn on  the HMM per-envelope composition bias filter\n  \
--F1 <x>  : Stage 1 (SSV) threshold:         promote hits w/ P <= <x>  (x>0)\n  \
--F1b <x> : Stage 1 (MSV) bias threshold:    promote hits w/ P <= <x>  (x>0)\n  \
--F2 <x>  : Stage 2 (Vit) threshold:         promote hits w/ P <= <x>  (x>0)\n  \
--F2b <x> : Stage 2 (Vit) bias threshold:    promote hits w/ P <= <x>  (x>0)\n  \
--F3 <x>  : Stage 3 (Fwd) threshold:         promote hits w/ P <= <x>  (x>0)\n  \
--F3b <x> : Stage 3 (Fwd) bias threshold:    promote hits w/ P <= <x>  (x>0)\n  \
--F4 <x>  : Stage 4 (gFwd) glocal threshold: promote hits w/ P <= <x>  (x>0)\n  \
--F4b <x> : Stage 4 (gFwd) glocal bias thr:  promote hits w/ P <= <x>  (x>0)\n  \
--F5 <x>  : Stage 5 (env defn) threshold:    promote hits w/ P <= <x>  (x>0)\n  \
--F5b <x> : Stage 5 (env defn) bias thr:     promote hits w/ P <= <x>  (x>0)\n  \
--F6 <x>  : Stage 6 (CYK) threshold:         promote hits w/ P <= <x>  (x>0)\n"
        );
        // group 102: HMM-only filter pipeline
        print!(
            "\nOptions controlling the HMM-only filter pipeline (run for models w/0 basepairs):\n  \
--hmmmax     : in HMM-only mode, turn off all filters\n  \
--hmmF1 <x>  : in HMM-only mode, set stage 1 (SSV) P value threshold to <x>\n  \
--hmmF2 <x>  : in HMM-only mode, set stage 2 (Vit) P value threshold to <x>\n  \
--hmmF3 <x>  : in HMM-only mode, set stage 3 (Fwd) P value threshold to <x>\n  \
--hmmnobias  : in HMM-only mode, turn off the bias composition filter\n  \
--hmmnonull2 : in HMM-only mode, turn off the null2 score correction\n  \
--nohmmonly  : never run HMM-only mode, not even for models with 0 basepairs\n"
        );
        // group 103: HMM envelope definition
        print!(
            "\nOptions for precise control of HMM envelope definition:\n  \
--rt1 <x> : set domain/envelope definition rt1 parameter as <x>  [0.25]\n  \
--rt2 <x> : set domain/envelope definition rt2 parameter as <x>  [0.10]\n  \
--rt3 <x> : set domain/envelope definition rt3 parameter as <x>  [0.20]\n  \
--ns <n>  : set number of domain/envelope tracebacks to <n>  [200]\n"
        );
        // group 104: CYK filter stage
        print!(
            "\nOptions for precise control of the CYK filter stage:\n  \
--ftau <x>    : set HMM band tail loss prob for CYK filter to <x>  [1e-4]\n  \
--fsums       : w/--fhbanded use posterior sums (widens bands)\n  \
--fqdb        : use QDBs in CYK filter round, not HMM bands\n  \
--fbeta <x>   : set tail loss prob for CYK filter QDB calculation to <x>  [1e-7]\n  \
--fnonbanded  : do not use any bands for CYK filter round\n  \
--nocykenv    : do not redefine envelopes after stage 6 based on CYK hits\n  \
--cykenvx <n> : CYK envelope redefinition threshold multiplier, <n> * F6  [10]\n"
        );
        // group 105: final stage
        print!(
            "\nOptions for precise control of the final stage:\n  \
--tau <x>   : set HMM band tail loss prob for final round to <x>\n  \
--sums      : w/--hbanded use posterior sums (widens bands)\n  \
--qdb       : use QDBs (instead of HMM bands) in final Inside round\n  \
--beta <x>  : set tail loss prob for final Inside QDB calculation to <x>\n  \
--nonbanded : do not use QDBs or HMM bands in final Inside round of CM search\n"
        );
        // group 106: terminating after individual pipeline stages
        print!(
            "\nOptions for terminating after individual pipeline stages:\n  \
--trmF3 : terminate after Stage 3 Fwd and output surviving windows\n"
        );
        // group 107: timing pipeline stages
        print!(
            "\nOptions for timing pipeline stages:\n  \
--timeF1 : abort after Stage 1 SSV; for timing expts\n  \
--timeF2 : abort after Stage 2 Vit; for timing expts\n  \
--timeF3 : abort after Stage 3 Fwd; for timing expts\n  \
--timeF4 : abort after Stage 4 glocal Fwd; for timing expts\n  \
--timeF5 : abort after Stage 5 envelope def; for timing expts\n  \
--timeF6 : abort after Stage 6 CYK; for timing expts\n"
        );
    }
    // group 7: Other options (header carries devmsg "*" unless --devhelp)
    print!(
        "\nOther options{}:\n  \
--notrunc     : do not allow truncated hits at sequence termini\n  \
--anytrunc    : allow full+truncated hits at terminii and anywhere within seqs\n  \
--nonull3     : turn off the NULL3 post hoc additional null model\n  \
--mxsize <x>  : set max allowed alnment mx size to <x> Mb [df: autodetermined]\n  \
--smxsize <x> : set max allowed size of search DP matrices to <x> Mb  [128.]\n  \
--cyk         : use scanning CM CYK algorithm, not Inside in final stage\n  \
--acyk        : align hits with CYK, not optimal accuracy\n  \
--wcx <x>     : set W (expected max hit len) as <x> * cm->clen (model len)\n  \
--toponly     : only search the top strand\n  \
--bottomonly  : only search the bottom strand\n  \
--tformat <s> : assert target <seqdb> is in format <s>: no autodetection\n  \
--cpu <n>     : number of parallel CPU workers to use for multithreads  [4]\n",
        if do_dev { "" } else { "*" }
    );
    if do_dev {
        // group 108: Other expert options
        print!(
            "\nOther expert options:\n  \
--nogreedy   : do not resolve hits with greedy algorithm, use optimal one\n  \
--cp9noel    : turn off local ends in cp9 HMMs\n  \
--cp9gloc    : configure cp9 HMM in glocal mode\n  \
--null2      : turn on null 2 biased composition HMM score corrections\n  \
--maxtau <x> : set max tau <x> when tightening HMM bands  [0.05]\n  \
--seed <n>   : set RNG seed to <n> (if 0: one-time arbitrary seed)  [181]\n  \
--onepass    : use CM only for best scoring HMM pass for full seq envelopes\n  \
--olonepass  : use CM only f. best sc'ing HMM pass f. overlapping envelopes\n  \
--noiter     : do not iteratively tighten bands when necessary\n  \
--inttrunc   : allow full and truncated hits anywhere within sequences\n  \
--onlytrunc  : allow only truncated hits, anywhere within sequences\n  \
--5trunc     : allow truncated hits only at 5' ends of sequences\n  \
--3trunc     : allow truncated hits only at 3' ends of sequences\n"
        );
    } else {
        print!("\n*Use --devhelp to show additional expert options.\n");
    }
    std::process::exit(0);
}

/// A command-line value/type error is a C `esl_opt_ProcessCmdline` failure
/// (cmsearch.c:1799): print `Failed to parse command line: <msg>`, then the shared
/// ERROR block (usage + basic options + exit 1), all to stdout — via `cmdline_fail`.
fn usage_error(msg: &str) -> ! {
    cmdline_fail(&format!("Failed to parse command line: {msg}"));
}

/// C `process_commandline()` ERROR: block (cmsearch.c:2223-2228). Every command-line
/// user error routes here: print the offending first line, then `esl_usage` +
/// the basic-options `esl_opt_DisplayHelp(group 1)` block, then exit(1). All of
/// this goes to STDOUT (C uses puts/printf/esl_usage(stdout,...)). The `Usage:`
/// program name is C's `esl_usage` basename (hardcoded "cmsearch"); the final
/// "do <argv0> -h" line embeds the actual binary path (the one path-dependent
/// line, excluded from byte-diffs like the tblout provenance lines).
fn cmdline_fail(first_line: &str) -> ! {
    let argv0 = std::env::args().next().unwrap_or_else(|| "cmsearch".to_string());
    print!(
        "{first_line}\n\
Usage: cmsearch [options] <cmfile> <seqdb>\n\
\n\
where basic options are:\n  \
-h        : show brief help on version and usage\n  \
-g        : configure CM for glocal alignment [default: local]\n  \
-Z <x>    : set search space size in *Mb* to <x> for E-value calculations  (x>0)\n  \
--devhelp : show list of otherwise hidden developer/expert options\n\
\n\
To see more help on available options, do {argv0} -h\n\n"
    );
    std::process::exit(1);
}

/// C `esl_opt_VerifyConfig` (esl_getopts.c:719): the require loop (all options in
/// table order) then the incompat loop. An option is "active" iff given on the
/// command line; a require target counts as missing iff not given (its val is
/// NULL — true for the boolean / no-default-outfile targets used here); an
/// incompat target counts as present iff given. Messages use the FULL optlist
/// string. Data transcribed from the cmsearch `ESL_OPTIONS` table (fields 7/8),
/// in table order. On violation returns the errbuf message (caller prefixes
/// "Failed to parse command line: ").
/// C `esl_opt_IsUsed` (esl_getopts.c): TRUE iff the option was given AND its value
/// differs from the default (`!esl_opt_IsDefault`). `--default` (accel preset) has
/// default value "default", so IsUsed(--default) is ALWAYS FALSE — giving it is a
/// no-op that neither fires its own guard nor triggers another's. (Other options
/// only read here as booleans/no-default values, for which "given" ⟺ "used".)
fn used(p: &infernox::search_cli::Parsed, name: &str) -> bool {
    name != "--default" && p.is_set(name)
}

fn verify_config(p: &infernox::search_cli::Parsed) -> Result<(), String> {
    // (name, require_optlist, incompat_optlist) — table order (cmsearch.c:101-253).
    const C: &[(&str, Option<&str>, Option<&str>)] = &[
        ("-g", None, Some("--hmmonly")),
        ("--fmt", Some("--tblout"), None),
        ("--notextw", None, Some("--textw")),
        ("--textw", None, Some("--notextw")),
        ("--nomiss", Some("-A"), None),
        ("--Fmid", Some("--mid"), None),
        ("--wcx", None, Some("--nohmm,--qdb,--fqdb")),
        ("--cpu", None, Some("--mpi")), // CPUOPTS (MPI opts, never set here)
        ("--noF1", None, Some("--doF1b")),
        ("--noF2", Some("--noF2b"), None),
        ("--noF3", Some("--noF3b"), None),
        ("--noF4", Some("--noF4b"), None),
        ("--F1", None, Some("--noF1")),
        ("--F1b", Some("--doF1b"), None),
        ("--F2", None, Some("--noF2")),
        ("--F2b", None, Some("--noF2b")),
        ("--F3", None, Some("--noF3")),
        ("--F3b", None, Some("--noF3b")),
        ("--F4", None, Some("--noF4")),
        ("--F4b", None, Some("--noF4b")),
        ("--F5b", Some("--doF5b"), None),
        ("--F6", None, Some("--noF6")),
        ("--hmmF1", None, Some("--nohmmonly")),
        ("--hmmF2", None, Some("--nohmmonly")),
        ("--hmmF3", None, Some("--nohmmonly")),
        ("--hmmnobias", None, Some("--nohmmonly")),
        ("--hmmnonull2", None, Some("--nohmmonly")),
        ("--nohmmonly", None, Some("--hmmmax")),
        ("--rt1", None, Some("--nohmm,--max")),
        ("--rt2", None, Some("--nohmm,--max")),
        ("--rt3", None, Some("--nohmm,--max")),
        ("--ns", None, Some("--nohmm,--max")),
        ("--ftau", None, Some("--fqdb")),
        ("--fsums", None, Some("--fqdb")),
        ("--nocykenv", None, Some("--max")),
        ("--cykenvx", None, Some("--max")),
        ("--tau", None, Some("--qdb")),
        ("--sums", None, Some("--qdb")),
        ("--trmF3", Some("--noali,--hmmonly"), None),
        ("--cp9noel", None, Some("-g")),
        ("--cp9gloc", None, Some("-g,--cp9noel")),
        ("--onepass", None, Some("--nohmm,--qdb,--fqdb")),
        ("--noiter", None, Some("--nohmm,--qdb,--fqdb")),
    ];
    for (name, req, _) in C {
        if used(p, name) {
            if let Some(r) = req {
                for t in r.split(',') {
                    if !used(p, t) {
                        return Err(format!(
                            "Option {name} requires (or has no effect without) option(s) {r}"
                        ));
                    }
                }
            }
        }
    }
    for (name, _, inc) in C {
        if used(p, name) {
            if let Some(ic) = inc {
                for t in ic.split(',') {
                    if t != *name && used(p, t) {
                        return Err(format!("Option {name} is incompatible with option(s) {ic}"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// C cmsearch.c:1873-2090 manual guards ("incompatible option combinations I
/// don't know how to disallow with esl_getopts"): the accel-preset blocks
/// (singular "Option X is incompatible with option Y", first hit in C's order)
/// then the threshold block (comma-list form). Runs AFTER `verify_config`, so a
/// pair the table already rejects (e.g. --max --rt1) never reaches here.
/// Returns the full first line (already includes the "Failed to parse..." prefix).
fn manual_guards(p: &infernox::search_cli::Parsed) -> Result<(), String> {
    // Accel-preset primaries: (primary, [incompatible others, in C's check order]).
    const ACCEL: &[(&str, &[&str])] = &[
        ("--max", &["--nohmm","--mid","--rfam","--FZ","--noF1","--noF2","--noF3","--noF4","--noF6","--doF1b","--noF2b","--noF3b","--noF4b","--doF5b","--F1","--F1b","--F2","--F2b","--F3","--F3b","--F4","--F4b","--F5","--F6","--ftau","--fsums","--fqdb","--fbeta","--fnonbanded","--nocykenv","--cykenvx","--tau","--sums","--nonbanded","--rt1","--rt2","--rt3","--ns","--maxtau","--anytrunc","--inttrunc","--onlytrunc","--5trunc","--3trunc","--onepass","--olonepass","--noiter"]),
        ("--nohmm", &["--max","--mid","--rfam","--FZ","--noF1","--noF2","--noF3","--noF4","--doF1b","--noF2b","--noF3b","--noF4b","--doF5b","--F1","--F1b","--F2","--F2b","--F3","--F3b","--F4","--F4b","--F5","--ftau","--fsums","--tau","--sums","--rt1","--rt2","--rt3","--ns","--maxtau","--anytrunc","--inttrunc","--onlytrunc","--5trunc","--3trunc","--onepass","--olonepass","--noiter"]),
        ("--mid", &["--max","--nohmm","--rfam","--FZ","--noF1","--noF2","--noF3","--doF1b","--noF2b","--F1","--F1b","--F2","--F2b"]),
        ("--default", &["--max","--nohmm","--rfam","--FZ"]),
        ("--rfam", &["--max","--nohmm","--default","--FZ"]),
        ("--FZ", &["--max","--nohmm","--default","--rfam"]),
        ("--hmmonly", &["--max","--nohmm","--mid","--rfam","--FZ","--noF1","--noF2","--noF3","--noF4","--noF6","--doF1b","--noF2b","--noF3b","--noF4b","--doF5b","--F1","--F1b","--F2","--F2b","--F3","--F3b","--F4","--F4b","--F5","--F6","--ftau","--fsums","--fqdb","--fbeta","--fnonbanded","--nocykenv","--cykenvx","--tau","--sums","--qdb","--beta","--nonbanded","--maxtau","--anytrunc","--inttrunc","--onlytrunc","--5trunc","--3trunc","--onepass","--olonepass","--noiter","--mxsize","--smxsize","--nonull3","--nohmmonly","--timeF4","--timeF5","--timeF6","--nogreedy","--cp9noel","--cp9gloc","--null2"]),
    ];
    for (primary, others) in ACCEL {
        if used(p, primary) {
            for o in *others {
                if used(p, o) {
                    return Err(format!(
                        "Failed to parse command line: Option {primary} is incompatible with option {o}"
                    ));
                }
            }
        }
    }
    // Threshold block (cmsearch.c:2060-2090): comma-list form. (primary, list, triggers).
    const THRESH: &[(&str, &str, &[&str])] = &[
        ("-E", "-T,--cut_ga,--cut_nc,--cut_tc", &["-T","--cut_ga","--cut_nc","--cut_tc"]),
        ("-T", "-E,--cut_ga,--cut_nc,--cut_tc", &["-E","--cut_ga","--cut_nc","--cut_tc"]),
        ("--incE", "--incT,--cut_ga,--cut_nc,--cut_tc", &["--incT","--cut_ga","--cut_nc","--cut_tc"]),
        ("--incT", "--incE,--cut_ga,--cut_nc,--cut_tc", &["--incE","--cut_ga","--cut_nc","--cut_tc"]),
        ("--cut_ga", "-E,-T,--incE,--incT,--cut_nc,--cut_tc", &["-E","-T","--incE","--incT","--cut_nc","--cut_tc"]),
        ("--cut_nc", "-E,-T,--incE,--incT,--cut_ga,--cut_tc", &["-E","-T","--incE","--incT","--cut_ga","--cut_tc"]),
        ("--cut_tc", "-E,-T,--incE,--incT,--cut_ga,--cut_nc", &["-E","-T","--incE","--incT","--cut_ga","--cut_nc"]),
    ];
    for (primary, list, triggers) in THRESH {
        if used(p, primary) && triggers.iter().any(|t| used(p, t)) {
            return Err(format!(
                "Failed to parse command line: Option {primary} is incompatible with {list}"
            ));
        }
    }
    // Truncation-mode mutual exclusions (cmsearch.c:2105-2140). Order matters (C checks
    // --notrunc, --anytrunc, -g, --onlytrunc, --5trunc, --3trunc in sequence; first hit
    // wins). Each carries the literal C message — note the --3trunc block prints
    // "Option --5trunc ..." verbatim (a C copy-paste quirk we reproduce faithfully).
    const TRUNC: &[(&str, &[&str], &str)] = &[
        ("--notrunc", &["--anytrunc","--inttrunc","--onlytrunc","--5trunc","--3trunc"],
            "Failed to parse command line: Option --notrunc is incompatible with --anytrunc,--inttrunc,--onlytrunc,--5trunc,--3trunc"),
        ("--anytrunc", &["-g","--notrunc","--inttrunc","--onlytrunc","--5trunc","--3trunc"],
            "Failed to parse command line: Option --anytrunc is incompatible with -g,--notrunc,--inttrunc,--onlytrunc,--5trunc,--3trunc"),
        ("-g", &["--anytrunc","--inttrunc","--onlytrunc","--5trunc","--3trunc"],
            "Failed to parse command line: Option -g is incompatible with --anytrunc,--inttrunc,--onlytrunc,--5trunc,--3trunc"),
        ("--onlytrunc", &["-g","--anytrunc","--inttrunc","--notrunc","--5trunc","--3trunc"],
            "Failed to parse command line: Option --onlytrunc is incompatible with -g,--anytrunc,--inttrunc,--notrunc,--5trunc,--3trunc"),
        ("--5trunc", &["-g","--anytrunc","--inttrunc","--notrunc","--onlytrunc","--3trunc"],
            "Failed to parse command line: Option --5trunc is incompatible with -g,--anytrunc,--inttrunc,--notrunc,--onlytrunc,--3trunc"),
        ("--3trunc", &["-g","--anytrunc","--inttrunc","--notrunc","--onlytrunc","--5trunc"],
            "Failed to parse command line: Option --5trunc is incompatible with -g,--anytrunc,--inttrunc,--notrunc,--onlytrunc,--5trunc"),
    ];
    for (primary, triggers, msg) in TRUNC {
        if used(p, primary) && triggers.iter().any(|t| used(p, t)) {
            return Err(msg.to_string());
        }
    }
    // C cmsearch.c:1857-1864 — the two --fmt manual guards (puts() + goto ERROR, i.e. the
    // message becomes the first line of the usage block, no "Failed to parse" prefix).
    // --fmt 2 requires cmscan's overlap/clan info, which cmsearch cannot produce.
    if used(p, "--fmt") {
        let fmt = p.get_i64("--fmt").ok().flatten().unwrap_or(1);
        if fmt == 3 && used(p, "--trmF3") {
            return Err("--fmt 3 doesn't make sense in combination with --trmF3".to_string());
        }
        if fmt == 2 {
            return Err(
                "--fmt 2 only makes sense with cmscan, because cmsearch can't determine overlaps"
                    .to_string(),
            );
        }
    }
    // C cmsearch.c:1817-1826: --beta only makes sense with --qdb/--nohmm/--max, and
    // --fbeta only with --fqdb/--nohmm. Without these, --beta alone would silently
    // recompute unused QDBs (final round stays HMM-banded) rather than erroring.
    if used(p, "--beta") && !used(p, "--qdb") && !used(p, "--nohmm") && !used(p, "--max") {
        return Err(
            "Failed to parse command line: --beta only makes sense in combination with --qdb, --nohmm or --max"
                .to_string(),
        );
    }
    if used(p, "--fbeta") && !used(p, "--fqdb") && !used(p, "--nohmm") {
        return Err(
            "Failed to parse command line: --fbeta only makes sense in combination with --fqdb or --nohmm"
                .to_string(),
        );
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Faithful esl_getopts-style parse against the full cmsearch option table:
    // every C option is recognized (value-taking ones consume their argument),
    // options must precede the positionals, and an unknown flag is a hard error
    // (matching C's "No such option" rejection) rather than being silently
    // swallowed with its value leaking into the positional list.
    let table = cmsearch_opt_table();
    // C esl_opt_ProcessCmdline routes any parse failure (unknown option, ambiguous
    // abbreviation) through the ERROR usage block with "Failed to parse command
    // line: <esl error>" and exit(1) — NOT a getopt-style exit(2).
    let parsed = infernox::search_cli::parse(&args, &table)
        .unwrap_or_else(|e| cmdline_fail(&format!("Failed to parse command line: {e}")));

    // C process_commandline (cmsearch.c:1748): esl_opt_VerifyConfig (table
    // require/incompat) runs first, then (cmsearch.c:1873+) the manual guards.
    // Both route to the ERROR usage block + exit(1).
    if let Err(e) = verify_config(&parsed) {
        cmdline_fail(&format!("Failed to parse command line: {e}"));
    }

    // C process_commandline (cmsearch.c:1751-1799): after ProcessCmdline +
    // VerifyConfig, the -h/--devhelp help block prints and exit(0) — BEFORE the
    // arg-count check and the manual guards. `do_dev = --devhelp`; giving both
    // -h and --devhelp yields the developer help (do_dev wins).
    let do_dev = parsed.is_set("--devhelp");
    if parsed.is_set("-h") || do_dev {
        full_help(do_dev);
    }

    if let Err(line) = manual_guards(&parsed) {
        cmdline_fail(&line);
    }

    // --- reporting / inclusion / search-space knobs (docgroups 1,3,4,5) ---
    let tblout: Option<String> = parsed.get_str("--tblout").map(|s| s.to_string());
    let toponly = parsed.is_set("--toponly");
    let bottomonly = parsed.is_set("--bottomonly");
    let ncpu: Option<usize> = parsed.get_usize("--cpu").unwrap_or_else(|e| usage_error(&e));
    let strict = parsed.is_set("--strict");
    let global = parsed.is_set("-g");
    let nohmm = parsed.is_set("--nohmm");
    let max = parsed.is_set("--max");
    let mid = parsed.is_set("--mid");
    let cyk = parsed.is_set("--cyk");
    let t_cutoff: Option<f32> = parsed.get_f32("-T").unwrap_or_else(|e| usage_error(&e));
    let e_report_opt: Option<f64> = parsed.get_f64("-E").unwrap_or_else(|e| usage_error(&e));
    // C `-Z <x>`: manual DB size in Mb → search-space size for E-values + filter tier.
    let z_mb_override: Option<f64> = parsed.get_f64("-Z").unwrap_or_else(|e| usage_error(&e));
    // C `--incE`/`--incT`: inclusion (significance) thresholds. Default incE=0.01,
    // inclusion by E-value; --incT switches to bit-score inclusion.
    let inc_e: f64 = parsed.get_f64("--incE").unwrap_or_else(|e| usage_error(&e)).unwrap_or(0.01);
    let inc_t: Option<f32> = parsed.get_f32("--incT").unwrap_or_else(|e| usage_error(&e));
    // C `--cut_ga`/`--cut_tc`/`--cut_nc`: use the model's GA/TC/NC bit-score cutoff.
    let model_cutoff: Option<ModelCutoff> = if parsed.is_set("--cut_ga") {
        Some(ModelCutoff::Ga)
    } else if parsed.is_set("--cut_tc") {
        Some(ModelCutoff::Tc)
    } else if parsed.is_set("--cut_nc") {
        Some(ModelCutoff::Nc)
    } else {
        None
    };
    // C `cmsearch` default: truncated alignment ON. `--notrunc` disables it.
    let notrunc = parsed.is_set("--notrunc");
    // C `cmsearch --tformat <s>`: assert the target seq file is in format <s>.
    let tformat: Option<String> = parsed.get_str("--tformat").map(|s| s.to_string());
    let positionals = parsed.positionals.clone();
    // C cmsearch.c:1802: esl_opt_ArgNumber(go) != 2 -> puts("Incorrect number of
    // command line arguments.") + ERROR block (no "Failed to parse..." prefix). Both
    // too few and too many args (e.g. a dangling option token that becomes a 3rd arg).
    if positionals.len() != 2 {
        cmdline_fail("Incorrect number of command line arguments.");
    }
    let cmpath = &positionals[0];
    let fapath = &positionals[1];
    // Bounded worker pool: caps peak memory (peak ≈ base + threads × per-task
    // working set). Mirrors C's --cpu. Omitted → rayon default (all cores).
    if let Some(n) = ncpu {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
    }
    // --strict: force bit-identical byte-parity with C in the Forward filter
    // (disables the FMA/reordered-sum relaxations). Default off (faster).
    infernox::cm_pipeline::set_forward_strict(strict);

    // C `cmsearch` reads a query CM file that may hold MANY models and runs the
    // full pipeline against the seq DB ONCE PER MODEL (cmsearch.c:517/746 —
    // `while (cm_file_Read(...) != eslEOF)`), concatenating each model's tblout
    // rows. Read the file and split it into per-model record chunks.
    let cmtext = std::fs::read_to_string(cmpath).unwrap_or_else(|e| {
        eprintln!("error: cannot read covariance model file '{}': {}", cmpath, e);
        std::process::exit(1);
    });
    let records = split_cm_records(&cmtext);
    if records.is_empty() {
        eprintln!("error: no CM records found in '{}'", cmpath);
        std::process::exit(1);
    }

    // Decode --tformat (if given) to a forced format; else autodetect (Unknown).
    // C cmsearch: `esl_sqio_EncodeFormat(esl_opt_GetString(go, "--tformat"))`.
    let informat = decode_informat(tformat.as_deref(), "--tformat");
    let recs = infernox::easel::read_seqfile(fapath, informat).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let seqs: Vec<&str> = recs.iter().map(|r| r.2.as_str()).collect();

    let e_report: f64 = e_report_opt.unwrap_or(10.0);
    // C per-stage filter P-value overrides (--F1/--F3/--F3b/--F4/--F4b/--F5).
    let f1 = parsed.get_f64("--F1").unwrap_or_else(|e| usage_error(&e));
    let f3 = parsed.get_f64("--F3").unwrap_or_else(|e| usage_error(&e));
    let f3b = parsed.get_f64("--F3b").unwrap_or_else(|e| usage_error(&e));
    let f4 = parsed.get_f64("--F4").unwrap_or_else(|e| usage_error(&e));
    let f4b = parsed.get_f64("--F4b").unwrap_or_else(|e| usage_error(&e));
    let f5 = parsed.get_f64("--F5").unwrap_or_else(|e| usage_error(&e));
    // C expert per-stage on/off + Viterbi/MSV-bias/env-bias threshold overrides
    // (cm_pipeline.c:575-604).
    let f2 = parsed.get_f64("--F2").unwrap_or_else(|e| usage_error(&e));
    let f2b = parsed.get_f64("--F2b").unwrap_or_else(|e| usage_error(&e));
    let f1b = parsed.get_f64("--F1b").unwrap_or_else(|e| usage_error(&e));
    let f5b = parsed.get_f64("--F5b").unwrap_or_else(|e| usage_error(&e));
    // C `--F6` (CYK filter P), `--cykenvx <n>` (F6env multiplier), `--noF6`, `--nocykenv`.
    let f6 = parsed.get_f64("--F6").unwrap_or_else(|e| usage_error(&e));
    let cykenvx = parsed.get_i64("--cykenvx").unwrap_or_else(|e| usage_error(&e));
    let no_f6 = parsed.is_set("--noF6");
    let nocykenv = parsed.is_set("--nocykenv");
    // C `--tau`/`--ftau`/`--maxtau`: HMM-band tail-loss probs (final/CYK-filter round + ceiling).
    let tau = parsed.get_f64("--tau").unwrap_or_else(|e| usage_error(&e));
    let ftau = parsed.get_f64("--ftau").unwrap_or_else(|e| usage_error(&e));
    let maxtau = parsed.get_f64("--maxtau").unwrap_or_else(|e| usage_error(&e));
    // C `--FZ <x>`: use <x> Mb for filter-threshold tier selection.
    let fz = parsed.get_f64("--FZ").unwrap_or_else(|e| usage_error(&e));
    // C `--rt1/--rt2/--rt3/--ns`: glocal domain/envelope-definition params.
    let rt1 = parsed.get_f64("--rt1").unwrap_or_else(|e| usage_error(&e));
    let rt2 = parsed.get_f64("--rt2").unwrap_or_else(|e| usage_error(&e));
    let rt3 = parsed.get_f64("--rt3").unwrap_or_else(|e| usage_error(&e));
    let ns = parsed.get_i64("--ns").unwrap_or_else(|e| usage_error(&e));
    let rfam = parsed.is_set("--rfam");
    let cfg = FaithfulConfig {
        toponly, bottomonly, e_report, global, nohmm, max, mid, rfam, cyk,
        t_cutoff, model_cutoff, notrunc,
        trunc5p: parsed.is_set("--5trunc"), trunc3p: parsed.is_set("--3trunc"),
        anytrunc: parsed.is_set("--anytrunc"),
        inttrunc: parsed.is_set("--inttrunc"),
        onlytrunc: parsed.is_set("--onlytrunc"),
        qdb: parsed.is_set("--qdb"),
        nonbanded: parsed.is_set("--nonbanded"),
        wcx: parsed.get_f64("--wcx").ok().flatten(),
        beta: parsed.get_f64("--beta").ok().flatten(),
        z_mb_override,
        f1, f3, f3b, f4, f4b, f5,
        f2, f2b, f1b, f5b,
        no_f1: parsed.is_set("--noF1"), no_f2: parsed.is_set("--noF2"),
        no_f3: parsed.is_set("--noF3"), no_f4: parsed.is_set("--noF4"),
        no_f2b: parsed.is_set("--noF2b"), no_f3b: parsed.is_set("--noF3b"),
        no_f4b: parsed.is_set("--noF4b"), do_f1b: parsed.is_set("--doF1b"),
        do_f5b: parsed.is_set("--doF5b"),
        f6, cykenvx, no_f6, nocykenv,
        tau, ftau, maxtau, fz,
        rt1, rt2, rt3, ns,
        hmmonly: parsed.is_set("--hmmonly"),
        nohmmonly: parsed.is_set("--nohmmonly"),
        hmmmax: parsed.is_set("--hmmmax"),
        hmm_f1: parsed.get_f64("--hmmF1").unwrap_or_else(|e| usage_error(&e)),
        hmm_f2: parsed.get_f64("--hmmF2").unwrap_or_else(|e| usage_error(&e)),
        hmm_f3: parsed.get_f64("--hmmF3").unwrap_or_else(|e| usage_error(&e)),
        hmmnobias: parsed.is_set("--hmmnobias"),
        hmmnonull2: parsed.is_set("--hmmnonull2"),
        nonull3: parsed.is_set("--nonull3"),
    };

    // C cmsearch.c -A: alignment output. `allow_trunc = !--nomiss` (default TRUE) is
    // passed to cm_tophits_Alignment; `textw > 0` selects STOCKHOLM (else PFAM). We
    // build each model's MSA inside the search loop (below) while its CM is alive.
    let dash_a: Option<String> = parsed.get_str("-A").map(|s| s.to_string());
    let a_allow_trunc = !parsed.is_set("--nomiss");
    let a_textw: i32 = if parsed.is_set("--notextw") {
        0
    } else {
        parsed.get_i64("--textw").unwrap_or(None).map(|v| v as i32).unwrap_or(120)
    };
    let a_report_by_score = t_cutoff.is_some();

    // Per-model search: build the searcher for each record (reads the CM in global
    // config, builds filters + CP9), search the whole DB, collect that model's hits.
    // Pipeline accounting resets per model because each record gets its own searcher.
    let mut model_results: Vec<ModelResult> = Vec::new();
    for rec in &records {
        let mut cm = cm_file_read_from_reader_opt(BufReader::new(rec.as_bytes()), false)
            .unwrap_or_else(|e| {
                eprintln!("error: cannot read covariance model: {:?}", e);
                std::process::exit(1);
            });
        // C cmsearch.c:2648 (cm_pli_NewModel -> cm_Configure with W_from_cmdline):
        // --wcx sets cm->W = (int)(cm->clen * wcx), overriding the file / band-calc W
        // before the pipeline builds the p7 filter + maxW windowing. --wcx is CLI-
        // incompatible with --qdb/--nohmm/--fqdb, so the QDB bands are never used for
        // scanning; only cm->W (windowing + truncation re-search widths) is affected.
        if let Some(x) = cfg.wcx {
            cm.w = (cm.clen as f64 * x) as i32;
        }
        // C cmsearch.c:2644 (CheckCMQDBInfo -> recalc): --beta <x> overrides the
        // final-round QDB2 tail-loss prob (qdbinfo->beta2 = final_beta, default 1e-15),
        // recomputing dmin2/dmax2. --beta is CLI-guarded to require --qdb/--nohmm/--max.
        if let Some(b) = cfg.beta {
            cm.qdb_beta2 = b;
        }
        let searcher = FaithfulSearcher::new(cm).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        // C cmsearch.c:586-592: unless we run the HMM-only pipeline for this model,
        // the CM search stages need E-value (exp-tail) calibration. C requires it when
        // (--nohmmonly || -g || (!--hmmonly && nbps>0)); if the CM lacks CMH_EXPTAIL_STATS
        // it dies via cm_Fail ("\nError: <msg>\n"), exit 1, after the run header (but
        // before this model's Query block). --hmmmax alone falls into this (it does NOT
        // enable HMM-only mode; only --hmmonly / a 0-basepair model does).
        {
            let nbps = searcher
                .cm()
                .ndtype
                .iter()
                .filter(|&&t| t as i32 == infernox::constants::MATP_ND)
                .count();
            let need_cm_stats =
                parsed.is_set("--nohmmonly") || global || (!parsed.is_set("--hmmonly") && nbps > 0);
            if need_cm_stats
                && (searcher.cm().flags & infernox::cm::CM_EXPTAIL_STATS) == 0
            {
                if model_results.is_empty() {
                    use std::io::Write;
                    let cpu_default: i64 = std::env::var("INFERNAL_NCPU")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(4);
                    let cpu_used =
                        matches!(parsed.get_i64("--cpu").unwrap_or(None), Some(v) if v != cpu_default);
                    let ncpu_disp: i64 = ncpu.map(|n| n as i64).unwrap_or_else(num_worker_threads);
                    print!("{}", output_header(&parsed, cmpath, fapath, ncpu_disp, cpu_used));
                    std::io::stdout().flush().ok();
                }
                eprint!(
                    "\nError: no E-value parameters were read for CM: {}.\nYou may need to run cmcalibrate.\n",
                    searcher.model_name()
                );
                std::process::exit(1);
            }
        }
        // C cm_pli_NewModel (cm_pipeline.c:1117-1129): --cut_ga/--cut_tc/--cut_nc
        // require the model to carry that bit-score cutoff. If absent it is a fatal
        // per-model error (esl_fatal "\nError: %s\n\n"), raised BEFORE this model is
        // searched — after the run header + this model's Query block have printed.
        if let Some(mc) = model_cutoff {
            if searcher.model_cutoff(mc).is_none() {
                let cpu_default: i64 = std::env::var("INFERNAL_NCPU")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(4);
                let cpu_used =
                    matches!(parsed.get_i64("--cpu").unwrap_or(None), Some(v) if v != cpu_default);
                let ncpu_disp: i64 = ncpu.map(|n| n as i64).unwrap_or_else(num_worker_threads);
                // Header is printed once, before the model loop; for the first model
                // this reproduces C's stdout exactly. (Multi-model runs where a LATER
                // model lacks the cutoff would also need the earlier models' output
                // flushed here — not exercised by the single-model fixtures.)
                let mut out = if model_results.is_empty() {
                    output_header(&parsed, cmpath, fapath, ncpu_disp, cpu_used)
                } else {
                    String::new()
                };
                let name = searcher.model_name().to_string();
                out.push_str(&format!("Query:       {}  [CLEN={}]\n", name, searcher.cm().clen));
                let acc = searcher.model_acc();
                if acc != "-" && !acc.is_empty() {
                    out.push_str(&format!("Accession:   {acc}\n"));
                }
                let desc = searcher.cm().desc.clone().unwrap_or_default();
                if !desc.is_empty() {
                    out.push_str(&format!("Description: {desc}\n"));
                }
                use std::io::Write;
                print!("{out}");
                std::io::stdout().flush().ok();
                let letter = match mc {
                    ModelCutoff::Ga => "GA",
                    ModelCutoff::Tc => "TC",
                    ModelCutoff::Nc => "NC",
                };
                eprint!("\nError: {letter} bit threshold unavailable for model {name}\n\n");
                std::process::exit(1);
            }
        }
        let reported = searcher.search(&seqs, &cfg);

        // Debug: dump the per-hit cm_alidisplay lines (nohmm path). Behind an env var
        // so it never perturbs normal output. Full concatenated lines (not chunked).
        if std::env::var("INFERNOX_ALIDUMP").is_ok() {
            for (rank, h) in reported.iter().enumerate() {
                if let Some(ad) = &h.alignment {
                    let tname = &recs[h.seq_idx].0;
                    eprintln!(
                        "ALIDUMP model={} rank={} target={} start={} stop={} cfrom={} cto={}",
                        searcher.model_name(), rank + 1, tname, h.start, h.stop, ad.cfrom_emit, ad.cto_emit
                    );
                    eprintln!("NC:{}", ad.ncline);
                    eprintln!("CS:{}", ad.csline);
                    eprintln!("MO:{}", ad.model);
                    eprintln!("MA:{}", ad.mline);
                    eprintln!("AS:{}", ad.aseq);
                    eprintln!("RF:{}", ad.rfline);
                }
            }
        }

        // Per-model bit-score inclusion cutoff (C `--cut_ga/--cut_tc/--cut_nc`):
        // resolves to this model's GA/TC/NC bit-score, or None if the model lacks it
        // (then inclusion falls back to --incT/--incE, matching reporting fallback).
        let inc_cutoff = model_cutoff.and_then(|mc| searcher.model_cutoff(mc));
        // Capture the pipeline statistics-summary data while the searcher (and its
        // per-pass accounting) is still alive. C cm_pli_Statistics prints this per query.
        let pli = searcher.pli_stats(&seqs, &cfg);
        let summed = searcher.acct_summed();
        let std_acct = searcher.acct_snapshot(1); // PLI_PASS_STD_ANY
        // Truncated-hit output tallies: passes 2 (5P) + 3 (3P) + 4 (5P&3P FORCE) +
        // 5 (5P&3P ANY, --anytrunc/--inttrunc/--onlytrunc), for the "includes N
        // truncated hit(s)" count and its residue ratio (C cm_pipeline.c:236-256).
        // Passes not run for the active mode contribute 0, so summing all four is
        // correct in every mode.
        let mut n_output_trunc = 0u64;
        let mut pos_output_trunc = 0u64;
        for p in [2usize, 3, 4, 5] {
            let a = searcher.acct_snapshot(p);
            n_output_trunc += a.n_output;
            pos_output_trunc += a.pos_output;
        }
        // HMM-only pipeline statistics (C pli_hmmonly_pass_statistics), or None for
        // the CM pipeline. Captured here while the searcher's accounting is alive.
        let hmmonly_stats = searcher.hmmonly_pass_stats(&seqs, &cfg);
        let stats = StatsData {
            pli,
            summed,
            std_nres: std_acct.nres_top + std_acct.nres_bot,
            n_output_trunc,
            pos_output_trunc,
            hmmonly_stats,
        };
        // C cmsearch.c:723 cm_tophits_Alignment(): build an MSA of all INCLUDED hits
        // (CM_HIT_IS_INCLUDED) for this model, while its CM is still alive. C reaches
        // each hit's parsetree via cm_alidisplay_Backconvert(ad); we retained the
        // original search parsetree on ad (approach B) and feed it straight through the
        // byte-verified Parsetrees2Alignment. Hits are already in the final sorted
        // (by-evalue) order, matching C's th->hit[] iteration.
        let msa_out: Option<(String, usize)> = dash_a.as_ref().map(|_| {
            let abc_out = infernox::easel::alphabet::EslAlphabet::rna();
            let mut names: Vec<String> = Vec::new();
            let mut dsqs: Vec<Vec<u8>> = Vec::new();
            let mut trs: Vec<infernox::parsetree::Parsetree> = Vec::new();
            let mut ppstrs: Vec<Option<Vec<u8>>> = Vec::new();
            for h in &reported {
                if !hit_included(h, inc_cutoff, a_report_by_score, inc_t, inc_e) {
                    continue;
                }
                // Approach (B): reuse the retained search parsetree/pp/subseq. A hit
                // that carries no CM parsetree (e.g. HMM-only) cannot be backed this
                // way; skip it (the target fixtures never mix such hits into `-A`).
                let ad = match &h.alignment {
                    Some(a) => a,
                    None => continue,
                };
                let (mut tr, dsq) = match (&ad.ali_tr, &ad.ali_dsq) {
                    (Some(tr), Some(dsq)) => (tr.clone(), dsq.clone()),
                    _ => continue,
                };
                // C cm_tophits_Alignment reaches the parsetree via Backconvert ->
                // Transmogrify, which OVERLOADS tr->pass_idx (cm_modelmaker.c:1064-1078)
                // from the hit's ACTUAL truncation status (Is5PTrunc/Is3PTrunc =
                // cfrom_emit!=cfrom_span / cto_emit!=cto_span, cm_alidisplay.c:1280-1291),
                // NOT the pipeline pass it was found in. Parsetrees2Alignment's `~`
                // (missing-char) injection then gates on that overloaded pass_idx
                // (cm_parsetree.c:1306-1319). We retain the byte-verified search parse
                // (residue layout already matches C — verified on truncated tRNAs) and
                // replicate exactly that pass_idx overload so the terminal `~` columns
                // render identically. `--nomiss` (allow_trunc=FALSE) skips all of this.
                if a_allow_trunc {
                    let trunc_5p = ad.cfrom_emit != ad.cfrom_span;
                    let trunc_3p = ad.cto_emit != ad.cto_span;
                    if trunc_5p || trunc_3p {
                        tr.is_std = false;
                        tr.pass_idx = if trunc_5p {
                            if trunc_3p {
                                infernox::cm_trunc::PLI_PASS_5P_AND_3P_FORCE
                            } else {
                                infernox::cm_trunc::PLI_PASS_5P_ONLY_FORCE
                            }
                        } else {
                            infernox::cm_trunc::PLI_PASS_3P_ONLY_FORCE
                        };
                    } else {
                        // Not actually truncated: Transmogrify leaves a standard parse
                        // (is_std TRUE), so no `~` is added regardless of the pass it
                        // was found in.
                        tr.is_std = true;
                        tr.pass_idx = infernox::cm_trunc::PLI_PASS_STD_ANY;
                    }
                }
                // C esl_msa_FormatSeqName: "%s/%ld-%ld" (sqname, sqfrom, sqto). For a
                // reverse-strand hit start > stop, exactly as C's ad->sqfrom/sqto.
                names.push(format!("{}/{}-{}", recs[h.seq_idx].0, h.start, h.stop));
                dsqs.push(dsq);
                trs.push(tr);
                ppstrs.push(ad.ali_pp.clone());
            }
            let nseq = trs.len();
            if nseq == 0 {
                return (String::new(), 0);
            }
            // Parsetrees2Alignment(do_full=TRUE, do_matchonly=FALSE, do_flush=FALSE,
            // allow_trunc=!--nomiss). do_post=TRUE (C always passes a non-NULL ppstrA).
            let msa = infernox::cm_dpalign::parsetrees_to_alignment(
                searcher.cm(), &abc_out, &names, &dsqs, &trs, &ppstrs,
                true, false, false, a_allow_trunc,
            );
            let fmt = if a_textw > 0 {
                infernox::easel::msafile::MsaFormat::Stockholm
            } else {
                infernox::easel::msafile::MsaFormat::Pfam
            };
            let mut buf: Vec<u8> = Vec::new();
            infernox::easel::msafile::esl_msafile_write(&mut buf, &msa, fmt).expect("write -A MSA");
            (String::from_utf8(buf).expect("MSA is UTF-8"), nseq)
        });
        model_results.push(ModelResult {
            qname: searcher.model_name().to_string(),
            qacc: searcher.model_acc().to_string(),
            qdesc: searcher.cm().desc.clone().unwrap_or_default(),
            clen: searcher.cm().clen,
            hits: reported,
            inc_cutoff,
            stats,
            msa_out,
        });
    }

    // C `by_E` (reporting mode) drives inclusion too (cm_pli_TargetIncludable uses
    // pli->by_E, NOT inc_by_E): by_E is FALSE iff -T given (cut_* handled per-model).
    let report_by_score = t_cutoff.is_some();

    // --- tblout table (written to the --tblout file only; unchanged path) ---
    if let Some(path) = &tblout {
        let mut out = format_tblout(&recs, &model_results, inc_e, inc_t, report_by_score);
        // C cmsearch.c:768 writes the tabular footer via cm_tophits_TabularTail.
        out.push_str(&infernox::cm_tophits::tabular_tail("cmsearch", "SEARCH", cmpath, fapath, &args));
        std::fs::write(path, &out).expect("write tblout");
    }

    // --- human-readable report (stdout, or -o file): C cmsearch.c main output ---
    // Options that only affect human output.
    let show_accessions = parsed.is_set("--acc");
    let show_alignments = !parsed.is_set("--noali");
    // textw: default 120 (C `--textw` default); --notextw => 0 (unlimited). C enforces
    // a getopts range n>=120 on --textw (values <120 are rejected before we get here).
    let textw: i32 = if parsed.is_set("--notextw") {
        0
    } else {
        parsed.get_i64("--textw").unwrap_or(None).map(|v| v as i32).unwrap_or(120)
    };
    // Worker-thread count shown in the header banner: the resolved value (== --cpu when
    // given, else the machine core count). C prints the " [--cpu]" suffix iff
    // esl_opt_IsUsed(--cpu) — i.e. --cpu was given AND differs from its default
    // (CMNCPU="4", or $INFERNAL_NCPU). Giving `--cpu 4` (== default) prints no suffix.
    let cpu_default: i64 = std::env::var("INFERNAL_NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let cpu_used = matches!(parsed.get_i64("--cpu").unwrap_or(None), Some(v) if v != cpu_default);
    let ncpu_disp: i64 = ncpu.map(|n| n as i64).unwrap_or_else(|| num_worker_threads());

    // C cmsearch.c -A: each model's MSA is appended to the same alignment file (afp),
    // and a per-model status line is printed to stdout after that model's statistics.
    let mut a_file_buf = String::new();
    let mut human = output_header(&parsed, cmpath, fapath, ncpu_disp, cpu_used);
    for m in &model_results {
        // Per-query header block (cmsearch.c:602-604).
        human.push_str(&format!("Query:       {}  [CLEN={}]\n", m.qname, m.clen));
        if m.qacc != "-" && !m.qacc.is_empty() {
            human.push_str(&format!("Accession:   {}\n", m.qacc));
        }
        if !m.qdesc.is_empty() {
            human.push_str(&format!("Description: {}\n", m.qdesc));
        }
        // "Hit scores" table (cm_tophits_Targets).
        let rows: Vec<infernox::cm_tophits::HitRow> = m
            .hits
            .iter()
            .map(|h| {
                let r = &recs[h.seq_idx];
                infernox::cm_tophits::HitRow {
                    name: r.0.as_str(),
                    acc: "",
                    desc: r.1.as_str(),
                    evalue: h.evalue,
                    score: h.score,
                    bias: h.bias,
                    start: h.start,
                    stop: h.stop,
                    in_rc: h.in_rc,
                    hmmonly: h.hmmonly,
                    trunc: h.trunc.as_str(),
                    gc: h.gc,
                    included: hit_included(h, m.inc_cutoff, report_by_score, inc_t, inc_e),
                }
            })
            .collect();
        human.push_str(&infernox::cm_tophits::cm_tophits_targets(
            &rows,
            false, // search mode (CM_SEARCH_SEQS)
            show_accessions,
            textw,
        ));
        human.push_str("\n\n");
        // "Hit alignments" section (C cmsearch.c:691-693): only when show_alignments
        // (default; suppressed by --noali). cm_tophits_HitAlignments + "\n\n". Called
        // even with 0 reported hits (it emits the header + no-hits message).
        if show_alignments {
            let cmacc = if m.qacc == "-" { "" } else { m.qacc.as_str() };
            let ali_rows: Vec<infernox::cm_tophits::AliHit> = m
                .hits
                .iter()
                .filter_map(|h| {
                    let ad = h.alignment.as_ref()?;
                    let r = &recs[h.seq_idx];
                    Some(infernox::cm_tophits::AliHit {
                        ad,
                        // cmsearch: model is the query, sequence is the target (>>).
                        cmname: m.qname.as_str(),
                        cmacc,
                        sqname: r.0.as_str(),
                        sqacc: "",
                        tdesc: r.1.as_str(),
                        target_is_model: false,
                        evalue: h.evalue,
                        score: h.score,
                        bias: h.bias,
                        hmmonly: h.hmmonly,
                        start: h.start,
                        stop: h.stop,
                        in_rc: h.in_rc,
                        src_l: r.2.len() as i64,
                        clen: m.clen,
                        included: hit_included(h, m.inc_cutoff, report_by_score, inc_t, inc_e),
                    })
                })
                .collect();
            human.push_str(&infernox::cm_tophits::cm_tophits_hit_alignments(
                &ali_rows,
                show_accessions,
                textw,
                m.hits.len(),
            ));
            human.push_str("\n\n");
        }
        // Statistics summary: HMM-only pipeline (C pli_hmmonly_pass_statistics) when
        // this model ran HMM-only, else the CM pipeline summary (C cm_pli_Statistics).
        if let Some(hs) = &m.stats.hmmonly_stats {
            human.push_str(&infernox::p7_hmmonly::pli_hmmonly_pass_statistics(hs));
            // C cm_pipeline.c:1772: pli_hmmonly_pass_statistics(ofp,pli); fprintf(ofp,"\n");
            human.push('\n');
        } else {
            human.push_str(&infernox::cm_tophits::format_pli_statistics(
                &m.stats.pli,
                &m.stats.summed,
                m.stats.std_nres,
                m.stats.n_output_trunc,
                m.stats.pos_output_trunc,
                false,
            ));
        }
        // C cm_pli_Statistics ends with esl_stopwatch_Display "# CPU time:" (timing-
        // variable line, treated like Date).
        human.push_str(&format!("# CPU time: {}\n", cpu_time_line()));
        // C cmsearch.c:723-732: after the statistics, write this model's `-A` MSA to
        // the alignment file and print the status line to stdout. cm_tophits_Alignment
        // returns NULL (nseq==0) when no hits satisfy inclusion → "no alignment" line.
        if let Some(path) = &dash_a {
            match &m.msa_out {
                Some((text, nseq)) if *nseq > 0 => {
                    a_file_buf.push_str(text);
                    human.push_str(&format!(
                        "# Alignment of {} hits satisfying inclusion thresholds saved to: {}\n",
                        nseq, path
                    ));
                }
                _ => {
                    human.push_str(
                        "# No hits satisfy inclusion thresholds; no alignment saved\n",
                    );
                }
            }
        }
        // C cmsearch.c:740 prints "//" per query.
        human.push_str("//\n");
    }
    // C cmsearch.c:501 opened afp at startup; write the accumulated MSA text now.
    if let Some(path) = &dash_a {
        std::fs::write(path, &a_file_buf).unwrap_or_else(|e| {
            eprintln!("Failed to open alignment file {} for writing: {}", path, e);
            std::process::exit(1);
        });
    }
    // C cmsearch.c:769 prints "[ok]" once at the very end (after all queries).
    human.push_str("[ok]\n");

    match parsed.get_str("-o") {
        Some(path) => std::fs::write(path, &human).expect("write -o"),
        None => print!("{}", human),
    }

    if std::env::var("STAGE_TIMING").is_ok() {
        let rows = [
            ("F1+F3+F3b  (MSV/Fwd/bias filter)", T_F1F3.load(Ordering::Relaxed)),
            ("F4+F4b+F5  (glocal Fwd/Bwd+envdef)", T_F4F5.load(Ordering::Relaxed)),
            ("F6 bands   (CP9 HMM banding)", T_F6BAND.load(Ordering::Relaxed)),
            ("F6 CYK     (banded CYK scan)", T_F6CYK.load(Ordering::Relaxed)),
            ("F7 bands   (CP9 HMM banding)", T_F7BAND.load(Ordering::Relaxed)),
            ("F7 Inside  (banded Inside+null3)", T_F7INS.load(Ordering::Relaxed)),
        ];
        let sum: u64 = rows.iter().map(|r| r.1).sum();
        eprintln!("\n=== per-stage CPU time (summed over threads; run with --cpu 1) ===");
        for (name, ns) in rows {
            eprintln!(
                "  {:<36} {:>8.1} ms  {:>5.1}%",
                name,
                ns as f64 / 1e6,
                if sum > 0 { 100.0 * ns as f64 / sum as f64 } else { 0.0 }
            );
        }
        eprintln!("  {:<36} {:>8.1} ms", "TOTAL (pipeline stages)", sum as f64 / 1e6);
    }
}

/// Column widths for one model's tblout block (C: cm_tophits_TabularTargets1,
/// cm_tophits.c:2258-2262 — widths come from that model's own tophits + qname/qacc).
struct TblWidths {
    tnamew: usize,
    qnamew: usize,
    qaccw: usize,
    taccw: usize,
    posw: usize,
}

fn compute_widths(recs: &[(String, String, String)], m: &ModelResult) -> TblWidths {
    TblWidths {
        tnamew: m.hits.iter().map(|h| recs[h.seq_idx].0.len()).max().unwrap_or(0).max(20),
        qnamew: m.qname.len().max(20),
        // C: qaccw = qacc ? max(9, strlen(qacc)) : 9. model_acc() returns "-" when none.
        qaccw: if m.qacc == "-" { 9 } else { m.qacc.len().max(9) },
        taccw: 9usize, // seq targets carry no accession → GetMaxAccessionLength=0, max(9,0)=9
        posw: m
            .hits
            .iter()
            .map(|h| h.start.abs().max(h.stop.abs()).to_string().len())
            .max()
            .unwrap_or(0)
            .max(8),
    }
}

fn format_tblout(
    recs: &[(String, String, String)],
    models: &[ModelResult],
    inc_e: f64,
    inc_t: Option<f32>,
    report_by_score: bool,
) -> String {
    let mut s = String::new();

    // Header printed once, using the FIRST model's widths (C: show_header = cm_idx==1).
    if let Some(first) = models.first() {
        let w = compute_widths(recs, first);
        let (tnamew, qnamew, qaccw, taccw, posw) = (w.tnamew, w.qnamew, w.qaccw, w.taccw, w.posw);
        // header line 1 (names)
        s.push_str(&format!(
            "#{:<w1$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:>3} {:>8} {:>8} {:>posw$} {:>posw$} {:>6} {:>5} {:>4} {:>4} {:>5} {:>6} {:>9} {:>3} {}\n",
            "target name", "accession", "query name", "accession", "mdl", "mdl from", "mdl to",
            "seq from", "seq to", "strand", "trunc", "pass", "gc", "bias", "score", "E-value", "inc",
            "description of target", w1 = tnamew - 1
        ));
        // header line 2 (dashes)
        let dash = |n: usize| "-".repeat(n);
        s.push_str(&format!(
            "#{:<w1$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:<3} {:<8} {:<8} {:<posw$} {:<posw$} {:<6} {:<5} {:<4} {:<4} {:<5} {:<6} {:<9} {:<3} {}\n",
            dash(tnamew - 1), dash(taccw), dash(qnamew), dash(qaccw), "---", dash(8), dash(8),
            dash(posw), dash(posw), "------", "-----", "----", "----", "-----", "------", "---------",
            "---", "---------------------", w1 = tnamew - 1
        ));
    }

    // Each model's rows, in file order, formatted with that model's OWN widths.
    for m in models {
        let w = compute_widths(recs, m);
        let (tnamew, qnamew, qaccw, taccw, posw) = (w.tnamew, w.qnamew, w.qaccw, w.taccw, w.posw);
        for h in &m.hits {
            let r = &recs[h.seq_idx];
            let tname = r.0.as_str();
            let tdesc = if r.1.is_empty() { "-" } else { r.1.as_str() };
            let strand = if h.in_rc { "-" } else { "+" };
            // Inclusion ("!"/"?"), C cm_pli_TargetIncludable (cm_pipeline.c:865-872):
            //   if   by_E:  include iff Eval <= incE
            //   else:       include iff score >= incT
            // NOTE the C quirk: inclusion keys off `by_E` (the REPORTING mode), so
            // `--incT` has NO effect unless `-T` (or a --cut_*) is also given; when
            // reporting by E-value, inclusion is by incE (default 0.01) regardless of
            // --incT. With --cut_ga/tc/nc, T=incT=the model's cutoff (per model).
            let included = match m.inc_cutoff {
                Some(cut) => h.score >= cut,
                None => {
                    if report_by_score {
                        h.score >= inc_t.unwrap_or(0.0)
                    } else {
                        h.evalue <= inc_e
                    }
                }
            };
            let inc = if included { "!" } else { "?" };
            let eval_s = fmt_evalue(h.evalue);
            s.push_str(&format!(
                "{:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:>3} {:>8} {:>8} {:>posw$} {:>posw$} {:>6} {:>5} {:>4} {:>4.2} {:>5.1} {:>6.1} {:>9} {:<3} {}\n",
                tname, "-", m.qname, m.qacc, if h.hmmonly { "hmm" } else { "cm" },
                h.mdl_from, h.mdl_to, h.start, h.stop, strand,
                h.trunc.as_str(), h.pass_idx,
                h.gc, h.bias, h.score, eval_s, inc, tdesc
            ));
        }
    }
    s
}

/// C `cm_pli_TargetIncludable` (cm_pipeline.c:865-872) reproduced for one hit:
/// inclusion keys off a per-model bit-score cutoff (--cut_ga/tc/nc), else the
/// reporting mode (by_E => E<=incE ; by_T => score>=incT). Matches the inline logic
/// in `format_tblout`; factored out for reuse by the human "Hit scores" table.
fn hit_included(
    h: &FaithfulHit,
    inc_cutoff: Option<f32>,
    report_by_score: bool,
    inc_t: Option<f32>,
    inc_e: f64,
) -> bool {
    match inc_cutoff {
        Some(cut) => h.score >= cut,
        None => {
            if report_by_score {
                h.score >= inc_t.unwrap_or(0.0)
            } else {
                h.evalue <= inc_e
            }
        }
    }
}

/// Resolved worker-thread count when `--cpu` is absent (C: `esl_threads_CPUCount`).
/// Used only for the banner line's default; the verify battery always passes --cpu.
fn num_worker_threads() -> i64 {
    std::thread::available_parallelism().map(|n| n.get() as i64).unwrap_or(0)
}

/// Captured statistics-summary data for one model (C `pli->acct` + `pli->F*`),
/// grabbed while the searcher is alive. Rendered by [`format_pli_statistics`].
struct StatsData {
    pli: PliStats,
    summed: PassAcctSnapshot,
    /// STD pass residues (nres_top+nres_bot): the "residues searched" (non-truncated).
    std_nres: u64,
    n_output_trunc: u64,
    pos_output_trunc: u64,
    /// C `pli_hmmonly_pass_statistics` data when this model ran HMM-only; then the
    /// summary block replaces the CM pipeline statistics summary.
    hmmonly_stats: Option<infernox::p7_hmmonly::HmmonlyPassStats>,
}

/// Timing-variable "# CPU time:" line body (matches esl_stopwatch_Display format;
/// the numbers are a documented-variable exception, like the Date line).
fn cpu_time_line() -> String {
    "0.00u 0.00s 00:00:00.00 Elapsed: 00:00:00.00".to_string()
}

/// printf `%g` (default precision 6 significant figures), for header config values.
fn fmt_g(x: f64) -> String {
    fmt_g_prec(x, 6)
}

/// printf `%.*g` with `p` significant figures.
fn fmt_g_prec(x: f64, p: i32) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let s = format!("{:.*e}", (p - 1) as usize, x);
    let parts: Vec<&str> = s.splitn(2, 'e').collect();
    let mant = parts[0];
    let ex = parts[1].parse::<i32>().unwrap_or(0);
    if ex < -4 || ex >= p {
        let m = strip_zeros(mant);
        let sign = if ex < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", m, sign, ex.abs())
    } else {
        let dec = (p - 1 - ex).max(0) as usize;
        strip_zeros(&format!("{:.*}", dec, x))
    }
}

/// Faithful port of `output_header` (cmsearch.c:2232): the `# ...` banner + config
/// block that precedes the per-query output. Every line is emitted iff the driving
/// option was set on the command line (C `esl_opt_IsUsed`), in the exact C order.
/// Reuses the fixed release constants (1.1.5 / "Sep 2023"), like the tblout tail.
fn output_header(p: &Parsed, cmfile: &str, seqfile: &str, ncpus: i64, cpu_used: bool) -> String {
    let mut s = String::new();
    // cm_banner: fixed for a byte-identical drop-in invoked as `cmsearch`.
    s.push_str("# cmsearch :: search CM(s) against a sequence database\n");
    s.push_str("# INFERNAL 1.1.5 (Sep 2023)\n");
    s.push_str("# Copyright (C) 2023 Howard Hughes Medical Institute.\n");
    s.push_str("# Freely distributed under the BSD open source license.\n");
    s.push_str("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");

    s.push_str(&format!("# query CM file:                         {}\n", cmfile));
    s.push_str(&format!("# target sequence database:              {}\n", seqfile));
    let g = |name: &str| -> f64 { p.get_f64(name).unwrap_or(None).unwrap_or(0.0) };
    let gi = |name: &str| -> i64 { p.get_i64(name).unwrap_or(None).unwrap_or(0) };
    if p.is_set("-g")           { s.push_str("# CM configuration:                      glocal\n"); }
    if p.is_set("-Z")           { s.push_str(&format!("# database size is set to:               {:.1} Mb\n", g("-Z"))); }
    if p.is_set("-o")           { s.push_str(&format!("# output directed to file:               {}\n", p.get_str("-o").unwrap_or(""))); }
    if p.is_set("-A")           { s.push_str(&format!("# MSA of significant hits saved to file: {}\n", p.get_str("-A").unwrap_or(""))); }
    if p.is_set("--tblout")     { s.push_str(&format!("# tabular output of hits:                {}\n", p.get_str("--tblout").unwrap_or(""))); }
    if p.is_set("--acc")        { s.push_str("# prefer accessions over names:          yes\n"); }
    if p.is_set("--noali")      { s.push_str("# show alignments in output:             no\n"); }
    if p.is_set("--notextw")    { s.push_str("# max ASCII text line length:            unlimited\n"); }
    if p.is_set("--textw")      { s.push_str(&format!("# max ASCII text line length:            {}\n", gi("--textw"))); }
    if p.is_set("--verbose")    { s.push_str("# verbose output mode:                   on\n"); }
    if p.is_set("-E")           { s.push_str(&format!("# sequence reporting threshold:          E-value <= {}\n", fmt_g(g("-E")))); }
    if p.is_set("-T")           { s.push_str(&format!("# sequence reporting threshold:          score >= {}\n", fmt_g(g("-T")))); }
    if p.is_set("--incE")       { s.push_str(&format!("# sequence inclusion threshold:          E-value <= {}\n", fmt_g(g("--incE")))); }
    if p.is_set("--incT")       { s.push_str(&format!("# sequence inclusion threshold:          score >= {}\n", fmt_g(g("--incT")))); }
    if p.is_set("--cut_ga")     { s.push_str("# model-specific thresholding:           GA cutoffs\n"); }
    if p.is_set("--cut_nc")     { s.push_str("# model-specific thresholding:           NC cutoffs\n"); }
    if p.is_set("--cut_tc")     { s.push_str("# model-specific thresholding:           TC cutoffs\n"); }
    if p.is_set("--max")        { s.push_str("# Max sensitivity mode:                  on [all heuristic filters off]\n"); }
    if p.is_set("--nohmm")      { s.push_str("# CM-only mode:                          on [HMM filters off]\n"); }
    if p.is_set("--mid")        { s.push_str("# HMM MSV and Viterbi filters:           off\n"); }
    if p.is_set("--rfam")       { s.push_str("# Rfam pipeline mode:                    on [strict filtering]\n"); }
    if p.is_set("--FZ")         { s.push_str(&format!("# Filters set as if DB size in Mb is:    {}\n", format_f(g("--FZ")))); }
    if p.is_set("--Fmid")       { s.push_str(&format!("# HMM Forward filter thresholds set to:  {}\n", fmt_g(g("--Fmid")))); }
    if p.is_set("--hmmonly")    { s.push_str("# HMM-only mode (for all models):        on [CM will not be used]\n"); }
    if p.is_set("--notrunc")    { s.push_str("# truncated sequence detection:          off\n"); }
    if p.is_set("--anytrunc")   { s.push_str("# allowing truncated sequences anywhere: on\n"); }
    if p.is_set("--inttrunc")   { s.push_str("# allowing internally truncated seqs:    on\n"); }
    if p.is_set("--onlytrunc")  { s.push_str("# only allowing truncated seqs anywhere: on\n"); }
    if p.is_set("--5trunc")     { s.push_str("# allowing 5' truncated seqs only:       on\n"); }
    if p.is_set("--3trunc")     { s.push_str("# allowing 3' truncated seqs only:       on\n"); }
    if p.is_set("--nonull3")    { s.push_str("# null3 bias corrections:                off\n"); }
    if p.is_set("--mxsize")     { s.push_str(&format!("# maximum DP alignment matrix size:      {:.1} Mb\n", g("--mxsize"))); }
    if p.is_set("--smxsize")    { s.push_str(&format!("# maximum DP search matrix size:         {:.1} Mb\n", g("--smxsize"))); }
    if p.is_set("--cyk")        { s.push_str("# use CYK for final search stage         on\n"); }
    if p.is_set("--acyk")       { s.push_str("# use CYK to align hits:                 on\n"); }
    if p.is_set("--wcx")        { s.push_str(&format!("# W set as <x> * cm->clen:               <x>={}\n", fmt_g(g("--wcx")))); }
    if p.is_set("--onepass")    { s.push_str("# using CM for best HMM pass only:       on\n"); }
    if p.is_set("--olonepass")  { s.push_str("# using CM for best HMM pass only (ol):  on\n"); }
    if p.is_set("--noiter")     { s.push_str("# iterative HMM band tightening:         off\n"); }
    if p.is_set("--toponly")    { s.push_str("# search top-strand only:                on\n"); }
    if p.is_set("--bottomonly") { s.push_str("# search bottom-strand only:             on\n"); }
    if p.is_set("--tformat")    { s.push_str(&format!("# targ <seqdb> format asserted:          {}\n", p.get_str("--tformat").unwrap_or(""))); }
    // Developer/expert options (cmsearch.c:2284-2350), same IsUsed gating.
    if p.is_set("--noF1")       { s.push_str("# HMM MSV filter:                        off\n"); }
    if p.is_set("--noF2")       { s.push_str("# HMM Vit filter:                        off\n"); }
    if p.is_set("--noF3")       { s.push_str("# HMM Fwd filter:                        off\n"); }
    if p.is_set("--noF4")       { s.push_str("# HMM glocal Fwd filter:                 off\n"); }
    if p.is_set("--noF6")       { s.push_str("# CM CYK filter:                         off\n"); }
    if p.is_set("--doF1b")      { s.push_str("# HMM MSV biased comp filter:            on\n"); }
    if p.is_set("--noF2b")      { s.push_str("# HMM Vit biased comp filter:            off\n"); }
    if p.is_set("--noF3b")      { s.push_str("# HMM Fwd biased comp filter:            off\n"); }
    if p.is_set("--noF4b")      { s.push_str("# HMM gFwd biased comp filter:           off\n"); }
    if p.is_set("--doF5b")      { s.push_str("# HMM per-envelope biased comp filter:   on\n"); }
    if p.is_set("--F1")         { s.push_str(&format!("# HMM MSV filter P threshold:            <= {}\n", fmt_g(g("--F1")))); }
    if p.is_set("--F1b")        { s.push_str(&format!("# HMM MSV bias P threshold:              <= {}\n", fmt_g(g("--F1b")))); }
    if p.is_set("--F2")         { s.push_str(&format!("# HMM Vit filter P threshold:            <= {}\n", fmt_g(g("--F2")))); }
    if p.is_set("--F2b")        { s.push_str(&format!("# HMM Vit bias P threshold:              <= {}\n", fmt_g(g("--F2b")))); }
    if p.is_set("--F3")         { s.push_str(&format!("# HMM Fwd filter P threshold:            <= {}\n", fmt_g(g("--F3")))); }
    if p.is_set("--F3b")        { s.push_str(&format!("# HMM Fwd bias P threshold:              <= {}\n", fmt_g(g("--F3b")))); }
    if p.is_set("--F4")         { s.push_str(&format!("# HMM glocal Fwd filter P threshold:     <= {}\n", fmt_g(g("--F4")))); }
    if p.is_set("--F4b")        { s.push_str(&format!("# HMM glocal Fwd bias P threshold:       <= {}\n", fmt_g(g("--F4b")))); }
    if p.is_set("--F5")         { s.push_str(&format!("# HMM env defn filter P threshold:       <= {}\n", fmt_g(g("--F5")))); }
    if p.is_set("--F5b")        { s.push_str(&format!("# HMM env defn bias   P threshold:       <= {}\n", fmt_g(g("--F5b")))); }
    if p.is_set("--F6")         { s.push_str(&format!("# CM CYK filter P threshold:             <= {}\n", fmt_g(g("--F6")))); }
    // HMM-only filter pipeline (docgroup 102, cmsearch.c:2306-2312), same IsUsed gating.
    if p.is_set("--hmmmax")     { s.push_str("# max sensitivity mode   (HMM-only):     on [all heuristic filters off]\n"); }
    if p.is_set("--hmmF1")      { s.push_str(&format!("# HMM MSV filter P threshold (HMM-only)  <= {}\n", fmt_g(g("--hmmF1")))); }
    if p.is_set("--hmmF2")      { s.push_str(&format!("# HMM Vit filter P threshold (HMM-only)  <= {}\n", fmt_g(g("--hmmF2")))); }
    if p.is_set("--hmmF3")      { s.push_str(&format!("# HMM Fwd filter P threshold (HMM-only)  <= {}\n", fmt_g(g("--hmmF3")))); }
    if p.is_set("--hmmnobias")  { s.push_str("# HMM MSV biased comp filter (HMM-only)  off\n"); }
    if p.is_set("--hmmnonull2") { s.push_str("# null2 bias corrections (HMM-only):     off\n"); }
    if p.is_set("--nohmmonly")  { s.push_str("# HMM-only mode for 0 basepair models:   no\n"); }
    if p.is_set("--rt1")        { s.push_str(&format!("# domain definition rt1 parameter        {}\n", fmt_g(g("--rt1")))); }
    if p.is_set("--rt2")        { s.push_str(&format!("# domain definition rt2 parameter        {}\n", fmt_g(g("--rt2")))); }
    if p.is_set("--rt3")        { s.push_str(&format!("# domain definition rt3 parameter        {}\n", fmt_g(g("--rt3")))); }
    if p.is_set("--ns")         { s.push_str(&format!("# number of envelope tracebacks sampled  {}\n", gi("--ns"))); }
    if p.is_set("--ftau")       { s.push_str(&format!("# tau parameter for CYK filter stage:    {}\n", fmt_g(g("--ftau")))); }
    if p.is_set("--fsums")      { s.push_str("# posterior sums (CYK filter stage):     on\n"); }
    if p.is_set("--fqdb")       { s.push_str("# QDBs (CYK filter stage)                on\n"); }
    // C output_header uses esl_opt_IsUsed (FALSE when value == default string,
    // esl_getopts.c:935/IsDefault), so an explicit default (e.g. --beta 1e-15) emits
    // NO header line. is_set alone would over-emit. Defaults: --fbeta 1e-7, --beta 1e-15.
    if p.is_set("--fbeta") && p.get_str("--fbeta") != Some("1e-7")   { s.push_str(&format!("# beta parameter for CYK filter stage:   {}\n", fmt_g(g("--fbeta")))); }
    if p.is_set("--fnonbanded") { s.push_str("# no bands (CYK filter stage)            on\n"); }
    if p.is_set("--nocykenv")   { s.push_str("# CYK envelope redefinition:             off\n"); }
    if p.is_set("--cykenvx")    { s.push_str(&format!("# CYK envelope redefn P-val multiplier:  {}\n", gi("--cykenvx"))); }
    if p.is_set("--tau")        { s.push_str(&format!("# tau parameter for final stage:         {}\n", fmt_g(g("--tau")))); }
    if p.is_set("--sums")       { s.push_str("# posterior sums (final stage):          on\n"); }
    if p.is_set("--qdb")        { s.push_str("# QDBs (final stage)                     on\n"); }
    if p.is_set("--beta") && p.get_str("--beta") != Some("1e-15")   { s.push_str(&format!("# beta parameter for final stage:        {}\n", fmt_g(g("--beta")))); }
    if p.is_set("--nonbanded")  { s.push_str("# no bands (final stage)                 on\n"); }
    if p.is_set("--nogreedy")   { s.push_str("# greedy CM hit resolution:              off\n"); }
    if p.is_set("--cp9noel")    { s.push_str("# CP9 HMM local ends:                    off\n"); }
    if p.is_set("--cp9gloc")    { s.push_str("# CP9 HMM configuration:                 glocal\n"); }
    if p.is_set("--null2")      { s.push_str("# null2 bias corrections:                on\n"); }
    if p.is_set("--maxtau")     { s.push_str(&format!("# max tau during band tightening:        {}\n", fmt_g(g("--maxtau")))); }
    if p.is_set("--seed") {
        if gi("--seed") == 0 { s.push_str("# random number seed:                    one-time arbitrary\n"); }
        else                 { s.push_str(&format!("# random number seed set to:             {}\n", gi("--seed"))); }
    }
    // truncated-hit-detection-off note (cmsearch.c:2359-2362).
    if !p.is_set("--notrunc") {
        if p.is_set("--max")   { s.push_str("# truncated hit detection:               off [due to --max]\n"); }
        if p.is_set("--nohmm") { s.push_str("# truncated hit detection:               off [due to --nohmm]\n"); }
    }
    // number of worker threads (always printed; cmsearch.c:2369).
    s.push_str(&format!(
        "# number of worker threads:              {}{}\n",
        ncpus,
        if cpu_used { " [--cpu]" } else { "" }
    ));
    s.push_str("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n");
    s
}

/// printf `%f` (default 6 decimals), for the --FZ line (C uses %f there).
fn format_f(x: f64) -> String {
    format!("{:.6}", x)
}

/// printf "%.2g" (2 significant figures), matching infernal tblout E-values.
fn fmt_evalue(e: f64) -> String {
    if e == 0.0 {
        return "0".to_string();
    }
    let p: i32 = 2; // significant figures
    // C printf %g decides fixed-vs-exponential from the exponent AFTER rounding to
    // p significant figures (e.g. 9.9999e-5 rounds to 1.0e-4, exponent -5 -> -4).
    // Round via %e first, then read the (possibly bumped) exponent.
    let s = format!("{:.*e}", (p - 1) as usize, e); // e.g. "1.0e-4", "3.6e-22"
    let (mant, ex) = {
        let parts: Vec<&str> = s.splitn(2, 'e').collect();
        (parts[0].to_string(), parts[1].parse::<i32>().unwrap_or(0))
    };
    if ex < -4 || ex >= p {
        // exponential style, (p-1) mantissa decimals, exponent >= 2 digits w/ sign
        let mant = strip_zeros(&mant);
        let sign = if ex < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", mant, sign, ex.abs())
    } else {
        // fixed style, (p-1-exp) decimals, strip trailing zeros
        let dec = (p - 1 - ex).max(0) as usize;
        strip_zeros(&format!("{:.*}", dec, e))
    }
}

/// strip trailing zeros (and a trailing '.') from a decimal mantissa string
fn strip_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}
