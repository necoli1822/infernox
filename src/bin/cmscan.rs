// SPDX-License-Identifier: BSD-3-Clause
//! infernox-cmscan — search sequence(s) against a covariance model database.
//!
//! Byte-parity port of C Infernal 1.1.5 `cmscan` (cmscan.c) `--tblout` (fmt 1)
//! output. cmscan runs the SAME per-(model, sequence) pipeline as cmsearch, so
//! this binary reuses [`infernox::cm_search::FaithfulSearcher`] — the exact
//! engine cmsearch uses — once per model, per query sequence.
//!
//! ## The Z (search-space) difference vs cmsearch
//! For E-values, cmscan uses `pli->Z = nmodels * qZ` where `qZ = 2*L` (both
//! strands) for each query sequence of length `L` (cmscan.c:497,592-598;
//! cm_pipeline.c:388). cmsearch uses `Z = total_db_residues`. Because the CM
//! exp-tail E-value is *linear* in Z (`E = (Z/dbsize)*nrandhits * exp(...)`,
//! UpdateExpsForDBSize / cm_pipeline.c:1074), and the DP scores/coordinates are
//! Z-independent, we run the searcher per single query sequence (its internal
//! `z = 2*L`) and multiply each hit's full-precision E-value by the scale factor
//! `Z_cmscan / (2*L)` before formatting. Default: scale = `nmodels`.
//!
//! The searcher's per-sequence reporting bar (E_searcher <= 10) is *lower* than
//! cmscan's (E_cmscan <= 10, i.e. E_searcher <= 10/scale) whenever scale >= 1
//! (always true by default, scale = nmodels), so the searcher returns a superset
//! and we re-threshold to cmscan's bar exactly. Caveat: the filter-tier
//! thresholds (cm_pipeline.c:497-560) key off Z_Mb too; for the common case
//! Z_cmscan and 2*L land in the same tier (identical filters). For extremely
//! large databases where nmodels*2*L crosses a tier boundary that 2*L does not,
//! the filters could differ — see report.

use infernox::cm_file::cm_file_read_binary;
use infernox::cm_search::{
    FaithfulConfig, FaithfulHit, FaithfulSearcher, ModelCutoff, PassAcctSnapshot, PliStats,
};
use infernox::search_cli::{ArgKind, OptSpec, Parsed};
use infernox::CM;
use std::io::{Cursor, Write};

/// The complete `cmscan` option table (name + arity), mirroring cmscan.c's
/// `ESL_OPTIONS options[]`. Every option is listed so value-taking options
/// consume their argument (e.g. `--fmt 2 db seq` no longer misparses) and unknown
/// flags are rejected instead of silently swallowed.
fn cmscan_opt_table() -> Vec<OptSpec> {
    use ArgKind::{None as N, Value as V};
    let mk = |name, kind| OptSpec { name, kind };
    vec![
        // docgroup 1
        mk("-h", N), mk("-g", N), mk("-Z", V), mk("--devhelp", N),
        // docgroup 2 (output) — cmscan has no -A / --nomiss
        mk("-o", V), mk("--tblout", V), mk("--fmt", V), mk("--acc", N), mk("--noali", N),
        mk("--notextw", N), mk("--textw", V), mk("--verbose", N),
        // docgroup 3 / 4 / 5
        mk("-E", V), mk("-T", V), mk("--incE", V), mk("--incT", V),
        mk("--cut_ga", N), mk("--cut_nc", N), mk("--cut_tc", N),
        // docgroup 6
        mk("--max", N), mk("--nohmm", N), mk("--mid", N), mk("--default", N), mk("--rfam", N),
        mk("--hmmonly", N), mk("--FZ", V), mk("--Fmid", V),
        // docgroup 7 — cmscan-specific: --qformat, --glist/--clanin/--oclan/--oskip real
        mk("--notrunc", N), mk("--anytrunc", N), mk("--nonull3", N), mk("--mxsize", V),
        mk("--smxsize", V), mk("--cyk", N), mk("--acyk", N), mk("--wcx", V), mk("--toponly", N),
        mk("--bottomonly", N), mk("--qformat", V), mk("--glist", V), mk("--clanin", V),
        mk("--oclan", N), mk("--oskip", N), mk("--cpu", V),
        // docgroup 101
        mk("--noF1", N), mk("--noF2", N), mk("--noF3", N), mk("--noF4", N), mk("--noF6", N),
        mk("--doF1b", N), mk("--noF2b", N), mk("--noF3b", N), mk("--noF4b", N), mk("--doF5b", N),
        mk("--F1", V), mk("--F1b", V), mk("--F2", V), mk("--F2b", V), mk("--F3", V), mk("--F3b", V),
        mk("--F4", V), mk("--F4b", V), mk("--F5", V), mk("--F5b", V), mk("--F6", V),
        // docgroup 102
        mk("--hmmmax", N), mk("--hmmF1", V), mk("--hmmF2", V), mk("--hmmF3", V), mk("--hmmnobias", N),
        mk("--hmmnonull2", N), mk("--nohmmonly", N),
        // docgroup 103
        mk("--rt1", V), mk("--rt2", V), mk("--rt3", V), mk("--ns", V),
        // docgroup 104
        mk("--ftau", V), mk("--fsums", N), mk("--fqdb", N), mk("--fbeta", V), mk("--fnonbanded", N),
        mk("--nocykenv", N), mk("--cykenvx", V),
        // docgroup 105
        mk("--tau", V), mk("--sums", N), mk("--qdb", N), mk("--beta", V), mk("--nonbanded", N),
        // docgroup 106 / 107
        mk("--trmF3", N), mk("--timeF1", N), mk("--timeF2", N), mk("--timeF3", N), mk("--timeF4", N),
        mk("--timeF5", N), mk("--timeF6", N),
        // docgroup 108
        mk("--nogreedy", N), mk("--cp9noel", N), mk("--cp9gloc", N), mk("--null2", N), mk("--maxtau", V),
        mk("--seed", V), mk("--block", V), mk("--onepass", N), mk("--olonepass", N), mk("--noiter", N),
        mk("--inttrunc", N), mk("--onlytrunc", N), mk("--5trunc", N), mk("--3trunc", N),
    ]
}


/// C `cmscan.c` process_commandline `-h`/`--devhelp` block (cmscan.c:2050-2098):
/// `cm_banner` + `esl_usage`, then each docgroup as `puts("\n<header>")` +
/// `esl_opt_DisplayHelp(stdout, go, group, 2, width)`. `--devhelp` (`do_dev`) adds
/// the hidden groups 101-108 and suppresses the `devmsg = "*"` suffixes on the
/// "acceleration heuristics"/"Other options" headers and the "*Use --devhelp..."
/// trailer. Text transcribed verbatim from C's DisplayHelp over the cmscan
/// ESL_OPTIONS table (groups 1,3,4,5,6,101-107 are identical to cmsearch; groups
/// 2,7,108 differ). Banner/Usage program name hardcoded "cmscan" to byte-match C.
fn full_help(do_dev: bool) -> ! {
    print!(
        "# cmscan :: search sequence(s) against a CM database\n\
# INFERNAL 1.1.5 (Sep 2023)\n\
# Copyright (C) 2023 Howard Hughes Medical Institute.\n\
# Freely distributed under the BSD open source license.\n\
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\
Usage: cmscan [-options] <cmdb> <seqfile>\n"
    );
    // group 1: Basic options
    print!(
        "\nBasic options:\n  \
-h        : show brief help on version and usage\n  \
-g        : configure CM for glocal alignment [default: local]\n  \
-Z <x>    : set search space size in *Mb* to <x> for E-value calculations  (x>0)\n  \
--devhelp : show list of otherwise hidden developer/expert options\n"
    );
    // group 2: Options directing output (cmscan: no -A / --nomiss)
    print!(
        "\nOptions directing output:\n  \
-o <f>       : direct output to file <f>, not stdout\n  \
--tblout <f> : save parseable table of hits to file <s>\n  \
--fmt <n>    : set hit table format to <n>  (1<=n<=3)\n  \
--acc        : prefer accessions over names in output\n  \
--noali      : don't output alignments, so output is smaller\n  \
--notextw    : unlimit ASCII text output line width\n  \
--textw <n>  : set max width of ASCII text output lines  [120]  (n>=120)\n  \
--verbose    : report extra information; mainly useful for debugging\n"
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
    // group 7: Other options (cmscan: --qformat + --glist/--clanin/--oclan/--oskip;
    // header carries devmsg "*" unless --devhelp)
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
--qformat <s> : assert query <seqfile> is in format <s>: no autodetection\n  \
--glist <f>   : configure CMs listed in file <f> in glocal mode, others in local\n  \
--clanin <f>  : read clan information from file <f>\n  \
--oclan       : w/'--fmt 2' and '--tblout', only mark overlaps within clans\n  \
--oskip       : w/'--fmt 2' and '--tblout', do not output lower scoring overlaps\n  \
--cpu <n>     : number of parallel CPU workers to use for multithreads  [4]\n",
        if do_dev { "" } else { "*" }
    );
    if do_dev {
        // group 108: Other expert options (cmscan: extra --block line)
        print!(
            "\nOther expert options:\n  \
--nogreedy   : do not resolve hits with greedy algorithm, use optimal one\n  \
--cp9noel    : turn off local ends in cp9 HMMs\n  \
--cp9gloc    : configure cp9 HMM in glocal mode\n  \
--null2      : turn on null 2 biased composition HMM score corrections\n  \
--maxtau <x> : set max tau <x> when tightening HMM bands  [0.05]\n  \
--seed <n>   : set RNG seed to <n> (if 0: one-time arbitrary seed)  [181]\n  \
--block <n>  : set block size (number of models per worker/thread) to <n>\n  \
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

// C `process_commandline()` value-validation errors (esl_getopts verify_type_and_range,
// e.g. "Option -E takes real-valued arg; got x on cmdline") all route through the
// same ERROR: block: `printf("Failed to parse command line: %s\n", errbuf)` +
// esl_usage + basic-options DisplayHelp + exit(1), all to STDOUT (cmscan.c:2223-2228).
fn usage_error(msg: &str) -> ! {
    infernox::search_cli::cmdline_fail(&format!("Failed to parse command line: {msg}"), CMSCAN_USAGE);
}

/// cmscan `esl_usage` line (cmscan.c:286) for the ERROR usage block. Note the
/// "[-options]" spelling and the <cmdb> <seqfile> positionals differ from cmsearch.
const CMSCAN_USAGE: &str = "cmscan [-options] <cmdb> <seqfile>";

/// cmscan `ESL_OPTIONS` require/incompat fields (fields 7/8), table order
/// (cmscan.c:126-...). Identical to cmsearch except: --glist/--clanin/--oclan/
/// --oskip are REAL cmscan options (--glist incompat -g; --clanin/--oskip require
/// --fmt; --oclan requires --fmt,--clanin).
const CMSCAN_CONSTRAINTS: &[infernox::search_cli::OptConstraint] = &[
    ("-g", None, Some("--hmmonly")),
    ("--fmt", Some("--tblout"), None),
    ("--notextw", None, Some("--textw")),
    ("--textw", None, Some("--notextw")),
    ("--nomiss", Some("-A"), None),
    ("--Fmid", Some("--mid"), None),
    ("--wcx", None, Some("--nohmm,--qdb,--fqdb")),
    ("--glist", None, Some("-g")),
    ("--clanin", Some("--fmt"), None),
    ("--oclan", Some("--fmt,--clanin"), None),
    ("--oskip", Some("--fmt"), None),
    ("--cpu", None, Some("--mpi")),
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
    ("--cp9noel", None, Some("-g")),
    ("--cp9gloc", None, Some("-g,--cp9noel")),
    ("--onepass", None, Some("--nohmm,--qdb,--fqdb")),
    ("--noiter", None, Some("--nohmm,--qdb,--fqdb")),
];

/// Accel-preset manual guards (cmscan.c:2174+): identical to cmsearch.
const CMSCAN_ACCEL: &[infernox::search_cli::AccelGuard] = &[
    ("--max", &["--nohmm","--mid","--rfam","--FZ","--noF1","--noF2","--noF3","--noF4","--noF6","--doF1b","--noF2b","--noF3b","--noF4b","--doF5b","--F1","--F1b","--F2","--F2b","--F3","--F3b","--F4","--F4b","--F5","--F6","--ftau","--fsums","--fqdb","--fbeta","--fnonbanded","--nocykenv","--cykenvx","--tau","--sums","--nonbanded","--rt1","--rt2","--rt3","--ns","--maxtau","--anytrunc","--inttrunc","--onlytrunc","--5trunc","--3trunc","--onepass","--olonepass","--noiter"]),
    ("--nohmm", &["--max","--mid","--rfam","--FZ","--noF1","--noF2","--noF3","--noF4","--doF1b","--noF2b","--noF3b","--noF4b","--doF5b","--F1","--F1b","--F2","--F2b","--F3","--F3b","--F4","--F4b","--F5","--ftau","--fsums","--tau","--sums","--rt1","--rt2","--rt3","--ns","--maxtau","--anytrunc","--inttrunc","--onlytrunc","--5trunc","--3trunc","--onepass","--olonepass","--noiter"]),
    ("--mid", &["--max","--nohmm","--rfam","--FZ","--noF1","--noF2","--noF3","--doF1b","--noF2b","--F1","--F1b","--F2","--F2b"]),
    ("--default", &["--max","--nohmm","--rfam","--FZ"]),
    ("--rfam", &["--max","--nohmm","--default","--FZ"]),
    ("--FZ", &["--max","--nohmm","--default","--rfam"]),
    ("--hmmonly", &["--max","--nohmm","--mid","--rfam","--FZ","--noF1","--noF2","--noF3","--noF4","--noF6","--doF1b","--noF2b","--noF3b","--noF4b","--doF5b","--F1","--F1b","--F2","--F2b","--F3","--F3b","--F4","--F4b","--F5","--F6","--ftau","--fsums","--fqdb","--fbeta","--fnonbanded","--nocykenv","--cykenvx","--tau","--sums","--qdb","--beta","--nonbanded","--maxtau","--anytrunc","--inttrunc","--onlytrunc","--5trunc","--3trunc","--onepass","--olonepass","--noiter","--mxsize","--smxsize","--nonull3","--nohmmonly","--timeF4","--timeF5","--timeF6","--nogreedy","--cp9noel","--cp9gloc","--null2"]),
];

/// Threshold manual guards (cmscan.c): identical to cmsearch.
const CMSCAN_THRESH: &[infernox::search_cli::ThreshGuard] = &[
    ("-E", "-T,--cut_ga,--cut_nc,--cut_tc", &["-T","--cut_ga","--cut_nc","--cut_tc"]),
    ("-T", "-E,--cut_ga,--cut_nc,--cut_tc", &["-E","--cut_ga","--cut_nc","--cut_tc"]),
    ("--incE", "--incT,--cut_ga,--cut_nc,--cut_tc", &["--incT","--cut_ga","--cut_nc","--cut_tc"]),
    ("--incT", "--incE,--cut_ga,--cut_nc,--cut_tc", &["--incE","--cut_ga","--cut_nc","--cut_tc"]),
    ("--cut_ga", "-E,-T,--incE,--incT,--cut_nc,--cut_tc", &["-E","-T","--incE","--incT","--cut_nc","--cut_tc"]),
    ("--cut_nc", "-E,-T,--incE,--incT,--cut_ga,--cut_tc", &["-E","-T","--incE","--incT","--cut_ga","--cut_tc"]),
    ("--cut_tc", "-E,-T,--incE,--incT,--cut_ga,--cut_nc", &["-E","-T","--incE","--incT","--cut_ga","--cut_nc"]),
];

/// Truncation-mode mutual exclusions (cmscan.c:2406-2440), identical to cmsearch.
/// Order matters (first hit wins); the --3trunc block reproduces C's literal
/// "Option --5trunc ..." message verbatim (a C copy-paste quirk).
const CMSCAN_TRUNC: &[infernox::search_cli::TruncGuard] = &[
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

/// Decode a `--qformat` option string to a forced [`infernox::easel::SqFormat`], or
/// [`infernox::easel::SqFormat::Unknown`] for autodetection when absent. C
/// `esl_sqio_EncodeFormat`: an unrecognized name is a fatal argument error.
fn decode_informat(opt: Option<&str>, optname: &str) -> infernox::easel::SqFormat {
    match opt {
        None => infernox::easel::SqFormat::Unknown,
        Some(s) => infernox::easel::esl_sqio_encode_format(s).unwrap_or_else(|| {
            eprintln!("error: {} '{}' is not a recognized sequence file format", optname, s);
            std::process::exit(1);
        }),
    }
}

/// Read every CM (and its p7 filter) from a pressed `.i1m` binary database.
fn read_pressed_db(dbarg: &str) -> Vec<CM> {
    let candidates = if dbarg.ends_with(".i1m") {
        vec![dbarg.to_string()]
    } else {
        vec![format!("{}.i1m", dbarg), dbarg.to_string()]
    };
    let mut path = None;
    for c in &candidates {
        if std::path::Path::new(c).exists() {
            path = Some(c.clone());
            break;
        }
    }
    let path = path.unwrap_or_else(|| {
        eprintln!(
            "error: no pressed CM database found for '{}' (expected '{}.i1m'). Run cmpress first.",
            dbarg, dbarg
        );
        std::process::exit(1);
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("error: cannot read '{}': {}", path, e);
        std::process::exit(1);
    });
    if bytes.len() < 4 || u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0xe3edb0b2
    {
        eprintln!(
            "error: '{}' is not a binary pressed CM database (bad magic). Run cmpress on the CM file.",
            path
        );
        std::process::exit(1);
    }
    let mut cur = Cursor::new(&bytes[..]);
    let total = bytes.len() as u64;
    let mut cms = Vec::new();
    while cur.position() < total {
        match cm_file_read_binary(&mut cur) {
            Ok(cm) => cms.push(cm),
            Err(e) => {
                eprintln!("error: failed to read CM #{} from '{}': {}", cms.len() + 1, path, e);
                std::process::exit(1);
            }
        }
    }
    if cms.is_empty() {
        eprintln!("error: no CMs found in '{}'", path);
        std::process::exit(1);
    }
    cms
}

/// One resolved cmscan hit ready for output (post E-value scaling + threshold).
struct ScanHit {
    model_idx: usize,
    model_name: String,
    model_acc: Option<String>,
    model_desc: Option<String>,
    model_clen: i32,
    hit: FaithfulHit,
    evalue: f64, // cmscan-scaled E-value (full precision)
    included: bool,
    /// Full source-sequence length (C `hit->srcL`); `seq len` column of fmt 2/3.
    src_l: i64,
    /// Stable per-hit id in build (pipeline creation) order (C `hit->hit_idx`);
    /// the join key between the overlap-markup pass and the fmt-2 output indices.
    hit_idx: usize,
    /// Clan index (C `hit->clan_idx`); -1 if the model is in no clan / no `--clanin`.
    clan_idx: i64,
    /// C `hit->any_oidx`: hit_idx of the best-scoring hit that overlaps this one
    /// (any overlap), or -1. Set by the overlap-markup pass.
    any_oidx: i64,
    /// C `hit->win_oidx`: hit_idx of the best-scoring *winning* (itself un-marked)
    /// overlap, or -1.
    win_oidx: i64,
    /// C `CM_HIT_IS_MARKED_OVERLAP`: this hit overlaps a better-scoring hit.
    marked_overlap: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Faithful esl_getopts-style parse: recognize every cmscan option (consuming
    // value-taking ones' arguments), require options before positionals, and error
    // on unknown flags. This fixes the misparse where e.g. `cmscan --fmt 2 db seq`
    // leaked "2" into the positional list.
    let table = cmscan_opt_table();
    // C esl_opt_ProcessCmdline routes any parse failure (unknown option, ambiguous
    // abbreviation) through the ERROR usage block with "Failed to parse command
    // line: <esl error>" and exit(1) — NOT a getopt-style exit(2).
    let parsed = infernox::search_cli::parse(&args, &table).unwrap_or_else(|e| {
        infernox::search_cli::cmdline_fail(&format!("Failed to parse command line: {e}"), CMSCAN_USAGE)
    });

    // C process_commandline (cmscan.c:2039/2043): esl_opt_VerifyConfig then the
    // manual guards, both routing to the ERROR usage block + exit(1).
    if let Err(e) = infernox::search_cli::verify_config(&parsed, CMSCAN_CONSTRAINTS) {
        infernox::search_cli::cmdline_fail(&format!("Failed to parse command line: {e}"), CMSCAN_USAGE);
    }

    // C process_commandline (cmscan.c:2050-2098): after ProcessCmdline +
    // VerifyConfig, the -h/--devhelp help block prints and exit(0) — BEFORE the
    // arg-count check and manual guards. do_dev (=--devhelp) wins when both given.
    let do_dev = parsed.is_set("--devhelp");
    if parsed.is_set("-h") || do_dev {
        full_help(do_dev);
    }

    if let Err(line) = infernox::search_cli::manual_guards(&parsed, CMSCAN_ACCEL, CMSCAN_THRESH, CMSCAN_TRUNC) {
        infernox::search_cli::cmdline_fail(&line, CMSCAN_USAGE);
    }

    let tblout: Option<String> = parsed.get_str("--tblout").map(|s| s.to_string());
    // C `--fmt <n>` (1<=n<=3, default 1). fmt 2 adds overlap+clan columns; fmt 3 adds
    // mdl-len/seq-len. --oclan (mark only within-clan overlaps) and --oskip (omit
    // lower-scoring overlaps from the table) modify fmt 2; both require --fmt.
    let fmt: i64 = parsed.get_i64("--fmt").unwrap_or_else(|e| usage_error(&e)).unwrap_or(1);
    let oclan = parsed.is_set("--oclan");
    let oskip = parsed.is_set("--oskip");
    // C cmscan.c:2142-2158 (after VerifyConfig/manual guards): --clanin, --oclan and
    // --oskip each require --fmt with the value exactly 2 (esl_getopts enforces --fmt is
    // *present* but not its value). Checked in this order so the message matches C.
    for (opt, present) in [
        ("--clanin", parsed.is_set("--clanin")),
        ("--oclan", oclan),
        ("--oskip", oskip),
    ] {
        if present && (!parsed.is_set("--fmt") || fmt != 2) {
            infernox::search_cli::cmdline_fail(
                &format!(
                    "Failed to parse command line: with {opt}, the additional option of --fmt <n> is required with <n> == 2"
                ),
                CMSCAN_USAGE,
            );
        }
    }
    // C `--clanin <f>`: read clan membership. clan_names[c] = clan name, clan_of_model
    // maps a model name to its clan index (C determine_clan_index / clan_mapA).
    let (clan_names, clan_of_model): (Vec<String>, std::collections::HashMap<String, i64>) =
        match parsed.get_str("--clanin") {
            Some(path) => read_clan_info(path).unwrap_or_else(|e| {
                eprintln!("\nError: {e}\n");
                std::process::exit(1);
            }),
            None => (Vec::new(), std::collections::HashMap::new()),
        };
    let ofile: Option<String> = parsed.get_str("-o").map(|s| s.to_string());
    let toponly = parsed.is_set("--toponly");
    let bottomonly = parsed.is_set("--bottomonly");
    let global = parsed.is_set("-g");
    let ncpu: Option<usize> = parsed.get_usize("--cpu").unwrap_or_else(|e| usage_error(&e));
    // C `-Z <x>`: manual search-space size in Mb (E-value denominator for cmscan).
    let z_mb: Option<f64> = parsed.get_f64("-Z").unwrap_or_else(|e| usage_error(&e));
    let t_cutoff: Option<f32> = parsed.get_f32("-T").unwrap_or_else(|e| usage_error(&e));
    let e_report_opt: Option<f64> = parsed.get_f64("-E").unwrap_or_else(|e| usage_error(&e));
    let inc_e: f64 = parsed.get_f64("--incE").unwrap_or_else(|e| usage_error(&e)).unwrap_or(0.01);
    let inc_t: Option<f32> = parsed.get_f32("--incT").unwrap_or_else(|e| usage_error(&e));
    let model_cutoff: Option<ModelCutoff> = if parsed.is_set("--cut_ga") {
        Some(ModelCutoff::Ga)
    } else if parsed.is_set("--cut_tc") {
        Some(ModelCutoff::Tc)
    } else if parsed.is_set("--cut_nc") {
        Some(ModelCutoff::Nc)
    } else {
        None
    };
    // C `cmscan --qformat <s>`: assert the query seq file is in format <s>.
    let qformat: Option<String> = parsed.get_str("--qformat").map(|s| s.to_string());
    let positionals = parsed.positionals.clone();
    // C `esl_opt_ArgNumber(go) != 2` → `puts("Incorrect number of command line
    // arguments.")` (no "Failed to parse..." prefix) + usage block + exit(1)
    // (cmscan.c:2231-2235).
    if positionals.len() != 2 {
        infernox::search_cli::cmdline_fail("Incorrect number of command line arguments.", CMSCAN_USAGE);
    }
    let dbarg = &positionals[0];
    let seqpath = &positionals[1];

    if let Some(n) = ncpu {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
    }

    // ---- Read pressed database + build one searcher per model (in DB order) ----
    let cms = read_pressed_db(dbarg);
    let nmodels = cms.len();
    // C cmsearch.c:2648 / cm_Configure(W_from_cmdline): --wcx overrides cm->W =
    // (int)(cm->clen * wcx) before the pipeline builds filters/maxW. Extract it here
    // since it must be applied before FaithfulSearcher::new (cfg is built below).
    let wcx_opt = parsed.get_f64("--wcx").ok().flatten();
    // C --beta <x> (cmsearch.c:2644): override final-round QDB2 tail-loss prob before
    // FaithfulSearcher::new recomputes dmin2/dmax2.
    let beta_opt = parsed.get_f64("--beta").ok().flatten();
    let searchers: Vec<FaithfulSearcher> = cms
        .into_iter()
        .map(|mut cm| {
            if let Some(x) = wcx_opt {
                cm.w = (cm.clen as f64 * x) as i32;
            }
            if let Some(b) = beta_opt {
                cm.qdb_beta2 = b;
            }
            FaithfulSearcher::new(cm).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            })
        })
        .collect();
    let model_meta: Vec<(String, Option<String>, Option<String>)> = searchers
        .iter()
        .map(|s| {
            let cm = s.cm();
            (cm.name.clone(), cm.acc.clone(), cm.desc.clone())
        })
        .collect();

    // Decode --qformat (if given) to a forced format; else autodetect.
    // C cmscan: `esl_sqio_EncodeFormat(esl_opt_GetString(go, "--qformat"))`.
    let informat = decode_informat(qformat.as_deref(), "--qformat");
    let recs = infernox::easel::read_seqfile(seqpath, informat).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let e_report: f64 = e_report_opt.unwrap_or(10.0);
    let use_bit_cutoffs = model_cutoff.is_some();
    let cfg = FaithfulConfig {
        toponly,
        bottomonly,
        e_report,
        global,
        // C accel presets (cmscan.c:2174+, same as cmsearch): --nohmm turns off the
        // HMM filters (CM-only), --max turns off all heuristic filters, --mid turns off
        // MSV+Viterbi. These MUST flow into the pipeline config or cmscan silently runs
        // the default HMM pipeline for --nohmm/--max/--mid.
        nohmm: parsed.is_set("--nohmm"),
        max: parsed.is_set("--max"),
        mid: parsed.is_set("--mid"),
        rfam: parsed.is_set("--rfam"),
        cyk: parsed.is_set("--cyk"),
        t_cutoff,
        model_cutoff,
        // C `--notrunc`: disable truncated (5P/3P/53) passes (cmscan default: ON).
        notrunc: parsed.is_set("--notrunc"),
        // C `--5trunc`/`--3trunc`: restrict to the 5' (resp 3') terminal force pass.
        trunc5p: parsed.is_set("--5trunc"),
        trunc3p: parsed.is_set("--3trunc"),
        // C `--anytrunc`/`--inttrunc`/`--onlytrunc`: add the internal 5P_AND_3P_ANY pass.
        anytrunc: parsed.is_set("--anytrunc"),
        inttrunc: parsed.is_set("--inttrunc"),
        onlytrunc: parsed.is_set("--onlytrunc"),
        qdb: parsed.is_set("--qdb"),
        nonbanded: parsed.is_set("--nonbanded"),
        wcx: parsed.get_f64("--wcx").ok().flatten(),
        beta: parsed.get_f64("--beta").ok().flatten(),
        // cmscan derives Z per query (see below); the manual -Z applies there.
        z_mb_override: None,
        // C per-stage filter P-value overrides (--F1/--F3/--F3b/--F4/--F4b/--F5).
        f1: parsed.get_f64("--F1").unwrap_or_else(|e| usage_error(&e)),
        f3: parsed.get_f64("--F3").unwrap_or_else(|e| usage_error(&e)),
        f3b: parsed.get_f64("--F3b").unwrap_or_else(|e| usage_error(&e)),
        f4: parsed.get_f64("--F4").unwrap_or_else(|e| usage_error(&e)),
        f4b: parsed.get_f64("--F4b").unwrap_or_else(|e| usage_error(&e)),
        f5: parsed.get_f64("--F5").unwrap_or_else(|e| usage_error(&e)),
        // C expert per-stage on/off + Viterbi/MSV-bias/env-bias threshold overrides.
        f2: parsed.get_f64("--F2").unwrap_or_else(|e| usage_error(&e)),
        f2b: parsed.get_f64("--F2b").unwrap_or_else(|e| usage_error(&e)),
        f1b: parsed.get_f64("--F1b").unwrap_or_else(|e| usage_error(&e)),
        f5b: parsed.get_f64("--F5b").unwrap_or_else(|e| usage_error(&e)),
        no_f1: parsed.is_set("--noF1"),
        no_f2: parsed.is_set("--noF2"),
        no_f3: parsed.is_set("--noF3"),
        no_f4: parsed.is_set("--noF4"),
        no_f2b: parsed.is_set("--noF2b"),
        no_f3b: parsed.is_set("--noF3b"),
        no_f4b: parsed.is_set("--noF4b"),
        do_f1b: parsed.is_set("--doF1b"),
        do_f5b: parsed.is_set("--doF5b"),
        // C `--F6`/`--cykenvx`/`--noF6`/`--nocykenv` (CYK filter stage controls).
        f6: parsed.get_f64("--F6").unwrap_or_else(|e| usage_error(&e)),
        cykenvx: parsed.get_i64("--cykenvx").unwrap_or_else(|e| usage_error(&e)),
        no_f6: parsed.is_set("--noF6"),
        nocykenv: parsed.is_set("--nocykenv"),
        // C `--tau`/`--ftau`/`--maxtau`: HMM-band tail-loss probs.
        tau: parsed.get_f64("--tau").unwrap_or_else(|e| usage_error(&e)),
        ftau: parsed.get_f64("--ftau").unwrap_or_else(|e| usage_error(&e)),
        maxtau: parsed.get_f64("--maxtau").unwrap_or_else(|e| usage_error(&e)),
        // Filter-tier Z (Mb). C selects the F1/F2/F3 tier from pli->Z; when `-Z <x>` is
        // given, cmscan sets pli->Z = x*1e6 (fixed, not the per-query nmodels*2*L), so the
        // tier must come from -Z too. --FZ (explicit filter Z) takes priority; otherwise
        // fall back to -Z. Without either, the searcher uses the per-query size.
        fz: parsed
            .get_f64("--FZ")
            .unwrap_or_else(|e| usage_error(&e))
            .or(z_mb),
        // C `--rt1/--rt2/--rt3/--ns`: glocal domain/envelope-definition params.
        rt1: parsed.get_f64("--rt1").unwrap_or_else(|e| usage_error(&e)),
        rt2: parsed.get_f64("--rt2").unwrap_or_else(|e| usage_error(&e)),
        rt3: parsed.get_f64("--rt3").unwrap_or_else(|e| usage_error(&e)),
        ns: parsed.get_i64("--ns").unwrap_or_else(|e| usage_error(&e)),
        // C `--hmmonly` family (HMM-only pipeline; cm_pipeline.c:613-631, 1028-1030).
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

    // Human-output knobs (mirror cmsearch): default stdout is human-readable.
    let show_alignments = !parsed.is_set("--noali");
    let show_accessions = parsed.is_set("--acc");
    let textw: i32 = if parsed.is_set("--notextw") {
        0
    } else {
        parsed.get_i64("--textw").unwrap_or(None).map(|v| v as i32).unwrap_or(120)
    };
    // esl_opt_IsUsed("--cpu") is TRUE only if given AND value != default (CMNCPU="4",
    // or $INFERNAL_NCPU). So `--cpu 4` must NOT print the "[--cpu]" suffix.
    let cpu_default: i64 = std::env::var("INFERNAL_NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let cpu_used = matches!(parsed.get_i64("--cpu").unwrap_or(None), Some(v) if v != cpu_default);
    let ncpu_disp: i64 = ncpu.map(|n| n as i64).unwrap_or_else(num_worker_threads);
    let report_by_score = t_cutoff.is_some();

    let mut tbl = String::new();
    let mut header_emitted = false;
    // C output_header is printed ONCE at the top of the human stream.
    let mut human = output_header_scan(&parsed, seqpath, dbarg, ncpu_disp, cpu_used);

    for rec in recs.iter() {
        let seqname = &rec.0;
        let seqacc = ""; // FASTA records carry no accession here
        let seq = rec.2.as_str();
        let l = seq.len() as f64;
        // C: Z per strand-count — single strand for --toponly OR --bottomonly.
        let one_strand = toponly || bottomonly;
        let q_z = if one_strand { l } else { 2.0 * l };
        let z_searcher = if one_strand { l } else { 2.0 * l };
        let z_cmscan = match z_mb {
            Some(mb) => (mb as i64 as f64) * 1_000_000.0, // C: (int64_t)Real, then *1e6
            None => (nmodels as f64) * q_z,
        };
        let scale = if z_searcher > 0.0 { z_cmscan / z_searcher } else { 1.0 };

        // Run every model against this single sequence, aggregating the per-model
        // CM_PLI_ACCT counters (cmscan's statistics summary is over ALL models of the
        // query) and collecting the reported hits.
        let mut scan_hits: Vec<ScanHit> = Vec::new();
        let mut agg = PassAcctSnapshot::default();
        let mut std_nres_total = 0u64;
        let mut n_out_trunc_total = 0u64;
        let mut pos_out_trunc_total = 0u64;
        let mut nnodes_total = 0i64;
        // HMM-only pipeline accounting, aggregated across the DB models that ran
        // HMM-only for this query (C cm_pipeline.c:1771 pli->nmodels_hmmonly path).
        // Each model contributes one per-model HmmonlyPassStats; we sum the counters
        // and re-orient to SCAN mode below.
        let mut hmmonly_agg: Option<infernox::p7_hmmonly::HmmonlyPassStats> = None;
        let mut pli_template: Option<PliStats> = None;
        // C cm_pli_NewModel (cm_pipeline.c:1117): --cut_ga/--cut_tc/--cut_nc require
        // every scanned model to carry the CMH_GA/TC/NC flag. The pipeline fatals on
        // the FIRST model lacking it — AFTER the header + "Query:" block are printed.
        // cmscan routes it through cm_Fail("cm_Pipeline() failed unexpected with status
        // code %d\n%s", status, errbuf) with status 11 (eslEINCOMPAT), errbuf being the
        // "X bit threshold unavailable for model <name>" line (cmscan.c:910/1192).
        // C cm_pli_NewModel (cm_pipeline.c:1073): for every CM-path (non-HMM-only) model,
        // the pipeline calls UpdateExpsForDBSize(), which ESL_FAILs with eslEINCOMPAT (10)
        // when the CM lacks exponential-tail stats (uncalibrated). cmscan routes it through
        // cm_Fail("cm_Pipeline() failed unexpected with status code %d\n%s", 10, errbuf).
        // This precedes the --cut_ga/tc/nc check (cm_pipeline.c:1117), and fires on the
        // FIRST such model AFTER the header + "Query:" block are printed. HMM-only models
        // (incl. 0-basepair models and --hmmonly) never call UpdateExpsForDBSize, so an
        // uncalibrated model searched with --hmmonly runs fine (exit 0), matching cmsearch.
        if let Some(mi) = searchers.iter().position(|s| {
            s.hmmonly_pass_stats(&[seq], &cfg).is_none()
                && (s.cm().flags & infernox::cm::CM_EXPTAIL_STATS) == 0
        }) {
            let _ = mi;
            human.push_str(&format!("Query:       {}  [L={}]\n", seqname, seq.len()));
            if !seqacc.is_empty() {
                human.push_str(&format!("Accession:   {}\n", seqacc));
            }
            if !rec.1.is_empty() {
                human.push_str(&format!("Description: {}\n", rec.1));
            }
            match &ofile {
                Some(path) => { let _ = std::fs::write(path, &human); }
                None => { print!("{}", human); let _ = std::io::stdout().flush(); }
            }
            // The ESL_FAIL message does NOT end in "\n"; cm_Fail prepends "\nError: " and
            // appends a single "\n" (so no trailing blank line, unlike the --cut_ga case
            // whose errbuf ends in "\n").
            eprint!(
                "\nError: cm_Pipeline() failed unexpected with status code 10\nUpdateExpsForDBSize(), cm does not have Exp stats\nYou may need to run cmcalibrate.\n"
            );
            std::process::exit(1);
        }
        if let Some(mc) = model_cutoff {
            if let Some(mi) = searchers.iter().position(|s| s.model_cutoff(mc).is_none()) {
                let letter = match mc {
                    ModelCutoff::Ga => "GA",
                    ModelCutoff::Tc => "TC",
                    ModelCutoff::Nc => "NC",
                };
                let mname = &model_meta[mi].0;
                human.push_str(&format!("Query:       {}  [L={}]\n", seqname, seq.len()));
                if !seqacc.is_empty() {
                    human.push_str(&format!("Accession:   {}\n", seqacc));
                }
                if !rec.1.is_empty() {
                    human.push_str(&format!("Description: {}\n", rec.1));
                }
                match &ofile {
                    Some(path) => { let _ = std::fs::write(path, &human); }
                    None => { print!("{}", human); let _ = std::io::stdout().flush(); }
                }
                // errbuf itself ends in "\n" (cm_pipeline.c:1129 sprintf), and cm_Fail
                // appends a final "\n" — hence the trailing blank line.
                eprint!(
                    "\nError: cm_Pipeline() failed unexpected with status code 11\n{} bit threshold unavailable for model {}\n\n",
                    letter, mname
                );
                std::process::exit(1);
            }
        }
        for (mi, searcher) in searchers.iter().enumerate() {
            let reported = searcher.search(&[seq], &cfg);
            // Accounting: sum this model's CM_SUMMED counters into the query totals.
            add_snapshot(&mut agg, &searcher.acct_summed());
            let std = searcher.acct_snapshot(1); // PLI_PASS_STD_ANY
            std_nres_total += std.nres_top + std.nres_bot;
            // Truncated-pass output counts (passes 2/3/4) are recomputed from the
            // re-thresholded reported hits below, not from the searcher's raw counters.
            nnodes_total += searcher.cm().clen as i64;
            if pli_template.is_none() {
                pli_template = Some(searcher.pli_stats(&[seq], &cfg));
            }
            // If this model ran HMM-only, fold its per-model HMM-only accounting into
            // the query aggregate (C: pli->nmodels_hmmonly++, plus counter sums).
            if let Some(hs) = searcher.hmmonly_pass_stats(&[seq], &cfg) {
                match hmmonly_agg.as_mut() {
                    None => hmmonly_agg = Some(hs),
                    Some(acc) => {
                        acc.nmodels_hmmonly += hs.nmodels_hmmonly;
                        acc.nnodes_hmmonly += hs.nnodes_hmmonly;
                        acc.nres_searched += hs.nres_searched;
                        acc.n_past_msv += hs.n_past_msv;
                        acc.pos_past_msv += hs.pos_past_msv;
                        acc.n_past_msvbias += hs.n_past_msvbias;
                        acc.pos_past_msvbias += hs.pos_past_msvbias;
                        acc.n_past_vit += hs.n_past_vit;
                        acc.pos_past_vit += hs.pos_past_vit;
                        acc.n_past_fwd += hs.n_past_fwd;
                        acc.pos_past_fwd += hs.pos_past_fwd;
                        acc.n_output += hs.n_output;
                        acc.pos_output += hs.pos_output;
                    }
                }
            }
            // Per-model bit-score inclusion cutoff (C --cut_ga/tc/nc): resolves to the
            // model's GA/TC/NC score, else None (fall back to --incT/--incE).
            let inc_cutoff = model_cutoff.and_then(|mc| searcher.model_cutoff(mc));
            for h in reported {
                let ev = h.evalue * scale; // E ∝ Z (full precision, scaled before %.2g)
                let report = if use_bit_cutoffs || t_cutoff.is_some() { true } else { ev <= e_report };
                let include = match inc_cutoff {
                    Some(cut) => h.score >= cut,
                    None => {
                        if report_by_score {
                            h.score >= inc_t.unwrap_or(0.0)
                        } else {
                            ev <= inc_e
                        }
                    }
                };
                if report {
                    let (mn, ma, md) = &model_meta[mi];
                    // C hit->clan_idx = determine_clan_index(omA[cm_idx]->name)
                    // (cmscan.c:833/995): clan membership is keyed off the pressed p7
                    // FILTER profile's name, NOT the CM display name (hit->name). These
                    // are identical for any cmbuild/cmpress'd model, but can differ in a
                    // hand-edited DB; matching C requires the p7 name here.
                    let clan_key = searcher
                        .cm()
                        .p7
                        .as_ref()
                        .map(|p| p.name.as_str())
                        .unwrap_or(mn.as_str());
                    let clan_idx = clan_of_model.get(clan_key).copied().unwrap_or(-1);
                    scan_hits.push(ScanHit {
                        model_idx: mi,
                        model_name: mn.clone(),
                        model_acc: ma.clone(),
                        model_desc: md.clone(),
                        model_clen: searcher.cm().clen,
                        evalue: ev,
                        included: include,
                        src_l: l as i64,
                        hit_idx: 0, // assigned below in build order, before markup
                        clan_idx,
                        any_oidx: -1,
                        win_oidx: -1,
                        marked_overlap: false,
                        hit: h,
                    });
                }
            }
        }

        // C hit->hit_idx (CreateNextHit): stable id in pipeline creation order. Rust
        // builds scan_hits in model order per sequence, matching C's per-sequence th
        // creation order. Assign here, BEFORE any markup re-sort, so any_oidx/win_oidx
        // reference stable ids. (C's tophits-level same-model overlap REMOVAL is already
        // performed per-model by the searcher's greedy dedup, so it is a no-op here.)
        for (i, sh) in scan_hits.iter_mut().enumerate() {
            sh.hit_idx = i;
        }
        // C cmscan.c:676-677: cm_tophits_SortForOverlapMarkup + RemoveOrMarkOverlaps
        // (do_remove=FALSE) marks cross-model overlaps (any_oidx/win_oidx/marked) for
        // fmt-2 output. --oclan restricts markup to within-clan overlaps.
        mark_overlaps(&mut scan_hits, oclan);

        // cmscan SortByEvalue: evalue asc, score desc, seq_idx asc (equal here),
        // start asc, pass_idx desc; final model_idx tiebreak for determinism.
        scan_hits.sort_by(|a, b| {
            a.evalue
                .partial_cmp(&b.evalue)
                .unwrap()
                .then(b.hit.score.partial_cmp(&a.hit.score).unwrap())
                .then(a.hit.start.cmp(&b.hit.start))
                .then(b.hit.pass_idx.cmp(&a.hit.pass_idx))
                .then(a.model_idx.cmp(&b.model_idx))
        });

        let show_header = !header_emitted;
        match fmt {
            2 => append_tblout2(&mut tbl, seqname, seqacc, &scan_hits, show_header, &clan_names, oskip),
            3 => append_tblout3(&mut tbl, seqname, seqacc, &scan_hits, show_header),
            _ => append_tblout1(&mut tbl, seqname, seqacc, &scan_hits, show_header),
        }
        header_emitted = true;

        // "Total CM hits reported" counters. The searcher accumulates n_output/pos_output
        // at its own z=2*L threshold, but cmscan re-thresholds each hit's E-value by the
        // per-query `scale` (pli->Z = nmodels*2*L). C's pipeline counts at the cmscan Z
        // natively, so we must recompute the CM output counters from the actually-reported
        // (re-thresholded) hits — the searcher's raw counts over-report under loose filters
        // (--max/--nohmm). Split STD (pass 1) vs truncated (passes 2/3/4 = 5P/3P/53) to
        // match C's "includes N truncated hit(s)". HMM-only hits are counted separately.
        let mut cm_n_output = 0u64;
        let mut cm_pos_std = 0u64;
        n_out_trunc_total = 0;
        pos_out_trunc_total = 0;
        for h in &scan_hits {
            if h.hit.hmmonly {
                continue;
            }
            let len = (h.hit.stop - h.hit.start).unsigned_abs() + 1;
            cm_n_output += 1;
            // Passes 2/3/4 (terminal FORCE) and 5 (internal ANY, --anytrunc/--inttrunc/
            // --onlytrunc) are truncated hits; pass 1 (STD) is not.
            if (2..=5).contains(&h.hit.pass_idx) {
                n_out_trunc_total += 1;
                pos_out_trunc_total += len;
            } else {
                cm_pos_std += len;
            }
        }
        agg.n_output = cm_n_output;
        // C CM_SUMMED.pos_output = Σ over ALL passes = std + truncated (pli_sum_statistics).
        // The "Total CM hits" ratio numerator is CM_SUMMED.pos_output + pos_output_trunc,
        // i.e. std + 2×truncated — so the summed snapshot must include truncated residues
        // (else the ratio halves for any truncated hit; a pre-existing bug the trunc modes
        // and default-mode truncated hits both expose).
        agg.pos_output = cm_pos_std + pos_out_trunc_total;

        // Statistics: cmscan orientation. Split the DB models into those that ran the
        // CM pipeline (nmodels) and those that ran HMM-only (nmodels_hmmonly); C keeps
        // these as two separate counters (cm_pipeline.c:1771/1774) and prints each block
        // only when its counter is > 0.
        let nmodels_hmmonly = hmmonly_agg.as_ref().map(|a| a.nmodels_hmmonly).unwrap_or(0);
        let nmodels_cm = nmodels as i64 - nmodels_hmmonly;
        let mut pli = pli_template.unwrap_or_else(|| searchers[0].pli_stats(&[seq], &cfg));
        pli.nmodels = nmodels_cm;
        pli.nseqs = 1;
        pli.nnodes = nnodes_total;
        // Re-orient the HMM-only aggregate to SCAN mode: query = the sequence, target =
        // the models; match_cm_spacing keys off pli->nmodels (the CM-pipeline count).
        if let Some(acc) = hmmonly_agg.as_mut() {
            acc.search_mode = false;
            acc.nseqs = 1;
            acc.nmodels = nmodels_cm;
        }

        human.push_str(&scan_human_block(
            seqname,
            seqacc,
            rec.1.as_str(),
            seq.len(),
            &scan_hits,
            show_alignments,
            show_accessions,
            textw,
            &pli,
            &agg,
            std_nres_total,
            n_out_trunc_total,
            pos_out_trunc_total,
            hmmonly_agg.as_ref(),
        ));
    }
    // C cmscan.c:752 prints "[ok]" once at the very end.
    human.push_str("[ok]\n");

    // C cmscan.c:751 writes the tabular footer via cm_tophits_TabularTail after the rows.
    // For cmscan (CM_SCAN_MODELS): Query file = seqfile, Target file = cm db (note the swap vs cmsearch).
    tbl.push_str(&infernox::cm_tophits::tabular_tail("cmscan", "SCAN", seqpath, dbarg, &args));

    if let Some(path) = &tblout {
        std::fs::write(path, &tbl).expect("write tblout");
    }

    match &ofile {
        Some(path) => {
            std::fs::write(path, &human).expect("write -o output");
        }
        None => {
            print!("{}", human);
            let _ = std::io::stdout().flush();
        }
    }
}

/// Sum one pass snapshot's counters into an accumulator (query-level totals across
/// all models). Mirrors the integer-sum, order-independent CM_PLI_ACCT semantics.
fn add_snapshot(acc: &mut PassAcctSnapshot, s: &PassAcctSnapshot) {
    acc.npli_top += s.npli_top;
    acc.npli_bot += s.npli_bot;
    acc.nres_top += s.nres_top;
    acc.nres_bot += s.nres_bot;
    acc.n_past_msv += s.n_past_msv;
    acc.pos_past_msv += s.pos_past_msv;
    acc.n_past_msvbias += s.n_past_msvbias;
    acc.pos_past_msvbias += s.pos_past_msvbias;
    acc.n_past_vit += s.n_past_vit;
    acc.pos_past_vit += s.pos_past_vit;
    acc.n_past_vitbias += s.n_past_vitbias;
    acc.pos_past_vitbias += s.pos_past_vitbias;
    acc.n_past_fwd += s.n_past_fwd;
    acc.pos_past_fwd += s.pos_past_fwd;
    acc.n_past_fwdbias += s.n_past_fwdbias;
    acc.pos_past_fwdbias += s.pos_past_fwdbias;
    acc.n_past_gfwd += s.n_past_gfwd;
    acc.pos_past_gfwd += s.pos_past_gfwd;
    acc.n_past_gfwdbias += s.n_past_gfwdbias;
    acc.pos_past_gfwdbias += s.pos_past_gfwdbias;
    acc.n_past_edef += s.n_past_edef;
    acc.pos_past_edef += s.pos_past_edef;
    acc.n_past_edefbias += s.n_past_edefbias;
    acc.pos_past_edefbias += s.pos_past_edefbias;
    acc.n_past_cyk += s.n_past_cyk;
    acc.pos_past_cyk += s.pos_past_cyk;
    acc.n_output += s.n_output;
    acc.pos_output += s.pos_output;
}

fn num_worker_threads() -> i64 {
    std::thread::available_parallelism().map(|n| n.get() as i64).unwrap_or(0)
}

/// Append cmscan `--fmt 1` tabular rows for one query sequence. Byte-faithful to
/// cm_tophits_TabularTargets1 (cm_tophits.c): target = MODEL, query = SEQUENCE.
fn append_tblout1(s: &mut String, qname: &str, qacc: &str, hits: &[ScanHit], show_header: bool) {
    let tnamew = hits.iter().map(|h| h.model_name.len()).max().unwrap_or(0).max(20);
    let taccw = hits
        .iter()
        .map(|h| h.model_acc.as_deref().map(|a| a.len()).unwrap_or(0))
        .max()
        .unwrap_or(0)
        .max(9);
    let qnamew = qname.len().max(20);
    let qaccw = qacc.len().max(9);
    let posw = hits
        .iter()
        .map(|h| h.hit.start.abs().max(h.hit.stop.abs()).to_string().len())
        .max()
        .unwrap_or(0)
        .max(8);

    if show_header {
        s.push_str(&format!(
            "#{:<w1$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:>3} {:>8} {:>8} {:>posw$} {:>posw$} {:>6} {:>5} {:>4} {:>4} {:>5} {:>6} {:>9} {:>3} {}\n",
            "target name", "accession", "query name", "accession", "mdl", "mdl from", "mdl to",
            "seq from", "seq to", "strand", "trunc", "pass", "gc", "bias", "score", "E-value", "inc",
            "description of target", w1 = tnamew - 1
        ));
        let dash = |n: usize| "-".repeat(n);
        s.push_str(&format!(
            "#{:<w1$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:<3} {:<8} {:<8} {:<posw$} {:<posw$} {:<6} {:<5} {:<4} {:<4} {:<5} {:<6} {:<9} {:<3} {}\n",
            dash(tnamew - 1), dash(taccw), dash(qnamew), dash(qaccw), "---", dash(8), dash(8),
            dash(posw), dash(posw), "------", "-----", "----", "----", "-----", "------", "---------",
            "---", "---------------------", w1 = tnamew - 1
        ));
    }

    for h in hits {
        let tname = h.model_name.as_str();
        let tacc = match &h.model_acc {
            Some(a) if !a.is_empty() => a.as_str(),
            _ => "-",
        };
        let tdesc = match &h.model_desc {
            Some(d) if !d.is_empty() => d.as_str(),
            _ => "-",
        };
        let strand = if h.hit.in_rc { "-" } else { "+" };
        let inc = if h.included { "!" } else { "?" };
        let eval_s = fmt_evalue(h.evalue);
        s.push_str(&format!(
            "{:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:>3} {:>8} {:>8} {:>posw$} {:>posw$} {:>6} {:>5} {:>4} {:>4.2} {:>5.1} {:>6.1} {:>9} {:<3} {}\n",
            tname, tacc, qname, if qacc.is_empty() { "-" } else { qacc },
            // C cm_tophits.c: the "mdl" column is "hmm" for HMM-only hits, else "cm".
            if h.hit.hmmonly { "hmm" } else { "cm" }, h.hit.mdl_from, h.hit.mdl_to, h.hit.start, h.hit.stop, strand,
            h.hit.trunc.as_str(), h.hit.pass_idx, h.hit.gc, h.hit.bias, h.hit.score, eval_s, inc, tdesc
        ));
    }
}

/// C `integer_textwidth` (cm_tophits.c:580): number of chars to print `n` in base 10
/// (0 -> 0; negatives add 1 for the sign).
fn integer_textwidth(mut n: i64) -> usize {
    let mut w = if n < 0 { 1 } else { 0 };
    while n != 0 {
        n /= 10;
        w += 1;
    }
    w
}

/// C `cm_tophits_OverlapNres` (cm_tophits.c:1363): number of residues shared by the
/// (ordered) ranges [from1..to1] and [from2..to2]. Callers pass each range low..high.
fn overlap_nres(mut from1: i64, mut to1: i64, mut from2: i64, mut to2: i64) -> i64 {
    if from1 > from2 {
        std::mem::swap(&mut from1, &mut from2);
        std::mem::swap(&mut to1, &mut to2);
    }
    if to1 < from2 {
        0
    } else if to1 < to2 {
        to1 - from2 + 1
    } else {
        to2 - from2 + 1
    }
}

/// C `read_clan_info_file` (cmscan.c): parse the `--clanin` file. Each non-comment
/// line is `<clan-name> <model> <model> ...`. Returns the clan names (in file order,
/// one per line = clan index) and a map from model name to its clan index (C
/// `clan_mapA` via `determine_clan_index`). We skip C's SSI-based model-existence
/// check (an error guard transparent on valid input); duplicate model membership is
/// still rejected as C does.
fn read_clan_info(path: &str) -> Result<(Vec<String>, std::collections::HashMap<String, i64>), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to open {path} for reading clan information: {e}"))?;
    let mut clan_names: Vec<String> = Vec::new();
    let mut clan_of_model: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let mut nclan: i64 = 0;
    for raw in text.lines() {
        // esl_fileparser: '#' starts a comment to end of line; blank lines are skipped.
        let line = match raw.find('#') {
            Some(i) => &raw[..i],
            None => raw,
        };
        let mut toks = line.split_whitespace();
        let clan = match toks.next() {
            Some(t) => t,
            None => continue,
        };
        clan_names.push(clan.to_string());
        for m in toks {
            if clan_of_model.contains_key(m) {
                return Err(format!("model {m} listed twice in {path}"));
            }
            clan_of_model.insert(m.to_string(), nclan);
        }
        nclan += 1;
    }
    if nclan == 0 {
        return Err(format!("Error reading {path}, no clans present in file"));
    }
    Ok((clan_names, clan_of_model))
}

/// C `cm_tophits_SortForOverlapMarkup` + `RemoveOrMarkOverlaps(do_remove=FALSE)`
/// (cm_tophits.c:443/1097). cmscan processes one sequence per call, so all hits share
/// seq_idx; we group by strand (and clan, if `do_clans_only`) and, within each group
/// (sorted by the markup order), mark every hit that overlaps a better-scoring one:
/// setting `marked_overlap`, `any_oidx` (best overlap) and `win_oidx` (best overlap
/// that is itself un-marked). C's same-model overlap REMOVAL pass is already done by
/// the per-model searcher, so only this cross-model markup remains.
fn mark_overlaps(hits: &mut [ScanHit], do_clans_only: bool) {
    let n = hits.len();
    if n < 2 {
        return;
    }
    // Markup sort order (C hit_sorter_for_overlap_markup_clans_only / _agnostic):
    // seq_idx(const) -> in_rc -> [clan_idx] -> evalue -> score(desc) -> start -> cm_idx.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        let a = &hits[x];
        let b = &hits[y];
        (a.hit.in_rc as u8)
            .cmp(&(b.hit.in_rc as u8))
            .then_with(|| {
                if do_clans_only {
                    a.clan_idx.cmp(&b.clan_idx)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then(a.evalue.partial_cmp(&b.evalue).unwrap())
            .then(b.hit.score.partial_cmp(&a.hit.score).unwrap())
            .then(a.hit.start.cmp(&b.hit.start))
            .then(a.model_idx.cmp(&b.model_idx))
    });
    let mut i = 0;
    while i < n {
        // group = maximal run with equal in_rc (and clan_idx if do_clans_only)
        let mut j = i + 1;
        while j < n
            && hits[order[j]].hit.in_rc == hits[order[i]].hit.in_rc
            && (!do_clans_only || hits[order[j]].clan_idx == hits[order[i]].clan_idx)
        {
            j += 1;
        }
        // C: process the set only if it has >1 hit, and (not clans-only OR clan_idx != -1)
        if j != i + 1 && (!do_clans_only || hits[order[i]].clan_idx != -1) {
            for a in i..j {
                let (a_start, a_stop, a_rc, a_marked, a_hitidx) = {
                    let ha = &hits[order[a]];
                    (ha.hit.start, ha.hit.stop, ha.hit.in_rc, ha.marked_overlap, ha.hit_idx as i64)
                };
                for b in (a + 1)..j {
                    let (b_start, b_stop) = {
                        let hb = &hits[order[b]];
                        (hb.hit.start, hb.hit.stop)
                    };
                    // C overlap test: forward (start<stop) vs reverse (start>stop).
                    let overlap = if !a_rc {
                        !(b_stop < a_start) && !(a_stop < b_start)
                    } else {
                        !(b_start < a_stop) && !(a_start < b_stop)
                    };
                    if overlap {
                        let hb = &mut hits[order[b]];
                        hb.marked_overlap = true;
                        if hb.any_oidx == -1 {
                            hb.any_oidx = a_hitidx;
                        }
                        if !a_marked && hb.win_oidx == -1 {
                            hb.win_oidx = a_hitidx;
                        }
                    }
                }
            }
        }
        i = j;
    }
}

/// cmscan `--fmt 3` tabular rows (C `cm_tophits_TabularTargets3`, cm_tophits.c:2698):
/// identical to fmt 1 plus two columns, `mdl len` (C ad->clen) and `seq len` (srcL),
/// inserted before the description. target = MODEL, query = SEQUENCE.
fn append_tblout3(s: &mut String, qname: &str, qacc: &str, hits: &[ScanHit], show_header: bool) {
    let tnamew = hits.iter().map(|h| h.model_name.len()).max().unwrap_or(0).max(20);
    let taccw = hits
        .iter()
        .map(|h| h.model_acc.as_deref().map(|a| a.len()).unwrap_or(0))
        .max()
        .unwrap_or(0)
        .max(9);
    let qnamew = qname.len().max(20);
    let qaccw = qacc.len().max(9);
    let posw = hits
        .iter()
        .map(|h| integer_textwidth(h.hit.start).max(integer_textwidth(h.hit.stop)))
        .max()
        .unwrap_or(0)
        .max(8);
    let clenw = hits.iter().map(|h| integer_textwidth(h.model_clen as i64)).max().unwrap_or(0).max(7);
    let srclw = hits.iter().map(|h| integer_textwidth(h.src_l)).max().unwrap_or(0).max(7);
    let dash = |k: usize| "-".repeat(k);
    if show_header {
        s.push_str(&format!(
            "#{:<w1$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:>3} {:>8} {:>8} {:>posw$} {:>posw$} {:>6} {:>5} {:>4} {:>4} {:>5} {:>6} {:>9} {:>3} {:>clenw$} {:>srclw$} {}\n",
            "target name", "accession", "query name", "accession", "mdl", "mdl from", "mdl to",
            "seq from", "seq to", "strand", "trunc", "pass", "gc", "bias", "score", "E-value", "inc",
            "mdl len", "seq len", "description of target", w1 = tnamew - 1
        ));
        s.push_str(&format!(
            "#{:<w1$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:<3} {:<8} {:<8} {:<posw$} {:<posw$} {:<6} {:<5} {:<4} {:<4} {:<5} {:<6} {:<9} {:<3} {:<clenw$} {:<srclw$} {}\n",
            dash(tnamew - 1), dash(taccw), dash(qnamew), dash(qaccw), "---", dash(8), dash(8),
            dash(posw), dash(posw), "------", "-----", "----", "----", "-----", "------", "---------",
            "---", dash(clenw), dash(srclw), "---------------------", w1 = tnamew - 1
        ));
    }
    for h in hits {
        let tname = h.model_name.as_str();
        let tacc = match &h.model_acc {
            Some(a) if !a.is_empty() => a.as_str(),
            _ => "-",
        };
        let tdesc = match &h.model_desc {
            Some(d) if !d.is_empty() => d.as_str(),
            _ => "-",
        };
        let strand = if h.hit.in_rc { "-" } else { "+" };
        let inc = if h.included { "!" } else { "?" };
        let eval_s = fmt_evalue(h.evalue);
        s.push_str(&format!(
            "{:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:>3} {:>8} {:>8} {:>posw$} {:>posw$} {:>6} {:>5} {:>4} {:>4.2} {:>5.1} {:>6.1} {:>9} {:<3} {:>clenw$} {:>srclw$} {}\n",
            tname, tacc, qname, if qacc.is_empty() { "-" } else { qacc },
            if h.hit.hmmonly { "hmm" } else { "cm" }, h.hit.mdl_from, h.hit.mdl_to, h.hit.start, h.hit.stop,
            strand, h.hit.trunc.as_str(), h.hit.pass_idx, h.hit.gc, h.hit.bias, h.hit.score, eval_s, inc,
            h.model_clen, h.src_l, tdesc
        ));
    }
}

/// cmscan `--fmt 2` tabular rows (C `cm_tophits_TabularTargets2`, cm_tophits.c:2359).
/// Adds an output index, clan name, and overlap columns (olp / anyidx,afrct1,afrct2 /
/// winidx,wfrct1,wfrct2) plus mdl-len/seq-len. `skip_overlaps` (`--oskip`) omits hits
/// that overlap a better-scoring winner. Assumes the default (non `--trmF3`) mode.
#[allow(clippy::too_many_arguments)]
fn append_tblout2(
    s: &mut String,
    qname: &str,
    qacc: &str,
    hits: &[ScanHit],
    show_header: bool,
    clan_names: &[String],
    skip_overlaps: bool,
) {
    let n = hits.len();
    let tnamew = hits.iter().map(|h| h.model_name.len()).max().unwrap_or(0).max(20);
    let taccw = hits
        .iter()
        .map(|h| h.model_acc.as_deref().map(|a| a.len()).unwrap_or(0))
        .max()
        .unwrap_or(0)
        .max(9);
    let qnamew = qname.len().max(20);
    let qaccw = qacc.len().max(9);
    let posw = hits
        .iter()
        .map(|h| integer_textwidth(h.hit.start).max(integer_textwidth(h.hit.stop)))
        .max()
        .unwrap_or(0)
        .max(8);
    let idxw1 = integer_textwidth(n as i64).max(4);
    let idxw2 = integer_textwidth(n as i64).max(6);
    let clanw = hits
        .iter()
        .filter(|h| h.clan_idx != -1)
        .map(|h| clan_names[h.clan_idx as usize].len())
        .max()
        .unwrap_or(0)
        .max(9);
    let clenw = hits.iter().map(|h| integer_textwidth(h.model_clen as i64)).max().unwrap_or(0).max(7);
    let srclw = hits.iter().map(|h| integer_textwidth(h.src_l)).max().unwrap_or(0).max(7);
    let dash = |k: usize| "-".repeat(k);
    let iw1 = idxw1 - 1;

    if show_header {
        // C fprintf #1 (header labels)
        s.push_str(&format!(
            "#{:<iw1$} {:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:<clanw$} {:>3} {:>8} {:>8} {:>posw$} {:>posw$} {:>6} {:>5} {:>4} {:>4} {:>5} {:>6} {:>9} {:>3} {:>3} {:>idxw2$} {:>6} {:>6} {:>idxw2$} {:>6} {:>6} {:>clenw$} {:>srclw$} {}\n",
            "idx", "target name", "accession", "query name", "accession", "clan name",
            "mdl", "mdl from", "mdl to", "seq from", "seq to", "strand", "trunc", "pass", "gc", "bias",
            "score", "E-value", "inc", "olp", "anyidx", "afrct1", "afrct2", "winidx", "wfrct1", "wfrct2",
            "mdl len", "seq len", "description of target"
        ));
        // C fprintf #2 (dashes)
        s.push_str(&format!(
            "#{:<iw1$} {:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:<clanw$} {:<3} {:<8} {:<8} {:<posw$} {:<posw$} {:<6} {:<5} {:<4} {:<4} {:<5} {:<6} {:<9} {:<3} {:<3} {} {:>6} {:>6} {} {:>6} {:>6} {:<clenw$} {:<srclw$} {}\n",
            dash(idxw1 - 1), dash(tnamew), dash(taccw), dash(qnamew), dash(qaccw), dash(clanw),
            "---", dash(8), dash(8), dash(posw), dash(posw), "------", "-----", "----", "----", "-----",
            "------", "---------", "---", "---", dash(idxw2), "------", "------", dash(idxw2), "------",
            "------", dash(clenw), dash(srclw), "---------------------"
        ));
    }

    // Pre-pass: sorted position of each hit_idx, which hits are referenced as an
    // overlap, and the 1-based output index each reported (non-skipped) hit gets.
    let mut sorted_idx = vec![0usize; n];
    for (h, sh) in hits.iter().enumerate() {
        sorted_idx[sh.hit_idx] = h;
    }
    let mut has_overlap = vec![false; n]; // indexed by hit_idx
    let mut output_idx = vec![-1i64; n]; // indexed by sorted position h
    let mut noutput: i64 = 0;
    for (h, sh) in hits.iter().enumerate() {
        if sh.any_oidx != -1 {
            has_overlap[sh.any_oidx as usize] = true;
        }
        if sh.win_oidx != -1 {
            has_overlap[sh.win_oidx as usize] = true;
        }
        let maybe_skip = sh.marked_overlap && sh.win_oidx != -1;
        if !skip_overlaps || !maybe_skip {
            noutput += 1;
            output_idx[h] = noutput;
        }
    }

    noutput = 0;
    for sh in hits.iter() {
        let maybe_skip = sh.marked_overlap && sh.win_oidx != -1;
        if skip_overlaps && maybe_skip {
            continue;
        }
        let as_ = if sh.any_oidx == -1 { -1i64 } else { sorted_idx[sh.any_oidx as usize] as i64 };
        let ws_ = if sh.win_oidx == -1 { -1i64 } else { sorted_idx[sh.win_oidx as usize] as i64 };
        let ao = if as_ == -1 { -1 } else { output_idx[as_ as usize] };
        let wo = if ws_ == -1 { -1 } else { output_idx[ws_ as usize] };

        // any_* overlap fractions (C: computed iff as != -1)
        let (anyidx_s, afrct1_s, afrct2_s) = if as_ != -1 {
            let o = &hits[as_ as usize];
            let (len1, len2, nres) = if sh.hit.in_rc {
                (
                    sh.hit.start - sh.hit.stop + 1,
                    o.hit.start - o.hit.stop + 1,
                    overlap_nres(sh.hit.stop, sh.hit.start, o.hit.stop, o.hit.start),
                )
            } else {
                (
                    sh.hit.stop - sh.hit.start + 1,
                    o.hit.stop - o.hit.start + 1,
                    overlap_nres(sh.hit.start, sh.hit.stop, o.hit.start, o.hit.stop),
                )
            };
            (
                ao.to_string(),
                format!("{:.3}", nres as f32 / len1 as f32),
                format!("{:.3}", nres as f32 / len2 as f32),
            )
        } else {
            ("-".to_string(), "-".to_string(), "-".to_string())
        };

        // win_* overlap columns: "-" if none, "\"" if identical to any_*, else computed
        let (winidx_s, wfrct1_s, wfrct2_s) = if ws_ == -1 {
            ("-".to_string(), "-".to_string(), "-".to_string())
        } else if ws_ == as_ {
            ("\"".to_string(), "\"".to_string(), "\"".to_string())
        } else {
            let o = &hits[ws_ as usize];
            let (len1, len2, nres) = if sh.hit.in_rc {
                (
                    sh.hit.start - sh.hit.stop + 1,
                    o.hit.start - o.hit.stop + 1,
                    overlap_nres(sh.hit.stop, sh.hit.start, o.hit.stop, o.hit.start),
                )
            } else {
                (
                    sh.hit.stop - sh.hit.start + 1,
                    o.hit.stop - o.hit.start + 1,
                    overlap_nres(sh.hit.start, sh.hit.stop, o.hit.start, o.hit.stop),
                )
            };
            (
                wo.to_string(),
                format!("{:.3}", nres as f32 / len1 as f32),
                format!("{:.3}", nres as f32 / len2 as f32),
            )
        };

        // olp column (C: marked -> " $ "/" = ", else has-overlap -> " ^ ", else " * ")
        let olp = if sh.marked_overlap {
            if ws_ == -1 { " $ " } else { " = " }
        } else if has_overlap[sh.hit_idx] {
            " ^ "
        } else {
            " * "
        };

        let clanname = if sh.clan_idx == -1 { "-" } else { clan_names[sh.clan_idx as usize].as_str() };
        let tname = sh.model_name.as_str();
        let tacc = match &sh.model_acc {
            Some(a) if !a.is_empty() => a.as_str(),
            _ => "-",
        };
        let tdesc = match &sh.model_desc {
            Some(d) if !d.is_empty() => d.as_str(),
            _ => "-",
        };
        let strand = if sh.hit.in_rc { "-" } else { "+" };
        let inc = if sh.included { " ! " } else { " ? " };
        let eval_s = fmt_evalue(sh.evalue);
        noutput += 1;
        s.push_str(&format!(
            "{:<idxw1$} {:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:<clanw$} {:>3} {:>8} {:>8} {:>posw$} {:>posw$} {:>6} {:>5} {:>4} {:>4.2} {:>5.1} {:>6.1} {:>9} {:>3} {:>3} {:>idxw2$} {:>6} {:>6} {:>idxw2$} {:>6} {:>6} {:>clenw$} {:>srclw$} {}\n",
            noutput, tname, tacc, qname, if qacc.is_empty() { "-" } else { qacc }, clanname,
            if sh.hit.hmmonly { "hmm" } else { "cm" }, sh.hit.mdl_from, sh.hit.mdl_to, sh.hit.start,
            sh.hit.stop, strand, sh.hit.trunc.as_str(), sh.hit.pass_idx, sh.hit.gc, sh.hit.bias,
            sh.hit.score, eval_s, inc, olp, anyidx_s, afrct1_s, afrct2_s, winidx_s, wfrct1_s, wfrct2_s,
            sh.model_clen, sh.src_l, tdesc
        ));
    }
}

/// One query sequence's human-readable block (C cmscan.c:585-734): the "Query:" line,
/// the SCAN-mode "Hit scores" table, the "Hit alignments" section, and the pipeline
/// statistics summary + "# CPU time:" + "//". Mirrors the cmsearch block with the
/// SCAN orientation (target = MODEL). `hits` are the reported hits sorted for output.
#[allow(clippy::too_many_arguments)]
fn scan_human_block(
    seqname: &str,
    seqacc: &str,
    seqdesc: &str,
    l: usize,
    hits: &[ScanHit],
    show_alignments: bool,
    show_accessions: bool,
    textw: i32,
    pli: &PliStats,
    summed: &PassAcctSnapshot,
    std_nres: u64,
    n_output_trunc: u64,
    pos_output_trunc: u64,
    hmmonly_stats: Option<&infernox::p7_hmmonly::HmmonlyPassStats>,
) -> String {
    let mut b = String::new();
    // Per-query header block (cmscan.c:585-587): Query is the SEQUENCE.
    b.push_str(&format!("Query:       {}  [L={}]\n", seqname, l));
    if !seqacc.is_empty() {
        b.push_str(&format!("Accession:   {}\n", seqacc));
    }
    if !seqdesc.is_empty() {
        b.push_str(&format!("Description: {}\n", seqdesc));
    }

    // "Hit scores" table (cm_tophits_Targets, SCAN mode: targets are models).
    let rows: Vec<infernox::cm_tophits::HitRow> = hits
        .iter()
        .map(|h| infernox::cm_tophits::HitRow {
            name: h.model_name.as_str(),
            acc: h.model_acc.as_deref().unwrap_or(""),
            desc: h.model_desc.as_deref().unwrap_or(""),
            evalue: h.evalue,
            score: h.hit.score,
            bias: h.hit.bias,
            start: h.hit.start,
            stop: h.hit.stop,
            in_rc: h.hit.in_rc,
            hmmonly: h.hit.hmmonly,
            trunc: h.hit.trunc.as_str(),
            gc: h.hit.gc,
            included: h.included,
        })
        .collect();
    b.push_str(&infernox::cm_tophits::cm_tophits_targets(
        &rows, true, // mode_scan
        show_accessions, textw,
    ));
    b.push_str("\n\n");

    // "Hit alignments" section (SCAN orientation: >> shows the model).
    if show_alignments {
        let ali_rows: Vec<infernox::cm_tophits::AliHit> = hits
            .iter()
            .filter_map(|h| {
                let ad = h.hit.alignment.as_ref()?;
                Some(infernox::cm_tophits::AliHit {
                    ad,
                    cmname: h.model_name.as_str(),
                    cmacc: h.model_acc.as_deref().unwrap_or(""),
                    sqname: seqname,
                    sqacc: seqacc,
                    tdesc: h.model_desc.as_deref().unwrap_or(""),
                    target_is_model: true,
                    evalue: h.evalue,
                    score: h.hit.score,
                    bias: h.hit.bias,
                    hmmonly: h.hit.hmmonly,
                    start: h.hit.start,
                    stop: h.hit.stop,
                    in_rc: h.hit.in_rc,
                    src_l: l as i64,
                    clen: h.model_clen,
                    included: h.included,
                })
            })
            .collect();
        b.push_str(&infernox::cm_tophits::cm_tophits_hit_alignments(
            &ali_rows,
            show_accessions,
            textw,
            hits.len(),
        ));
        b.push_str("\n\n");
    }

    // Statistics summary — replicate C cm_pli_Statistics (cm_pipeline.c:1742):
    //   * CM-pipeline block only if pli->nmodels > 0 (models that used the CM);
    //   * HMM-only block only if pli->nmodels_hmmonly > 0, followed by a blank line;
    //   * a combined "Total CM and HMM hits reported" line when BOTH ran;
    //   * then the "# CPU time:" line and the "//" record terminator.
    if pli.nmodels > 0 {
        b.push_str(&infernox::cm_tophits::format_pli_statistics(
            pli, summed, std_nres, n_output_trunc, pos_output_trunc, true,
        ));
    }
    if let Some(hs) = hmmonly_stats {
        // C cm_pipeline.c:1772: pli_hmmonly_pass_statistics(ofp,pli); fprintf(ofp,"\n");
        b.push_str(&infernox::p7_hmmonly::pli_hmmonly_pass_statistics(hs));
        b.push('\n');
    }
    if pli.nmodels > 0 {
        if let Some(hs) = hmmonly_stats {
            // C cm_pipeline.c:1778-1782.
            b.push_str(&format!(
                "Total CM and HMM hits reported:                    {:15}\n\n",
                summed.n_output as i64 + hs.n_output
            ));
        }
    }
    // C cm_pli_Statistics ends with the "# CPU time:" line; then cmscan.c:734 prints "//".
    b.push_str(&format!("# CPU time: {}\n", cpu_time_line()));
    b.push_str("//\n");
    b
}

/// Timing-variable "# CPU time:" line body (documented-variable, like the Date line).
fn cpu_time_line() -> String {
    "0.00u 0.00s 00:00:00.00 Elapsed: 00:00:00.00".to_string()
}

/// printf `%g` (default precision 6 significant figures), for header config values.
fn fmt_g(x: f64) -> String {
    infernox::cm_tophits::fmt_g_prec(x, 6)
}

/// printf `%f` (default 6 decimals), for the --FZ line (C uses %f there).
fn format_f(x: f64) -> String {
    format!("{:.6}", x)
}

/// Faithful port of cmscan.c `output_header`: the `# ...` banner + config block that
/// precedes the per-query output. Every line is emitted iff the driving option was set
/// (C `esl_opt_IsUsed`), in cmscan.c order. Note the SCAN swap: "# query sequence file"
/// then "# target CM database" (opposite of cmsearch).
fn output_header_scan(p: &Parsed, seqfile: &str, cmfile: &str, ncpus: i64, cpu_used: bool) -> String {
    let mut s = String::new();
    s.push_str("# cmscan :: search sequence(s) against a CM database\n");
    s.push_str("# INFERNAL 1.1.5 (Sep 2023)\n");
    s.push_str("# Copyright (C) 2023 Howard Hughes Medical Institute.\n");
    s.push_str("# Freely distributed under the BSD open source license.\n");
    s.push_str("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");

    s.push_str(&format!("# query sequence file:                   {}\n", seqfile));
    s.push_str(&format!("# target CM database:                    {}\n", cmfile));
    let g = |name: &str| -> f64 { p.get_f64(name).unwrap_or(None).unwrap_or(0.0) };
    let gi = |name: &str| -> i64 { p.get_i64(name).unwrap_or(None).unwrap_or(0) };
    let gs = |name: &str| -> String { p.get_str(name).unwrap_or("").to_string() };
    if p.is_set("-g")           { s.push_str("# CM configuration:                      glocal\n"); }
    if p.is_set("-Z")           { s.push_str(&format!("# database size is set to:               {:.1} Mb\n", g("-Z"))); }
    if p.is_set("-o")           { s.push_str(&format!("# output directed to file:               {}\n", gs("-o"))); }
    if p.is_set("--tblout")     { s.push_str(&format!("# tabular output of hits:                {}\n", gs("--tblout"))); }
    if p.is_set("--fmt")        { s.push_str(&format!("# tabular output format:                 {}\n", gi("--fmt"))); }
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
    if p.is_set("--qformat")    { s.push_str(&format!("# query <seqfile> format asserted:       {}\n", gs("--qformat"))); }
    if p.is_set("--glist")      { s.push_str(&format!("# models for glocal mode scan read from: {}\n", gs("--glist"))); }
    if p.is_set("--block")      { s.push_str(&format!("# block size (# models) set to:          {}\n", gi("--block"))); }
    if p.is_set("--clanin")     { s.push_str(&format!("# clan information read from file:       {}\n", gs("--clanin"))); }
    if p.is_set("--oclan")      { s.push_str("# only mark overlaps within clans:       yes\n"); }
    if p.is_set("--oskip")      { s.push_str("# skipping overlaps in tbl output:       yes\n"); }
    // Developer/expert options (cmscan.c output_header, same IsUsed gating + order).
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
    // C esl_opt_IsUsed is FALSE when value == default string (esl_getopts.c:935),
    // so an explicit default emits no header line. Defaults: --fbeta 1e-7, --beta 1e-15.
    if p.is_set("--fbeta") && p.get_str("--fbeta") != Some("1e-7")   { s.push_str(&format!("# beta parameter for CYK filter stage:   {}\n", fmt_g(g("--fbeta")))); }
    if p.is_set("--fnonbanded") { s.push_str("# no bands (CYK filter stage)            on\n"); }
    if p.is_set("--nocykenv")   { s.push_str("# CYK envelope redefinition:             off\n"); }
    if p.is_set("--cykenvx")    { s.push_str(&format!("# CYK envelope redefn P-val multiplier:  {}\n", gi("--cykenvx"))); }
    if p.is_set("--tau")        { s.push_str(&format!("# tau parameter for final stage:         {}\n", fmt_g(g("--tau")))); }
    if p.is_set("--sums")       { s.push_str("# posterior sums (final stage):          on\n"); }
    if p.is_set("--qdb")        { s.push_str("# QDBs (final stage)                     on\n"); }
    if p.is_set("--beta") && p.get_str("--beta") != Some("1e-15")   { s.push_str(&format!("# beta parameter for final stage:        {}\n", fmt_g(g("--beta")))); }
    if p.is_set("--nonbanded")  { s.push_str("# no bands (final stage)                 on\n"); }
    if p.is_set("--timeF1")     { s.push_str("# abort after Stage 1 MSV (for timing)   on\n"); }
    if p.is_set("--timeF2")     { s.push_str("# abort after Stage 2 Vit (for timing)   on\n"); }
    if p.is_set("--timeF3")     { s.push_str("# abort after Stage 3 Fwd (for timing)   on\n"); }
    if p.is_set("--timeF4")     { s.push_str("# abort after Stage 4 gFwd (for timing)  on\n"); }
    if p.is_set("--timeF5")     { s.push_str("# abort after Stage 5 env defn (for timing) on\n"); }
    if p.is_set("--timeF6")     { s.push_str("# abort after Stage 6 CYK (for timing)   on\n"); }
    if p.is_set("--trmF3")      { s.push_str("# terminate after Stage 3 Fwd:           on\n"); }
    if p.is_set("--nogreedy")   { s.push_str("# greedy CM hit resolution:              off\n"); }
    if p.is_set("--cp9noel")    { s.push_str("# CP9 HMM local ends:                    off\n"); }
    if p.is_set("--cp9gloc")    { s.push_str("# CP9 HMM configuration:                 glocal\n"); }
    if p.is_set("--null2")      { s.push_str("# null2 bias corrections:                on\n"); }
    if p.is_set("--maxtau")     { s.push_str(&format!("# max tau during band tightening:        {}\n", fmt_g(g("--maxtau")))); }
    if p.is_set("--seed") {
        if gi("--seed") == 0 { s.push_str("# random number seed:                    one-time arbitrary\n"); }
        else                 { s.push_str(&format!("# random number seed set to:             {}\n", gi("--seed"))); }
    }
    if !p.is_set("--notrunc") {
        if p.is_set("--max")   { s.push_str("# truncated hit detection:               off [due to --max]\n"); }
        if p.is_set("--nohmm") { s.push_str("# truncated hit detection:               off [due to --nohmm]\n"); }
    }
    // number of worker threads (cmscan.c: ALWAYS printed).
    s.push_str(&format!(
        "# number of worker threads:              {}{}\n",
        ncpus,
        if cpu_used { " [--cpu]" } else { "" }
    ));
    s.push_str("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n");
    s
}

/// printf "%.2g" (2 significant figures), matching infernal tblout E-values.
fn fmt_evalue(e: f64) -> String {
    if e == 0.0 {
        return "0".to_string();
    }
    let p: i32 = 2;
    let s = format!("{:.*e}", (p - 1) as usize, e);
    let (mant, ex) = {
        let parts: Vec<&str> = s.splitn(2, 'e').collect();
        (parts[0].to_string(), parts[1].parse::<i32>().unwrap_or(0))
    };
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
