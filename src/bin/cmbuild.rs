// SPDX-License-Identifier: BSD-3-Clause
// infernox-cmbuild — build a covariance model from a structural MSA.
//
// Faithful port of the core of Infernal 1.1.5's cmbuild (cmbuild.c pipeline:
// build_model -> annotate -> set_effective_seqnumber -> parameterize ->
// configure_model -> cm_Validate -> cm_file_WriteASCII), using the ported
// cm_modelmaker / prior / eweight / cm_consensus / cm_rebalance modules.
//
// Scope: default `cmbuild <out.cm> <msa.sto>` with --wpb weighting and --eent
// entropy weighting. Options: -n, -F, -o, -O, --hand, --enone, --wnone.
//
// The emitted CM includes the p7 filter HMM (HMMER3/f block + EFP7GF line),
// built and calibrated via attach_p7_filter (build_and_calibrate_p7_filter,
// cmbuild.c:2269). The whole file — CM body (states/nodes/emissions/transitions/
// consensus/rf/map/QDBs) and p7 filter block — is byte-identical to C cmbuild
// (modulo the documented-variable INFERNAL1/a banner, DATE, COM, HMMER version).

use std::io::Write;

use infernox::cm::CM;
use infernox::cm_modelmaker as mm;
use infernox::easel::alphabet::EslAlphabet;
use infernox::easel::msa::EslMsa;

const DEFAULT_EL_SELFPROB: f64 = 0.94;
const DEFAULT_NULL2_OMEGA: f64 = 0.000015258791;
const DEFAULT_NULL3_OMEGA: f64 = 0.000015258791;
const V1P0_NULL_OMEGA: f64 = 0.03125; // 1/(2^5), null2/null3 omega for v0.56->v1.0.2

use infernox::msaweight::{BuildKnobs, WScheme};

struct Opts {
    cmfile: String,
    msafile: String,
    force: bool,
    hand: bool,
    name: Option<String>,
    ofile: Option<String>,
    omsafile: Option<String>,
    /// --rsearch <matrixfile>: RSEARCH parameterization with a RIBOSUM matrix.
    rsearch: Option<String>,
    // MSA-clustering options (build >1 CM per MSA)
    ctarget: Option<i32>, // --ctarget <n>: target number of clusters
    cmaxid: Option<f64>,  // --cmaxid <x>: max fractional id between clusters
    call: bool,           // --call: one CM per sequence
    corig: bool,          // --corig: also build a CM from the full MSA
    cdump: Option<String>, // --cdump <f>: dump per-cluster MSAs to <f>

    // --- Group A: weighting ---
    wscheme: WScheme,
    wid: f64, // --wid (for --wblosum), default 0.62

    // --- Group A: effective seq number ---
    enone: bool,
    ere: Option<f64>,    // --ere
    eset: Option<f64>,   // --eset
    ehmmre: Option<f64>, // --ehmmre
    eminseq: f64,        // --eminseq, default 0.1
    emaxseq: Option<f64>,// --emaxseq (default cm->nseq)
    esigma: f64,         // --esigma, default 45.0

    // --- Group A: priors ---
    noh3pri: bool,        // --noh3pri
    v1p0: bool,           // --v1p0
    p56: bool,            // --p56
    null: [f32; 4],       // --null <f> (default uniform 0.25); ACGU background
    null_file: Option<String>, // --null filename (for header echo)

    // --- Group A: construction ---
    symfrac: f64,     // --symfrac, default 0.5
    fragthresh: f64,  // --fragthresh, default 0.5
    fraggiven: bool,  // --fraggiven (use MSA's given ~ fragment annotation, don't infer)
    noss: bool,       // --noss
    nobalance: bool,  // --nobalance
    nodetach: bool,   // --nodetach
    iins: bool,       // --iins
    iflank: bool,     // --iflank
    elself: f64,      // --elself, default 0.94
    n2omega: f64,     // --n2omega, default 0.000015258791
    n3omega: f64,     // --n3omega, default 0.000015258791

    // --- Group B ---
    informat: Option<String>, // --informat

    // --- Group C dump files ---
    cmtbl: Option<String>,      // --cmtbl
    emap: Option<String>,       // --emap
    occfile: Option<String>,    // --occfile
    fp7occfile: Option<String>, // --fp7occfile

    // --- MSA refinement (--refine) ---
    refine: Option<String>,  // --refine <f>: refine input aln (EM/Gibbs), save to <f>
    refine_local: bool,      // -l: configure model local for aln refinement
    gibbs: bool,             // --gibbs: Gibbs sampling instead of EM
    seed: u32,               // --seed <n>: RNG seed (default 0 => arbitrary)
    refine_cyk: bool,        // --cyk: CYK instead of optimal accuracy
    notrunc: bool,           // --notrunc: no truncated alignment algorithm
    miss: bool,              // --miss: mark terminal-gap seqs as fragments
    sub: bool,               // --sub: sub CM for columns b/t HMM start/end
    nonbanded: bool,         // --nonbanded: no bands for refinement alignment
    indi: bool,              // --indi: print individual seq scores during refine
    fins: bool,              // --fins: flush inserts left/right in alignments
    tau: f64,                // --tau, default 1e-7
    mxsize: f64,             // --mxsize, default 2048.0
    rdump: Option<String>,   // --rdump <f>: dump intermediate alignments to <f>

    // --- HMM filter construction (p7) ---
    p7ml: bool,              // --p7ml: define the filter p7 HMM as the ML p7 HMM
    p7ere: Option<f64>,      // --p7ere <x>: filter p7 HMM min rel entropy/posn

    // --- HMM filter calibration sample counts (--E?N, all default 200) ---
    emn: i32,                // --EmN: # sampled seqs for local MSV calibration
    evn: i32,                // --EvN: # sampled seqs for local Vit calibration
    elfn: i32,               // --ElfN: # sampled seqs for local Fwd calibration
    egfn: i32,               // --EgfN: # sampled seqs for glocal Fwd calibration

    // Set of option names the user explicitly passed (esl_opt_IsUsed), for the
    // output_header echo lines. Canonical long-option names (or -n/-F/-o/-O).
    used: std::collections::HashSet<&'static str>,
}

/// C: CMReadNullModel (cm.c). Reads K(=4) lines whose first token is not "#";
/// first token = background prob for A,C,G,U in order. Checks sum≈1.0, FNorms.
fn read_null_model(path: &str) -> [f32; 4] {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Failed to open null model file {path}: {e}");
        std::process::exit(1);
    });
    let mut null = [0.0f32; 4];
    let mut x = 0usize;
    let mut sum = 0.0f32;
    for line in content.lines() {
        if x >= 4 {
            break;
        }
        let tok = match line.split([' ', '\t']).find(|t| !t.is_empty()) {
            Some(t) => t,
            None => continue,
        };
        if tok == "#" {
            continue;
        }
        let v: f32 = tok.parse().unwrap_or(0.0); // C uses atof (0.0 on parse fail)
        null[x] = v;
        sum += v;
        x += 1;
    }
    if x < 4 || sum > 1.00001 || sum < 0.99999 {
        eprintln!("{path} is not in CM null model file format.\nThere are not 4 background probabilities that sum to exactly 1.0");
        std::process::exit(1);
    }
    // esl_vec_FNorm(null, 4)
    let s: f32 = null.iter().sum();
    for v in &mut null {
        *v /= s;
    }
    null
}

/// C `cmbuild.c` process_commandline ERROR: block (cmbuild.c:653-658). Every
/// esl_getopts command-line error routes here: print the offending first line
/// (`Failed to parse command line: <errbuf>` or `Incorrect number of command
/// line arguments.`), then `esl_usage` + the basic-options
/// `esl_opt_DisplayHelp(stdout, go, 1, 2, 100)` block, then
/// `printf("\nTo see more help on available options, do %s -h\n\n", argv0)` and
/// exit(1). ALL of this goes to STDOUT (C uses esl_usage(stdout,...)/puts/printf).
/// The `Usage:` program name is C's `esl_usage` basename (hardcoded "cmbuild");
/// the final "do <argv0> -h" line embeds the actual binary path (the one
/// path-dependent line, excluded from byte-diffs).
fn cmdline_fail(first_line: &str) -> ! {
    let argv0 = std::env::args().next().unwrap_or_else(|| "cmbuild".to_string());
    print!(
        "{first_line}\n\
Usage: cmbuild [-options] <cmfile_out> <msafile>\n\
\n\
where basic options are:\n  \
-h        : show brief help on version and usage\n  \
-n <s>    : name the CM(s) <s>, (only if single aln in file)\n  \
-F        : force; allow overwriting of <cmfile_out>\n  \
-o <f>    : direct summary output to file <f>, not stdout\n  \
-O <f>    : resave consensus/insert column annotated MSA to file <f>\n  \
--devhelp : show list of otherwise hidden developer/expert options\n\
\n\
To see more help on available options, do {argv0} -h\n\n"
    );
    std::process::exit(1);
}

/// C `cm_Fail()` (errors.c:56): print `\nError: <msg>\n` to stderr and exit(1).
/// Used for the runtime clustering-option checks (--corig/--cdump without a
/// clustering mode, "More than one of ...") which are NOT esl_getopts errors and
/// carry no usage banner.
fn cm_fail(msg: &str) -> ! {
    eprintln!("\nError: {msg}");
    std::process::exit(1);
}

/// C `cmbuild.c` process_commandline `-h` path (cmbuild.c:625-648): print the
/// full banner + usage + every visible option group (esl_opt_DisplayHelp for
/// docgroups 1..8) to stdout, then exit(0). Transcribed verbatim from C's `-h`.
fn full_help() -> ! {
    print!(
        "# cmbuild :: covariance model construction from multiple sequence alignments\n\
# INFERNAL 1.1.5 (Sep 2023)\n\
# Copyright (C) 2023 Howard Hughes Medical Institute.\n\
# Freely distributed under the BSD open source license.\n\
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\
Usage: cmbuild [-options] <cmfile_out> <msafile>\n\
\n\
Basic options:\n  \
-h        : show brief help on version and usage\n  \
-n <s>    : name the CM(s) <s>, (only if single aln in file)\n  \
-F        : force; allow overwriting of <cmfile_out>\n  \
-o <f>    : direct summary output to file <f>, not stdout\n  \
-O <f>    : resave consensus/insert column annotated MSA to file <f>\n  \
--devhelp : show list of otherwise hidden developer/expert options\n\
\n\
Alternative model construction strategies:\n  \
--fast           : assign cols w/ >= symfrac residues as consensus\n  \
--hand           : use reference coordinate annotation to specify consensus\n  \
--symfrac <x>    : fraction of non-gaps to require in a consensus column [0..1]\n  \
--fragthresh <x> : if aligned seq spans <= x*alen, tag seq as a fragment\n  \
--fragnrfpos <n> : w/--hand, seqs w/ > <n> 5' or 3' consensus gaps are fragments\n  \
--fraggiven      : use fragment info, if any, in input MSA, don't infer frags\n  \
--noss           : ignore secondary structure annotation in input alignment\n  \
--rsearch <f>    : use RSEARCH parameterization with RIBOSUM matrix file <f>\n  \
--consrf         : with --hand, rewrite RF line with consensus sequence\n\
\n\
Other model construction options*:\n  \
--null <f>  : read null (random sequence) model from file <f>\n  \
--prior <f> : read priors from file <f>\n\
\n\
Alternative relative sequence weighting strategies:\n  \
--wpb     : Henikoff position-based weights  [default]\n  \
--wgsc    : Gerstein/Sonnhammer/Chothia tree weights\n  \
--wnone   : don't do any relative weighting; set all to 1\n  \
--wgiven  : use weights as given in MSA file\n  \
--wblosum : Henikoff simple filter weights\n  \
--wid <x> : for --wblosum: set identity cutoff  [0.62]  (0<=x<=1)\n\
\n\
Alternative effective sequence weighting strategies:\n  \
--eent        : adjust eff seq # to achieve relative entropy target  [default]\n  \
--enone       : no effective seq # weighting: just use nseq\n  \
--ere <x>     : for --eent: set CM target relative entropy to <x>\n  \
--eset <x>    : set eff seq # for all models to <x>\n  \
--eminseq <x> : for --eent: set minimum effective sequence number to <x>  [0.1]\n  \
--emaxseq <x> : for --eent: set maximum effective sequence number to <x>\n  \
--ehmmre <x>  : for --eent: set minimum HMM relative entropy to <x>\n  \
--esigma <x>  : for --eent: set sigma param to <x>  [45.0]\n\
\n\
Options for HMM filter construction*:\n  \
--p7ere <x> : for the filter p7 HMM, set minimum rel entropy/posn to <x>\n  \
--p7ml      : define the filter p7 HMM as the ML p7 HMM\n\
\n\
Options for HMM filter calibration*:\n  \
--EmN <n>  : number of sampled seqs to use for p7 local MSV calibration  [200]\n  \
--EvN <n>  : number of sampled seqs to use for p7 local Vit calibration  [200]\n  \
--ElfN <n> : number of sampled seqs to use for p7 local Fwd calibration  [200]\n  \
--EgfN <n> : number of sampled seqs to use for p7 glocal Fwd calibration  [200]\n\
\n\
Options for refining the input alignment*:\n  \
--refine <f> : refine input aln w/Expectation-Maximization, save to <f>\n  \
-l           : w/--refine, configure model for local alignment [default: global]\n  \
--gibbs      : w/--refine, use Gibbs sampling instead of EM\n  \
--seed <n>   : w/--gibbs, set RNG seed to <n> (if 0: one-time arbitrary seed)\n  \
--cyk        : w/--refine, use CYK instead of optimal accuracy\n  \
--notrunc    : w/--refine, do not use truncated alignment algorithm\n  \
--miss       : w/--refine, mark seqs w/terminal gaps as fragments\n\
\n\
*Use --devhelp to show additional expert options.\n"
    );
    std::process::exit(0);
}

fn parse_args() -> Opts {
    // C esl_getopts accepts attached short-opt values (`-oout.cm` == `-o out.cm`);
    // expand for the match parser. cmbuild value-taking short options: -n,-o,-O.
    let argv: Vec<String> =
        infernox::search_cli::expand_short_opts(&std::env::args().collect::<Vec<_>>(), &['n', 'o', 'O']);
    let mut o = Opts {
        cmfile: String::new(),
        msafile: String::new(),
        force: false,
        hand: false,
        name: None,
        ofile: None,
        omsafile: None,
        rsearch: None,
        ctarget: None,
        cmaxid: None,
        call: false,
        corig: false,
        cdump: None,
        wscheme: WScheme::Pb,
        wid: 0.62,
        enone: false,
        ere: None,
        eset: None,
        ehmmre: None,
        eminseq: 0.1,
        emaxseq: None,
        esigma: 45.0,
        noh3pri: false,
        v1p0: false,
        p56: false,
        null: [0.25, 0.25, 0.25, 0.25],
        null_file: None,
        symfrac: 0.5,
        fragthresh: 0.5,
        fraggiven: false,
        noss: false,
        nobalance: false,
        nodetach: false,
        iins: false,
        iflank: false,
        elself: DEFAULT_EL_SELFPROB,
        n2omega: DEFAULT_NULL2_OMEGA,
        n3omega: DEFAULT_NULL3_OMEGA,
        informat: None,
        cmtbl: None,
        emap: None,
        occfile: None,
        fp7occfile: None,
        refine: None,
        refine_local: false,
        gibbs: false,
        seed: 0,
        refine_cyk: false,
        notrunc: false,
        miss: false,
        sub: false,
        nonbanded: false,
        indi: false,
        fins: false,
        tau: 1e-7,
        mxsize: 2048.0,
        rdump: None,
        p7ml: false,
        p7ere: None,
        emn: 200,
        evn: 200,
        elfn: 200,
        egfn: 200,
        used: std::collections::HashSet::new(),
    };
    let mut positional = Vec::new();
    let mut i = 1;
    // Helper closure: fetch the argument to a flag. `$opt` is the flag name, used
    // to reproduce esl_getopts' exact "Option <opt> requires an argument" message.
    macro_rules! next_arg {
        ($i:expr, $opt:expr) => {{
            $i += 1;
            match argv.get($i) {
                Some(v) => v.clone(),
                // C esl_getopts: missing required arg. Long options
                // (process_longopt) end the message with a period; short options
                // (process_shortopt) do not.
                None => {
                    let dot = if $opt.starts_with("--") { "." } else { "" };
                    cmdline_fail(&format!(
                        "Failed to parse command line: Option {} requires an argument{}",
                        $opt, dot
                    ))
                }
            }
        }};
    }
    macro_rules! next_real {
        ($i:expr, $opt:expr) => {{
            let s = next_arg!($i, $opt);
            s.parse::<f64>().unwrap_or_else(|_| {
                // C esl_getopts: verify_type_and_range, eslARG_REAL.
                cmdline_fail(&format!(
                    "Failed to parse command line: Option {} takes real-valued arg; got {} on cmdline",
                    $opt, s
                ))
            })
        }};
    }
    macro_rules! next_int {
        ($i:expr, $opt:expr) => {{
            let s = next_arg!($i, $opt);
            s.parse().unwrap_or_else(|_| {
                // C esl_getopts: verify_type_and_range, eslARG_INT.
                cmdline_fail(&format!(
                    "Failed to parse command line: Option {} takes integer arg; got {} on cmdline",
                    $opt, s
                ))
            })
        }};
    }
    while i < argv.len() {
        let arg = argv[i].clone();
        match arg.as_str() {
            "-F" => { o.force = true; o.used.insert("-F"); }
            "--hand" => { o.hand = true; o.used.insert("--hand"); }
            "-n" => { let s = next_arg!(i, &arg); o.name = Some(s); o.used.insert("-n"); }
            "-o" => { let s = next_arg!(i, &arg); o.ofile = Some(s); o.used.insert("-o"); }
            "-O" => { let s = next_arg!(i, &arg); o.omsafile = Some(s); o.used.insert("-O"); }
            // --- RSEARCH parameterization ---
            "--rsearch" => { let s = next_arg!(i, &arg); o.rsearch = Some(s); o.used.insert("--rsearch"); }
            // --- MSA clustering (build >1 CM per MSA) ---
            "--ctarget" => { o.ctarget = Some(next_int!(i, &arg)); o.used.insert("--ctarget"); }
            "--cmaxid" => { o.cmaxid = Some(next_real!(i, &arg)); o.used.insert("--cmaxid"); }
            "--call" => { o.call = true; o.used.insert("--call"); }
            "--corig" => { o.corig = true; o.used.insert("--corig"); }
            "--cdump" => { let s = next_arg!(i, &arg); o.cdump = Some(s); o.used.insert("--cdump"); }
            // weighting group (WGTOPTS; mutually exclusive, last one wins)
            "--wpb" => { o.wscheme = WScheme::Pb; o.used.insert("--wpb"); }
            "--wgsc" => { o.wscheme = WScheme::Gsc; o.used.insert("--wgsc"); }
            "--wnone" => { o.wscheme = WScheme::None; o.used.insert("--wnone"); }
            "--wgiven" => { o.wscheme = WScheme::Given; o.used.insert("--wgiven"); }
            "--wblosum" => { o.wscheme = WScheme::Blosum; o.used.insert("--wblosum"); }
            "--wid" => { o.wid = next_real!(i, &arg); o.used.insert("--wid"); }
            // effective seq number
            "--eent" => { o.used.insert("--eent"); } // default; accept
            "--enone" => { o.enone = true; o.used.insert("--enone"); }
            "--ere" => { o.ere = Some(next_real!(i, &arg)); o.used.insert("--ere"); }
            "--eset" => { o.eset = Some(next_real!(i, &arg)); o.used.insert("--eset"); }
            "--ehmmre" => { o.ehmmre = Some(next_real!(i, &arg)); o.used.insert("--ehmmre"); }
            "--eminseq" => { o.eminseq = next_real!(i, &arg); o.used.insert("--eminseq"); }
            "--emaxseq" => { o.emaxseq = Some(next_real!(i, &arg)); o.used.insert("--emaxseq"); }
            "--esigma" => { o.esigma = next_real!(i, &arg); o.used.insert("--esigma"); }
            // priors
            "--noh3pri" => { o.noh3pri = true; o.used.insert("--noh3pri"); }
            "--v1p0" => { o.v1p0 = true; o.used.insert("--v1p0"); }
            "--p56" => { o.p56 = true; o.used.insert("--p56"); }
            "--null" => { let f = next_arg!(i, &arg); o.null = read_null_model(&f); o.used.insert("--null"); o.null_file = Some(f); }
            // construction
            "--fast" => { o.used.insert("--fast"); } // default; accept
            "--symfrac" => { o.symfrac = next_real!(i, &arg); o.used.insert("--symfrac"); }
            "--fragthresh" => {
                // C FRAGOPTS toggle group {--fragthresh,--fragnrfpos,--fraggiven}
                // (cmbuild.c:36,55). esl_getopts.c set_option: if --fraggiven was
                // already set on cmdline it toggled --fragthresh's setby to
                // CMDLINE, so re-setting it here hits the "already been set" guard
                // (esl_getopts.c:1271) BEFORE the toggle-conflict check.
                if o.used.contains("--fraggiven") {
                    cmdline_fail("Failed to parse command line: Option --fragthresh has already been set on cmdline.");
                }
                o.fragthresh = next_real!(i, &arg); o.used.insert("--fragthresh");
            }
            "--fraggiven" => {
                // C FRAGOPTS toggle group. esl_getopts.c set_option: double-set
                // guard (esl_getopts.c:1271) first, then the toggle-tie loop
                // (esl_getopts.c:1318-1333). --fragthresh has a default val so if it
                // was explicitly set (setby==CMDLINE) the tie loop reports a
                // conflict; if only at its default it is silently toggled off.
                if o.used.contains("--fraggiven") {
                    cmdline_fail("Failed to parse command line: Option --fraggiven has already been set on cmdline.");
                }
                if o.used.contains("--fragthresh") {
                    cmdline_fail("Failed to parse command line: Options --fragthresh and --fraggiven conflict, toggling each other.");
                }
                o.fraggiven = true; o.used.insert("--fraggiven");
            }
            "--noss" => { o.noss = true; o.used.insert("--noss"); }
            "--nobalance" => { o.nobalance = true; o.used.insert("--nobalance"); }
            "--nodetach" => { o.nodetach = true; o.used.insert("--nodetach"); }
            "--iins" => { o.iins = true; o.used.insert("--iins"); }
            "--iflank" => { o.iflank = true; o.used.insert("--iflank"); }
            "--elself" => { o.elself = next_real!(i, &arg); o.used.insert("--elself"); }
            "--n2omega" => { o.n2omega = next_real!(i, &arg); o.used.insert("--n2omega"); }
            "--n3omega" => { o.n3omega = next_real!(i, &arg); o.used.insert("--n3omega"); }
            "--informat" => { let s = next_arg!(i, &arg); o.informat = Some(s); o.used.insert("--informat"); }
            "--cmtbl" => { let s = next_arg!(i, &arg); o.cmtbl = Some(s); o.used.insert("--cmtbl"); }
            "--emap" => { let s = next_arg!(i, &arg); o.emap = Some(s); o.used.insert("--emap"); }
            "--occfile" => { let s = next_arg!(i, &arg); o.occfile = Some(s); o.used.insert("--occfile"); }
            "--fp7occfile" => { let s = next_arg!(i, &arg); o.fp7occfile = Some(s); o.used.insert("--fp7occfile"); }
            // --- MSA refinement (--refine and its sub-options) ---
            "--refine" => { let s = next_arg!(i, &arg); o.refine = Some(s); o.used.insert("--refine"); }
            "-l" => { o.refine_local = true; o.used.insert("-l"); }
            "--gibbs" => { o.gibbs = true; o.used.insert("--gibbs"); }
            "--seed" => { o.seed = next_int!(i, &arg); o.used.insert("--seed"); }
            "--cyk" => { o.refine_cyk = true; o.used.insert("--cyk"); }
            "--notrunc" => { o.notrunc = true; o.used.insert("--notrunc"); }
            "--miss" => { o.miss = true; o.used.insert("--miss"); }
            "--sub" => { o.sub = true; o.used.insert("--sub"); }
            "--nonbanded" => { o.nonbanded = true; o.used.insert("--nonbanded"); }
            "--indi" => { o.indi = true; o.used.insert("--indi"); }
            "--fins" => { o.fins = true; o.used.insert("--fins"); }
            "--tau" => { o.tau = next_real!(i, &arg); o.used.insert("--tau"); }
            "--mxsize" => { o.mxsize = next_real!(i, &arg); o.used.insert("--mxsize"); }
            "--rdump" => { let s = next_arg!(i, &arg); o.rdump = Some(s); o.used.insert("--rdump"); }
            // C cmbuild.c:102-103 (HMM filter construction, docgroup 6).
            "--p7ml" => { o.p7ml = true; o.used.insert("--p7ml"); }
            "--p7ere" => { o.p7ere = Some(next_real!(i, &arg)); o.used.insert("--p7ere"); }
            // C cmbuild.c:111-114 (HMM filter calibration sample counts, docgroup 7).
            "--EmN" => { o.emn = next_int!(i, &arg); o.used.insert("--EmN"); }
            "--EvN" => { o.evn = next_int!(i, &arg); o.used.insert("--EvN"); }
            "--ElfN" => { o.elfn = next_int!(i, &arg); o.used.insert("--ElfN"); }
            "--EgfN" => { o.egfn = next_int!(i, &arg); o.used.insert("--EgfN"); }
            // C cmbuild option table has only "-h" (no "--help"); "--help" falls
            // through to the unrecognized-option error below.
            "-h" => full_help(),
            // C esl_getopts: unrecognized option (process_argument/is_opt).
            s if s.starts_with('-') => {
                cmdline_fail(&format!("Failed to parse command line: No such option \"{s}\"."))
            }
            s => positional.push(s.to_string()),
        }
        i += 1;
    }
    // C cmbuild.c:600-603: exactly 2 non-option arguments required.
    if positional.len() != 2 {
        cmdline_fail("Incorrect number of command line arguments.");
    }
    // C esl_opt_VerifyConfig (esl_getopts.c): the require loop runs first over
    // all options in table order, then the incompat loop over all options in
    // table order; the first violation aborts with the "Failed to parse command
    // line: ..." banner (routed through the process_commandline ERROR: block).
    // --- require loop (table order) ---
    let has_refine = o.refine.is_some();
    let req_refine = |set: bool, name: &str| {
        if set && !has_refine {
            cmdline_fail(&format!(
                "Failed to parse command line: Option {name} requires (or has no effect without) option(s) --refine"
            ));
        }
    };
    // Refine group declaration order: -l, --gibbs, --seed, --cyk, --notrunc,
    // --miss, --sub, --nonbanded, --indi, --fins, --tau, --mxsize, --rdump.
    req_refine(o.used.contains("-l"), "-l");
    req_refine(o.used.contains("--gibbs"), "--gibbs");
    if o.used.contains("--seed") && !o.used.contains("--gibbs") {
        cmdline_fail(
            "Failed to parse command line: Option --seed requires (or has no effect without) option(s) --gibbs",
        );
    }
    req_refine(o.used.contains("--cyk"), "--cyk");
    req_refine(o.used.contains("--notrunc"), "--notrunc");
    req_refine(o.used.contains("--miss"), "--miss");
    req_refine(o.used.contains("--sub"), "--sub");
    req_refine(o.used.contains("--nonbanded"), "--nonbanded");
    req_refine(o.used.contains("--indi"), "--indi");
    req_refine(o.used.contains("--fins"), "--fins");
    req_refine(o.used.contains("--tau"), "--tau");
    req_refine(o.used.contains("--mxsize"), "--mxsize");
    req_refine(o.used.contains("--rdump"), "--rdump");
    // --- incompat loop (table order). --p7ere (cmbuild.c:102) is declared before
    // the refine/cluster groups, so its incompat check (with --p7ml) fires first.
    if o.p7ere.is_some() && o.p7ml {
        cmdline_fail(
            "Failed to parse command line: Option --p7ere is incompatible with option(s) --p7ml",
        );
    }
    // --ctarget/--cmaxid (docgroup 110), then --sub, then --tau. ---
    if o.ctarget.is_some() && o.call {
        cmdline_fail("Failed to parse command line: Option --ctarget is incompatible with option(s) --call");
    }
    if o.cmaxid.is_some() && o.call {
        cmdline_fail("Failed to parse command line: Option --cmaxid is incompatible with option(s) --call");
    }
    if o.used.contains("--sub") && (o.used.contains("--notrunc") || o.used.contains("-l")) {
        cmdline_fail(
            "Failed to parse command line: Option --sub is incompatible with option(s) --notrunc,-l",
        );
    }
    if o.used.contains("--tau") && o.used.contains("--nonbanded") {
        cmdline_fail(
            "Failed to parse command line: Option --tau is incompatible with option(s) --nonbanded",
        );
    }
    // Runtime cm_Fail checks (NOT esl_getopts errors; stderr "\nError: ..." with
    // no usage banner). C cmbuild.c:867 (--corig), :951 (--cdump). The
    // "More than one of --ctarget, --cmaxid, --call" guard (cmbuild.c:487) fires
    // later, after output_header, and is handled in main().
    let nmodes = o.ctarget.is_some() as i32 + o.cmaxid.is_some() as i32 + o.call as i32;
    if o.corig && nmodes == 0 {
        cm_fail("--corig only makes sense in combination with --ctarget, --cmaxid, OR --call");
    }
    if o.cdump.is_some() && nmodes == 0 {
        cm_fail("--cdump only makes sense in combination with --ctarget, --cmaxid, OR --call");
    }

    o.cmfile = positional[0].clone();
    o.msafile = positional[1].clone();
    // C build_model:1830-1841: if --n2omega/--n3omega not set, --p56 selects
    // V1P0_NULL2/3_OMEGA (1/32); otherwise the default 0.000015258791 stands.
    if o.p56 {
        if !o.used.contains("--n2omega") {
            o.n2omega = V1P0_NULL_OMEGA;
        }
        if !o.used.contains("--n3omega") {
            o.n3omega = V1P0_NULL_OMEGA;
        }
    }
    o
}

// ---------------------------------------------------------------------------
// stdout build summary (cmbuild.c: output_header / print_column_headings /
// output_result) — byte-identical column layout.
// ---------------------------------------------------------------------------

/// esl_vec_FRelEntropy (esl_vectorops.c), the KL divergence sum in bits.
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

/// eweight.c:378 cm_MeanMatchRelativeEntropy.
fn cm_mean_match_relative_entropy(cm: &CM) -> f64 {
    let k = cm.null.len();
    let mut pair_null = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            pair_null[i * k + j] = cm.null[i] * cm.null[j];
        }
    }
    let mut kl = 0.0f64;
    for v in 0..(cm.m as usize) {
        let stid = cm.stid[v] as i32;
        if stid == infernox::constants::MATP_MP {
            kl += esl_vec_f_rel_entropy(&cm.e[v], &pair_null, k * k) as f64;
        } else if stid == infernox::constants::MATL_ML || stid == infernox::constants::MATR_MR {
            kl += esl_vec_f_rel_entropy(&cm.e[v], &cm.null, k) as f64;
        }
    }
    kl / (cm.clen as f64)
}

/// eweight.c:669 cp9_MeanMatchRelativeEntropy.
fn cp9_mean_match_relative_entropy(cp9: &infernox::cp9::CP9) -> f64 {
    let k = cp9.null.len();
    let mut kl = 0.0f64;
    for kpos in 1..=(cp9.m as usize) {
        kl += esl_vec_f_rel_entropy(&cp9.mat[kpos], &cp9.null, k) as f64;
    }
    kl / (cp9.m as f64)
}

fn cm_count_statetype(cm: &CM, sttype: i32) -> i32 {
    (0..(cm.m as usize)).filter(|&v| cm.sttype[v] as i32 == sttype).count() as i32
}

/// Build the global CP9 ML HMM from a fully-parameterized CM (as cmstat does),
/// so we can report cp9_MeanMatchRelativeEntropy for the summary's HMM column.
fn build_cp9(cm: &CM) -> infernox::cp9::CP9 {
    let emap = infernox::cp9::create_emit_map(cm);
    let map = infernox::cp9::cp9_map_cm2hmm(cm);
    let psi = infernox::cp9::cm_expected_state_occupancy(cm);
    let tmap = infernox::cp9::cm_create_transition_map();
    infernox::cp9::cp9_build_and_configure_global(cm, &emap, &map, &psi, &tmap)
}

/// C printf "%g" (default precision 6): shortest of %e/%f, trailing zeros and a
/// bare decimal point stripped. Used for a handful of output_header echoes.
fn g_fmt(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    // C %g uses %e if exp < -4 or exp >= precision(6), else %f.
    if exp < -4 || exp >= 6 {
        // %e with precision 5 (P-1), then strip trailing zeros in mantissa.
        let s = format!("{:.5e}", x);
        // Rust "{:e}" -> e.g. "1.52588e-5"; C wants "1.52588e-05" (2-digit exp).
        let (mant, e) = s.split_once('e').unwrap();
        let mant = strip_trailing_zeros(mant);
        let ei: i32 = e.parse().unwrap();
        format!("{}e{}{:02}", mant, if ei < 0 { "-" } else { "+" }, ei.abs())
    } else {
        // %f with precision (6 - 1 - exp) significant→ digits after point = 5 - exp
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

/// C: output_header (cmbuild.c:662). Writes banner + echoed options + separator.
fn output_header(w: &mut dyn Write, opts: &Opts) {
    let u = &opts.used;
    let _ = writeln!(w, "# cmbuild :: covariance model construction from multiple sequence alignments");
    let _ = writeln!(w, "# INFERNAL 1.1.5 (Sep 2023)");
    let _ = writeln!(w, "# Copyright (C) 2023 Howard Hughes Medical Institute.");
    let _ = writeln!(w, "# Freely distributed under the BSD open source license.");
    let _ = writeln!(w, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
    let _ = writeln!(w, "# CM file:                                            {}", opts.cmfile);
    let _ = writeln!(w, "# alignment file:                                     {}", opts.msafile);
    if u.contains("-n") { let _ = writeln!(w, "# name (the single) CM:                               {}", opts.name.as_deref().unwrap_or("")); }
    if u.contains("-F") { let _ = writeln!(w, "# overwrite CM file if necessary:                     yes"); }
    if u.contains("-o") { let _ = writeln!(w, "# output directed to file:                            {}", opts.ofile.as_deref().unwrap_or("")); }
    if u.contains("-O") { let _ = writeln!(w, "# processed alignment resaved to:                     {}", opts.omsafile.as_deref().unwrap_or("")); }
    if u.contains("--symfrac") { let _ = writeln!(w, "# minimum symbol fraction in a consensus column:      {}", g_fmt(opts.symfrac)); }
    // C cmbuild.c:672: if(esl_opt_IsUsed(go,"--fragthresh")). esl_opt_IsUsed is
    // FALSE when the value equals its default (esl_getopts.c esl_opt_IsDefault
    // string-compares val vs defval "0.5"): passing "--fragthresh 0.5" prints
    // nothing; only a non-default value emits the line.
    if opts.fragthresh != 0.5 { let _ = writeln!(w, "# seq called frag if L <= x*alen:                     {:.3}", opts.fragthresh); }
    if u.contains("--hand") { let _ = writeln!(w, "# use #=GC RF annotation to define consensus columns: yes"); }
    if u.contains("--null") { let _ = writeln!(w, "# read null model from file:                          {}", opts.null_file.as_deref().unwrap_or("")); }
    if u.contains("--noss") { let _ = writeln!(w, "# ignore secondary structure, if any:                 yes"); }
    if u.contains("--rsearch") { let _ = writeln!(w, "# RSEARCH parameterization mode w/RIBOSUM mx file:    {}", opts.rsearch.as_deref().unwrap_or("")); }
    if u.contains("--informat") { let _ = writeln!(w, "# input format specified as:                          {}", opts.informat.as_deref().unwrap_or("")); }
    if u.contains("--v1p0") { let _ = writeln!(w, "# v1.0 parameterization mode:                         on"); }
    if u.contains("--p56") { let _ = writeln!(w, "# use default priors from v0.56 through v1.0.2:       yes"); }
    if u.contains("--iins") { let _ = writeln!(w, "# allowing informative insert emission probabilities: yes"); }
    if u.contains("--iflank") { let _ = writeln!(w, "# allowing informative ROOT_IL/IR transition probs:   yes"); }
    if u.contains("--nobalance") { let _ = writeln!(w, "# do default rebalancing of CM:                       no"); }
    if u.contains("--nodetach") { let _ = writeln!(w, "# do default detachment of flawed ambiguous inserts:  no"); }
    if u.contains("--elself") { let _ = writeln!(w, "# local end (EL) self loop probability:               {}", g_fmt(opts.elself)); }
    if u.contains("--n2omega") { let _ = writeln!(w, "# prior probability of null2 model (if used):         {}", g_fmt(opts.n2omega)); }
    if u.contains("--n3omega") { let _ = writeln!(w, "# prior probability of null3 model (if used):         {}", g_fmt(opts.n3omega)); }
    // C cmbuild.c:692: if(esl_opt_IsUsed(go,"--wpb")). --wpb is the default
    // weighting scheme (WGTOPTS toggle default "default"), so esl_opt_IsUsed is
    // always FALSE (val==defval even when explicitly passed) -> the line is
    // never emitted. Only the non-default schemes below print.
    if u.contains("--wgsc") { let _ = writeln!(w, "# relative weighting scheme:                          G/S/C"); }
    if u.contains("--wnone") { let _ = writeln!(w, "# relative weighting scheme:                          none"); }
    if u.contains("--wgiven") { let _ = writeln!(w, "# relative weighting scheme:                          wts from MSA file"); }
    if u.contains("--wblosum") { let _ = writeln!(w, "# relative weighting scheme:                          BLOSUM filter"); }
    if u.contains("--wid") { let _ = writeln!(w, "# frac id cutoff for BLOSUM wgts:                     {:.6}", opts.wid); }
    // C cmbuild.c:699: if(esl_opt_IsUsed(go,"--eent")). --eent is the default
    // effective-seq scheme (EFFOPTS toggle default), so esl_opt_IsUsed is always
    // FALSE and the line is never emitted. Only --enone/--eset below print.
    if u.contains("--enone") { let _ = writeln!(w, "# effective seq number scheme:                        none"); }
    if u.contains("--ere") { let _ = writeln!(w, "# minimum rel entropy target:                         {:.6} bits", opts.ere.unwrap()); }
    if u.contains("--eset") { let _ = writeln!(w, "# effective seq number:                               set to {:.6}", opts.eset.unwrap()); }
    if u.contains("--eminseq") { let _ = writeln!(w, "# minimum effective sequence number allowed:          {}", g_fmt(opts.eminseq)); }
    if u.contains("--emaxseq") { let _ = writeln!(w, "# maximum effective sequence number allowed:          {}", g_fmt(opts.emaxseq.unwrap())); }
    if u.contains("--ehmmre") { let _ = writeln!(w, "# minimum ML CP9 HMM rel entropy target:              {:.6} bits", opts.ehmmre.unwrap()); }
    if u.contains("--esigma") { let _ = writeln!(w, "# entropy target sigma parameter:                     {:.6} bits", opts.esigma); }
    if u.contains("--cmtbl") { let _ = writeln!(w, "# saving tabular description of CM topology to file:  {}", opts.cmtbl.as_deref().unwrap_or("")); }
    if u.contains("--emap") { let _ = writeln!(w, "# saving consensus emit map to file:                  {}", opts.emap.as_deref().unwrap_or("")); }
    if u.contains("--occfile") { let _ = writeln!(w, "# saving CM expected occupancy values to file:        {}", opts.occfile.as_deref().unwrap_or("")); }
    if u.contains("--fp7occfile") { let _ = writeln!(w, "# saving filter P7 expected occupancy values to file: {}", opts.fp7occfile.as_deref().unwrap_or("")); }
    // --- MSA refinement options (C cmbuild.c:708-723) ---
    if u.contains("--refine") { let _ = writeln!(w, "# input alignment refinement prior to model building: on"); }
    if u.contains("-l") { let _ = writeln!(w, "# model configuration for aln refinement:             local"); }
    if u.contains("--gibbs") { let _ = writeln!(w, "# Gibbs sampling (instead of EM) for aln refinement:  on"); }
    if u.contains("--seed") {
        if opts.seed == 0 { let _ = writeln!(w, "# random number seed:                                  one-time arbitrary"); }
        else { let _ = writeln!(w, "# random number seed set to:                           {}", opts.seed); }
    }
    if u.contains("--notrunc") { let _ = writeln!(w, "# use truncated aln algorithms for aln refinement:    no"); }
    if u.contains("--cyk") { let _ = writeln!(w, "# use the CYK algorithm instead of optimal accuracy:  yes"); }
    if u.contains("--sub") { let _ = writeln!(w, "# alternative truncated seq alignment 'sub' mode:     on"); }
    if u.contains("--nonbanded") { let _ = writeln!(w, "# use HMM bands for accelerating aln refinement:      no"); }
    if u.contains("--indi") { let _ = writeln!(w, "# print individual seq scores during aln refinement:  yes"); }
    if u.contains("--fins") { let _ = writeln!(w, "# flush inserts left/rigth during aln refinement:     yes"); }
    if u.contains("--tau") { let _ = writeln!(w, "# tail loss probability for HMM bands set to:         {}", g_fmt(opts.tau)); }
    if u.contains("--mxsize") { let _ = writeln!(w, "# maximum DP matrix size set to:                      {:.2} Mb", opts.mxsize); }
    if u.contains("--rdump") { let _ = writeln!(w, "# printing intermediate alns during aln refnment to:  {}", opts.rdump.as_deref().unwrap_or("")); }
    // HMM filter construction options (C cmbuild.c:725-726). %g for --p7ere.
    if u.contains("--p7ml") { let _ = writeln!(w, "# filter HMM is ML HMM created from CM:               yes"); }
    if u.contains("--p7ere") { let _ = writeln!(w, "# filter HMM minimum rel entropy target:              {} bits", g_fmt(opts.p7ere.unwrap())); }
    // HMM filter calibration sample counts (C cmbuild.c:731-734). esl_opt_IsUsed
    // string-compares val vs the default "200", so passing "--EmN 200" prints
    // nothing; only a non-default count emits the line (as with --fragthresh).
    if opts.emn != 200 { let _ = writeln!(w, "# seq number for filter HMM MSV Gumbel mu fit:        {}", opts.emn); }
    if opts.evn != 200 { let _ = writeln!(w, "# seq number for filter HMM Vit Gumbel mu fit:        {}", opts.evn); }
    if opts.elfn != 200 { let _ = writeln!(w, "# seq number for filter HMM local Fwd Gumbel mu fit:  {}", opts.elfn); }
    if opts.egfn != 200 { let _ = writeln!(w, "# seq number for filter HMM glocal Fwd Gumbel mu fit: {}", opts.egfn); }
    // Clustering options (C cmbuild.c:758-762). %g for --cmaxid.
    if u.contains("--ctarget") { let _ = writeln!(w, "# building >1 CMs from each MSA; target num CMs:      {}", opts.ctarget.unwrap()); }
    if u.contains("--cmaxid") { let _ = writeln!(w, "# building >1 CMs from each MSA; max id b/t clusters: {}", g_fmt(opts.cmaxid.unwrap())); }
    if u.contains("--call") { let _ = writeln!(w, "# building a CM from each sequence in each MSA:       yes"); }
    if u.contains("--corig") { let _ = writeln!(w, "# appending original CM to cluster CMs:               yes"); }
    if u.contains("--cdump") { let _ = writeln!(w, "# writing training alns for each cluster CM to file:  {}", opts.cdump.as_deref().unwrap_or("")); }
    let _ = writeln!(w, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
}

/// C: UniqueStatetype (cm.c). stid -> unique-state name.
fn uniquestatetype(stid: i32) -> &'static str {
    use infernox::constants::*;
    const END_EL: i32 = 20;
    match stid {
        -1 => "DUMMY",
        ROOT_S => "ROOT_S",
        ROOT_IL => "ROOT_IL",
        ROOT_IR => "ROOT_IR",
        BEGL_S => "BEGL_S",
        BEGR_S => "BEGR_S",
        BEGR_IL => "BEGR_IL",
        MATP_MP => "MATP_MP",
        MATP_ML => "MATP_ML",
        MATP_MR => "MATP_MR",
        MATP_D => "MATP_D",
        MATP_IL => "MATP_IL",
        MATP_IR => "MATP_IR",
        MATL_ML => "MATL_ML",
        MATL_D => "MATL_D",
        MATL_IL => "MATL_IL",
        MATR_MR => "MATR_MR",
        MATR_D => "MATR_D",
        MATR_IR => "MATR_IR",
        END_E => "END_E",
        BIF_B => "BIF_B",
        END_EL => "END_EL",
        _ => panic!("bogus unique state type {stid}"),
    }
}

/// C: Nodetype (cm.c). ndtype -> node name; "-" for DUMMY (-1).
fn nodetype_str(ndtype: i32) -> &'static str {
    if ndtype < 0 {
        "-"
    } else {
        infernox::cm::NODE_TYPE_NAMES[ndtype as usize]
    }
}

/// C: PrintCM (cm.c:1315). Tabular description of CM topology (--cmtbl).
fn dump_cmtbl(w: &mut dyn Write, cm: &CM) {
    let _ = writeln!(w, "{:>5} {:>6} {:>5} {:>6} {:>7} {:>6} {:>5} {:>5} {:>5}",
        " idx ", "sttype", "ndidx", "ndtype", "  stid ", "cfirst", " cnum", "plast", " pnum");
    let _ = writeln!(w, "{:>5} {:>6} {:>5} {:>6} {:>7} {:>5} {:>5} {:>5} {:>5}",
        "-----", "------", "-----", "------", "-------", "------", "-----", "-----", "-----");
    for x in 0..(cm.m as usize) {
        let _ = writeln!(w, "{:>5} {:<6} {:>5} {:>6} {:<7} {:>6} {:>5} {:>5} {:>5}",
            x,
            infernox::cm::STATE_TYPE_NAMES[cm.sttype[x] as usize],
            cm.ndidx[x],
            nodetype_str(cm.ndtype[cm.ndidx[x] as usize] as i32),
            uniquestatetype(cm.stid[x] as i32),
            cm.cfirst[x],
            cm.cnum[x],
            cm.plast[x],
            cm.pnum[x]);
    }
}

/// C: DumpEmitMap (display.c:1231). CM-to-consensus emit map (--emap).
fn dump_emap(w: &mut dyn Write, cm: &CM) {
    let map = infernox::cm_emitmap::create_emit_map(cm).expect("emit map");
    let _ = writeln!(w, "CM to consensus emit map; consensus length = {} ", map.clen);
    let _ = writeln!(w, "{:>4} {:>7} {:>9} {:>4} {:>4} {:>4}", "Node", "State 1", "Node type", "lpos", "rpos", "epos");
    let _ = writeln!(w, "{:>4} {:>7} {:>9} {:>4} {:>4} {:>4}", "----", "-------", "---------", "----", "----", "----");
    for nd in 0..(cm.nodes as usize) {
        let _ = writeln!(w, "{:>4} {:>7} {:>9} {:>4} {:>4} {:>4}",
            nd,
            cm.nodemap[nd],
            nodetype_str(cm.ndtype[nd] as i32),
            map.lpos[nd],
            map.rpos[nd],
            map.epos[nd]);
    }
}

/// C: dump_cm_occupancy_values (cmbuild.c). Expected occupancy per CM state.
fn dump_cm_occupancy(w: &mut dyn Write, cm: &CM) {
    let psi = infernox::cp9::cm_expected_state_occupancy(cm);
    let _ = writeln!(w, "# model_name: {}", cm.name);
    let _ = writeln!(w, "# number_of_states: {}", cm.m);
    let _ = writeln!(w, "# columns: <state_idx> <state_expected_occupancy>");
    for v in 0..(cm.m as usize) {
        let _ = writeln!(w, "{} {:.5}", v, psi[v]);
    }
    let _ = writeln!(w, "//");
}

/// C: dump_fp7_occupancy_values (cmbuild.c). Expected occupancy per filter-P7 node.
fn dump_fp7_occupancy(w: &mut dyn Write, name: &str, p7: &infernox::p7_hmm::P7Profile) {
    let (mocc, iocc) = infernox::p7_filter_emit::p7_hmm_calculate_occupancy(p7);
    let _ = writeln!(w, "# model_name: {}", name);
    let _ = writeln!(w, "# number_of_nodes: {}", p7.m);
    let _ = writeln!(w, "# columns: <node_idx> <expected_occupancy_match> <expected_occupancy_insert> <expected_occupancy_delete>");
    for k in 0..=(p7.m as usize) {
        let _ = writeln!(w, "{} {:.5} {:.5} {:.5}", k, mocc[k], iocc[k], 1.0 - mocc[k]);
    }
    let _ = writeln!(w, "//");
}

/// C: print_column_headings (cmbuild.c:1254).
fn print_column_headings(w: &mut dyn Write) {
    let _ = writeln!(w, "# {:<6} {:<20} {:>8} {:>8} {:>6} {:>5} {:>4} {:>4} {:>11}", "", "", "", "", "", "", "", "", "rel entropy");
    let _ = writeln!(w, "# {:<6} {:<20} {:>8} {:>8} {:>6} {:>5} {:>4} {:>4} {:>11}", "", "", "", "", "", "", "", "", "-----------");
    let _ = writeln!(w, "# {:<6} {:<20} {:>8} {:>8} {:>6} {:>5} {:>4} {:>4} {:>5} {:>5} {}", "idx", "name", "nseq", "eff_nseq", "alen", "clen", "bps", "bifs", "CM", "HMM", "description");
    let _ = writeln!(w, "# {:<6} {:<20} {:>8} {:>8} {:>6} {:>5} {:>4} {:>4} {:>5} {:>5} {}", "------", "--------------------", "--------", "--------", "------", "-----", "----", "----", "-----", "-----", "-----------");
}

fn main() {
    let opts = parse_args();

    if !opts.force && std::path::Path::new(&opts.cmfile).exists() {
        eprintln!("cmfile {} already exists; use -F to force overwrite", opts.cmfile);
        std::process::exit(1);
    }

    // C: --informat decode + validate (main, cmbuild.c:319-325).
    if let Some(ref fmt) = opts.informat {
        let f = fmt.to_ascii_lowercase();
        match f.as_str() {
            "stockholm" | "pfam" => {}
            "selex" => {
                eprintln!("infernox-cmbuild: --informat selex input reading is not yet supported");
                std::process::exit(1);
            }
            _ => {
                eprintln!("{fmt} is not a recognized/valid input format for cmbuild (must be Stockholm, Pfam, or Selex)");
                std::process::exit(1);
            }
        }
    }

    let abc = EslAlphabet::rna();

    // --- init_cfg: if --rsearch, read + probify the RIBOSUM matrix (cmbuild.c:852-861) ---
    let fullmat = opts.rsearch.as_ref().map(|matfile| {
        let matbytes = std::fs::read(matfile).unwrap_or_else(|e| {
            eprintln!("Failed to open matrix file {matfile}: {e}");
            std::process::exit(1);
        });
        let mut fm = infernox::rsearch::read_matrix(&abc, &matbytes);
        // cmbuild.c:860: overwrite score matrix scores w/target probs.
        infernox::rsearch::ribosum_calc_targets(&mut fm, &abc);
        fm
    });

    let input = match std::fs::read_to_string(&opts.msafile) {
        Ok(s) => s,
        Err(e) => { eprintln!("failed to read {}: {e}", opts.msafile); std::process::exit(1); }
    };
    let mut msas = match infernox::easel::msafile::read_all(&input, Some(&abc)) {
        Ok(m) => m,
        Err(e) => { eprintln!("failed to parse MSA {}: {e:?}", opts.msafile); std::process::exit(1); }
    };

    let mut out: Box<dyn Write> = Box::new(
        std::fs::File::create(&opts.cmfile).unwrap_or_else(|e| {
            eprintln!("cannot create {}: {e}", opts.cmfile);
            std::process::exit(1);
        }),
    );

    // Summary output: stdout, or -o <f>. (C: cfg->ofp)
    let mut sumbuf: Vec<u8> = Vec::new();
    output_header(&mut sumbuf, &opts);

    // C cmbuild.c:1328-1374 (-O): the consensus/insert-annotated Stockholm MSA
    // for each built CM is accumulated here (cfg->postmsafp is opened once and
    // each CM's omsa is written to it sequentially) and flushed to the -O file
    // at the end. Empty unless -O was used.
    let mut omsa_buf: Vec<u8> = Vec::new();

    // C cmbuild.c:487: after output_header, the "More than one of --ctarget,
    // --cmaxid, --call" guard cm_Fails. (--*+--call are already blocked by
    // esl_getopts incompat; only --ctarget together with --cmaxid reaches here.)
    // C has already written the header to cfg->ofp, so flush the accumulated
    // header (stdout or -o) first, then emit cm_Fail's stderr line and exit 1.
    if opts.ctarget.is_some() && opts.cmaxid.is_some() {
        match &opts.ofile {
            Some(f) => {
                let _ = std::fs::write(f, &sumbuf);
            }
            None => {
                let _ = std::io::stdout().write_all(&sumbuf);
            }
        }
        cm_fail("More than one of --ctarget, --cmaxid, --call were enabled, shouldn't happen.");
    }

    // --refine: Stage 1b now fully supports --nonbanded in both configurations:
    // C sets align_opts = SMALL|NONBANDED|CYK and dispatches through do_small
    // (cm_alndata.c:377-388): do_small && !do_trunc -> CYKDivideAndConquer (Rust
    // cyk_divide_and_conquer); do_small && do_trunc (the default) -> TrCYK_DnC (Rust
    // truncyk::tr_cyk_dnc). Both are byte-faithful ports, so no gating is needed.
    // --refine output file (C: cfg->refinefp, opened once). --rdump likewise.
    let mut refinefp: Option<Box<dyn Write>> = match &opts.refine {
        Some(path) => Some(Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Failed to open output file {path} for writing MSAs from --refine to: {e}");
            std::process::exit(1);
        }))),
        None => None,
    };
    let mut rdfp: Option<Box<dyn Write>> = match &opts.rdump {
        Some(path) => Some(Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Failed to open output file {path} for writing intermediate MSAs to: {e}");
            std::process::exit(1);
        }))),
        None => None,
    };

    // Group C dump-file buffers (appended per model, written at the end).
    let mut cmtbl_buf: Vec<u8> = Vec::new();
    let mut emap_buf: Vec<u8> = Vec::new();
    let mut occ_buf: Vec<u8> = Vec::new();
    let mut fp7occ_buf: Vec<u8> = Vec::new();

    let comlog = std::env::args().collect::<Vec<_>>().join(" ");

    // C: do_cluster = --ctarget || --cmaxid || --call (cmbuild.c:486).
    let do_cluster = opts.ctarget.is_some() || opts.cmaxid.is_some() || opts.call;
    // C: nc = --ctarget value; mindiff = 1 - --cmaxid (cmbuild.c:489-490).
    let nc_target = opts.ctarget.unwrap_or(0);
    let mindiff_opt = opts.cmaxid.map(|x| 1.0 - x).unwrap_or(0.0);

    // --cdump: open the cluster-MSA dump file (truncate).
    let mut cdfp: Option<Box<dyn Write>> = match &opts.cdump {
        Some(path) => Some(Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Failed to open output file {path} for writing MSAs to: {e}");
            std::process::exit(1);
        }))),
        None => None,
    };

    // C: cfg->ncm_total, the 1-based running index across ALL CMs built (may be
    // >1 per MSA under clustering). Used as the summary "idx" column and to print
    // the column headings once (msaidx==1 && cmidx==1 -> ncm_total==1).
    let mut ncm_total = 0i32;

    for (idx, base_msa) in msas.iter_mut().enumerate() {
        if base_msa.name.is_none() {
            base_msa.name = Some(opts.name.clone().unwrap_or_else(|| {
                std::path::Path::new(&opts.msafile)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("aln{}", idx + 1))
            }));
        }
        if let Some(ref n) = opts.name {
            base_msa.name = Some(n.clone());
        }

        // C: if(do_cluster) divide the master MSA into per-cluster MSAs; else
        // the single master MSA is the only one built (cmbuild.c:501-506).
        let cmsa: Vec<EslMsa> = if do_cluster {
            msa_divide(
                base_msa,
                &abc,
                opts.call,
                opts.cmaxid.is_some(),
                opts.ctarget.is_some(),
                mindiff_opt,
                nc_target,
                opts.corig,
            )
        } else {
            vec![base_msa.clone()]
        };

        // C: for(c = 0; c < ncm; c++) { ... } (cmbuild.c:507).
        for mut msa in cmsa {
            ncm_total += 1; // C: cfg->ncm_total++ (start of the per-CM loop)
            // C: if(do_cluster && --cdump) write this cluster's MSA, before build.
            if do_cluster {
                if let Some(fp) = cdfp.as_mut() {
                    if let Err(e) = infernox::easel::msafile::esl_msafile_write(
                        fp,
                        &msa,
                        infernox::easel::msafile::MsaFormat::Stockholm,
                    ) {
                        eprintln!("--cdump related esl_msafile_Write() call failed: {e}");
                        std::process::exit(1);
                    }
                }
            }

            // C process_build_workunit (cmbuild.c:995-997): with --fraggiven,
            // check_fragments() VALIDATES the MSA's given ~ annotation (it does
            // not modify the MSA) BEFORE build_model. It runs after
            // set_relative_weights in C, but is independent of the weights, so we
            // run it here (weighting happens inside build_one below). On failure C
            // cm_Fails; the header was already written to cfg->ofp, so flush the
            // accumulated header (stdout or -o) first, then emit "\nError: ...".
            if opts.fraggiven {
                if let Err(msg) = mm::check_fragments(&msa) {
                    match &opts.ofile {
                        Some(f) => {
                            let _ = std::fs::write(f, &sumbuf);
                        }
                        None => {
                            let _ = std::io::stdout().write_all(&sumbuf);
                        }
                    }
                    eprintln!("\nError: {msg}");
                    std::process::exit(1);
                }
            }

            // C build_model (cmbuild.c:1402): --hand requires #=GC RF annotation.
            // C's output_header already wrote the banner to cfg->ofp before this
            // fatal, so flush the accumulated header (stdout or -o) to match, then
            // emit cm_Fail's "\nError: ...\n" to stderr and exit 1.
            if opts.hand && msa.rf.is_none() {
                match &opts.ofile {
                    Some(f) => {
                        let _ = std::fs::write(f, &sumbuf);
                    }
                    None => {
                        let _ = std::io::stdout().write_all(&sumbuf);
                    }
                }
                eprintln!(
                    "\nError: --hand used, but alignment #{} has no reference coord annotation",
                    idx as i32 + 1
                );
                std::process::exit(1);
            }

            // fullmat = Some(..) only under --rsearch (RIBOSUM parameterization).
            let (mut cm, _gtr, trs) = build_one(&mut msa, &abc, &opts, &comlog, fullmat.as_ref());

            // C cmbuild.c:1328-1374 (-O): resave a consensus/insert-annotated
            // Stockholm MSA (WT + RF), built by realigning the input parsetrees.
            // Placed here (before refinement) so it uses the just-built <msa>/<trs>;
            // this is byte-faithful when --refine is not used. (With --refine, C
            // reassigns tr=new_tr and uses the refined alignment; that combination
            // is not yet reproduced here.)
            if opts.omsafile.is_some() && opts.refine.is_none() {
                append_omsa(&mut omsa_buf, &cm, &abc, &msa, &trs);
            }

            // --- optional MSA refinement (C cmbuild.c:533-548) ---
            if opts.refine.is_some() {
                let nali = idx as i32 + 1; // C cfg->nali (1-based MSA index)
                let _ = writeln!(sumbuf, "#");
                let _ = writeln!(
                    sumbuf,
                    "# Refining MSA for CM: {} (aln: {:>4} cm: {:>6})",
                    cm.name, nali, ncm_total
                );
                let (rcm, rmsa) = refine_msa(
                    &opts,
                    &abc,
                    &comlog,
                    cm,
                    msa,
                    trs,
                    &mut sumbuf,
                    &mut refinefp,
                    &mut rdfp,
                );
                cm = rcm;
                msa = rmsa;
            }

            if let Err(e) = cm.cm_validate(0.0001) {
                eprintln!("CM validation failed: {e}");
                std::process::exit(1);
            }

            // Pristine copy of the (possibly refined) MSA for the temp-CM emission
            // build (Stage B rebuilds the full pipeline internally).
            let mut msa_pristine = msa.clone();

            // --- build_and_calibrate_p7_filter (cmbuild.c:2269) ---
            attach_p7_filter(&mut cm, &msa, &mut msa_pristine, &abc, &opts, &comlog, fullmat.as_ref());

            // C output_result (cmbuild.c:1266): print column headings once, before
            // the first CM's summary line (msaidx==1 && cmidx==1 -> ncm_total==1),
            // then write the CM.
            if ncm_total == 1 {
                print_column_headings(&mut sumbuf);
            }
            if let Err(e) = infernox::cm_file::cm_file_write_ascii(&mut out, &cm) {
                eprintln!("CM save failed: {e}");
                std::process::exit(1);
            }

            // rel-entropy columns: CM (marginalized) + CP9 ML HMM.
            let cm_re = cm_mean_match_relative_entropy(&cm);
            let cp9 = build_cp9(&cm);
            let hmm_re = cp9_mean_match_relative_entropy(&cp9);

            // Group C dump files (C: output_result).
            if opts.cmtbl.is_some() {
                dump_cmtbl(&mut cmtbl_buf, &cm);
            }
            if opts.emap.is_some() {
                dump_emap(&mut emap_buf, &cm);
            }
            if opts.occfile.is_some() {
                dump_cm_occupancy(&mut occ_buf, &cm);
            }
            if opts.fp7occfile.is_some() {
                if let Some(ref p7) = cm.p7 {
                    dump_fp7_occupancy(&mut fp7occ_buf, &cm.name, p7);
                }
            }
            let bps = cm_count_statetype(&cm, infernox::constants::MP_ST);
            let bifs = cm_count_statetype(&cm, infernox::constants::B_ST);
            // C: "%8d %-20s %8d %8.2f %6"PRId64" %5d %4d %4d %5.3f %5.3f %s\n"
            let _ = writeln!(
                sumbuf,
                "{:>8} {:<20} {:>8} {:>8.2} {:>6} {:>5} {:>4} {:>4} {:>5.3} {:>5.3} {}",
                ncm_total,
                cm.name,
                msa.nseq,
                cm.eff_nseq,
                msa.alen,
                cm.clen,
                bps,
                bifs,
                cm_re,
                hmm_re,
                msa.desc.as_deref().unwrap_or(""),
            );
        }
    }

    // Write Group C dump files.
    if let Some(ref f) = opts.cmtbl {
        let _ = std::fs::write(f, &cmtbl_buf);
    }
    if let Some(ref f) = opts.emap {
        let _ = std::fs::write(f, &emap_buf);
    }
    if let Some(ref f) = opts.occfile {
        let _ = std::fs::write(f, &occ_buf);
    }
    if let Some(ref f) = opts.fp7occfile {
        let _ = std::fs::write(f, &fp7occ_buf);
    }

    // C main cleanup (cmbuild.c:370-428): if any output file was opened, print a
    // "#\n" then a "saved to file" message per file, in the fixed C order.
    let any_dump = opts.omsafile.is_some()
        || opts.cmtbl.is_some()
        || opts.emap.is_some()
        || opts.occfile.is_some()
        || opts.fp7occfile.is_some()
        || opts.refine.is_some()
        || opts.rdump.is_some();
    if any_dump {
        let _ = writeln!(sumbuf, "#");
    }
    // C cmbuild.c:373 — the -O (postmsafp) message is the FIRST saved-file line.
    if let Some(ref f) = opts.omsafile {
        let _ = std::fs::write(f, &omsa_buf);
        let _ = writeln!(sumbuf, "# Processed and annotated MSAs saved to file {f}.");
    }
    if let Some(ref f) = opts.cmtbl {
        let _ = writeln!(sumbuf, "# CM topology description saved in file {f}.");
    }
    if let Some(ref f) = opts.emap {
        let _ = writeln!(sumbuf, "# CM emit map saved in file {f}.");
    }
    // C cmbuild.c:409-416 (refine/rdump come after emap, before occfile).
    if let Some(ref f) = opts.refine {
        let _ = writeln!(sumbuf, "# Refined alignments used to build CMs saved in file {f}.");
    }
    if let Some(ref f) = opts.rdump {
        let _ = writeln!(sumbuf, "# Intermediate alignments from MSA refinement saved in file {f}.");
    }
    if let Some(ref f) = opts.occfile {
        let _ = writeln!(sumbuf, "# Expected occupancy values for each CM state saved in file {f}.");
    }
    if let Some(ref f) = opts.fp7occfile {
        let _ = writeln!(sumbuf, "# Expected occupancy values for each filter P7 HMM state saved in file {f}.");
    }

    // C footer (master, cmbuild.c:441-442): "#\n" then "# CPU time: ...".
    // The elapsed time is a documented (non-deterministic) variable.
    let _ = writeln!(sumbuf, "#");
    let _ = writeln!(sumbuf, "# CPU time: 0.00u 0.00s 00:00:00.00 Elapsed: 00:00:00.00");

    // Emit the summary to stdout, or to -o <f>.
    match &opts.ofile {
        Some(f) => {
            if let Err(e) = std::fs::write(f, &sumbuf) {
                eprintln!("Failed to open -o output file {f}: {e}");
                std::process::exit(1);
            }
        }
        None => {
            let _ = std::io::stdout().write_all(&sumbuf);
        }
    }
}

/// C: MSADivide() (cmbuild.c:2807). Split the master MSA into one MSA per
/// cluster of sequences (one CM will be built per returned MSA). Modes:
///   do_all (--call):     each seq is its own cluster.
///   do_mindiff (--cmaxid): maximize #clusters s.t. min inter-cluster diff >= mindiff.
///   do_nc (--ctarget):   binary-search mindiff to hit exactly target_nc clusters.
/// If do_orig (--corig), the full master MSA is appended as the last MSA (kept
/// with its original, un-suffixed name); cluster MSAs are renamed "<name>.<m+1>".
fn msa_divide(
    mmsa: &EslMsa,
    abc: &EslAlphabet,
    do_all: bool,
    do_mindiff: bool,
    do_nc: bool,
    mut mindiff: f64,
    mut target_nc: i32,
    do_orig: bool,
) -> Vec<EslMsa> {
    use infernox::easel::tree;

    let nseq = mmsa.nseq;
    // C contract: exactly one mode.
    debug_assert_eq!(do_all as i32 + do_nc as i32 + do_mindiff as i32, 1);

    if do_nc {
        mindiff = 0.0;
    }

    let nc: usize;
    let clust: Vec<i32>;

    if do_all {
        // Mode 1: each seq becomes its own cluster.
        nc = nseq;
        clust = (0..nseq as i32).collect();
        println!(
            "# Alignment split into {} clusters; each comprised of exactly 1 sequence",
            nc
        );
        println!("#");
    } else {
        // Mode 2 or 3: distance matrix + single-linkage tree.
        let d = infernox::easel::distance::esl_dst_x_diff_mx(abc, &mmsa.ax, nseq);
        let mut t = tree::esl_tree_single_linkage(&d);
        t.set_taxa_parents();

        // C: diff[n] = T->ld[n], rounded down to nearest 0.001 (cmbuild.c:2872).
        let mut diff = vec![0.0f64; (t.n - 1).max(1)];
        for n in (0..=(t.n - 2)).rev() {
            let mut v = t.ld[n];
            v *= 1000.0;
            v = ((v as i32) as f32) as f64;
            v /= 1000.0;
            diff[n] = v;
        }

        let (this_nc, this_clust);
        if do_mindiff {
            // Mode 2.
            let (c, ncl, _best) = select_node(&mut t, &diff, mindiff);
            this_nc = ncl;
            this_clust = c;
            println!(
                "# Alignment split into {} clusters; each will be used to train a CM.",
                ncl
            );
            println!(
                "# Maximum identity b/t any 2 seqs in different clusters: {:.2}",
                1.0 - mindiff
            );
            println!("#");
        } else {
            // Mode 3, do_nc.
            if target_nc > t.n as i32 {
                target_nc = t.n as i32;
            }
            let (c, ncl, md) = find_mindiff(&mut t, &diff, target_nc);
            mindiff = md;
            this_nc = ncl;
            this_clust = c;
            println!(
                "# Alignment split into {} clusters; each will be used to train a CM.",
                ncl
            );
            println!(
                "# Maximum identity b/t any 2 seqs in different clusters: {:.2}",
                1.0 - mindiff
            );
            println!("#");
        }
        nc = this_nc;
        clust = this_clust;
    }

    // useme[m][i]: whether seq i goes into cluster MSA m. For do_orig, cluster
    // nc keeps all seqs (the full master MSA).
    let mut result: Vec<EslMsa> = Vec::with_capacity(if do_orig { nc + 1 } else { nc });
    for m in 0..nc {
        let mut useme = vec![false; nseq];
        for i in 0..nseq {
            if clust[i] != -1 && clust[i] as usize == m {
                useme[i] = true;
            }
        }
        let mut cmsa_m = mmsa.sequence_subset(&useme);
        // C: rename by appending ".<m+1>".
        let base = cmsa_m.name.clone().unwrap_or_default();
        cmsa_m.name = Some(format!("{}.{}", base, m + 1));
        result.push(cmsa_m);
    }
    if do_orig {
        let useme = vec![true; nseq];
        result.push(mmsa.sequence_subset(&useme));
    }

    result
}

/// C: select_node() (cmbuild.c:3013). Partition taxa (seqs) into clusters such
/// that the minimum inter-cluster disparity exceeds <mindiff> while maximizing
/// the number of clusters. Returns (clust[0..N-1], nc, best_node).
fn select_node(t: &mut infernox::easel::tree::EslTree, diff: &[f64], mindiff: f64) -> (Vec<i32>, usize, i32) {
    t.set_cladesizes();
    let cladesize = t.cladesize.as_ref().unwrap();
    let parent = &t.parent;
    let left = &t.left;
    let right = &t.right;

    let mut clust = vec![0i32; t.n];

    // Two LIFO stacks (esl_stack_I*): ns1 traverses to find cluster roots, ns2
    // collects all taxa within a chosen cluster's clade.
    let mut ns1: Vec<i32> = Vec::new();
    let mut ns2: Vec<i32> = Vec::new();

    ns1.push(0); // push root
    let mut maxsize = 0i32;
    let mut best = 0i32;
    let mut c = 0i32;

    while let Some(n) = ns1.pop() {
        let nn = n as usize;
        if (n == 0 || diff[parent[nn] as usize] > mindiff) && diff[nn] <= mindiff {
            // We're at a cluster.
            if cladesize[nn] > maxsize {
                maxsize = cladesize[nn];
                best = n;
            }
            ns2.push(n);
            while let Some(np) = ns2.pop() {
                let npi = np as usize;
                if left[npi] <= 0 {
                    clust[(-left[npi]) as usize] = c;
                } else {
                    ns2.push(left[npi]);
                }
                if right[npi] <= 0 {
                    clust[(-right[npi]) as usize] = c;
                } else {
                    ns2.push(right[npi]);
                }
            }
            c += 1;
        } else {
            // Not a cluster; keep traversing.
            if left[nn] <= 0 {
                clust[(-left[nn]) as usize] = c;
                c += 1;
            } else {
                ns1.push(left[nn]);
            }
            if right[nn] <= 0 {
                clust[(-right[nn]) as usize] = c;
                c += 1;
            } else {
                ns1.push(right[nn]);
            }
        }
    }

    (clust, c as usize, best)
}

/// C: find_mindiff() (cmbuild.c:3104). Binary-search the minimum fractional
/// difference (mindiff) that yields >= target_nc clusters, then define clusters.
/// diff values are rounded to 0.001, guaranteeing an exact target is reachable.
fn find_mindiff(
    t: &mut infernox::easel::tree::EslTree,
    diff: &[f64],
    target_nc: i32,
) -> (Vec<i32>, usize, f64) {
    let mut high = 1.0f32;
    let mut low = 0.0f32;
    let mut high_nc = 0i32;
    let mut mindiff = 0.5f32;
    let mut curr_nc = -1i32;
    let mut keep_going = true;
    let thresh = 0.001f32;
    let mut clust: Vec<i32> = Vec::new();

    while keep_going {
        let (c, ncl, _best) = select_node(t, diff, mindiff as f64);
        clust = c;
        curr_nc = ncl as i32;
        if curr_nc < target_nc {
            high = mindiff;
            high_nc = curr_nc;
            mindiff -= (mindiff - low) / 2.0;
            if (high - 0.0).abs() < thresh && (low - 0.0).abs() < thresh {
                keep_going = false;
            }
        } else {
            low = mindiff;
            mindiff += (high - mindiff) / 2.0;
            if (high - low).abs() < thresh {
                keep_going = false;
            }
        }
    }
    // If we couldn't hit target exactly, adjust as C does.
    if curr_nc != target_nc {
        if high_nc < target_nc {
            mindiff = high;
            let (c, ncl, _best) = select_node(t, diff, mindiff as f64);
            clust = c;
            curr_nc = ncl as i32;
        } else {
            while high_nc > target_nc {
                high += thresh;
                if high > 1.0 {
                    eprintln!("find_mindiff(), mindiff has risen above 1.0");
                    std::process::exit(1);
                }
                mindiff = high;
                let (c, ncl, _best) = select_node(t, diff, mindiff as f64);
                clust = c;
                curr_nc = ncl as i32;
                high_nc = curr_nc;
            }
        }
    }

    (clust, curr_nc as usize, mindiff as f64)
}

/// C: cmbuild.c:build_and_calibrate_p7_filter() (default path, use_mlp7_as_filter=FALSE).
/// Builds the filter HMM (cm->fp7) and its EFP7GF/STATS calibration, then attaches
/// it to the CM. `msa_a` is the post-build MSA (fed to Stage A p7_Builder, which
/// clones+re-weights it). `msa_b` is a pristine clone (Stage B rebuilds the temp CM).
fn attach_p7_filter(
    cm: &mut CM,
    msa_a: &EslMsa,
    msa_b: &mut EslMsa,
    abc: &EslAlphabet,
    opts: &Opts,
    comlog: &str,
    fullmat: Option<&infernox::rsearch::FullMat>,
) {
    use infernox::p7_hmm::{P7H_CONS, P7H_CS, P7H_MAP, P7H_STATS};

    let clen = cm.clen as usize;

    // C build_and_calibrate_p7_filter (cmbuild.c:1015-1017): use_mlp7_as_filter is
    // (pretend_cm_is_hmm || --p7ml). pretend_cm_is_hmm is TRUE iff the model has
    // zero basepairs and --noh3pri/--v1p0/--p56 were not used; --p7ml forces the
    // ML-p7 filter even for a with-basepairs model. In that case the filter is the
    // CM's ML p7 HMM (cm->mlp7) directly — no p7_Builder, no temp CM.
    let use_mlp7 = determine_pretend_cm_is_hmm(opts, cm) || opts.p7ml;

    let mut fhmm = if use_mlp7 {
        // mlp7-as-filter path: fhmm = cm->mlp7 = cm_cp9_to_p7(cm). No p7_Builder,
        // no temp CM. cm_cp9_to_p7 sets emissions/transitions/RF/CS/COMPO; the
        // consensus is then set at threshold 0.5 by cm_p7_hmm_SetConsensus.
        let mut fhmm = infernox::p7_filter_emit::cm_cp9_to_p7(cm);
        infernox::p7_filter_emit::cm_p7_hmm_set_consensus(&mut fhmm);
        fhmm
    } else {
        // Default path: Stage A (p7_Builder) transitions + Stage B (temp-CM
        // marginalized) emissions.
        // C: fp7_bld->re_target = --p7ere ? <x> : DEFAULT_ETARGET_HMMFILTER
        // (cmbuild.c:897), consumed by p7_Builder's entropy weighting.
        let (mut fhmm, fhmm_re) =
            infernox::p7_builder::build_filter_p7_with_re(cm, msa_a, opts.p7ere);

        // ACC/DESC: C p7_Builder annotate() copies them from msa->acc/desc (=
        // cm->acc/desc). Stage A leaves them unset.
        fhmm.acc = cm.acc.clone();
        fhmm.desc = cm.desc.clone();

        // Stage B: temp-CM marginalized ML-p7 emissions (C !--p7hemit branch,
        // cmbuild.c:2402-2446).
        let emis = infernox::p7_filter_emit::build_filter_emissions(
            msa_b, abc, fhmm_re, &build_knobs(opts), fullmat,
        );
        for k in 0..=clen {
            fhmm.mat[k] = emis.mat[k];
            fhmm.ins[k] = emis.ins[k];
        }
        // reset composition (needs Stage A transitions for occupancy).
        infernox::p7_filter_emit::p7_hmm_set_composition(&mut fhmm);
        fhmm.eff_nseq = emis.neff as f32; // C: fhmm->eff_nseq = acm->eff_nseq

        // Consensus (cm_p7_hmm_SetConsensus @0.5) from Stage B.
        fhmm.consensus = emis.consensus;
        fhmm.flags |= P7H_CONS;

        // MAP: same consensus columns as the CM (amsa->rf set from cm->map).
        for k in 1..=clen {
            fhmm.map[k] = cm.map[k];
        }
        fhmm.flags |= P7H_MAP;

        // CS: overwrite with the CM's WUSS consensus structure (build_and_calibrate:2390).
        if let Some(cons) = infernox::cm_consensus::create_cm_consensus_full(cm) {
            fhmm.cs = vec![b' '; clen + 2];
            fhmm.cs[0] = b' ';
            for k in 1..=clen {
                fhmm.cs[k] = cons.cstr[k - 1];
            }
            fhmm.flags |= P7H_CS;
        }

        // RF: C copies the CM's RF if it has one (build_and_calibrate:2385-2389).
        if cm.flags & infernox::cm::CM_RF != 0 && (cm.rf.len() as i32) > cm.clen {
            use infernox::p7_hmm::P7H_RF;
            fhmm.rf = vec![b' '; clen + 2];
            for k in 1..=clen {
                fhmm.rf[k] = cm.rf[k];
            }
            fhmm.flags |= P7H_RF;
        }
        fhmm
    };

    fhmm.ctime = Some(current_ctime());
    // C: p7_hmm_AppendComlog(cm->fp7, argc, argv) — APPENDS the command. For the
    // default path fhmm->comlog is NULL → 1 COM line; for the mlp7 path it already
    // carries cm->comlog (copied in cm_cp9_to_p7) → the command is duplicated (2
    // COM lines). Reproduce that structure faithfully.
    fhmm.comlog = Some(match fhmm.comlog.take() {
        Some(c) => format!("{c}\n{comlog}"),
        None => comlog.to_string(),
    });

    // Stage C: cm_p7_Calibrate → STATS LOCAL + glocal EFP7GF. Sample counts come
    // from --EmN/--EvN/--ElfN/--EgfN (cmbuild.c:2299-2302, GetInteger => 200 default).
    let ns = infernox::cm_p7_calibrate::P7CalN {
        emn: opts.emn as usize,
        evn: opts.evn as usize,
        elfn: opts.elfn as usize,
        egfn: opts.egfn as usize,
    };
    let cal = infernox::cm_p7_calibrate::cm_p7_calibrate(&mut fhmm, cm.clen, ns);
    fhmm.flags |= P7H_STATS;

    // cm_SetFilterHMM: attach + record glocal-forward params (cm_p7_modelmaker.c:431).
    cm.efp7gf_tau = cal.gfmu;
    cm.efp7gf_lambda = cal.gflambda;
    cm.p7 = Some(fhmm);
    cm.flags |= infernox::cm::CM_FP7;
}

/// TRUE if --v1p0/--p56/--noh3pri/--prior force the standard (non-zerobp) prior,
/// so a 0-bp model is NOT treated as HMM-like. (--prior not yet supported.)
fn force_standard_prior(opts: &Opts) -> bool {
    opts.noh3pri || opts.v1p0 || opts.p56
}

/// C: determine_pretend_cm_is_hmm() (cmbuild.c).
fn determine_pretend_cm_is_hmm(opts: &Opts, cm: &CM) -> bool {
    cm.cm_count_nodetype(infernox::constants::MATP_ND) == 0 && !force_standard_prior(opts)
}

/// C init_cfg prior selection + pri2use. Returns the prior to use given `pretend`.
///   --p56/--v1p0 -> v0p56 prior (pri_zerobp NULL).
///   --noh3pri    -> Prior_Default(FALSE) (pri_zerobp NULL).
///   default      -> pretend ? Prior_Default(TRUE) : Prior_Default(FALSE).
fn select_prior(opts: &Opts, pretend: bool) -> infernox::prior::Prior {
    if opts.p56 || opts.v1p0 {
        infernox::prior::prior_v0p56_through_v1p02()
    } else {
        infernox::prior::prior_default(pretend)
    }
}

/// Assemble the shared build knobs consumed by both build_one and the p7 filter's
/// temp donor CM (build_filter_emissions).
fn build_knobs(opts: &Opts) -> BuildKnobs {
    BuildKnobs {
        hand: opts.hand,
        noss: opts.noss,
        wscheme: opts.wscheme,
        wid: opts.wid,
        symfrac: opts.symfrac as f32,
        fragthresh: opts.fragthresh as f32,
        fraggiven: opts.fraggiven,
        nobalance: opts.nobalance,
        nodetach: opts.nodetach,
        iins: opts.iins,
        iflank: opts.iflank,
        eminseq: opts.eminseq,
        emaxseq: opts.emaxseq,
        null: opts.null,
        force_standard_prior: force_standard_prior(opts),
        v1p0: opts.v1p0,
        use_v0p56_prior: opts.p56 || opts.v1p0,
    }
}

/// C: process_build_workunit() (minus refine; rsearch supported).
fn build_one(
    msa: &mut EslMsa,
    abc: &EslAlphabet,
    opts: &Opts,
    comlog: &str,
    fullmat: Option<&infernox::rsearch::FullMat>,
) -> (CM, mm::Ptree, Vec<mm::Ptree>) {
    let alen = msa.alen as usize;

    // --- check_and_clean_msa: clean SS_cons ---
    // C: check_and_clean_msa (cmbuild.c:1403-1406). With --noss, strip all base
    // pairs from SS_cons (create it if absent), making the CM HMM-like.
    if opts.noss {
        msa.ss_cons = Some(".".repeat(alen));
    }
    {
        let mut ss: Vec<u8> = msa.ss_cons.as_ref().expect("no SS_cons").as_bytes().to_vec();
        ss.resize(alen, b'.');
        if !mm::clean_cs(&mut ss, alen) {
            eprintln!("Failed to parse consensus structure annotation");
            std::process::exit(1);
        }
        msa.ss_cons = Some(String::from_utf8_lossy(&ss).into_owned());
    }

    // --- check_and_clean_msa: --rsearch resolves degeneracies (cmbuild.c:1410-1417).
    // Runs BEFORE the checksum, as in C's check_and_clean_msa (precedes
    // esl_msa_Checksum in process_build_workunit). Requires nseq == 1.
    if let Some(fm) = fullmat {
        if msa.nseq != 1 {
            eprintln!("with --rsearch option, all of the input alignments must have exactly 1 sequence");
            std::process::exit(1);
        }
        infernox::rsearch::ribosum_msa_resolve_degeneracies(fm, msa, abc);
    }

    // --- checksum (before weighting/fragment marking) ---
    let checksum = mm::msa_checksum(msa);

    // --- set_relative_weights (cmbuild.c:set_relative_weights) ---
    // C: mw_cfg->ignore_rf = esl_opt_GetBoolean("--hand") ? FALSE : TRUE. With
    // --hand the PB weighting takes consensus columns from the RF annotation.
    let ignore_rf = !opts.hand;
    infernox::msaweight::apply_weights(msa, opts.wscheme, opts.wid, ignore_rf);

    // --- mark_fragments / check_fragments (cmbuild.c:995-1000) ---
    // C: if(--fraggiven) check_fragments(); else mark_fragments(). With
    // --fraggiven the MSA's given ~ (missing) annotation is validated (in
    // main(), where the header can be flushed on failure) and used as-is; the
    // MSA is NOT modified here. Otherwise infer fragments from aligned span.
    // C passes esl_opt_GetReal("--fragthresh") (double) to a float param.
    if !opts.fraggiven {
        mm::mark_fragments(msa, opts.fragthresh as f32);
    }

    // --- build_model ---
    let use_rf = opts.hand;
    // C: use_wts = (use_rf || --v1p0) ? FALSE : TRUE
    let use_wts = !use_rf && !opts.v1p0;
    // C passes esl_opt_GetReal("--symfrac") (double) to a float param.
    let symfrac = opts.symfrac as f32;
    let (mut cm, gtr) = mm::hand_modelmaker(msa, abc, use_rf, use_wts, symfrac);

    // C build_model (cmbuild.c:1710-1715): rsearch uses the RIBOSUM background g
    // as the null model and raises CM_RSEARCHEMIT; else the (possibly
    // --null-overridden) background model.
    if let Some(fm) = fullmat {
        let mut g = [0.0f32; 4];
        g.copy_from_slice(&fm.g[..4]);
        cm.cm_set_null_model(&g);
        cm.flags |= infernox::cm::CM_RSEARCHEMIT;
    } else {
        cm.cm_set_null_model(&opts.null); // C: CMSetNullModel(cm, cfg->null) (--null)
    }

    // rebalance (default; --nobalance skips)
    if !opts.nobalance {
        cm = infernox::cm_rebalance::cm_rebalance(&cm);
    }

    // determine_pretend_cm_is_hmm (cmbuild.c): 0-basepair models use the mimic_h3
    // prior and have their parsetrees "doctored" (D<->I ambiguity removed) — but
    // NOT if --noh3pri / --v1p0 / --p56 / --prior force the standard prior.
    let pretend_cm_is_hmm = determine_pretend_cm_is_hmm(opts, &cm);

    // counts. Two passes (C build_model): first count everything except
    // truncated MP emissions; snapshot the MP counts into dbl_e; then count the
    // truncated (half-observed) MP emissions via mean-posterior estimates.
    let pri_for_counts = select_prior(opts, pretend_cm_is_hmm);
    let used_el = vec![false; alen + 1];
    let mut trs: Vec<_> = (0..msa.nseq)
        .map(|i| mm::transmogrify(&cm, &gtr, &msa.ax[i], &used_el, alen))
        .collect();
    if pretend_cm_is_hmm {
        for tr in &mut trs {
            mm::cm_parsetree_doctor(&cm, tr);
        }
    }
    for i in 0..msa.nseq {
        mm::parsetree_count(&mut cm, &trs[i], &msa.ax[i], msa.wgt[i] as f32);
    }
    // dbl_e: frozen double copy of MP emission counts (C: build_model dbl_e[]).
    let dbl_e: Vec<Vec<f64>> = (0..cm.m as usize)
        .map(|v| {
            if cm.sttype[v] as i32 == infernox::constants::MP_ST {
                cm.e[v].iter().map(|&x| x as f64).collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    for i in 0..msa.nseq {
        mm::parsetree_count_only_truncated_mps(
            &mut cm,
            &trs[i],
            &msa.ax[i],
            msa.wgt[i] as f32,
            &dbl_e,
            &pri_for_counts,
        );
    }
    cm.nseq = msa.nseq as i32;
    cm.eff_nseq = msa.nseq as f32;

    // C build_model:1797 — zero ROOT_IL/IR transition counts unless --v1p0/--iflank.
    if !opts.iflank && !opts.v1p0 {
        mm::cm_zero_flanking_insert_counts(&mut cm);
    }
    // C build_model:1802 — detach-check unless --nodetach.
    if !opts.nodetach {
        mm::cm_find_and_detach_dual_inserts(&mut cm, true, false); // check only
    }

    // C: cm->el_selfsc = sreLOG2(esl_opt_GetReal(go, "--elself")); default 0.94.
    cm.el_selfsc = (opts.elself.ln() * 1.44269504) as f32; // sreLOG2(elself)
    cm.n2_omega = opts.n2omega;
    cm.n3_omega = opts.n3omega;

    // --- annotate ---
    if let Some(ref n) = msa.name {
        cm.name = n.clone();
    }
    cm.acc = msa.acc.clone();
    cm.desc = msa.desc.clone();
    // The writer's multiline() prepends the "[n] " command index itself.
    cm.comlog = Some(comlog.to_string());
    cm.ctime = Some(current_ctime());
    cm.checksum = checksum;
    cm.flags |= infernox::cm::CM_CHKSUM;

    apply_cutoffs(msa, &mut cm);

    let pri = select_prior(opts, pretend_cm_is_hmm);

    // --- set_effective_seqnumber (cmbuild.c:set_effective_seqnumber) ---
    // C cmbuild.c:1993: --enone OR --rsearch => neff = nseq (no entropy weighting,
    // no rescale; eff_nseq already equals nseq from the count stage).
    if opts.enone || fullmat.is_some() {
        cm.eff_nseq = msa.nseq as f32;
    } else if let Some(eset) = opts.eset {
        // C: neff = --eset; cm->eff_nseq = neff; cm_Rescale(cm, neff/(float)nseq).
        cm.eff_nseq = eset as f32;
        infernox::eweight::cm_rescale(&mut cm, (eset / msa.nseq as f64) as f32);
    } else {
        // --eent (default)
        let mut clen = 0i32;
        for nd in 0..cm.nodes as usize {
            match cm.ndtype[nd] as i32 {
                x if x == infernox::constants::MATP_ND => clen += 2,
                x if x == infernox::constants::MATL_ND => clen += 1,
                x if x == infernox::constants::MATR_ND => clen += 1,
                _ => {}
            }
        }
        let nbps = cm.cm_count_nodetype(infernox::constants::MATP_ND);
        // C: --v1p0 uses version_1p0_default_target_relent(clen, 6.0).
        let etarget = if opts.v1p0 {
            infernox::eweight::version_1p0_default_target_relent(clen, 6.0)
        } else {
            infernox::eweight::set_target_relent(clen, nbps, opts.esigma, opts.ere)
        };
        let min_neff = opts.eminseq; // C: --eminseq (default 0.1)
        let max_neff = opts.emaxseq.unwrap_or(cm.nseq as f64); // C: --emaxseq else cm->nseq
        let (hmm_re, mut neff) =
            infernox::eweight::cm_entropy_weight(&mut cm, &pri, etarget, min_neff, max_neff, false);
        // --ehmmre: ensure HMM rel entropy/match column >= <x>; if not, recompute
        // neff with pretend_cm_is_hmm=TRUE. (cmbuild.c:2032-2044)
        if let Some(hmm_etarget) = opts.ehmmre {
            // C has a stray debug printf here (to stdout, not cfg->ofp):
            println!("hmm_etarget: {:.6}", hmm_etarget);
            if hmm_re < hmm_etarget {
                let (_hmm_re2, neff2) = infernox::eweight::cm_entropy_weight(
                    &mut cm, &pri, hmm_etarget, min_neff, max_neff, true,
                );
                neff = neff2;
            }
        }
        cm.eff_nseq = neff as f32;
        // C: cm_Rescale(cm, neff / (float) msa->nseq) — division in double, then
        // truncated to float for the scale arg.
        infernox::eweight::cm_rescale(&mut cm, (neff / msa.nseq as f64) as f32);
    }

    // --- parameterize (cmbuild.c:parameterize) ---
    infernox::prior::priorify_cm(&mut cm, &pri);
    // C cmbuild.c:2079: rsearch overwrites emission probs from RIBOSUM targets.
    if let Some(fm) = fullmat {
        infernox::rsearch::rsearch_cm_probify_emissions(&mut cm, fm, abc);
    }
    // C: detach dual inserts unless --nodetach.
    if !opts.nodetach {
        mm::cm_find_and_detach_dual_inserts(&mut cm, false, true); // detach
    }
    // C: flatten insert emissions unless --iins (informative inserts).
    if !opts.iins {
        mm::flatten_insert_emissions(&mut cm);
    }
    cm.cm_renormalize();

    // --- configure_model (QDB + W + logodds; p7 deferred) ---
    mm::configure_qdb_and_w(&mut cm);

    // --- set_consensus ---
    infernox::cm_consensus::cm_set_consensus(&mut cm);

    (cm, gtr, trs)
}

// ===========================================================================
// --refine: iterative MSA refinement (EM / Gibbs). C: cmbuild.c:refine_msa.
// ===========================================================================

/// Digital codes for the easel RNA alphabet (sym="ACGU-RYMKSWHBVDN*~", Kp=18):
/// canonical 0..3, gap('-'/'_'/'.')=4, degenerate 5..15, missing('*'=16,'~'=17).
const ABC_K: i32 = 4;
const ABC_GAP: u8 = 4;

/// C: esl_abc_XIsResidue — a residue is canonical (0..K-1) or degenerate
/// (K+1..Kp-3), i.e. NOT gap (K=4) and NOT missing (16,17). Matches
/// cm_modelmaker::xis_residue.
#[inline]
fn abc_is_residue(x: u8) -> bool {
    (x as i32) < ABC_K || ((x as i32) > ABC_K && (x as i32) < 16)
}

/// C singlet emission with degenerate averaging (esl_abc_FAvgScore over the
/// full alphabet — matches the codebase's established handling, exact for
/// canonical residues and N).
#[inline]
fn refine_sing_sc(esc: &[f32], di: u8) -> f32 {
    if (di as i32) < ABC_K {
        esc[di as usize]
    } else {
        let mut s = 0.0f32;
        for a in 0..ABC_K as usize {
            s += esc[a];
        }
        s / ABC_K as f32
    }
}

/// C pair emission with degenerate averaging (DegeneratePairScore).
#[inline]
fn refine_pair_sc(esc: &[f32], di: u8, dj: u8) -> f32 {
    if (di as i32) < ABC_K && (dj as i32) < ABC_K {
        esc[di as usize * ABC_K as usize + dj as usize]
    } else {
        let mut s = 0.0f32;
        for a in 0..ABC_K as usize {
            for c in 0..ABC_K as usize {
                s += esc[a * ABC_K as usize + c];
            }
        }
        s / (ABC_K * ABC_K) as f32
    }
}

/// C: ParsetreeScore (cm_parsetree.c:445), specialized to the refine_msa call
/// (emap=NULL, do_null2=FALSE) — returns only the log-odds score `sc`. Faithful
/// preorder accumulation over the standard (global) parsetree `tr`.
fn parsetree_score(cm: &CM, tr: &mm::Ptree, dsq: &[u8]) -> f32 {
    use infernox::constants::{B_ST, E_ST, IL_ST, IR_ST, ML_ST, MP_ST, MR_ST};
    const CMH_LOCAL_BEGIN: u32 = 1 << 10;
    let mut sc = 0.0f32;
    for tidx in 0..tr.n as usize {
        let v = tr.state[tidx];
        let mode = tr.mode[tidx] as i32;
        if v == cm.m {
            continue; // EL, local alignment end
        }
        let vv = v as usize;
        let st = cm.sttype[vv] as i32;
        if st == E_ST || st == B_ST {
            continue; // no scores in B, E
        }
        // --- transition score ---
        if tr.nxtl[tidx] == -1 {
            // truncated end: no transition contribution (sc += 0.)
        } else {
            let y = tr.state[tr.nxtl[tidx] as usize];
            if v == 0 {
                if cm.flags & CMH_LOCAL_BEGIN != 0 {
                    sc += cm.beginsc[y as usize];
                } else {
                    sc += cm.tsc[vv][(y - cm.cfirst[vv]) as usize];
                }
            } else if y == cm.m {
                // local end (EL): endsc + el_selfsc * (emitr-emitl+1 - StateDelta)
                let sd = match mode {
                    3 => match st {
                        MP_ST => 2,
                        ML_ST | MR_ST | IL_ST | IR_ST => 1,
                        _ => 0,
                    },
                    2 => match st {
                        MP_ST | ML_ST | IL_ST => 1,
                        _ => 0,
                    },
                    _ => match st {
                        MP_ST | MR_ST | IR_ST => 1,
                        _ => 0,
                    },
                };
                sc += cm.endsc[vv]
                    + cm.el_selfsc * (tr.emitr[tidx] - tr.emitl[tidx] + 1 - sd) as f32;
            } else {
                sc += cm.tsc[vv][(y - cm.cfirst[vv]) as usize];
            }
        }
        // --- emission score ---
        if st == MP_ST {
            let symi = dsq[tr.emitl[tidx] as usize];
            let symj = dsq[tr.emitr[tidx] as usize];
            match mode {
                3 => sc += refine_pair_sc(&cm.esc[vv], symi, symj), // TRMODE_J
                2 => sc += cm.lmesc[vv][symi as usize],             // TRMODE_L
                1 => sc += cm.rmesc[vv][symj as usize],             // TRMODE_R
                _ => {}
            }
        } else if (st == ML_ST || st == IL_ST) && (mode == 3 || mode == 2) {
            // ModeEmitsLeft(J or L)
            let symi = dsq[tr.emitl[tidx] as usize];
            sc += refine_sing_sc(&cm.esc[vv], symi);
        } else if (st == MR_ST || st == IR_ST) && (mode == 3 || mode == 1) {
            // ModeEmitsRight(J or R)
            let symj = dsq[tr.emitr[tidx] as usize];
            sc += refine_sing_sc(&cm.esc[vv], symj);
        }
    }
    sc
}

/// C: convert_parsetrees_to_unaln_coords (cmbuild.c:2716). Rewrite each build
/// parsetree's emitl/emitr from alignment columns to unaligned residue coords
/// (1-based), so they can be scored against the unaligned sequences.
fn convert_parsetrees_to_unaln_coords(trs: &mut [mm::Ptree], msa: &EslMsa) {
    let alen = msa.alen as usize;
    for (i, tr) in trs.iter_mut().enumerate() {
        // map[apos] = unaligned position (1-based), or -1 for a gap column.
        let mut map = vec![-1i32; alen + 1];
        let mut uapos = 1i32;
        for apos in 1..=alen {
            if msa.ax[i][apos] != ABC_GAP {
                map[apos] = uapos;
                uapos += 1;
            }
        }
        for x in 0..tr.n as usize {
            if tr.emitl[x] != -1 {
                tr.emitl[x] = map[tr.emitl[x] as usize];
            }
            if tr.emitr[x] != -1 {
                tr.emitr[x] = map[tr.emitr[x] as usize];
            }
        }
    }
}

/// C: esl_sq_GetFromMSA — extract the unaligned digital sequence for row `i`
/// (strip gap/missing columns), 1-based with leading/trailing sentinels.
fn unaligned_dsq_from_ax(ax: &[u8], alen: usize) -> Vec<u8> {
    let mut dsq = vec![255u8]; // sentinel
    for apos in 1..=alen {
        let x = ax[apos];
        if abc_is_residue(x) {
            dsq.push(x);
        }
    }
    dsq.push(255u8);
    dsq
}

/// C: refine_msa (cmbuild.c:1039). Iteratively realign the input MSA to the CM
/// and rebuild until the summed parse bit-scores converge (EM). Prints the
/// interleaved iteration table to `sumbuf`, writes the final refined MSA to
/// `refinefp` (and intermediate alignments to `rdfp` if --rdump). Returns the
/// refined (CM, MSA). Supported and byte-verified against C on tRNA5.sto:
/// EM default (global/local + optacc/cyk + truncated-default/notrunc), --gibbs
/// (stochastic sampling, faithful RNG), --sub (sub-CM), and --nonbanded --notrunc
/// (D&C CYK). The ONLY gated combination is default (truncated) --nonbanded, which
/// C routes to TrCYK_DnC (truncyk.c) -- refused by the caller until that engine is
/// ported (see the gate near cmbuild.c's refine loop).
#[allow(clippy::too_many_arguments)]
fn refine_msa(
    opts: &Opts,
    abc: &EslAlphabet,
    comlog: &str,
    mut cm: CM,
    input_msa: EslMsa,
    mut input_trs: Vec<mm::Ptree>,
    sumbuf: &mut Vec<u8>,
    refinefp: &mut Option<Box<dyn Write>>,
    rdfp: &mut Option<Box<dyn Write>>,
) -> (CM, EslMsa) {
    let threshold = 0.01f32;
    let eslsmallx1 = 5e-9f32; // C eslSMALLX1
    let max_niter = 200;
    let nseq = input_msa.nseq;
    let alen = input_msa.alen as usize;
    let msa_name = input_msa.name.clone().unwrap_or_default();
    let do_trunc = !(opts.notrunc || opts.sub);

    // Alignment configuration (C cm_ConfigureSub/cm_Configure options block,
    // cmbuild.c:2168-2201): global unless -l, truncated unless --notrunc/--sub.
    let aln_opts = infernox::cm_alndata::AlnOpts {
        do_global: !opts.refine_local,
        do_sub: opts.sub,
        do_notrunc: opts.notrunc,
        do_nonbanded: opts.nonbanded,
        // C cmbuild.c:2178-2182: --nonbanded => align_opts |= SMALL|NONBANDED|CYK,
        // and &= ~OPTACC (CYK forced on, optimal-accuracy off). So do_cyk is TRUE
        // whenever --nonbanded, regardless of the (mutually-exclusive) --cyk flag.
        do_cyk: opts.refine_cyk || opts.nonbanded,
        do_sample: opts.gibbs,
        do_small: opts.nonbanded, // C: --nonbanded => SMALL|NONBANDED|CYK
        want_pp: false,           // C: refine passes ppstrA=NULL
        // C configure_model:2169 cm->tau = --tau; DispatchSqAlignment uses --mxsize.
        tau: opts.tau,
        mxsize: opts.mxsize as f32,
        // cmbuild --refine has no --fixedtau/--maxtau; defaults preserve the
        // byte-verified do_xtau=on / maxtau=0.05 refine behavior.
        maxtau: 0.05,
        do_fixedtau: false,
    };
    let mut aln_cfg = infernox::cm_alndata::configure_for_alignment(&mut cm, &aln_opts);
    let mut rng = infernox::easel::random::EslRandom::new(opts.seed);

    // Unaligned digital sequences (fixed across iterations; the MSA columns
    // change but the underlying sequences don't).
    let names: Vec<String> = input_msa.sqname.clone();
    let dsqs: Vec<Vec<u8>> = (0..nseq)
        .map(|i| unaligned_dsq_from_ax(&input_msa.ax[i], alen))
        .collect();

    // Initial score: implicit parsetrees of the input MSA seqs to the initial CM.
    convert_parsetrees_to_unaln_coords(&mut input_trs, &input_msa);
    let mut oldscore = 0.0f32;
    for i in 0..nseq {
        oldscore += parsetree_score(&cm, &input_trs[i], &dsqs[i]);
    }

    // Header for the tabular output (C print_refine_column_headings).
    print_refine_column_headings(sumbuf);
    // Only print the iter-0 (implicit parse) score when NOT doing truncated
    // alignment (C cmbuild.c:1114): truncated scores are not comparable.
    if !do_trunc {
        let _ = writeln!(sumbuf, "  {:>5} {:>13.2} {:>10}", 0, oldscore, "-");
    }
    // Initial alignment to --rdump.
    if let Some(rd) = rdfp.as_deref_mut() {
        write_stockholm(rd, &input_msa);
    }

    let mut msa = input_msa;
    let mut iter = 0;
    while iter <= max_niter {
        iter += 1;

        // 1. cm -> parsetrees (DispatchSqBlockAlignment).
        let mut trs_a: Vec<infernox::parsetree::Parsetree> = Vec::with_capacity(nseq);
        let mut totscore = 0.0f32;
        for i in 0..nseq {
            let l = (dsqs[i].len() - 2) as i32;
            let (tr, _pp, sc) =
                infernox::cm_alndata::dispatch_sq_alignment(&cm, &aln_cfg, &aln_opts, &dsqs[i], l, &mut rng);
            totscore += sc;
            trs_a.push(tr);
        }

        // convergence check
        let delta = (totscore - oldscore) / totscore.abs();
        if opts.indi {
            print_refine_column_headings(sumbuf);
        }
        if iter > 1 || !do_trunc {
            let _ = writeln!(sumbuf, "  {:>5} {:>13.2} {:>10.3}", iter, totscore, delta);
        } else {
            let _ = writeln!(sumbuf, "  {:>5} {:>13.2} {:>10}", iter, totscore, "-");
        }
        if delta <= threshold && delta > (-1.0 * eslsmallx1) {
            break;
        }
        oldscore = totscore;

        // 2. parsetrees -> msa (Parsetrees2Alignment, do_full=TRUE, no PP).
        let nopp: Vec<Option<Vec<u8>>> = vec![None; nseq];
        // C refine_msa (cmbuild.c:1154): do_full=TRUE, do_matchonly=FALSE,
        // allow_trunc=--miss; do_flush=--fins (from cm->align_opts FLUSHINSERTS).
        let mut new_msa = infernox::cm_dpalign::parsetrees_to_alignment(
            &cm, abc, &names, &dsqs, &trs_a, &nopp, false, false, opts.fins, opts.miss,
        );
        new_msa.name = Some(msa_name.clone());
        new_msa.digitize(abc);
        if let Some(rd) = rdfp.as_deref_mut() {
            write_stockholm(rd, &new_msa);
        }

        // 3. msa -> cm (process_build_workunit, minus the p7 filter — that is
        // built once on the FINAL CM by the caller; intermediate p7 filters are
        // discarded and affect neither alignment nor RNG, so we skip them).
        let (mut new_cm, _gtr, _trs) = build_one(&mut new_msa, abc, opts, comlog, None);
        aln_cfg = infernox::cm_alndata::configure_for_alignment(&mut new_cm, &aln_opts);
        cm = new_cm;
        msa = new_msa;
    }

    // Write the final refined alignment to the --refine output file.
    if let Some(rf) = refinefp.as_deref_mut() {
        write_stockholm(rf, &msa);
    }

    // If the model is local (-l), convert it back to global before returning,
    // because CMs can only be written in global mode. This is the one place in
    // Infernal where a model goes local -> global (C cmbuild.c:1187-1222).
    // The alignment path uses CMH_LOCAL_BEGIN=1<<10, CMH_LOCAL_END=1<<11
    // (cp9.rs cm_config_local / cm_alndata configure_for_alignment).
    const CMH_LOCAL_BEGIN: u32 = 1 << 10;
    const CMH_LOCAL_END: u32 = 1 << 11;
    if cm.flags & CMH_LOCAL_BEGIN != 0 {
        // C: restore t[0] from root_trans, zero all begin[], drop local-begin flag.
        let root_trans = cm
            .root_trans
            .clone()
            .expect("trying to globalize model but cm.root_trans is None");
        for v in 0..cm.m as usize {
            cm.begin[v] = 0.0;
        }
        let cnum0 = cm.cnum[0] as usize;
        for v in 0..cnum0 {
            cm.t[0][v] = root_trans[v];
        }
        cm.flags &= !CMH_LOCAL_BEGIN;
    }
    if cm.flags & CMH_LOCAL_END != 0 {
        // C: zero all end[], renormalize transitions of every internal exit node
        // (MATP/MATL/MATR/BEGL/BEGR not adjacent to an END node), drop local-end
        // + bits flags, recompute log odds.
        for v in 0..cm.m as usize {
            cm.end[v] = 0.0;
        }
        use infernox::constants::{BEGL_ND, BEGR_ND, END_ND, MATL_ND, MATP_ND, MATR_ND};
        for nd in 1..cm.nodes as usize {
            let t = cm.ndtype[nd] as i32;
            let is_exit = (t == MATP_ND || t == MATL_ND || t == MATR_ND || t == BEGL_ND
                || t == BEGR_ND)
                && cm.ndtype[nd + 1] as i32 != END_ND;
            if is_exit {
                let v = cm.nodemap[nd] as usize;
                let cnum = cm.cnum[v] as usize;
                // C esl_vec_FNorm: sum != 0 -> divide; else uniform 1/n.
                let sum: f32 = cm.t[v][..cnum].iter().sum();
                if sum != 0.0 {
                    for x in 0..cnum {
                        cm.t[v][x] /= sum;
                    }
                } else if cnum > 0 {
                    for x in 0..cnum {
                        cm.t[v][x] = 1.0 / cnum as f32;
                    }
                }
            }
        }
        cm.flags &= !CMH_LOCAL_END;
        cm.cm_logoddsify();
    }

    (cm, msa)
}

/// Write an MSA in Stockholm format to a `dyn Write` (adapts the generic
/// `esl_msafile_write` — `&mut &mut dyn Write` is `Sized` and implements `Write`).
fn write_stockholm(w: &mut dyn Write, msa: &EslMsa) {
    let mut ww = w;
    let _ = infernox::easel::msafile::esl_msafile_write(&mut ww, msa, infernox::easel::msafile::MsaFormat::Stockholm);
}

/// C cmbuild.c:1333-1372 (-O output). Build the consensus/insert-annotated
/// Stockholm MSA for one CM and append it to `buf`. For each sequence, C fetches
/// the dealigned residues from the input MSA (esl_sq_FetchFromMSA), remaps the
/// (aligned-coordinate) parsetree emit positions to unaligned coordinates via an
/// aligned->unaligned map (a2ua_map), then calls Parsetrees2Alignment with
/// do_full=TRUE, do_matchonly=FALSE, allow_trunc=TRUE. The WT (sequence weight),
/// name, desc and accession annotation is transferred from the input MSA.
fn append_omsa(
    buf: &mut Vec<u8>,
    cm: &CM,
    abc: &EslAlphabet,
    msa: &EslMsa,
    trs: &[mm::Ptree],
) {
    use infernox::parsetree::Parsetree;
    let alen = msa.alen as usize;
    let mut names: Vec<String> = Vec::with_capacity(msa.nseq);
    let mut dsqs: Vec<Vec<u8>> = Vec::with_capacity(msa.nseq);
    let mut ptrees: Vec<Parsetree> = Vec::with_capacity(msa.nseq);
    let ppstrs: Vec<Option<Vec<u8>>> = vec![None; msa.nseq];

    for i in 0..msa.nseq {
        // Dealign row i and build a2ua_map[apos] (1..alen) -> unaligned pos
        // (1-based), 0 for gap positions. (C: esl_sq_FetchFromMSA + a2ua_map.)
        let mut a2ua = vec![0i32; alen + 1];
        let mut dsq: Vec<u8> = Vec::with_capacity(alen + 2);
        dsq.push(255u8); // 1-based leading sentinel
        let mut uapos = 1i32;
        for apos in 1..=alen {
            let x = msa.ax[i][apos];
            if infernox::cm::abc_is_residue(x as usize) {
                a2ua[apos] = uapos;
                uapos += 1;
                dsq.push(x);
            }
        }
        dsq.push(255u8); // trailing sentinel

        // Convert the build parsetree (mm::Ptree) into a parsetree::Parsetree and
        // remap its emit coordinates from aligned to unaligned (C: the tr->emitl/
        // emitr rewrite loop). Guide-tree parses are standard (is_std = true).
        let p = &trs[i];
        let mut t = Parsetree::new(p.n as usize);
        t.n = p.n;
        t.state = p.state.clone();
        t.nxtl = p.nxtl.clone();
        t.nxtr = p.nxtr.clone();
        t.prv = p.prv.clone();
        t.mode = p.mode.clone();
        t.is_std = true;
        t.pass_idx = 0;
        t.emitl = p
            .emitl
            .iter()
            .map(|&e| if e != -1 { a2ua[e as usize] } else { e })
            .collect();
        t.emitr = p
            .emitr
            .iter()
            .map(|&e| if e != -1 { a2ua[e as usize] } else { e })
            .collect();

        names.push(msa.sqname[i].clone());
        dsqs.push(dsq);
        ptrees.push(t);
    }

    // C: Parsetrees2Alignment(cm, abc, sq, msa->wgt, tr, NULL, nseq, NULL, NULL,
    //    do_full=TRUE, do_matchonly=FALSE, allow_trunc=TRUE, &omsa).
    // do_post=false: C passes ppstr=NULL for -O (no #=GR PP annotation). This also
    // keeps msa->pp NULL so the Stockholm writer's name-column margin is not widened
    // by a reserved PP field (byte-exact column alignment).
    let mut omsa = infernox::cm_dpalign::parsetrees_to_alignment(
        cm, abc, &names, &dsqs, &ptrees, &ppstrs,
        /*do_post=*/ false, /*do_matchonly=*/ false, /*do_flush=*/ false,
        /*allow_trunc=*/ true,
    );

    // C: transfer name/desc/accession from the input MSA and the sequence weights.
    omsa.name = msa.name.clone();
    omsa.desc = msa.desc.clone();
    omsa.acc = msa.acc.clone();
    omsa.wgt = msa.wgt.clone();
    omsa.flags |= infernox::easel::msa::ESL_MSA_HASWGTS;

    write_stockholm(buf, &omsa);
}

/// C: print_refine_column_headings (cmbuild.c:3209).
fn print_refine_column_headings(w: &mut dyn Write) {
    let _ = writeln!(w, "#");
    let _ = writeln!(w, "# {:<5} {:<13} {:>10}", "iter", "bit score sum", "fract diff");
    let _ = writeln!(w, "# {:<5} {:<13} {:>10}", "-----", "-------------", "----------");
}

/// C: cmbuild.c:set_model_cutoffs(). Transfer GA/TC/NC from the MSA if present.
fn apply_cutoffs(msa: &EslMsa, cm: &mut CM) {
    use infernox::easel::msa::{ESL_MSA_GA1, ESL_MSA_NC1, ESL_MSA_TC1};
    if msa.cutset[ESL_MSA_TC1] {
        cm.tc = msa.cutoff[ESL_MSA_TC1];
        cm.flags |= infernox::cm::CM_TC;
    }
    if msa.cutset[ESL_MSA_GA1] {
        cm.ga = msa.cutoff[ESL_MSA_GA1];
        cm.flags |= infernox::cm::CM_GA;
    }
    if msa.cutset[ESL_MSA_NC1] {
        cm.nc = msa.cutoff[ESL_MSA_NC1];
        cm.flags |= infernox::cm::CM_NC;
    }
}

fn current_ctime() -> String {
    "Thu Jan  1 00:00:00 1970".to_string()
}
