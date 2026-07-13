// SPDX-License-Identifier: BSD-3-Clause
// infernox-cmconvert — convert covariance model file formats.
//
// Faithful port of Infernal's `cmconvert` (cmconvert.c) for the ASCII path: it
// reads a .cm file (possibly containing multiple models) and re-emits each model
// in Infernal 1.1 ASCII format (`INFERNAL1/a`), including the trailing HMMER3/f
// p7 filter block. Output is byte-identical to C `cmconvert -a` except for the
// two tool-version banner lines (`INFERNAL1/a [...]` and `HMMER3/f [...]`), which
// are legitimately tool-version dependent.
//
// Supported options:
//   -a            output models in ASCII format (default)
//   -b            output models in INFERNAL 1.1 binary format
//   --fhmm        output the filter p7 HMM in HMMER3/f ASCII format
//   --mlhmm       output the ML p7 HMM in HMMER3/f ASCII format
//   -1            output backward-compatible Infernal v0.7->v1.0.2 ASCII format
//   -o <file>     send output to <file> instead of stdout
//   -h, --help    show usage
//
// -1 drives the ported cm_file_Write1p0ASCII; verified byte-identical to C
// `cmconvert -1` (minifam 3-model file) modulo the `INFERNAL-1 [converted from
// <version>]` banner (tool-version dependent).
//
// --mlhmm builds the CM's ML p7 HMM (cm_cp9_to_p7), calibrates it via the ported
// cm_p7_Calibrate (RNG-matched seed-42 stream), and writes it in HMMER3/f ASCII.
// Verified byte-identical to C `cmconvert --mlhmm` (minifam 3-model file and the
// 0-basepair snR75 model) modulo the DATE line and the COM cmconvert invocation.
//
// Binary output (-b) drives the ported cm_file_write_binary, which the library
// already byte-verifies against C `cmconvert -b` for the tRNA fixtures. The only
// run-to-run-variable region is the embedded COM (command-line) string, whose
// bytes differ by argv[0] path exactly as the ASCII COM line does.

use std::io::{BufReader, Cursor, Write};
use std::process::exit;

use infernox::cm_file::{cm_file_read_from_reader_opt, cm_file_write_ascii, cm_file_write_binary};
use infernox::p7_hmm::p7_hmmfile_write_ascii;

/// C `cm_CreateDefaultApp()` (cm.c) command-line ERROR block, used by cmconvert
/// (and cmpress). Both the ProcessCmdline/VerifyConfig failure path
/// (`Failed to parse command line: <errbuf>`) and the `esl_opt_ArgNumber != 1`
/// path (`Incorrect number of command line arguments.`) print the offending first
/// line, then `esl_usage`, then `"\nTo see more help on available options, do
/// <argv0> -h\n\n"`, and exit(1) — all to STDOUT. The `Usage:` program name is
/// esl_usage's basename (hardcoded "cmconvert"); the final "do <argv0> -h" line
/// embeds the real binary path (the one path-dependent line, excluded from diffs).
fn cmdline_fail(first_line: &str) -> ! {
    let argv0 = std::env::args().next().unwrap_or_else(|| "cmconvert".to_string());
    print!(
        "{first_line}\n\
Usage: cmconvert [-options] <cmfile>\n\
\n\
To see more help on available options, do {argv0} -h\n\n"
    );
    exit(1);
}

/// C `cmconvert.c` process_commandline `-h` path: full banner + usage + grouped
/// esl_opt_DisplayHelp, verbatim (byte-identical to C's `-h`).
fn full_help() {
    print!(
        "# cmconvert :: convert CM file to a different Infernal format\n\
# INFERNAL 1.1.5 (Sep 2023)\n\
# Copyright (C) 2023 Howard Hughes Medical Institute.\n\
# Freely distributed under the BSD open source license.\n\
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\
Usage: cmconvert [-options] <cmfile>\n\
\n\
Options:\n  \
-h      : show brief help on version and usage\n  \
-a      : ascii:  output models in INFERNAL 1.1 ASCII format  [default]\n  \
-b      : binary: output models in INFERNAL 1.1 binary format\n  \
-1      : output backward compatible Infernal v0.7-->v1.0.2 ASCII format\n  \
-o <f>  : save CM file to file <f>, not stdout\n  \
--mlhmm : output maximum likelihood HMM for CM in HMMER3 format\n  \
--fhmm  : output filter HMM for CM in HMMER3 format\n"
    );
}

fn main() {
    // C esl_getopts accepts attached short-opt values (`-oout.cm` == `-o out.cm`);
    // expand for the match parser. cmconvert's only value-taking short option is -o.
    let args: Vec<String> =
        infernox::search_cli::expand_short_opts(&std::env::args().collect::<Vec<_>>(), &['o']);
    let prog = args
        .first()
        .map(|s| s.as_str())
        .unwrap_or("infernox-cmconvert");

    let mut outfile: Option<String> = None;
    let mut cmfile: Option<String> = None;
    let mut ascii = true; // -a is the default
    let mut binary = false; // -b
    let mut fhmm = false; // --fhmm (write the filter p7 HMM in HMMER3/f format)
    let mut mlhmm = false; // --mlhmm (write the ML p7 HMM in HMMER3/f format)
    let mut legacy1p0 = false; // -1 (write legacy Infernal v0.7->v1.0.2 ASCII format)

    // C esl_getopts stops option processing at the first non-option token (a
    // bare "-", "--", or any token not starting with '-'); everything after is a
    // positional argument (esl_getopts.c:1428-1440). We mirror that with `done_opts`.
    let mut positionals: Vec<String> = Vec::new();
    let mut done_opts = false;
    let mut i = 1;
    while i < args.len() {
        let a = args[i].clone();
        if !done_opts && a == "--" {
            // "--" explicitly ends option processing (not itself an argument).
            done_opts = true;
        } else if !done_opts && a.starts_with('-') && a != "-" {
            match a.as_str() {
                "-h" => {
                    full_help();
                    exit(0);
                }
                "-a" => {
                    // C: OUTOPTS toggle group — -a/-b are mutually exclusive; last wins.
                    ascii = true;
                    binary = false;
                }
                "-b" => {
                    // cmconvert.c:95  cm_file_WriteBinary(ofp, fmtcode, cm, NULL).
                    binary = true;
                    ascii = false;
                }
                "-o" => {
                    i += 1;
                    match args.get(i) {
                        Some(v) => outfile = Some(v.clone()),
                        // C esl_getopts process_stdopt: short-option missing arg,
                        // no trailing period.
                        None => cmdline_fail(
                            "Failed to parse command line: Option -o requires an argument",
                        ),
                    }
                }
                // cmconvert.c:98  --fhmm -> p7_hmmfile_WriteASCII(ofp, -1, cm->fp7).
                "--fhmm" => {
                    fhmm = true;
                    ascii = false;
                    binary = false;
                }
                // cmconvert.c:97  --mlhmm -> p7_hmmfile_WriteASCII(ofp, -1, cm->mlp7).
                // Builds+calibrates the CM's ML p7 HMM (configure_model -> cm_cp9_to_p7
                // -> cm_p7_Calibrate) and writes it in HMMER3/f ASCII format.
                "--mlhmm" => {
                    mlhmm = true;
                    ascii = false;
                    binary = false;
                }
                // cmconvert.c:96  -1 -> cm_file_Write1p0ASCII(ofp, cm). Legacy
                // Infernal v0.7->v1.0.2 ASCII output.
                "-1" => {
                    legacy1p0 = true;
                    ascii = false;
                    binary = false;
                    fhmm = false;
                }
                // C esl_getopts: unrecognized option (process_longopt/process_stdopt).
                other => cmdline_fail(&format!(
                    "Failed to parse command line: No such option \"{other}\"."
                )),
            }
        } else {
            done_opts = true;
            positionals.push(a);
        }
        i += 1;
    }

    let _ = ascii; // -a is the default; -b selects binary below.
    let _ = prog;

    // cm_CreateDefaultApp(options, 1, ...): exactly 1 non-option argument required.
    if positionals.len() != 1 {
        cmdline_fail("Incorrect number of command line arguments.");
    }
    cmfile = Some(positionals.into_iter().next().unwrap());
    let cmfile = cmfile.unwrap();

    // Read the whole input, then split it into per-model chunks. Each model in an
    // Infernal 1.1 file begins with an `INFERNAL1/a` banner and ends after its
    // embedded p7 `//`. C's cmconvert loops `while (cm_file_Read(...) == eslOK)`;
    // our reader consumes one model per call, so we drive it once per chunk.
    let content = match std::fs::read_to_string(&cmfile) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: failed to open {}: {}", prog, cmfile, e);
            exit(1);
        }
    };

    let chunks = split_models(&content);
    if chunks.is_empty() {
        eprintln!("{}: no models found in {}", prog, cmfile);
        exit(1);
    }

    // Open output sink.
    let mut out: Box<dyn Write> = match &outfile {
        Some(path) => match std::fs::File::create(path) {
            Ok(f) => Box::new(f),
            Err(e) => {
                eprintln!("{}: failed to create {}: {}", prog, path, e);
                exit(1);
            }
        },
        None => Box::new(std::io::stdout()),
    };

    // Command-line string appended to each model's comlog, mirroring C
    // cm_AppendComlog (cm.c:2777): the argv joined by single spaces.
    let invocation = args.join(" ");

    for chunk in chunks {
        // A legacy (v0.7->v1.0.2) record begins with the `INFERNAL-1` banner; a
        // current record begins with `INFERNAL1/a`. C decides on cmfp->format.
        let is_legacy = chunk
            .split_whitespace()
            .next()
            .map(|t| t == "INFERNAL-1")
            .unwrap_or(false);

        // Read globally (do_localize=false), matching what C's cmconvert writes.
        let mut cm = match cm_file_read_from_reader_opt(BufReader::new(Cursor::new(chunk)), false) {
            Ok(cm) => cm,
            Err(e) => {
                eprintln!("{}: failed to parse a model from {}: {:?}", prog, cmfile, e);
                exit(1);
            }
        };

        // cmconvert.c:74-78: `if (cmfp->format == CM_FILE_1 || --mlhmm) configure_model(cm)`.
        // A legacy file lacks QDBs/W/consensus/filter-HMM, so we compute them here
        // (which also refreshes the log-odds scores after the reader detached +
        // renormalized). Must run BEFORE the comlog append so the filter HMM keeps
        // only the original build/calibrate commands. (For a 1.1-format input under
        // --mlhmm, the existing mlhmm branch below already matches C byte-for-byte,
        // so we do not reconfigure it here.)
        if is_legacy {
            configure_model(&mut cm);
        }
        // cmconvert.c:82-92: append this invocation to the appropriate comlog.
        // --fhmm appends to cm->fp7's comlog (p7_hmm_AppendComlog); the CM/binary
        // paths append to the CM comlog (cm_AppendComlog).
        let wr = if mlhmm {
            // cmconvert.c:79-83 configure_model(): build cm->mlp7 (cm_cp9_to_p7),
            // calibrate it (cm_p7_Calibrate), set consensus (p7_hmm_SetConsensus,
            // RNA threshold 0.9) and ctime. C then appends the invocation to the
            // mlp7 comlog (p7_hmm_AppendComlog) and writes it in HMMER3/f ASCII.
            let mut mlp7 = infernox::p7_filter_emit::cm_cp9_to_p7(&cm);
            // cm_cp9_to_p7 -> p7_hmm_SetConsensus(mlp7, NULL) (cm_p7_modelmaker.c:189).
            infernox::p7_filter_emit::p7_hmm_set_consensus(&mut mlp7);
            // cm_cp9_to_p7 -> p7_hmm_SetCtime(mlp7) (DATE line; documented-variable).
            mlp7.ctime = Some("Thu Jan  1 00:00:00 1970".to_string());
            // configure_model(): cm_p7_Calibrate(cm->mlp7, ..., lmsvL=lvitL=200,
            // lfwdL=100, gfwdL=max(100,2*clen), N=200 each, lftailp=0.055,
            // gftailp=0.065). Populates evparam; raise P7H_STATS so STATS lines emit.
            let _cal = infernox::cm_p7_calibrate::cm_p7_calibrate(&mut mlp7, cm.clen, infernox::cm_p7_calibrate::P7CalN::default());
            mlp7.flags |= infernox::p7_hmm::P7H_STATS;
            // cmconvert.c:84  p7_hmm_AppendComlog(cm->mlp7, argc, argv). mlp7.comlog
            // already holds cm->comlog (copied inside cm_cp9_to_p7); append the
            // cmconvert invocation as a new [n] line.
            mlp7.comlog = Some(match mlp7.comlog.take() {
                Some(mut prev) => {
                    prev.push('\n');
                    prev.push_str(&invocation);
                    prev
                }
                None => invocation.clone(),
            });
            // cmconvert.c:97  p7_hmmfile_WriteASCII(ofp, -1, cm->mlp7).
            p7_hmmfile_write_ascii(&mut out, &mlp7)
        } else if fhmm {
            // cmconvert.c:87  p7_hmm_AppendComlog(cm->fp7, argc, argv).
            match cm.p7.as_mut() {
                Some(fp7) => {
                    fp7.comlog = Some(match fp7.comlog.take() {
                        Some(mut prev) => {
                            prev.push('\n');
                            prev.push_str(&invocation);
                            prev
                        }
                        None => invocation.clone(),
                    });
                    // cmconvert.c:98  p7_hmmfile_WriteASCII(ofp, -1, cm->fp7).
                    p7_hmmfile_write_ascii(&mut out, fp7)
                }
                None => {
                    eprintln!("{}: model has no filter p7 HMM to write with --fhmm", prog);
                    exit(1);
                }
            }
        } else {
            // cmconvert.c:91  cm_AppendComlog(cm, argc, argv, FALSE, 0).
            cm.comlog = Some(match cm.comlog.take() {
                Some(mut prev) => {
                    prev.push('\n');
                    prev.push_str(&invocation);
                    prev
                }
                None => invocation.clone(),
            });
            // cmconvert.c:94-96: -a -> WriteASCII, -b -> WriteBinary, -1 ->
            // Write1p0ASCII. (Write1p0ASCII prints only the first comlog line as
            // BCOM, so the appended cmconvert command does not appear there.)
            if legacy1p0 {
                infernox::cm_file::cm_file_write_1p0_ascii(&mut out, &cm)
            } else if binary {
                cm_file_write_binary(&mut out, &cm)
            } else {
                cm_file_write_ascii(&mut out, &cm)
            }
        };
        if let Err(e) = wr {
            eprintln!("{}: write error: {}", prog, e);
            exit(1);
        }
    }

    if let Err(e) = out.flush() {
        eprintln!("{}: write error: {}", prog, e);
        exit(1);
    }
}

/// C: cmconvert.c configure_model() (cmconvert.c:108-159). For a model read from a
/// v1.0->v1.0.2 file, the new 1.1 format needs QDBs (cm->dmin/dmax), W, consensus
/// and a filter p7 HMM — none of which the old file stored. This reproduces
/// cm_Configure(CM_CONFIG_QDB) + cm_SetConsensus + (build ML p7 as filter,
/// cm_p7_Calibrate, cm_SetFilterHMM) using the same ported pieces cmbuild uses.
fn configure_model(cm: &mut infernox::cm::CM) {
    // cm->config_opts |= CM_CONFIG_QDB; cm_Configure(cm): compute dmin/dmax + W and
    // logoddsify (this also fixes tsc/esc after the reader's renormalization).
    infernox::cm_modelmaker::configure_qdb_and_w(cm);
    // cm_SetConsensus(cm, cm->cmcons, NULL): sets cm->consensus + CMH_CONS.
    infernox::cm_consensus::cm_set_consensus(cm);

    // "define the filter HMM as the ML p7 HMM" (cmconvert.c:135-157). cm->mlp7 is
    // built by cm_cp9_to_p7 (in cm_Configure); we build it here from the now-
    // configured CM. Its comlog is cm->comlog copied inside cm_cp9_to_p7 (the
    // BCOM/CCOM lines only — the cmconvert invocation is appended to the CM comlog
    // AFTER this, so the filter block keeps just [1]/[2]).
    let mut fp7 = infernox::p7_filter_emit::cm_cp9_to_p7(cm);
    // cm_cp9_to_p7 -> p7_hmm_SetConsensus(mlp7, NULL) (cm_p7_modelmaker.c:189).
    infernox::p7_filter_emit::p7_hmm_set_consensus(&mut fp7);
    // p7_hmm_SetCtime (DATE line; documented-variable, stripped in verification).
    fp7.ctime = Some("Thu Jan  1 00:00:00 1970".to_string());
    // cm_p7_Calibrate(cm->mlp7, ... lmsvL=lvitL=200, lfwdL=100, gfwdL=max(100,
    // 2*clen), N=200, lftailp=0.055, gftailp=0.065): populates STATS + glocal
    // forward mu/lambda. Raise P7H_STATS so the STATS lines emit.
    let cal = infernox::cm_p7_calibrate::cm_p7_calibrate(&mut fp7, cm.clen, infernox::cm_p7_calibrate::P7CalN::default());
    fp7.flags |= infernox::p7_hmm::P7H_STATS;
    // cm_SetFilterHMM(cm, cm->mlp7, gfmu, gflambda): attach + record EFP7GF params.
    cm.efp7gf_tau = cal.gfmu;
    cm.efp7gf_lambda = cal.gflambda;
    cm.p7 = Some(fp7);
    cm.flags |= infernox::cm::CM_FP7;
}

/// True if `line` begins a new CM record. C `open_engine`/`cm_file_OpenBuffer`
/// (cm_file.c:226-227,475-476) recognizes two ASCII magics on the first token:
/// the current `INFERNAL1/a` banner and the legacy `INFERNAL-1` banner (which
/// covers v0.7/0.71/0.81/1.0/1.0.2 — they all share the `INFERNAL-1 [...]` tag).
/// The token match is on the whitespace-delimited first token, so we compare the
/// leading token rather than a raw prefix to avoid matching `INFERNAL1/a` as an
/// `INFERNAL-1` record and vice versa.
fn is_record_banner(line: &str) -> bool {
    let tok = line.split_whitespace().next().unwrap_or("");
    tok == "INFERNAL1/a" || tok == "INFERNAL-1"
}

/// Split a multi-model .cm file into per-model text chunks. Each chunk starts at
/// a CM record banner line (`INFERNAL1/a` for v1.1, or `INFERNAL-1` for legacy
/// v0.7->v1.0.2) and runs up to (but not including) the next one.
fn split_models(content: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current: Option<String> = None;
    for line in content.split_inclusive('\n') {
        if is_record_banner(line) {
            if let Some(prev) = current.take() {
                chunks.push(prev);
            }
            current = Some(String::new());
        }
        if let Some(buf) = current.as_mut() {
            buf.push_str(line);
        }
    }
    if let Some(prev) = current.take() {
        chunks.push(prev);
    }
    chunks
}
