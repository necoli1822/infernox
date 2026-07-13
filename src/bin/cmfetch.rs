// SPDX-License-Identifier: BSD-3-Clause
// infernox-cmfetch — faithful Rust port of Infernal 1.1.5's src/cmfetch.c.
//
// cmfetch: retrieve one or more covariance models from a CM database (such as
// Rfam), or build an SSI index over such a database.
//
// This binary is a 1:1 transcription of the three modes implemented by
// cmfetch.c (default single fetch, `-f` multifetch, `--index`). Porting notes
// cite `cmfetch.c:<function>:<line>` next to each transcribed block.
//
// Byte-parity note on the version banner
// --------------------------------------
// C's onefetch()/multifetch() emit each fetched model with
// cm_file_WriteASCII() (cmfetch.c:323, 275), which re-serializes the model —
// it is NOT a verbatim byte copy of the source record. The re-serialization
// stamps the running tool's own INFERNAL banner:
//     "INFERNAL1/a [<INFERNAL_VERSION> | <INFERNAL_DATE>]"
// For the golden `cmfetch` (Infernal 1.1.5, linked against HMMER 3.4) that is
//     "INFERNAL1/a [1.1.5 | Sep 2023]"
//     "HMMER3/f [3.4 | Aug 2023]"      (the embedded p7 filter banner)
// The infernox library writer (cm_file::cm_file_write_ascii) is byte-identical
// to C's cm_file_WriteASCII for the model body AND the embedded p7 filter block
// (already verified against `cmconvert -a`), and its HMMER3/f banner already
// reads "3.4 | Aug 2023". Its INFERNAL1/a banner, however, is stamped with the
// crate version (env!("CARGO_PKG_VERSION") | "Jul 2014"), a library-level
// decision owned by cm_file.rs which this binary must not edit. So to be
// byte-identical to the golden `cmfetch`, this binary patches only the first
// (INFERNAL1/a) banner line of each serialized record to the C string. Nothing
// else is altered.

use infernox::easel::ssi::{EslNewSsi, EslSsi};
use infernox::cm::CM;
use infernox::cm_file::{cm_file_read_from_reader_opt, cm_file_write_ascii};
use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::exit;

// The INFERNAL banner emitted by the golden cmfetch (Infernal 1.1.5, Sep 2023).
// cm_file_WriteASCII writes "INFERNAL1/a [%s | %s]\n" with INFERNAL_VERSION /
// INFERNAL_DATE (cm_file.c:606). We reproduce the 1.1.5 stamp verbatim.
const INFERNAL_BANNER: &str = "INFERNAL1/a [1.1.5 | Sep 2023]";

// cmfetch.c:22-25
const BANNER: &str = "retrieve CMs from a file";
const USAGE1: &str = "[options] <cmfile> <key>         (retrieves CM named <key>)";
const USAGE2: &str = "[options] -f <cmfile> <keyfile>  (retrieves all CMs in <keyfile>)";
const USAGE3: &str = "[options] --index <cmfile>       (indexes <cmfile>)";

/// One parsed model record from the CM file.
struct Record {
    /// Byte offset of the record's first line ("INFERNAL1/a ..."). This is C's
    /// cm->offset (cm_file.c sets cm->offset to the ftello at the record start),
    /// used as the SSI r_off.
    offset: u64,
    name: String,
    acc: Option<String>,
    /// The parsed model, kept so we can re-serialize it exactly as C does.
    cm: CM,
}

fn main() {
    // C esl_getopts accepts attached short-opt values (`-oout` == `-o out`); expand
    // for the match parser. cmfetch's only value-taking short option is -o.
    let argv: Vec<String> =
        infernox::search_cli::expand_short_opts(&std::env::args().collect::<Vec<_>>(), &['o']);
    let argv0 = "cmfetch";

    // ---- parse command line (cmfetch.c:83-121) ----
    let mut opt_h = false;
    let mut opt_f = false;
    let mut opt_index = false;
    let mut opt_o: Option<String> = None; // -o <f>
    let mut opt_bigo = false; // -O
    let mut args: Vec<String> = Vec::new();

    // C esl_getopts stops option processing at the first non-option token (bare
    // "-", "--", or a token not starting with '-'); the rest are positional
    // arguments (esl_getopts.c:1428-1440). `done_opts` mirrors that.
    let mut done_opts = false;
    let mut i = 1;
    while i < argv.len() {
        let a = &argv[i];
        if !done_opts && a == "--" {
            done_opts = true;
            i += 1;
            continue;
        }
        if done_opts || !(a.starts_with('-') && a.as_str() != "-") {
            args.push(a.clone());
            done_opts = true;
            i += 1;
            continue;
        }
        match a.as_str() {
            "-h" => opt_h = true,
            "-f" => opt_f = true,
            "-O" => opt_bigo = true,
            "--index" => opt_index = true,
            "-o" => {
                i += 1;
                if i >= argv.len() {
                    // esl_getopts short-option missing arg (no period), routed
                    // through cmfetch.c:83's "Failed to parse command line: %s".
                    cmdline_failure(argv0, "Failed to parse command line: Option -o requires an argument\n");
                }
                opt_o = Some(argv[i].clone());
            }
            // C esl_getopts: unrecognized option (No such option "X".).
            s => {
                cmdline_failure(argv0, &format!("Failed to parse command line: No such option \"{s}\".\n"));
            }
        }
        i += 1;
    }

    // cmfetch.c:86  -h
    if opt_h {
        cmdline_help(argv0);
    }

    // esl_opt_VerifyConfig incompatibilities (cmfetch.c option table):
    //   -f  incompat --index
    //   -o  incompat -O, --index
    //   -O  incompat -o, -f, --index
    if opt_f && opt_index {
        cmdline_failure(argv0, "Error in configuration: Option -f is incompatible with option(s) --index\n");
    }
    if opt_o.is_some() && (opt_bigo || opt_index) {
        cmdline_failure(argv0, "Error in configuration: Option -o is incompatible with option(s) -O,--index\n");
    }
    if opt_bigo && (opt_o.is_some() || opt_f || opt_index) {
        cmdline_failure(argv0, "Error in configuration: Option -O is incompatible with option(s) -o,-f,--index\n");
    }

    // cmfetch.c:87
    if args.is_empty() {
        cmdline_failure(argv0, "Incorrect number of command line arguments.\n");
    }

    let cmfile: String;
    let keyfile: Option<String>;
    let keyname: Option<String>;

    if opt_index {
        // cmfetch.c:92-101
        if args.len() != 1 {
            cmdline_failure(argv0, "Incorrect number of command line arguments.\n");
        }
        cmfile = args[0].clone();
        keyfile = None;
        keyname = None;
        if cmfile == "-" {
            cmdline_failure(argv0, "Can't use - with --index, can't index <stdin>.\n");
        }
    } else if opt_f {
        // cmfetch.c:103-113
        if args.len() != 2 {
            cmdline_failure(argv0, "Incorrect number of command line arguments.\n");
        }
        cmfile = args[0].clone();
        keyfile = Some(args[1].clone());
        keyname = None;
        if cmfile == "-" && keyfile.as_deref() == Some("-") {
            cmdline_failure(argv0, "Either <cmfile> or <keyfile> can be - but not both.\n");
        }
    } else {
        // cmfetch.c:114-121
        if args.len() != 2 {
            cmdline_failure(argv0, "Incorrect number of command line arguments.\n");
        }
        cmfile = args[0].clone();
        keyfile = None;
        keyname = Some(args[1].clone());
    }

    // ---- open the CM file (cmfetch.c:124) ----
    // We do not have a CM_FILE abstraction available here; read the whole file
    // and split it into per-model records ourselves.
    let file_bytes = match std::fs::read(&cmfile) {
        Ok(b) => b,
        Err(_) => cm_fail(&format!(
            "File existence/permissions problem in trying to open CM file {}.\n",
            cmfile
        )),
    };

    // cm_file_Open detects a pressed index (<cmfile>.i1i). cmfetch.c:133-135
    // refuses to --index an already-pressed file.
    let is_pressed = Path::new(&format!("{}.i1i", cmfile)).exists();
    if is_pressed && opt_index {
        cm_fail(&format!(
            "Looks like {} has already been SSI-indexed using cmpress; no need to index again",
            cmfile
        ));
    }

    // Parse every model record.
    let records = match parse_records(&file_bytes) {
        Ok(r) => r,
        Err(e) => cm_fail(&format!("Failed to parse CM file {}: {}\n", cmfile, e)),
    };

    // ---- open output stream (cmfetch.c:138-148) ----
    // For -O the output file is named by <key> (== args[1]); handled below.
    // For -o it's the named file; else stdout.
    // We open the concrete Writer inside each mode.

    if opt_index {
        // cmfetch.c:152
        create_ssi_index(&cmfile, &records);
    } else if opt_f {
        // cmfetch.c:153
        let kf = keyfile.unwrap();
        // -O/-o output selection: -f is incompatible with -O; -o allowed.
        let to_stdout = opt_o.is_none();
        let mut out: Box<dyn Write> = match &opt_o {
            Some(f) => Box::new(BufWriter::new(open_out(f))),
            None => Box::new(BufWriter::new(std::io::stdout())),
        };
        multifetch(&cmfile, &records, &kf, &mut out, to_stdout);
        out.flush().ok();
    } else {
        // default: single fetch (cmfetch.c:154-158)
        let key = keyname.unwrap();
        let (mut out, to_stdout): (Box<dyn Write>, bool) = if opt_bigo {
            // cmfetch.c:138-142  -O: output to a file named by <key> (args[1]).
            (Box::new(BufWriter::new(open_out(&args[1]))), false)
        } else if let Some(f) = &opt_o {
            (Box::new(BufWriter::new(open_out(f))), false)
        } else {
            (Box::new(BufWriter::new(std::io::stdout())), true)
        };
        onefetch(&cmfile, &records, &key, &mut out);
        out.flush().ok();
        // cmfetch.c:157
        if !to_stdout {
            println!("\n\nRetrieved CM {}.", key);
        }
    }
}

/// Split the raw CM file into records. A record begins at each line that starts
/// with "INFERNAL1/a" and runs until the next such line (or EOF); it contains
/// the CM ASCII block (through its `//`) and the trailing p7 HMM filter block
/// (through its `//`). This mirrors cm_file_Read()'s notion of a record and its
/// cm->offset. (cm_file.c)
fn parse_records(bytes: &[u8]) -> Result<Vec<Record>, String> {
    let marker: &[u8] = b"INFERNAL1/a";
    let mut starts: Vec<usize> = Vec::new();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        if (i == 0 || bytes[i - 1] == b'\n') && bytes[i..].starts_with(marker) {
            starts.push(i);
        }
        i += 1;
    }
    if starts.is_empty() {
        return Err("no INFERNAL1/a records found".to_string());
    }

    let mut records = Vec::with_capacity(starts.len());
    for (k, &start) in starts.iter().enumerate() {
        let end = if k + 1 < starts.len() { starts[k + 1] } else { n };
        let slice = &bytes[start..end];
        // Read the model without localizing (matches C's CMFileRead, which reads
        // the global model; cm_file::cm_file_read_global uses do_localize=false).
        let cm = cm_file_read_from_reader_opt(std::io::BufReader::new(slice), false)
            .map_err(|e| format!("{:?}", e))?;
        let name = cm.name.clone();
        let acc = cm.acc.clone();
        records.push(Record {
            offset: start as u64,
            name,
            acc,
            cm,
        });
    }
    Ok(records)
}

/// Serialize one model exactly as C's cm_file_WriteASCII does, then patch the
/// INFERNAL1/a banner line to the golden cmfetch's version string.
fn serialize_record(cm: &CM) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    cm_file_write_ascii(&mut buf, cm).expect("cm_file_write_ascii failed");
    // Replace the first line ("INFERNAL1/a [<crate ver> | ...]") with the C banner.
    if let Some(nl) = buf.iter().position(|&b| b == b'\n') {
        let mut out = Vec::with_capacity(buf.len());
        out.extend_from_slice(INFERNAL_BANNER.as_bytes());
        out.extend_from_slice(&buf[nl..]); // keep the '\n' and everything after
        out
    } else {
        buf
    }
}

/// create_ssi_index() — cmfetch.c:170-225
///
/// Build an SSI index; store both NAME (primary) and ACC (secondary) as keys.
fn create_ssi_index(cmfile: &str, records: &[Record]) {
    // cmfetch.c:181
    let ssifile = format!("{}.ssi", cmfile);

    // cmfetch.c:183-186  esl_newssi_Open(ssifile, FALSE, &ns)
    let mut ns = match EslNewSsi::open(&ssifile, false) {
        Ok(ns) => ns,
        Err(infernox::easel::error::InfernalError::Overwrite) => {
            cm_fail(&format!("SSI index {} already exists; delete or rename it", ssifile))
        }
        Err(_) => cm_fail(&format!("failed to open SSI index {}", ssifile)),
    };

    // cmfetch.c:188-189  esl_newssi_AddFile(ns, cmfp->fname, 0, &fh)
    let fh = match ns.add_file(cmfile, 0) {
        Ok(fh) => fh,
        Err(_) => cm_fail(&format!("Failed to add CM file {} to new SSI index\n", cmfile)),
    };

    // cmfetch.c:191-192
    print!("Working...    ");
    std::io::stdout().flush().ok();

    // cmfetch.c:194-208  loop over models.
    let mut ncm = 0i32;
    let mut nprimary: u64 = 0; // == ns->nprimary
    let mut nsecondary: u64 = 0; // == ns->nsecondary
    for r in records {
        ncm += 1;
        // cmfetch.c:198
        if r.name.is_empty() {
            cm_fail(&format!(
                "Every CM must have a name to be indexed. Failed to find name of CM #{}\n",
                ncm
            ));
        }
        // cmfetch.c:200-201  primary key = name, r_off = cm->offset, d_off=0, L=0.
        if ns.add_key(&r.name, fh, r.offset, 0, 0).is_err() {
            cm_fail(&format!("Failed to add key {} to SSI index", r.name));
        }
        nprimary += 1;
        // cmfetch.c:203-206  secondary key (alias) = accession, if present.
        if let Some(ref acc) = r.acc {
            if ns.add_alias(acc, &r.name).is_err() {
                cm_fail(&format!("Failed to add secondary key {} to SSI index", acc));
            }
            nsecondary += 1;
        }
    }

    // cmfetch.c:211-212
    if ns.write().is_err() {
        cm_fail(&format!("Failed to write keys to ssi file {}\n", ssifile));
    }

    // cmfetch.c:214-219
    println!("done.");
    if nsecondary > 0 {
        println!(
            "Indexed {} CMs ({} names and {} accessions).",
            ncm, nprimary, nsecondary
        );
    } else {
        println!("Indexed {} CMs ({} names).", ncm, nprimary);
    }
    println!("SSI index written to file {}", ssifile);
}

/// multifetch() — cmfetch.c:241-289
fn multifetch(cmfile: &str, records: &[Record], keyfile: &str, out: &mut dyn Write, to_stdout: bool) {
    // cmfetch.c:254  open the keyfile.
    let kf_bytes = match std::fs::read(keyfile) {
        Ok(b) => b,
        Err(_) => cm_fail(&format!("Failed to open key file {}\n", keyfile)),
    };
    let kf_text = String::from_utf8_lossy(&kf_bytes);

    // esl_fileparser with comment char '#'; one token (first) per non-blank line.
    let mut keys_in_order: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in kf_text.lines() {
        // strip comment (esl_fileparser_SetCommentChar '#', cmfetch.c:255)
        let content = match line.find('#') {
            Some(p) => &line[..p],
            None => line,
        };
        let mut toks = content.split_whitespace();
        if let Some(tok) = toks.next() {
            // cmfetch.c:262-263  duplicate key is a fatal error.
            if !seen.insert(tok.to_string()) {
                cm_fail(&format!("CM key {} occurs more than once in file {}\n", tok, keyfile));
            }
            keys_in_order.push(tok.to_string());
        }
    }

    // Do we have an SSI index? (cmfetch.c:265,268 branch on cmfp->ssi)
    let have_ssi = open_ssi(cmfile).is_some();

    let mut ncm = 0i32;
    if have_ssi {
        // cmfetch.c:265  with SSI: fetch in keyfile order.
        for key in &keys_in_order {
            onefetch(cmfile, records, key, out);
            ncm += 1;
        }
    } else {
        // cmfetch.c:268-282  without SSI: single pass over the file, output the
        // models whose name or accession is in the keylist, in FILE order.
        for r in records {
            let hit = seen.contains(&r.name)
                || r.acc.as_ref().map(|a| seen.contains(a)).unwrap_or(false);
            if hit {
                out.write_all(&serialize_record(&r.cm)).ok();
                ncm += 1;
            }
        }
    }

    // cmfetch.c:284
    if !to_stdout {
        println!("\nRetrieved {} CMs.", ncm);
    }
}

/// onefetch() — cmfetch.c:299-334
///
/// Fetch the single model whose name or accession == <key>.
fn onefetch(cmfile: &str, records: &[Record], key: &str, out: &mut dyn Write) {
    // cmfetch.c:306-312  if an SSI index exists, position by key first.
    let mut start_idx = 0usize;
    if let Some(mut ssi) = open_ssi(cmfile) {
        match ssi.find_name(key) {
            Ok(e) => {
                // Positioned at r_off; locate the record starting there.
                start_idx = records
                    .iter()
                    .position(|r| r.offset == e.roff)
                    .unwrap_or(0);
            }
            Err(infernox::easel::error::InfernalError::NotFound) => {
                cm_fail(&format!("CM {} not found in SSI index for file {}\n", key, cmfile));
            }
            Err(infernox::easel::error::InfernalError::Format) => {
                cm_fail(&format!("Failed to parse SSI index for {}\n", cmfile));
            }
            Err(_) => cm_fail(&format!(
                "Failed to look up location of CM {} in SSI index of file {}\n",
                key, cmfile
            )),
        }
    }

    // cmfetch.c:314-320  read forward from the positioned record until name/acc
    // matches the key.
    let mut found: Option<&Record> = None;
    for r in &records[start_idx..] {
        if r.name == key || r.acc.as_deref() == Some(key) {
            found = Some(r);
            break;
        }
    }

    match found {
        // cmfetch.c:322-325
        Some(r) => {
            out.write_all(&serialize_record(&r.cm)).ok();
        }
        // cmfetch.c:329-331
        None => {
            cm_fail(&format!("CM {} not found in file {}\n", key, cmfile));
        }
    }
}

/// Open the SSI index for <cmfile> if present: a pressed index (<cmfile>.i1i)
/// takes precedence over <cmfile>.ssi, matching cm_file_Open.
fn open_ssi(cmfile: &str) -> Option<EslSsi> {
    let i1i = format!("{}.i1i", cmfile);
    if Path::new(&i1i).exists() {
        if let Ok(s) = EslSsi::open(&i1i) {
            return Some(s);
        }
    }
    let ssi = format!("{}.ssi", cmfile);
    if Path::new(&ssi).exists() {
        if let Ok(s) = EslSsi::open(&ssi) {
            return Some(s);
        }
    }
    None
}

fn open_out(path: &str) -> std::fs::File {
    match std::fs::File::create(path) {
        Ok(f) => f,
        Err(_) => cm_fail(&format!("Failed to open output file {}\n", path)),
    }
}

// ---- error / usage helpers (cmfetch.c:27-52) ----

fn cm_fail(msg: &str) -> ! {
    // cm_Fail(): print to stderr and exit(1).
    eprint!("\nError: {}", msg);
    if !msg.ends_with('\n') {
        eprintln!();
    }
    exit(1);
}

fn cmdline_failure(argv0: &str, msg: &str) -> ! {
    // cmfetch.c:27-40 cmdline_failure(): the message goes to STDERR (vfprintf to
    // stderr), then the three esl_usage lines + the "-h" pointer to STDOUT, exit 1.
    // esl_usage stamps argv[0]'s basename (hardcoded "cmfetch" via `argv0`), but the
    // final `printf(... do %s -h ..., argv[0])` embeds the FULL argv[0] path - so we
    // echo the real argv[0] there (matching C under any invocation), as the other
    // infernox tools do; it is the one path-dependent line (excluded from byte-diffs).
    let real_argv0 = std::env::args().next().unwrap_or_else(|| argv0.to_string());
    eprint!("{}", msg);
    println!("Usage: {} {}", argv0, USAGE1);
    println!("Usage: {} {}", argv0, USAGE2);
    println!("Usage: {} {}", argv0, USAGE3);
    println!("\nTo see more help on available options, do {} -h\n", real_argv0);
    exit(1);
}

fn cmdline_help(argv0: &str) -> ! {
    // cmfetch.c:42-52  cmdline_help() opens with esl_banner(stdout, argv0, banner).
    // Unlike the Infernal tools (cm_banner), cmfetch uses Easel's esl_banner, which
    // stamps the Easel library version/copyright, not INFERNAL's. esl_banner emits
    // (easel.c:esl_banner): the "# <prog> :: <banner>" title, then the Easel
    // version line, the two copyright lines, and the dashed separator.
    println!("# {} :: {}", argv0, BANNER);
    println!("# Easel 0.49 (Aug 2023)");
    println!("# Copyright (C) 2023 Howard Hughes Medical Institute.");
    println!("# Freely distributed under the BSD open source license.");
    println!("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
    println!("Usage: {} {}", argv0, USAGE1);
    println!("Usage: {} {}", argv0, USAGE2);
    println!("Usage: {} {}", argv0, USAGE3);
    println!("\n where options are:");
    println!("  -h      : help; show brief info on version and usage");
    println!("  -f      : second cmdline arg is a file of names to retrieve");
    println!("  -o <f>  : output CM to file <f> instead of stdout");
    println!("  -O      : output CM to file named <key>");
    println!("  --index : index the <cmfile>, creating <cmfile>.ssi");
    exit(0);
}
