// SPDX-License-Identifier: BSD-3-Clause
//! infernox-cmpress — prepare a CM database for faster cmscan searches.
//!
//! Faithful port of C Infernal 1.1.5 `cmpress` (cmpress.c). cmpress creates four
//! output files from a CM file:
//!   - `.i1m`  binary core CMs + p7 HMM filters   (byte-parity target)
//!   - `.i1i`  SSI index over `.i1m`              (byte-parity target)
//!   - `.i1f`  binary vectorized MSV filter part  (byte-parity target)
//!   - `.i1p`  binary vectorized profile remainder (byte-parity target)
//!
//! `.i1m` reuses `infernox::cm_file::cm_file_write_binary_vec` (byte-identical to
//! C's cm_file_WriteBinary — cmpress does NOT append to comlog, unlike
//! cmconvert). `.i1i` reuses `infernox::easel::ssi::EslNewSsi` (esl_newssi_* port).
//!
//! ## `.i1f` / `.i1p`
//! These hold HMMER's SSE-striped optimized profile. cmpress builds the generic
//! p7 profile (p7_ProfileConfig LOCAL, L=400), converts it to the striped
//! oprofile (p7_oprofile_Convert), and serializes both files via
//! `infernox::p7_oprofile::cm_p7_oprofile_write` — byte-identical to C
//! cm_p7_oprofile_Write, so a stock C `cmscan` can consume an infernox-pressed DB.

use infernox::cm_file::{cm_file_read_from_reader_opt, cm_file_write_binary_vec};
use infernox::p7_oprofile::cm_p7_oprofile_write;
use infernox::easel::ssi::EslNewSsi;
use std::io::{BufReader, Cursor};

const USAGE: &str = "Usage: infernox-cmpress [-F] <cmfile>";

/// C `cmpress.c` process_commandline `-h` path: full banner + usage + grouped
/// esl_opt_DisplayHelp, verbatim (byte-identical to C's `-h`). Note C's banner
/// text has the "an CM" typo — reproduced faithfully.
fn print_help() {
    print!(
        "# cmpress :: prepare an CM database for faster cmscan searches\n\
# INFERNAL 1.1.5 (Sep 2023)\n\
# Copyright (C) 2023 Howard Hughes Medical Institute.\n\
# Freely distributed under the BSD open source license.\n\
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\
Usage: cmpress [-options] <cmfile>\n\
\n\
Options:\n  \
-h : show brief help on version and usage\n  \
-F : force: overwrite any previous pressed files\n"
    );
}

fn usage_error(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    eprintln!("{}", USAGE);
    std::process::exit(2);
}

/// Split a (possibly multi-model) ASCII CM file into per-model text chunks. Each
/// model begins with the format banner line ("INFERNAL1/a ..." or "INFERNAL-1
/// ..."). Everything from one banner up to (not including) the next is one model.
fn split_models(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut started = false;
    for line in text.lines() {
        if line.starts_with("INFERNAL") {
            if started && !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
            }
            started = true;
        }
        if started {
            cur.push_str(line);
            cur.push('\n');
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().skip(1).any(|a| a == "-h" || a == "--help") {
        print_help();
        std::process::exit(0);
    }

    let mut force = false;
    let mut positionals: Vec<String> = Vec::new();
    let mut ai = 1;
    while ai < args.len() {
        match args[ai].as_str() {
            "-F" => { force = true; ai += 1; }
            other => {
                if other.starts_with('-') && other.len() > 1 {
                    ai += 1;
                } else {
                    positionals.push(other.to_string());
                    ai += 1;
                }
            }
        }
    }
    if positionals.is_empty() {
        usage_error("missing required argument: <cmfile>");
    }
    let cmfile = &positionals[0];
    if cmfile == "-" {
        eprintln!("Can't use - for <cmfile> argument: can't index standard input");
        std::process::exit(1);
    }

    let mfile = format!("{}.i1m", cmfile);
    let ffile = format!("{}.i1f", cmfile);
    let pfile = format!("{}.i1p", cmfile);
    let ssifile = format!("{}.i1i", cmfile);

    // Overwrite protection (matches C open_dbfiles).
    if !force {
        for f in [&ssifile, &mfile, &ffile, &pfile] {
            if std::path::Path::new(f).exists() {
                eprintln!(
                    "Pressed index file {} already exists;\nDelete old cmpress indices first, or use -F to overwrite",
                    f
                );
                std::process::exit(1);
            }
        }
    }

    let text = std::fs::read_to_string(cmfile).unwrap_or_else(|e| {
        eprintln!("File existence/permissions problem in trying to open CM file {}: {}", cmfile, e);
        std::process::exit(1);
    });
    let chunks = split_models(&text);
    if chunks.is_empty() {
        eprintln!("No CMs found in {}", cmfile);
        std::process::exit(1);
    }

    // Build SSI index over the .i1m file. The stored filename is the cmfile path
    // exactly as passed (C stores cmfp->fname), fmt code 0 (CMs have none yet).
    let mut nssi = EslNewSsi::open(&ssifile, force).unwrap_or_else(|e| {
        eprintln!("failed to open SSI index {}: {:?}", ssifile, e);
        std::process::exit(1);
    });
    let fh = nssi.add_file(cmfile, 0).unwrap_or_else(|e| {
        eprintln!("Failed to add CM file {} to new SSI index: {:?}", cmfile, e);
        std::process::exit(1);
    });

    print!("Working...    ");
    use std::io::Write as _;
    let _ = std::io::stdout().flush();

    // Write each CM's binary block to .i1m, recording its start offset for the SSI.
    // .i1f/.i1p accumulate the SSE striped optimized-profile caches in lockstep
    // (cmpress.c:97-114), byte-identical to C cm_p7_oprofile_Write.
    let mut mbytes: Vec<u8> = Vec::new();
    let mut fbytes: Vec<u8> = Vec::new();
    let mut pbytes: Vec<u8> = Vec::new();
    let mut ncm = 0usize;
    let mut nsecondary = 0usize;
    for chunk in &chunks {
        let cm = cm_file_read_from_reader_opt(BufReader::new(Cursor::new(chunk.as_bytes())), false)
            .unwrap_or_else(|e| {
                eprintln!("\nfailed to parse CM #{} in {}: {:?}", ncm + 1, cmfile, e);
                std::process::exit(1);
            });
        if cm.name.is_empty() {
            eprintln!("\nEvery CM must have a name to be indexed. CM #{} has none.", ncm + 1);
            std::process::exit(1);
        }
        let fp7 = match cm.p7.as_ref() {
            Some(p) => p,
            None => {
                eprintln!("\nCM {} (#{}) does not have a filter HMM", cm.name, ncm + 1);
                std::process::exit(1);
            }
        };

        // cmpress.c:106-108 — capture disk positions BEFORE writing this record.
        // cm_offset = ftello(mfp); offs[FOFFSET] = ftello(ffp); offs[POFFSET] = ftello(pfp).
        let cm_offset = mbytes.len() as u64;
        let foffset = fbytes.len() as i64;
        let poffset = pbytes.len() as i64;
        nssi
            .add_key(&cm.name, fh, cm_offset, 0, 0)
            .unwrap_or_else(|e| {
                eprintln!("\nFailed to add key {} to SSI index: {:?}", cm.name, e);
                std::process::exit(1);
            });
        if let Some(acc) = cm.acc.as_deref() {
            nssi.add_alias(acc, &cm.name).unwrap_or_else(|e| {
                eprintln!("\nFailed to add secondary key {} to SSI index: {:?}", acc, e);
                std::process::exit(1);
            });
            nsecondary += 1;
        }

        // Write the CM (+ trailing p7) in v1.1 binary format, capturing fp7_offset
        // (= om->offs[p7_MOFFSET], cmpress.c:114). cmpress does NOT modify comlog
        // (unlike cmconvert), so the bytes match C cmpress exactly.
        let fp7_offset = cm_file_write_binary_vec(&mut mbytes, &cm).unwrap_or_else(|e| {
            eprintln!("\nfailed to write binary CM {}: {}", cm.name, e);
            std::process::exit(1);
        });

        // om->offs = { p7_MOFFSET, p7_FOFFSET, p7_POFFSET } (hmmer.h:74).
        let offs: [i64; 3] = [fp7_offset as i64, foffset, poffset];
        let gfmu = cm.efp7gf_tau as f32; // fp7_evparam[CM_p7_GFMU]
        let gflambda = cm.efp7gf_lambda as f32; // fp7_evparam[CM_p7_GFLAMBDA]
        let nbp = cm.cm_count_nodetype(infernox::constants::MATP_ND);
        cm_p7_oprofile_write(
            &mut fbytes,
            &mut pbytes,
            cm_offset as i64,
            cm.clen,
            cm.w,
            nbp,
            gfmu,
            gflambda,
            offs,
            fp7,
        );
        ncm += 1;
    }

    std::fs::write(&mfile, &mbytes).unwrap_or_else(|e| {
        eprintln!("\nFailed to write {}: {}", mfile, e);
        std::process::exit(1);
    });

    // Write the SSI index (.i1i). esl_newssi_Write sorts keys internally.
    nssi.write().unwrap_or_else(|e| {
        eprintln!("\nSSI indexing failed: {:?}", e);
        std::process::exit(1);
    });

    // .i1f/.i1p: SIMD-striped optimized-profile caches, byte-identical to C
    // cm_p7_oprofile_Write (accumulated above). A stock C cmscan can consume an
    // infernox-pressed DB.
    std::fs::write(&ffile, &fbytes).unwrap_or_else(|e| {
        eprintln!("\nFailed to write {}: {}", ffile, e);
        std::process::exit(1);
    });
    std::fs::write(&pfile, &pbytes).unwrap_or_else(|e| {
        eprintln!("\nFailed to write {}: {}", pfile, e);
        std::process::exit(1);
    });

    println!("done.");
    if nsecondary > 0 {
        println!(
            "Pressed and indexed {} CMs and p7 HMM filters ({} names and {} accessions).",
            ncm, ncm, nsecondary
        );
    } else {
        println!("Pressed and indexed {} CMs and p7 HMM filters ({} names).", ncm, ncm);
    }
    println!("Covariance models and p7 filters pressed into binary file:  {}", mfile);
    println!("SSI index for binary covariance model file:                 {}", ssifile);
    println!("Optimized p7 filter profiles (MSV part)  pressed into:      {}", ffile);
    println!("Optimized p7 filter profiles (remainder) pressed into:      {}", pfile);
}
