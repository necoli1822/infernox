//! infernox-cmsearch — search a sequence database with a covariance model.
//!
//! Byte-parity implementation: reproduces C Infernal 1.1.5 `cmsearch` (STD_ANY
//! pipeline) `--tblout` output exactly. This binary is now a thin CLI wrapper:
//! the faithful pipeline lives in the reusable library entry point
//! [`infernal::faithful_search::FaithfulSearcher`], so the exact same search can
//! run in-process from other crates (e.g. tRNAscan-SE).
//!
//! Usage: infernox-cmsearch <cm> <fasta> [--tblout out] [--toponly] [--cpu N]

use infernal::faithful_search::{
    FaithfulConfig, FaithfulHit, FaithfulSearcher, T_F1F3, T_F4F5, T_F6BAND, T_F6CYK, T_F7BAND,
    T_F7INS,
};
use std::sync::atomic::Ordering;

fn read_fasta(path: &str) -> Vec<(String, String, String)> {
    // (name, description, seq)
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error: cannot read sequence file '{}': {}", path, e);
        std::process::exit(1);
    });
    let mut recs = Vec::new();
    let mut name = String::new();
    let mut desc = String::new();
    let mut seq = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            if !name.is_empty() {
                recs.push((name.clone(), desc.clone(), seq.clone()));
            }
            // Skip leading whitespace after '>' (esl_sqio treats "> name" == ">name").
            let mut it = rest.trim_start().splitn(2, char::is_whitespace);
            name = it.next().unwrap_or("").to_string();
            desc = it.next().unwrap_or("").trim().to_string();
            seq.clear();
        } else {
            seq.push_str(line.trim());
        }
    }
    if !name.is_empty() {
        recs.push((name, desc, seq));
    }
    recs
}

const USAGE: &str = "Usage: infernox-cmsearch <cm> <fasta> [--tblout out] [--toponly] [--cpu N]";

fn print_help() {
    println!("infernox-cmsearch — search a sequence database with a covariance model.");
    println!();
    println!("{}", USAGE);
    println!();
    println!("Arguments:");
    println!("  <cm>              covariance model file (must contain a p7 filter)");
    println!("  <fasta>           sequence database to search (FASTA)");
    println!();
    println!("Options:");
    println!("  --tblout <file>   write tabular hit output to <file> (default: stdout)");
    println!("  --toponly         search only the top (given) strand, not the reverse complement");
    println!("  --cpu <N>         number of worker threads to use (default: all cores)");
    println!("  --strict          bit-identical Forward filter (disable FMA/reorder speedups)");
    println!("  -h, --help        print this help message and exit");
}

/// Print an error message plus usage to stderr and exit with code 2.
fn usage_error(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    eprintln!("{}", USAGE);
    eprintln!("Try 'infernox-cmsearch --help' for more information.");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Handle -h/--help anywhere on the command line before any other parsing.
    if args.iter().skip(1).any(|a| a == "-h" || a == "--help") {
        print_help();
        std::process::exit(0);
    }

    // Parse flags anywhere on the command line; the two positional arguments
    // (<cm> <fasta>, in order) may appear before, after, or interspersed with
    // flags (matching C cmsearch / esl_getopts behavior).
    let mut tblout: Option<String> = None;
    let mut toponly = false;
    let mut ncpu: Option<usize> = None;
    let mut strict = false;
    let mut global = false;
    let mut nohmm = false;
    let mut max = false;
    let mut mid = false;
    let mut t_cutoff: Option<f32> = None;
    // C `cmsearch` default: truncated alignment ON. `--notrunc` disables it.
    let mut notrunc = false;
    let mut positionals: Vec<String> = Vec::new();
    let mut ai = 1;
    while ai < args.len() {
        match args[ai].as_str() {
            "--tblout" => {
                let v = args
                    .get(ai + 1)
                    .unwrap_or_else(|| usage_error("--tblout requires a file argument"));
                tblout = Some(v.clone());
                ai += 2;
            }
            "--toponly" => {
                toponly = true;
                ai += 1;
            }
            "-g" => {
                global = true;
                ai += 1;
            }
            "--nohmm" => {
                nohmm = true;
                ai += 1;
            }
            "--max" => {
                max = true;
                ai += 1;
            }
            "--mid" => {
                mid = true;
                ai += 1;
            }
            "-T" => {
                let v = args
                    .get(ai + 1)
                    .unwrap_or_else(|| usage_error("-T requires a numeric bit-score argument"));
                let t: f32 = v
                    .parse()
                    .unwrap_or_else(|_| usage_error("-T argument must be a number"));
                t_cutoff = Some(t);
                ai += 2;
            }
            "--notrunc" => {
                notrunc = true;
                ai += 1;
            }
            "--strict" => {
                strict = true;
                ai += 1;
            }
            "--cpu" => {
                let v = args
                    .get(ai + 1)
                    .unwrap_or_else(|| usage_error("--cpu requires a numeric argument N"));
                let n: usize = v
                    .parse()
                    .unwrap_or_else(|_| usage_error("--cpu argument must be a positive integer"));
                ncpu = Some(n);
                ai += 2;
            }
            other => {
                if other.starts_with('-') && other.len() > 1 {
                    // Unknown flag: skip (C ignores some; keep permissive for CLI parity).
                    ai += 1;
                } else {
                    positionals.push(other.to_string());
                    ai += 1;
                }
            }
        }
    }
    if positionals.len() < 2 {
        usage_error("missing required arguments: <cm> <fasta>");
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
    infernal::cm_pipeline::set_forward_strict(strict);

    // Build the searcher (reads the CM in global config, builds filters + CP9).
    let searcher = FaithfulSearcher::from_cm_file(cmpath).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let recs = read_fasta(fapath);
    let seqs: Vec<&str> = recs.iter().map(|r| r.2.as_str()).collect();

    let e_report: f64 = 10.0;
    let e_inc: f64 = 0.01;
    let cfg = FaithfulConfig { toponly, e_report, global, nohmm, max, mid, t_cutoff, notrunc };

    let reported = searcher.search(&seqs, &cfg);

    // Debug: dump the per-hit cm_alidisplay lines (nohmm path). Behind an env var
    // so it never perturbs normal output. Full concatenated lines (not chunked).
    if std::env::var("INFERNOX_ALIDUMP").is_ok() {
        for (rank, h) in reported.iter().enumerate() {
            if let Some(ad) = &h.alignment {
                let tname = &recs[h.seq_idx].0;
                eprintln!(
                    "ALIDUMP rank={} target={} start={} stop={} cfrom={} cto={}",
                    rank + 1, tname, h.start, h.stop, ad.cfrom_emit, ad.cto_emit
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

    // tblout
    let out = format_tblout(&recs, &reported, e_inc, searcher.model_name(), searcher.model_acc());
    if let Some(path) = tblout {
        std::fs::write(&path, &out).expect("write tblout");
        eprintln!("wrote {} hits to {}", reported.len(), path);
    } else {
        print!("{}", out);
    }
    eprintln!("reported={}", reported.len());

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

fn format_tblout(
    recs: &[(String, String, String)],
    reported: &[FaithfulHit],
    e_inc: f64,
    qname: &str,
    qacc: &str,
) -> String {
    let mut s = String::new();
    // widths (cm_tophits_TabularTargets1)
    let tnamew = reported.iter().map(|h| recs[h.seq_idx].0.len()).max().unwrap_or(0).max(20);
    let qnamew = qname.len().max(20);
    let taccw = 9usize;
    let qaccw = qacc.len().max(9);
    let posw = reported
        .iter()
        .map(|h| h.start.abs().max(h.stop.abs()).to_string().len())
        .max()
        .unwrap_or(0)
        .max(8);

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

    for h in reported {
        let r = &recs[h.seq_idx];
        let tname = r.0.as_str();
        let tdesc = if r.1.is_empty() { "-" } else { r.1.as_str() };
        let strand = if h.in_rc { "-" } else { "+" };
        let inc = if h.evalue <= e_inc { "!" } else { "?" };
        let eval_s = fmt_evalue(h.evalue);
        s.push_str(&format!(
            "{:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:>3} {:>8} {:>8} {:>posw$} {:>posw$} {:>6} {:>5} {:>4} {:>4.2} {:>5.1} {:>6.1} {:>9} {:<3} {}\n",
            tname, "-", qname, qacc, "cm", h.mdl_from, h.mdl_to, h.start, h.stop, strand,
            h.trunc.as_str(), h.pass_idx,
            h.gc, h.bias, h.score, eval_s, inc, tdesc
        ));
    }
    s
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
