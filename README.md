# Infernox

**A Rust reimplementation of [Infernal](http://eddylab.org/infernal/) — covariance-model search for structured RNA.**

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

Infernox reimplements Infernal's complete covariance-model (CM) stack — Easel, HMMER (P7), and the Infernal pipeline and tools — in pure Rust. **All ten tool binaries are byte-parity drop-in replacements** for C Infernal 1.1.5: given the same inputs they produce byte-for-byte identical output.

## Overview

[Infernal](http://eddylab.org/infernal/) (INFERence of RNA ALignment) finds structured-RNA homologs (tRNAs, rRNAs, riboswitches, and other Rfam families) by searching sequences against **covariance models** — profile stochastic context-free grammars that jointly model sequence and secondary structure. Its central tool, `cmsearch`, searches a sequence database with a CM.

Infernox reproduces the entire Infernal 1.1.5 tool suite faithfully. The port is a *literal* transcription of the C reference: it mirrors C's exact operations, evaluation order, `float`-vs-`double` arithmetic, and off-by-one boundaries — and it deliberately reproduces the reference's latent quirks rather than "fixing" them — so that every tool's stdout, tabular output, and on-disk files match C exactly.

## Tool status

Every tool is verified byte-for-byte against C Infernal 1.1.5:

| Tool | Purpose | Status |
|------|---------|--------|
| **`infernox-cmsearch`** | Search a sequence database with a CM | ✅ Byte-parity |
| **`infernox-cmscan`** | Search sequence(s) vs a CM database | ✅ Byte-parity |
| **`infernox-cmalign`** | Align sequences to a CM | ✅ Byte-parity |
| **`infernox-cmbuild`** | Build a CM from an alignment | ✅ Byte-parity |
| **`infernox-cmcalibrate`** | Calibrate E-value parameters | ✅ Byte-parity |
| **`infernox-cmstat`** | CM summary statistics | ✅ Byte-parity |
| **`infernox-cmconvert`** | Convert CM file formats | ✅ Byte-parity |
| **`infernox-cmfetch`** | Retrieve a CM from a database | ✅ Byte-parity |
| **`infernox-cmpress`** | Format/compress a CM database | ✅ Byte-parity |
| **`infernox-cmemit`** | Sample sequences from a CM | ✅ Byte-parity |

The only intentional differences are values that *cannot* match by construction: live wall-clock/CPU-time fields, the embedded command-line path (`argv[0]`), and dates.

## Correctness (byte-parity with C Infernal 1.1.5)

Correctness was the primary goal: establish byte-for-byte identity with the C reference as the golden standard before any optimization. The suite is checked against C 1.1.5 with a whole-file diff (normalizing only the unmatchable time/path/date fields):

- **`cmsearch`** — default, `-g` (glocal), `--max`, `--nohmm`, `--mid`, `--rfam`, `--hmmonly`, `--anytrunc`/`--inttrunc`/`--onlytrunc`, `--cyk`, `--notrunc`, `--nonull3`, plus `--tblout`; verified on *E. coli* K-12 MG1655 (NC_000913.3, 4.6 Mb) and *M. rufus* genome-scale searches.
- **`cmscan`** — default, `--anytrunc`, `--rfam`, `--nonull3`, `--fmt 2`.
- **`cmalign`** — default (optimal-accuracy), `--cyk`, `--notrunc`, `-g`, `--nonbanded`, `--matchonly`, `--dnaout`, `--noprob`, `--small` (divide-and-conquer), `--sub` (sub-CM), `--sample` (stochastic traceback).
- **`cmbuild`** — default, `-F`, `--enone`, `--hand`, `--p7ml`, `--refine`(`--gibbs`); stdout report **and** the on-disk `.cm` are identical.
- **`cmcalibrate`** — all four exponential-tail fits (local/glocal × CYK/Inside), identical to the last digit, driven by the same MT19937 RNG stream as C.
- **`cmconvert`** — `-a`/`-b`/`-1`, `--mlhmm`/`--fhmm`; **`cmpress`** — all four index files (`.i1m`/`.i1i`/`.i1f`/`.i1p`) are `cmp`-identical; **`cmfetch`**, **`cmstat`**, **`cmemit`** likewise.

The full two-pass `cm_Pipeline` is reproduced stage by stage (F1 MSV → F2 Viterbi → F3 local Forward → F3b bias → F4/F5 glocal envelope definition → F6 CYK → F7 Inside + null3), including E-value computation, overlap removal, and thresholding.

## Improvements over C Infernal

- **Memory-safe** — Rust's ownership model eliminates the buffer overflows, use-after-free, and out-of-bounds reads that C cannot check at runtime.
- **Self-contained** — the Easel + HMMER + Infernal stack is reimplemented as a **single Rust crate**. `cargo build` produces standalone binaries with **no autotools, no external Easel/HMMER, no C toolchain**. The only third-party crates are `thiserror`, `flate2` (pure-Rust gzip input), and `rayon`.
- **Lower memory footprint** on many workloads — e.g. on *M. genitalium* (580 kb) infernox used **15.3 MB vs C's 35.9 MB (~57% less)**.
- **Verifiable output** — byte-identical to C Infernal, checkable with a golden-output regression harness.
- **Modern tooling** — Cargo build/dependency management; ordinary `cargo install`.

## Performance

Correctness came first; the filters are already AVX2/FMA-accelerated and byte-parity is preserved. Infernox parallelizes a single search over `(chunk × strand)` **windows** (rather than C's per-sequence workers), so it spreads even a single-genome search across all cores; at genome scale on multiple threads it is competitive with C.

Single-threaded, the residual gap is the striped-vs-serial layout of the SIMD filters (C uses Farrar-striped SSE/AVX profiles; infernox's are not yet striped). Representative single-thread run (MG1655, `--cpu 1`):

| Implementation | Time | Peak RSS |
|----------------|------|----------|
| C Infernal 1.1.5 | 2.26 s | 18.5 MB |
| Infernox | 6.04 s | 22.6 MB |

Both are multithreaded (`--cpu <n>`).

## Installation

Requires a Rust toolchain (1.70+). From source:

```bash
git clone https://github.com/necoli1822/infernox.git
cd infernox
cargo build --release
# binaries in target/release/infernox-*
```

For a machine-specific (non-portable) build with native SIMD codegen:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Usage

The binaries are drop-in equivalents of their C counterparts (same options, same output):

```bash
# Search a sequence database with a covariance model
infernox-cmsearch --tblout hits.tbl <cm_file> <seq_file>

# Build, calibrate, and press a CM for scanning
infernox-cmbuild      my.cm my.sto
infernox-cmcalibrate  my.cm
infernox-cmpress      my.cm

# Scan sequences against a pressed CM database
infernox-cmscan --tblout scan.tbl my.cm <seq_file>

# Align sequences to a CM
infernox-cmalign my.cm <seq_file>
```

Any option accepted by the corresponding C tool is accepted here; run a tool with `-h` for the full list.

## Architecture

Infernox is a **single crate** (`infernox`) that mirrors Infernal's library layering as modules:

| Module | Role |
|--------|------|
| `infernox::easel` | Port of the Easel sequence/statistics library (alphabet, MSA/sequence I/O, Gumbel/exponential distributions, MT19937 RNG, SSI index) |
| `infernox::hmmer` | HMMER3 P7 profile-HMM constants used by the CM filters |
| crate root | Covariance-model types, CM file I/O (ASCII/binary 1.0/1.1), the faithful `cm_Pipeline`, the alignment/calibration DP engines, and the ten tool binaries (`src/bin/`) |

Each Rust source file is annotated with the exact C source file, function, and line range it transcribes (`// C <file>:<func>:<lines>:`), so the port can be audited against the reference line by line.

## Changelog

### 0.2.0
- **Striped-SSE (Farrar) Viterbi filter.** The P7 Viterbi filter now has an
  SSE2 SIMD kernel with a striped (Farrar) profile layout, dispatched at runtime
  (`is_x86_feature_detected!("sse2")`) with a scalar fallback on other targets.
  Output is byte-identical to the scalar path (verified by `sse_matches_scalar`)
  and to C 1.1.5 — this is a pure speedup that closes the single-thread
  striped-vs-serial gap noted below.
- **Log-sum lookup table** (`ilogsum_lut`/`ilogsum_with`) for the CP9 DP hot path.
- **Full cross-crate LTO** (`lto = "fat"`): ~1.3% faster DP kernels (+~5s compile).

### 0.1.4
- **`--mid` search pass fix.** The `--mid` pass now sets the F3/F3b/F4/F4b/F5
  filter thresholds to `--Fmid` (default `0.02`), matching C Infernal's
  `cm_pipeline.c` `--mid` handling. Without this, the stricter default
  thresholds dropped hits that C keeps in a `--mid` scan (e.g. an archaeal
  tRNA-Gln in a truncated-contig-end scan). Applies to `cmsearch`, `cmscan`,
  and the library search API.
- **Internal cleanup.** Removed three never-read bindings (a duplicate counter
  reset in `cmscan`, a redundant `Option` unwrap in `cmconvert`). Behaviorally
  inert — all tool output remains byte-identical to C 1.1.5.

### 0.1.3
- Rebased the `do_notrunc_cm` fix onto the published source so the `-g`
  (glocal) `--notrunc` isotype-scan path is byte-identical to C.

### 0.1.2
- First crates.io release of the consolidated single-crate `infernox`.
  CP9 HMM-band subtract-overflow fix (faithful `wrapping_sub`) that removes a
  panic on tRNA-length glocal band paths.

## License

Infernox is distributed under the **BSD 3-Clause License**, the same license as the original Infernal. See [LICENSE](LICENSE).

## Authors

- **Sunju Kim** — Rust reimplementation

## Acknowledgments

- **Infernal** and the **Easel**/**HMMER** libraries by Sean Eddy, Eric Nawrocki, and the Eddy/Rivas laboratory (Harvard University / Howard Hughes Medical Institute).
- Nawrocki EP, Eddy SR. *Infernal 1.1: 100-fold faster RNA homology searches.* Bioinformatics. 2013;29(22):2933–2935.
