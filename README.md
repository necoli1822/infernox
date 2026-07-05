# Infernox

**A Rust reimplementation of [Infernal](http://eddylab.org/infernal/) — covariance-model search for structured RNA.**

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

Infernox reimplements Infernal's covariance-model (CM) search stack — Easel, HMMER (P7), and the Infernal pipeline — in pure Rust. This release ships a **complete, byte-parity `cmsearch`**; the remaining Infernal tools are stubbed and under active development.

## Overview

[Infernal](http://eddylab.org/infernal/) (INFERence of RNA ALignment) finds structured-RNA homologs (tRNAs, rRNAs, riboswitches, and other Rfam families) by searching sequences against **covariance models** — profile stochastic context-free grammars that jointly model sequence and secondary structure. Its central tool, `cmsearch`, searches a sequence database with a CM.

Infernox reproduces `cmsearch` faithfully: for the default (non-truncated) search its `--tblout` output is **byte-for-byte identical** to C Infernal 1.1.5.

## Tool status

| Tool | Purpose | Status |
|------|---------|--------|
| **`infernox-cmsearch`** | Search a sequence database with a CM | ✅ **Complete** (byte-parity with C Infernal 1.1.5) |
| `infernox-cmbuild` | Build a CM from an alignment | 🚧 Under development (stub) |
| `infernox-cmcalibrate` | Calibrate E-value parameters | 🚧 Under development (stub) |
| `infernox-cmalign` | Align sequences to a CM | 🚧 Under development (stub) |
| `infernox-cmscan` | Search sequence(s) vs a CM database | 🚧 Under development (stub) |
| `infernox-cmpress` | Format/compress a CM database | 🚧 Under development (stub) |
| `infernox-cmfetch` | Retrieve a CM from a database | 🚧 Under development (stub) |
| `infernox-cmemit` | Sample sequences from a CM | 🚧 Under development (stub) |
| `infernox-cmstat` | CM summary statistics | 🚧 Under development (stub) |
| `infernox-cmconvert` | Convert CM file formats | 🚧 Under development (stub) |

Stub tools compile and are installed, but print an "under development" message and exit non-zero.

## Correctness (byte-parity with C Infernal 1.1.5)

`infernox-cmsearch` is verified against C Infernal 1.1.5 `cmsearch` (STD_ANY pipeline), `--tblout` output:

- **E. coli K-12 MG1655** (NC_000913.3, 4,641,652 bp): **87 hits, header + data byte-identical** to C (md5 of hit rows matches).
- **Small case** (tRNA5 CM × sample tRNAs): **10 hits byte-identical**.

The full two-pass `cm_Pipeline` is reproduced stage by stage (F1 MSV → F3 local Forward → F3b bias → F4 glocal Forward → F5 envelope definition → F6 CYK → F7 Inside + null3), including E-value computation, overlap removal, and thresholding.

## Improvements over C Infernal

- **Memory-safe** — Rust's ownership model eliminates buffer overflows, use-after-free, and out-of-bounds reads that C cannot check at runtime.
- **Lower memory footprint** on many workloads — e.g. on *M. genitalium* (580 kb) infernox used **15.3 MB vs C's 35.9 MB (~57% less)**.
- **Self-contained** — the Easel + HMMER + Infernal stack is reimplemented in a single Rust workspace. `cargo build` produces a standalone binary with **no autotools, no external Easel/HMMER, no C toolchain**.
- **Verifiable output** — byte-identical to C Infernal, checkable with a golden-output regression harness.
- **Modern tooling** — Cargo build/dependency management; ordinary `cargo install`.

## Performance

Correctness came first; performance work is ongoing. On the same box, MG1655, single-threaded (`--cpu 1`):

| Implementation | Time | Peak RSS |
|----------------|------|----------|
| C Infernal 1.1.5 | 2.26 s | 18.5 MB |
| Infernox | 6.04 s | 22.6 MB |

So single-threaded infernox is currently **~2.7× slower** than C (same order of magnitude, not drastically slower). Both C Infernal and infernox are multithreaded (`--cpu <n>`); they differ in granularity — C parallelizes over sequence workers, while infernox parallelizes over `(chunk × strand)` **windows**, so it can spread a single-genome search across all cores. SIMD acceleration of the MSV/Forward filters (C uses AVX2; infernox is currently scalar) is the main remaining gap and is under development.

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

```bash
# Search a sequence database with a covariance model
infernox-cmsearch <cm_file> <seq_file> --tblout hits.tbl

# Options
#   --tblout <file>   write tabular per-hit output
#   --toponly         search only the top (given) strand
#   --cpu <n>         number of worker threads (default: all cores)
```

Example with the bundled sample CM:

```bash
cargo run --release --bin infernox-cmsearch -- \
    infernal/tests/data/trna-5.cm genome.fna --tblout trna_hits.tbl
```

## Architecture

Infernox is a Cargo workspace mirroring Infernal's library layering:

| Crate | Role |
|-------|------|
| `easel` | Port of the Easel sequence/statistics library (alphabet, Gumbel/exponential distributions, RNG) |
| `hmmer` | Port of the HMMER3 P7 profile-HMM components used by the CM filters |
| `infernal` | Covariance-model types, CM file I/O, the faithful `cm_Pipeline`, and the tool binaries |

Pre-parity modules retained during the port live under `infernal/src/legacy/` and are not part of the verified `cmsearch` path.

## License

Infernox is distributed under the **BSD 3-Clause License**, the same license as the original Infernal. See [LICENSE](LICENSE).

## Authors

- **Sunju Kim** — Rust reimplementation

## Acknowledgments

- **Infernal** and the **Easel**/**HMMER** libraries by Sean Eddy, Eric Nawrocki, and the Eddy/Rivas laboratory (Harvard University / Howard Hughes Medical Institute).
- Nawrocki EP, Eddy SR. *Infernal 1.1: 100-fold faster RNA homology searches.* Bioinformatics. 2013;29(22):2933–2935.
