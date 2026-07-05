# Changelog — infernox-infernal

All notable changes to the `infernox-infernal` crate are documented here.

## [0.1.2] - 2026-07-05

### Removed

- **Development golden-test scaffolding**: dropped the `tests/golden_phase*.rs`
  integration tests (and the orphaned `tests/data/trna-5.cm`) from the crate.
  They depended on externally C-generated golden fixtures that were never
  vendored (one even hardcoded an absolute developer path), so the published
  package shipped tests that could not pass. The self-contained `src/` unit
  tests (114) remain and pass; `cargo test` is now clean. No library code
  changed — the byte-parity fix from 0.1.1 is included unchanged.

## [0.1.1] - 2026-07-05

### Fixed

- **cp9_faithful (CP9 HMM band gap-fill)**: the `I_k -> M_k+1` gap check now uses
  `wrapping_sub` to faithfully reproduce Infernal C's `int` overflow. C
  (`hmmband.c:2513`) does not guard this transition with `if(r_mn[k+1] != INT_MAX)`,
  so when `M_k+1` is unreached the subtraction `INT_MIN - INT_MAX` overflows and
  wraps to `+1` (≥ -1), leaving the gap unfilled. The previous plain `-` panicked
  in debug and could diverge from C; every reachable case is unaffected (operands
  are small residue coordinates). Restores byte-parity with C Infernal cmsearch.

## [0.1.0] - 2026-07-02

### Initial release

- Byte-parity covariance-model search (`infernox-cmsearch`) vs C Infernal 1.1.5.
- Core library (`infernal`) plus `cm*` tool binaries (cmsearch complete;
  remaining `cm*` tools are documented stubs, under development).
