//! Legacy pre-parity code.
//!
//! These modules predate the strict C-parity (byte-for-byte) porting effort and
//! are NOT verified faithful transcriptions of the C Infernal source. They back
//! the older live `cmsearch` path and are kept compiling here while the faithful
//! replacement is built out top-level (`cm_pipeline`, `p7_generic`,
//! `cp9_faithful`, ...). Do NOT extend these for parity work; port fresh from C
//! into the top-level modules instead (see memory `methodology-c-parity-before-speed`).
//!
//! For backward compatibility, `lib.rs` re-exports these with `pub use legacy::*;`
//! so existing `crate::<mod>` / `infernal::<mod>` paths keep resolving; new code
//! should prefer the explicit `crate::legacy::<mod>` path.

pub mod alignment;
pub mod cm_dp;
pub mod cm_dp_hb;
pub mod cm_emitmap;
pub mod cm_qdband;
pub mod cm_subinfo;
pub mod cmsearch;
pub mod cp9;
pub mod cp9_bands;
pub mod cp9_dp;
pub mod cp9_map;
pub mod cp9_modelmaker;
pub mod cp9_mx;
pub mod output;
pub mod p7_dp;
pub mod p7_glocal;
pub mod p7_simd;
pub mod parsetree;
pub mod pipeline;
pub mod simd;
pub mod truncated;
