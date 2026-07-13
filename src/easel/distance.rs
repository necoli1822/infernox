//! Pairwise sequence distances.
//!
//! Faithful port of the digital-sequence distance functions used by cmbuild's
//! MSA clustering: `esl_dst_XPairId`, `esl_dst_XPairIdMx`, `esl_dst_XDiffMx`.
//!
//! C reference: original/easel/esl_distance.c

use crate::easel::alphabet::EslAlphabet;
use crate::easel::constants::ESL_DSQ_SENTINEL;

/// C: esl_abc_XIsResidue() macro (esl_alphabet.h:103):
///   (x < K) || (x > K && x < Kp-2)
#[inline]
fn x_is_residue(abc: &EslAlphabet, x: u8) -> bool {
    let x = x as i32;
    x < abc.K || (x > abc.K && x < abc.Kp - 2)
}

/// C: esl_dst_XPairId() (esl_distance.c:277)
///
/// Fractional identity of two aligned digital seqs. Only exactly matching codes
/// count as identities. Denominator is MIN(len1,len2), where len is the number
/// of residues in each seq. Returns (pid, nid, n).
pub fn esl_dst_x_pair_id(abc: &EslAlphabet, ax1: &[u8], ax2: &[u8]) -> (f64, i32, i32) {
    let mut nid = 0i32;
    let mut len1 = 0i32;
    let mut len2 = 0i32;

    // for (i = 1; ax1[i] != SENTINEL && ax2[i] != SENTINEL; i++)
    let mut i = 1usize;
    while ax1[i] != ESL_DSQ_SENTINEL && ax2[i] != ESL_DSQ_SENTINEL {
        let r1 = x_is_residue(abc, ax1[i]);
        let r2 = x_is_residue(abc, ax2[i]);
        if r1 {
            len1 += 1;
        }
        if r2 {
            len2 += 1;
        }
        if r1 && r2 && ax1[i] == ax2[i] {
            nid += 1;
        }
        i += 1;
    }
    len1 = len1.min(len2);

    let pid = if len1 == 0 {
        0.0
    } else {
        nid as f64 / len1 as f64
    };
    (pid, nid, len1)
}

/// C: esl_dst_XPairIdMx() (esl_distance.c:638)
///
/// NxN fractional-identity matrix. S[i][i]=1, symmetric.
pub fn esl_dst_x_pair_id_mx(abc: &EslAlphabet, ax: &[Vec<u8>], n: usize) -> Vec<Vec<f64>> {
    let mut s = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        s[i][i] = 1.0;
        for j in (i + 1)..n {
            let (pid, _, _) = esl_dst_x_pair_id(abc, &ax[i], &ax[j]);
            s[i][j] = pid;
            s[j][i] = pid;
        }
    }
    s
}

/// C: esl_dst_XDiffMx() (esl_distance.c:688)
///
/// NxN fractional-difference matrix (1 - pairwise identity). D[i][i]=0, symmetric.
pub fn esl_dst_x_diff_mx(abc: &EslAlphabet, ax: &[Vec<u8>], n: usize) -> Vec<Vec<f64>> {
    let mut d = esl_dst_x_pair_id_mx(abc, ax, n);
    for i in 0..n {
        d[i][i] = 0.0;
        for j in (i + 1)..n {
            d[i][j] = 1.0 - d[i][j];
            d[j][i] = d[i][j];
        }
    }
    d
}
