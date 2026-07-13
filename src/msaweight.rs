// SPDX-License-Identifier: BSD-3-Clause
//! Relative sequence weighting schemes for cmbuild, plus the shared build-knob
//! bundle used by both the cmbuild binary (`build_one`) and the p7 filter's
//! temporary donor CM (`p7_filter_emit::build_temp_cm`), so a knob that changes
//! the CM emissions changes the filter block identically.
//!
//! Faithful ports of Easel's esl_msaweight.c:
//!   - esl_msaweight_GSC    (Gerstein/Sonnhammer/Chothia tree weights)  [--wgsc]
//!   - esl_msaweight_BLOSUM (Henikoff simple filter weights)             [--wblosum]
//! (esl_msaweight_PB_adv, the default --wpb path, lives in cm_modelmaker.rs.)

use crate::cm_modelmaker as mm;
use crate::easel::msa::EslMsa;

/// Relative weighting scheme (C: --wpb default, --wgsc, --wnone, --wgiven, --wblosum).
#[derive(Clone, Copy, PartialEq)]
pub enum WScheme {
    Pb,
    Gsc,
    None,
    Given,
    Blosum,
}

/// Build-time knobs shared between the CM-body build and the p7 filter donor CM.
#[derive(Clone, Copy)]
pub struct BuildKnobs {
    pub hand: bool,
    pub noss: bool,
    pub wscheme: WScheme,
    pub wid: f64,
    pub symfrac: f32,
    pub fragthresh: f32,
    /// --fraggiven: use the MSA's given ~ fragment annotation, don't infer (so
    /// the temp p7-filter build must NOT re-run mark_fragments either).
    pub fraggiven: bool,
    pub nobalance: bool,
    pub nodetach: bool,
    pub iins: bool,
    pub iflank: bool,
    pub eminseq: f64,
    pub emaxseq: Option<f64>,
    /// --null background (default uniform 0.25); ACGU. Used for the temp CM's null.
    pub null: [f32; 4],
    /// determine_pretend_cm_is_hmm guard: TRUE if --noh3pri/--v1p0/--p56/--prior
    /// force the standard prior (so a 0-bp model is NOT treated as an HMM).
    pub force_standard_prior: bool,
    /// --v1p0: use_wts=FALSE, don't zero ROOT_IL/IR flanking counts.
    pub v1p0: bool,
    /// --p56 or --v1p0: the temp CM uses the v0.56->v1.0.2 prior (cfg->pri).
    pub use_v0p56_prior: bool,
}

/// C: set_relative_weights (cmbuild.c). Sets msa->wgt per the chosen scheme.
pub fn apply_weights(msa: &mut EslMsa, scheme: WScheme, wid: f64, ignore_rf: bool) {
    match scheme {
        WScheme::None => {
            for w in &mut msa.wgt {
                *w = 1.0;
            }
        }
        WScheme::Given => { /* use weights as given in MSA file */ }
        WScheme::Pb => mm::msaweight_pb(msa, ignore_rf), // default --wpb (ignore_rf = !--hand)
        WScheme::Gsc => msaweight_gsc(msa),
        WScheme::Blosum => msaweight_blosum(msa, wid),
    }
}

// --- digital-alphabet residue predicate (RNA K=4, Kp=18) ---
const K_RNA: usize = 4;
const KP_RNA: usize = 18;

#[inline]
fn xis_residue(x: u8) -> bool {
    (x as usize) < K_RNA || ((x as usize) > K_RNA && (x as usize) < KP_RNA - 2)
}

/// esl_dst_XPairId (esl_distance.c): fractional identity = nid / min(len1,len2).
/// IUPAC-degenerate residues count as identity only on exact code match.
fn x_pair_id(ax1: &[u8], ax2: &[u8], alen: usize) -> f64 {
    let mut nid = 0i32;
    let mut len1 = 0i32;
    let mut len2 = 0i32;
    for i in 1..=alen {
        let r1 = xis_residue(ax1[i]);
        let r2 = xis_residue(ax2[i]);
        if r1 {
            len1 += 1;
        }
        if r2 {
            len2 += 1;
        }
        if r1 && r2 && ax1[i] == ax2[i] {
            nid += 1;
        }
    }
    let len = len1.min(len2); // ESL_MIN(len1, len2)
    if len == 0 {
        0.0
    } else {
        nid as f64 / len as f64
    }
}

/// esl_cluster_SingleLinkage (esl_cluster.c) specialized to the msacluster
/// pairwise-identity linkage predicate (link iff pid >= maxid). Returns
/// (assignments[0..nseq], nc). Faithful stack-based traversal, backward scan.
fn single_linkage(msa: &EslMsa, maxid: f64) -> (Vec<usize>, usize) {
    let n = msa.nseq;
    let alen = msa.alen as usize;
    // a = available stack, b = connected-but-unextended stack, c = assignments
    let mut a: Vec<usize> = (0..n).map(|v| n - v - 1).collect(); // push all backwards
    let mut na = n;
    let mut b: Vec<usize> = vec![0; n];
    let mut nb = 0usize;
    let mut c: Vec<usize> = vec![0; n];
    let mut nc = 0usize;

    while na > 0 {
        let v = a[na - 1];
        na -= 1; // pop off a
        b[nb] = v;
        nb += 1; // push onto b
        while nb > 0 {
            let v = b[nb - 1];
            nb -= 1; // pop off b
            c[v] = nc; // assign to cluster nc
            // iterate available list backwards (deletion/swap safe)
            let mut i = na as isize - 1;
            while i >= 0 {
                let w = a[i as usize];
                let do_link = x_pair_id(&msa.ax[v], &msa.ax[w], alen) >= maxid;
                if do_link {
                    a[i as usize] = a[na - 1];
                    na -= 1; // delete w from a
                    b[nb] = w;
                    nb += 1; // push onto b
                }
                i -= 1;
            }
        }
        nc += 1;
    }
    (c, nc)
}

/// A rooted binary tree (esl_tree.c ESL_TREE), enough for GSC weights.
/// Internal nodes 0..N-2. left[v]/right[v]: <=0 => taxon (-value), >0 => node.
struct Tree {
    n: usize,           // number of taxa
    left: Vec<isize>,   // [0..N-2]
    right: Vec<isize>,  // [0..N-2]
    ld: Vec<f64>,       // [0..N-2] left branch length (height for UPGMA)
    rd: Vec<f64>,       // [0..N-2]
    cladesize: Vec<i32>,// [0..N-2]
}

/// esl_dst_XDiffMx (via XPairIdMx): NxN difference matrix, diff = 1 - pid,
/// 0 on the diagonal.
fn x_diff_mx(msa: &EslMsa) -> Vec<Vec<f64>> {
    let n = msa.nseq;
    let alen = msa.alen as usize;
    let mut d = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let pid = x_pair_id(&msa.ax[i], &msa.ax[j], alen);
            let diff = 1.0 - pid;
            d[i][j] = diff;
            d[j][i] = diff;
        }
    }
    d
}

/// esl_tree_UPGMA (cluster_engine with mode==eslUPGMA), then SetCladesizes.
fn upgma(d_original: &[Vec<f64>], n: usize) -> Tree {
    // Clone the distance matrix (whittled to 2x2).
    let mut d: Vec<Vec<f64>> = d_original.to_vec();
    let mut t = Tree {
        n,
        left: vec![0; n - 1],
        right: vec![0; n - 1],
        ld: vec![0.0; n - 1],
        rd: vec![0.0; n - 1],
        cladesize: vec![0; n - 1],
    };
    let mut height = vec![0.0f64; n - 1];
    let mut idx: Vec<isize> = (0..n).map(|i| -(i as isize)).collect(); // idx[i] = -i
    let mut nin: Vec<i32> = vec![1; n];

    let mut cur_n = n;
    while cur_n >= 2 {
        // find minimum in the current cur_n x cur_n submatrix
        let mut min_d = d[0][1];
        let mut i = 0usize;
        let mut j = 1usize;
        for row in 0..cur_n {
            for col in (row + 1)..cur_n {
                if d[row][col] < min_d {
                    min_d = d[row][col];
                    i = row;
                    j = col;
                }
            }
        }
        let node = cur_n - 2; // new internal node index
        t.left[node] = idx[i];
        t.right[node] = idx[j];
        height[node] = min_d / 2.0; // UPGMA (additive tree)
        t.ld[node] = height[node];
        t.rd[node] = height[node];
        // additive tree: subtract child heights (max with 0 for fp roundoff)
        if idx[i] > 0 {
            t.ld[node] = (t.ld[node] - height[idx[i] as usize]).max(0.0);
        }
        if idx[j] > 0 {
            t.rd[node] = (t.rd[node] - height[idx[j] as usize]).max(0.0);
        }

        // move j to cur_n-1, i to cur_n-2 (swapping rows/cols + idx/nin)
        if j != cur_n - 1 {
            for row in 0..cur_n {
                d[row].swap(cur_n - 1, j);
            }
            d.swap(cur_n - 1, j);
            idx.swap(j, cur_n - 1);
            nin.swap(j, cur_n - 1);
        }
        if i != cur_n - 2 {
            for row in 0..cur_n {
                d[row].swap(cur_n - 2, i);
            }
            d.swap(cur_n - 2, i);
            idx.swap(i, cur_n - 2);
            nin.swap(i, cur_n - 2);
        }
        let i = cur_n - 2;
        let j = cur_n - 1;
        // merge i (N-2) with j (N-1): UPGMA weighted mean.
        for col in 0..cur_n {
            d[i][col] = (nin[i] as f64 * d[i][col] + nin[j] as f64 * d[j][col])
                / ((nin[i] + nin[j]) as f64);
            d[col][i] = d[i][col];
        }
        nin[i] += nin[j];
        idx[i] = node as isize;

        cur_n -= 1;
    }

    // esl_tree_SetCladesizes
    for v in (0..=(n - 2)).rev() {
        if t.left[v] <= 0 {
            t.cladesize[v] += 1;
        } else {
            t.cladesize[v] += t.cladesize[t.left[v] as usize];
        }
        if t.right[v] <= 0 {
            t.cladesize[v] += 1;
        } else {
            t.cladesize[v] += t.cladesize[t.right[v] as usize];
        }
    }
    t
}

/// esl_msaweight_GSC — Gerstein/Sonnhammer/Chothia tree weights via UPGMA.
pub fn msaweight_gsc(msa: &mut EslMsa) {
    let nseq = msa.nseq;
    if nseq == 1 {
        msa.wgt[0] = 1.0;
        return;
    }
    let d = x_diff_mx(msa);
    let t = upgma(&d, nseq);
    let mut x = vec![0.0f64; nseq - 1];

    // Postorder: total branch length under each internal node.
    for i in (0..=(nseq - 2)).rev() {
        x[i] = t.ld[i] + t.rd[i];
        if t.left[i] > 0 {
            x[i] += x[t.left[i] as usize];
        }
        if t.right[i] > 0 {
            x[i] += x[t.right[i] as usize];
        }
    }

    // Preorder: apportion weight above each node to its children.
    x[0] = 0.0;
    for i in 0..=(nseq - 2) {
        let mut lw = t.ld[i];
        if t.left[i] > 0 {
            lw += x[t.left[i] as usize];
        }
        let mut rw = t.rd[i];
        if t.right[i] > 0 {
            rw += x[t.right[i] as usize];
        }
        let (lx, rx);
        if lw + rw == 0.0 {
            // all-zero-branch clade: split x[i] by cladesize.
            lx = if t.left[i] > 0 {
                x[i] * (t.cladesize[t.left[i] as usize] as f64 / t.cladesize[i] as f64)
            } else {
                x[i] / t.cladesize[i] as f64
            };
            rx = if t.right[i] > 0 {
                x[i] * (t.cladesize[t.right[i] as usize] as f64 / t.cladesize[i] as f64)
            } else {
                x[i] / t.cladesize[i] as f64
            };
        } else {
            lx = x[i] * lw / (lw + rw);
            rx = x[i] * rw / (lw + rw);
        }
        if t.left[i] <= 0 {
            msa.wgt[(-t.left[i]) as usize] = lx + t.ld[i];
        } else {
            x[t.left[i] as usize] = lx + t.ld[i];
        }
        if t.right[i] <= 0 {
            msa.wgt[(-t.right[i]) as usize] = rx + t.rd[i];
        } else {
            x[t.right[i] as usize] = rx + t.rd[i];
        }
    }

    // Renormalize to sum to N.
    let sum: f64 = msa.wgt.iter().sum();
    for w in &mut msa.wgt {
        *w /= sum;
    }
    for w in &mut msa.wgt {
        *w *= nseq as f64;
    }
}

/// esl_msaweight_BLOSUM (esl_msaweight.c): Henikoff simple filter weights.
///   cluster by single-linkage at fractional-id >= maxid;
///   wgt[i] = 1 / (size of i's cluster); then DNorm and DScale up to nseq.
pub fn msaweight_blosum(msa: &mut EslMsa, maxid: f64) {
    let nseq = msa.nseq;
    if nseq == 1 {
        msa.wgt[0] = 1.0;
        return;
    }
    let (assign, nc) = single_linkage(msa, maxid);
    let mut nmem = vec![0i32; nc];
    for i in 0..nseq {
        nmem[assign[i]] += 1;
    }
    for i in 0..nseq {
        msa.wgt[i] = 1.0 / (nmem[assign[i]] as f64);
    }
    // esl_vec_DNorm(wgt, nseq); esl_vec_DScale(wgt, nseq, nseq)
    let sum: f64 = msa.wgt.iter().sum();
    for w in &mut msa.wgt {
        *w /= sum;
    }
    for w in &mut msa.wgt {
        *w *= nseq as f64;
    }
}
