//! Phylogenetic / linkage trees.
//!
//! Faithful port of the pieces of `esl_tree.c` used by cmbuild's MSA
//! clustering: the ESL_TREE object, single-linkage `cluster_engine`, and the
//! `SetTaxaParents` / `SetCladesizes` derived-info builders.
//!
//! C reference: original/easel/esl_tree.c, original/easel/esl_tree.h

/* esl_tree.h:75-78 : clustering strategy flags */
pub const ESL_UPGMA: i32 = 0;
pub const ESL_WPGMA: i32 = 1;
pub const ESL_SINGLE_LINKAGE: i32 = 2;
pub const ESL_COMPLETE_LINKAGE: i32 = 3;

/// C: ESL_TREE (esl_tree.h:20). Only the fields cmbuild's clustering needs.
///
/// For N taxa there are N-1 internal nodes, numbered 0..N-2 (root = 0). When a
/// left/right child is a taxon it is stored as a value <= 0 (negated taxon
/// index); internal-node children are stored as 1..N-2.
#[derive(Debug, Clone)]
pub struct EslTree {
    pub n: usize,           /* number of taxa */
    pub parent: Vec<i32>,   /* [0..N-2] index of parent node */
    pub left: Vec<i32>,     /* [0..N-2] left child  (<=0 taxon, >0 node) */
    pub right: Vec<i32>,    /* [0..N-2] right child (<=0 taxon, >0 node) */
    pub ld: Vec<f64>,       /* [0..N-2] left branch length / linkage height */
    pub rd: Vec<f64>,       /* [0..N-2] right branch length / linkage height */
    pub taxaparent: Option<Vec<i32>>, /* [0..N-1] set by set_taxa_parents */
    pub cladesize: Option<Vec<i32>>,  /* [0..N-2] set by set_cladesizes */
    pub is_linkage_tree: bool,
}

impl EslTree {
    /// C: esl_tree_Create() (esl_tree.c:46). Allocates the N-1 mandatory node
    /// arrays, zero-initialized.
    pub fn create(ntaxa: usize) -> Self {
        debug_assert!(ntaxa >= 2);
        let m = ntaxa - 1;
        EslTree {
            n: ntaxa,
            parent: vec![0; m],
            left: vec![0; m],
            right: vec![0; m],
            ld: vec![0.0; m],
            rd: vec![0.0; m],
            taxaparent: None,
            cladesize: None,
            is_linkage_tree: false,
        }
    }

    /// C: esl_tree_SetTaxaParents() (esl_tree.c:256)
    pub fn set_taxa_parents(&mut self) {
        if self.taxaparent.is_some() {
            return;
        }
        let mut tp = vec![0i32; self.n];
        for i in 0..(self.n - 1) {
            if self.left[i] <= 0 {
                tp[(-self.left[i]) as usize] = i as i32;
            }
            if self.right[i] <= 0 {
                tp[(-self.right[i]) as usize] = i as i32;
            }
        }
        self.taxaparent = Some(tp);
    }

    /// C: esl_tree_SetCladesizes() (esl_tree.c:292)
    pub fn set_cladesizes(&mut self) {
        if self.cladesize.is_some() {
            return;
        }
        let m = self.n - 1;
        let mut cs = vec![0i32; m];
        // for (i = T->N-2; i >= 0; i--)
        for i in (0..m).rev() {
            if self.left[i] <= 0 {
                cs[i] += 1;
            } else {
                cs[i] += cs[self.left[i] as usize];
            }
            if self.right[i] <= 0 {
                cs[i] += 1;
            } else {
                cs[i] += cs[self.right[i] as usize];
            }
        }
        self.cladesize = Some(cs);
    }
}

/// C: cluster_engine() (esl_tree.c, the static engine behind the four public
/// clustering entry points). `d_original` is an NxN symmetric distance matrix.
///
/// Faithful transcription of the O(N^3) agglomerative merge, including the
/// exact min-selection tie-breaking (first strict-minimum wins) and the
/// row/col swap bookkeeping that determines final tree topology.
fn cluster_engine(d_original: &[Vec<f64>], mode: i32) -> EslTree {
    let n0 = d_original.len();
    debug_assert!(n0 >= 2);

    // NxN working copy of the distance matrix.
    let mut d: Vec<Vec<f64>> = d_original.iter().map(|r| r.clone()).collect();
    let mut t = EslTree::create(n0);

    let mut idx: Vec<i32> = (0..n0 as i32).map(|i| -i).collect(); // idx[i] = -i
    let mut nin: Vec<i32> = vec![1; n0];
    let mut height: Vec<f64> = vec![0.0; n0 - 1];

    if mode == ESL_SINGLE_LINKAGE || mode == ESL_COMPLETE_LINKAGE {
        t.is_linkage_tree = true;
    }

    // for (N = D->n; N >= 2; N--)
    let mut cap_n = n0;
    while cap_n >= 2 {
        // Find minimum in the current N x N matrix.
        let mut min_d = d[0][1];
        let mut i = 0usize;
        let mut j = 1usize;
        for row in 0..cap_n {
            for col in (row + 1)..cap_n {
                if d[row][col] < min_d {
                    min_d = d[row][col];
                    i = row;
                    j = col;
                }
            }
        }

        // Add node (index = N-2), joining row/col i and j.
        let node = cap_n - 2;
        t.left[node] = idx[i];
        t.right[node] = idx[j];
        height[node] = if t.is_linkage_tree { min_d } else { min_d / 2.0 };

        t.ld[node] = height[node];
        t.rd[node] = height[node];
        if !t.is_linkage_tree {
            if idx[i] > 0 {
                t.ld[node] = f64::max(0.0, t.ld[node] - height[idx[i] as usize]);
            }
            if idx[j] > 0 {
                t.rd[node] = f64::max(0.0, t.rd[node] - height[idx[j] as usize]);
            }
        }

        if idx[i] > 0 {
            t.parent[idx[i] as usize] = node as i32;
        }
        if idx[j] > 0 {
            t.parent[idx[j] as usize] = node as i32;
        }

        // Build new matrix by merging row/col i+j:
        //  1. move j to N-1 (unless already there)
        //  2. move i to N-2 (unless already there)
        if j != cap_n - 1 {
            for row in 0..cap_n {
                d[row].swap(cap_n - 1, j);
            }
            for col in 0..cap_n {
                let tmp = d[cap_n - 1][col];
                d[cap_n - 1][col] = d[j][col];
                d[j][col] = tmp;
            }
            idx.swap(j, cap_n - 1);
            nin.swap(j, cap_n - 1);
        }
        if i != cap_n - 2 {
            for row in 0..cap_n {
                d[row].swap(cap_n - 2, i);
            }
            for col in 0..cap_n {
                let tmp = d[cap_n - 2][col];
                d[cap_n - 2][col] = d[i][col];
                d[i][col] = tmp;
            }
            idx.swap(i, cap_n - 2);
            nin.swap(i, cap_n - 2);
        }
        i = cap_n - 2;
        j = cap_n - 1;

        // 3. merge i (now N-2) with j (now N-1) per clustering rule.
        for col in 0..cap_n {
            let v = match mode {
                ESL_UPGMA => {
                    (nin[i] as f64 * d[i][col] + nin[j] as f64 * d[j][col])
                        / (nin[i] + nin[j]) as f64
                }
                ESL_WPGMA => (d[i][col] + d[j][col]) / 2.0,
                ESL_SINGLE_LINKAGE => f64::min(d[i][col], d[j][col]),
                ESL_COMPLETE_LINKAGE => f64::max(d[i][col], d[j][col]),
                _ => unreachable!("no such strategy"),
            };
            d[i][col] = v;
            d[col][i] = v;
        }

        nin[i] += nin[j];
        idx[i] = node as i32;

        cap_n -= 1;
    }

    t
}

/// C: esl_tree_SingleLinkage() (esl_tree.c:1776)
pub fn esl_tree_single_linkage(d: &[Vec<f64>]) -> EslTree {
    cluster_engine(d, ESL_SINGLE_LINKAGE)
}
