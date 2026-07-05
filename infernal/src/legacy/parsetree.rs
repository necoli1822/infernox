//! Parse tree structure for CM alignment traceback
//!
//! A parse tree represents the alignment of a sequence to a covariance model,
//! showing which states were visited and which residues were emitted.

/// Parse tree node representing one step in the alignment
///
/// The parse tree is a doubly-linked tree structure where each node
/// represents a state visit in the CM alignment. Nodes are connected
/// via nxtl (next left sibling), nxtr (next right sibling), and prv (parent).
#[derive(Debug, Clone)]
pub struct Parsetree {
    /// Number of nodes in the tree
    pub n: i32,
    /// Left sequence position emitted (1-indexed, 0 if no emission)
    pub emitl: Vec<i32>,
    /// Right sequence position emitted (1-indexed, 0 if no emission)
    pub emitr: Vec<i32>,
    /// State index in the CM
    pub state: Vec<i32>,
    /// Next left child index (-1 if none)
    pub nxtl: Vec<i32>,
    /// Next right child index (-1 if none)
    pub nxtr: Vec<i32>,
    /// Previous node index (-1 if root)
    pub prv: Vec<i32>,
    /// Per-node truncation marginal mode (C `tr->mode`): TRMODE_J(3), TRMODE_L(2),
    /// TRMODE_R(1), TRMODE_T(0). Defaults to TRMODE_J for non-truncated parses.
    pub mode: Vec<i8>,
}

impl Parsetree {
    /// Create new parsetree with given capacity
    pub fn new(capacity: usize) -> Self {
        Parsetree {
            n: 0,
            emitl: Vec::with_capacity(capacity),
            emitr: Vec::with_capacity(capacity),
            state: Vec::with_capacity(capacity),
            nxtl: Vec::with_capacity(capacity),
            nxtr: Vec::with_capacity(capacity),
            prv: Vec::with_capacity(capacity),
            mode: Vec::with_capacity(capacity),
        }
    }

    /// Add a node to the parsetree
    ///
    /// # Arguments
    /// * `emitl` - Left sequence position (1-indexed, 0 if no left emission)
    /// * `emitr` - Right sequence position (1-indexed, 0 if no right emission)
    /// * `state` - State index in the CM
    /// * `nxtl` - Index of next left child (-1 if none)
    /// * `nxtr` - Index of next right child (-1 if none)
    /// * `prv` - Index of parent node (-1 if root)
    ///
    /// # Returns
    /// The index of the newly added node
    pub fn add_node(
        &mut self,
        emitl: i32,
        emitr: i32,
        state: i32,
        nxtl: i32,
        nxtr: i32,
        prv: i32,
    ) -> i32 {
        // Default marginal mode is TRMODE_J (3) — correct for non-truncated parses.
        self.add_node_mode(emitl, emitr, state, nxtl, nxtr, prv, 3)
    }

    /// Add a node carrying an explicit truncation marginal mode (C
    /// `InsertTraceNodewithMode`). See [`Parsetree::add_node`] for the shared args.
    #[allow(clippy::too_many_arguments)]
    pub fn add_node_mode(
        &mut self,
        emitl: i32,
        emitr: i32,
        state: i32,
        nxtl: i32,
        nxtr: i32,
        prv: i32,
        mode: i8,
    ) -> i32 {
        let idx = self.n;
        self.emitl.push(emitl);
        self.emitr.push(emitr);
        self.state.push(state);
        self.nxtl.push(nxtl);
        self.nxtr.push(nxtr);
        self.prv.push(prv);
        self.mode.push(mode);
        self.n += 1;
        idx
    }

    /// Get the root node index (always 0)
    pub fn root(&self) -> Option<i32> {
        if self.n > 0 {
            Some(0)
        } else {
            None
        }
    }

    /// Get state at node index
    pub fn get_state(&self, idx: usize) -> Option<i32> {
        if idx < self.state.len() {
            Some(self.state[idx])
        } else {
            None
        }
    }

    /// Get emissions at node index (emitl, emitr)
    pub fn get_emissions(&self, idx: usize) -> Option<(i32, i32)> {
        if idx < self.emitl.len() {
            Some((self.emitl[idx], self.emitr[idx]))
        } else {
            None
        }
    }

    /// Get hit boundaries from the parsetree
    ///
    /// Returns (start, end) positions in the sequence (1-indexed).
    /// Scans all nodes to find the minimum left emission and maximum right emission.
    pub fn get_hit_bounds(&self) -> Option<(i32, i32)> {
        if self.n == 0 {
            return None;
        }

        let mut min_l = i32::MAX;
        let mut max_r = i32::MIN;

        for i in 0..self.n as usize {
            let l = self.emitl[i];
            let r = self.emitr[i];
            // Only consider actual emissions (non-zero positions)
            if l > 0 && l < min_l {
                min_l = l;
            }
            if r > 0 && r > max_r {
                max_r = r;
            }
        }

        if min_l == i32::MAX || max_r == i32::MIN {
            // No emissions found (shouldn't happen in valid parsetree)
            Some((1, 1))
        } else {
            Some((min_l, max_r))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsetree_new() {
        let pt = Parsetree::new(100);
        assert_eq!(pt.n, 0);
        assert_eq!(pt.state.capacity(), 100);
    }

    #[test]
    fn test_parsetree_add_node() {
        let mut pt = Parsetree::new(10);

        // Add root node (state 0, no parent)
        let idx = pt.add_node(1, 76, 0, -1, -1, -1);
        assert_eq!(idx, 0);
        assert_eq!(pt.n, 1);
        assert_eq!(pt.state[0], 0);
        assert_eq!(pt.emitl[0], 1);
        assert_eq!(pt.emitr[0], 76);

        // Add child node
        let idx2 = pt.add_node(2, 75, 1, -1, -1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(pt.n, 2);
        assert_eq!(pt.prv[1], 0); // parent is node 0
    }

    #[test]
    fn test_parsetree_root() {
        let mut pt = Parsetree::new(10);
        assert_eq!(pt.root(), None);

        pt.add_node(1, 76, 0, -1, -1, -1);
        assert_eq!(pt.root(), Some(0));
    }

    #[test]
    fn test_parsetree_get_state() {
        let mut pt = Parsetree::new(10);
        pt.add_node(1, 76, 42, -1, -1, -1);

        assert_eq!(pt.get_state(0), Some(42));
        assert_eq!(pt.get_state(1), None);
    }

    #[test]
    fn test_parsetree_get_emissions() {
        let mut pt = Parsetree::new(10);
        pt.add_node(5, 10, 0, -1, -1, -1);

        assert_eq!(pt.get_emissions(0), Some((5, 10)));
        assert_eq!(pt.get_emissions(1), None);
    }
}
