//! CM Emit Map - maps model nodes to consensus positions
//!
//! Port of display.c:CreateEmitMap() from Infernal 1.1.5
//!
//! The emit map tracks where each node's subtree aligns in the consensus sequence.
//! - lpos[nd]: leftmost consensus position for subtree under node nd
//! - rpos[nd]: rightmost consensus position for subtree under node nd
//! - epos[nd]: EL (local end) insertions follow this consensus position

use crate::cm::CM;
use crate::constants::{BIF_ND, END_ND, MATL_ND, MATP_ND, MATR_ND};

/// Maps model nodes to consensus positions.
///
/// Consensus positions are indexed 1..clen.
/// Each array (lpos, rpos, epos) is indexed 0..nodes-1.
///
/// - Residues from an MP go into lpos and rpos in the consensus.
/// - Residues from an IL follow lpos.
/// - Residues from an IR precede rpos.
/// - Residues from an EL follow epos[nd] for the nd that went to EL.
/// - For non-emitters, rpos and lpos are non-inclusive bounds:
///   for example, rpos[0], lpos[0] are 0, clen+1.
/// - All rpos, lpos, epos are valid coords 0..clen+1 in the consensus.
#[derive(Debug, Clone)]
pub struct CMEmitMap {
    /// Left bound of consensus for subtree under node nd [0..nodes-1]
    pub lpos: Vec<i32>,
    /// Right bound of consensus for subtree under node nd [0..nodes-1]
    pub rpos: Vec<i32>,
    /// EL inserts come after this consensus position [0..nodes-1]
    pub epos: Vec<i32>,
    /// Consensus length
    pub clen: i32,
}

impl CMEmitMap {
    /// Create a new emit map with the given number of nodes
    pub fn new(nodes: usize) -> Self {
        CMEmitMap {
            lpos: vec![-1; nodes],
            rpos: vec![-1; nodes],
            epos: vec![-1; nodes],
            clen: 0,
        }
    }

    /// Calculate and return the size of the emit map in megabytes
    pub fn size_mb(&self) -> f32 {
        let bytes = std::mem::size_of::<CMEmitMap>()
            + self.lpos.len() * std::mem::size_of::<i32>()
            + self.rpos.len() * std::mem::size_of::<i32>()
            + self.epos.len() * std::mem::size_of::<i32>();
        bytes as f32 / 1_000_000.0
    }

    /// Dump the emit map to a string for debugging
    pub fn dump(&self, cm: &CM) -> String {
        use crate::cm::node_type_to_str;

        let mut result = format!(
            "CM to consensus emit map; consensus length = {}\n",
            self.clen
        );
        result.push_str(&format!(
            "{:>4} {:>7} {:>9} {:>4} {:>4} {:>4}\n",
            "Node", "State 1", "Node type", "lpos", "rpos", "epos"
        ));
        result.push_str(&format!(
            "{:>4} {:>7} {:>9} {:>4} {:>4} {:>4}\n",
            "----", "-------", "---------", "----", "----", "----"
        ));

        for nd in 0..cm.nodes as usize {
            result.push_str(&format!(
                "{:>4} {:>7} {:>9} {:>4} {:>4} {:>4}\n",
                nd,
                cm.nodemap[nd],
                node_type_to_str(cm.ndtype[nd]),
                self.lpos[nd],
                self.rpos[nd],
                self.epos[nd]
            ));
        }

        result
    }
}

/// Create and fill an emit map for a given CM.
///
/// This is a 1:1 port of CreateEmitMap() from display.c.
///
/// The algorithm uses a stack-based depth-first traversal:
/// 1. Push each node twice: once for left-side processing, once for right-side
/// 2. On left-side visit: record lpos after incrementing cpos for left-emitters
/// 3. On right-side visit: record rpos, then increment cpos for right-emitters
/// 4. For BIF nodes: push both children for recursive processing
/// 5. After traversal, compute epos by reverse iteration
///
/// # Arguments
/// * `cm` - The covariance model
///
/// # Returns
/// * `Some(CMEmitMap)` on success
/// * `None` if the map couldn't be created or validated
pub fn create_emit_map(cm: &CM) -> Option<CMEmitMap> {
    let nodes = cm.nodes as usize;
    let mut map = CMEmitMap::new(nodes);

    // Stack stores pairs: (on_right, node_index)
    // on_right: false = left side, true = right side
    let mut stack: Vec<(bool, i32)> = Vec::with_capacity(nodes * 2);

    let mut cpos: i32 = 0;
    let nd: i32 = 0;

    // Push initial node: left side first
    stack.push((false, nd)); // (on_right=false, nd=0)

    while let Some((on_right, nd)) = stack.pop() {
        let nd_usize = nd as usize;

        if on_right {
            // Right side processing
            map.rpos[nd_usize] = cpos + 1;
            let ndtype = cm.ndtype[nd_usize] as i32;
            if ndtype == MATP_ND || ndtype == MATR_ND {
                cpos += 1;
            }
        } else {
            // Left side processing
            let ndtype = cm.ndtype[nd_usize] as i32;
            if ndtype == MATP_ND || ndtype == MATL_ND {
                cpos += 1;
            }
            map.lpos[nd_usize] = cpos;

            if ndtype == BIF_ND {
                // Push the BIF back on for its right side
                stack.push((true, nd));

                // Push node index for right child
                // For BIF: cnum[v] is the right child S_st
                let bif_state = cm.nodemap[nd_usize];
                let right_child_state = cm.cnum[bif_state as usize];
                let right_child_nd = cm.ndidx[right_child_state as usize];
                stack.push((false, right_child_nd));

                // Push node index for left child
                // For BIF: cfirst[v] is the left child S_st
                let left_child_state = cm.cfirst[bif_state as usize];
                let left_child_nd = cm.ndidx[left_child_state as usize];
                stack.push((false, left_child_nd));
            } else {
                // Push the node back on for right side
                stack.push((true, nd));

                // Push child node on (if not END)
                if ndtype != END_ND {
                    stack.push((false, nd + 1));
                }
            }
        }
    }

    // Construct the epos map: if we do a v->EL transition,
    // the EL follows what consensus position (and its IL insertions, if any)
    let mut epos_cpos: i32 = 0;
    for nd in (0..nodes).rev() {
        let ndtype = cm.ndtype[nd] as i32;
        if ndtype == END_ND {
            epos_cpos = map.lpos[nd];
        } else if ndtype == BIF_ND {
            // Propagate epos for *right* branch
            let bif_state = cm.nodemap[nd];
            let right_child_state = cm.cnum[bif_state as usize];
            let right_child_nd = cm.ndidx[right_child_state as usize] as usize;
            epos_cpos = map.epos[right_child_nd];
        }
        map.epos[nd] = epos_cpos;
    }

    // Consensus length is rpos[0] - 1
    map.clen = map.rpos[0] - 1;

    // Validate that we've filled in the map correctly
    for nd in 0..nodes {
        if map.lpos[nd] == -1 || map.rpos[nd] == -1 || map.epos[nd] == -1 {
            return None;
        }
    }

    Some(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test emit map creation with struct initialization
    #[test]
    fn test_emit_map_basic() {
        let map = CMEmitMap::new(5);
        assert_eq!(map.lpos.len(), 5);
        assert_eq!(map.rpos.len(), 5);
        assert_eq!(map.epos.len(), 5);
        assert_eq!(map.clen, 0);
        assert!(map.lpos.iter().all(|&x| x == -1));
        assert!(map.rpos.iter().all(|&x| x == -1));
        assert!(map.epos.iter().all(|&x| x == -1));
    }

    #[test]
    fn test_emit_map_size() {
        let map = CMEmitMap::new(60);
        let size = map.size_mb();
        // 3 vectors * 60 elements * 4 bytes = 720 bytes + struct overhead
        assert!(size > 0.0);
        assert!(size < 0.001); // Should be less than 1KB
    }
}
