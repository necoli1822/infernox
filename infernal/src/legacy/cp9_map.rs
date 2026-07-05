//! CP9 Map - CM to HMM bidirectional mapping
//!
//! Port of cp9_modelmaker.c:CP9_map_cm2hmm() from Infernal 1.1.5
//!
//! The CP9Map creates a bidirectional mapping between Covariance Model (CM)
//! states and CP9 HMM nodes/states. This mapping is essential for building
//! a Plan 9 HMM that mirrors the CM's behavior.

use crate::cm::CM;
use crate::cm_emitmap::{create_emit_map, CMEmitMap};
use crate::constants::{
    E_ST, IL_ST, IR_ST, MATL_ND, MATP_ND, MATR_ND, BEGR_ND,
};

/// HMM state types for mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HmmStateType {
    Match = 0,
    Insert = 1,
    Delete = 2,
}

/// Mapping between CM states/nodes and CP9 HMM nodes/states
///
/// This structure provides bidirectional mapping essential for HMM banding.
/// The mapping accounts for the fact that:
/// - MATP nodes contribute to TWO consensus columns (left and right)
/// - Insert states must be carefully mapped to maintain proper emission semantics
/// - The CM's bifurcation structure requires special handling
#[derive(Debug, Clone)]
pub struct CP9Map {
    /// Left consensus position for each CM node [-1 for non-{MATL,MATR,MATP}]
    /// Index: [0..cm_nodes]
    pub nd2lpos: Vec<i32>,

    /// Right consensus position for each CM node [-1 for non-{MATR,MATP}]
    /// Index: [0..cm_nodes]
    pub nd2rpos: Vec<i32>,

    /// CM node that models each consensus position
    /// Index: [0..=hmm_m] (position 0 is node 0, positions 1..=hmm_m are consensus)
    pub pos2nd: Vec<i32>,

    /// CM state -> HMM node mapping (up to 2 nodes per CM state)
    /// cs2hn[v][0]: first HMM node, cs2hn[v][1]: second or -1
    /// Index: [0..cm_m][0..2]
    pub cs2hn: Vec<[i32; 2]>,

    /// CM state -> HMM state type mapping (0=Match, 1=Insert, 2=Delete)
    /// Index: [0..cm_m][0..2]
    pub cs2hs: Vec<[i32; 2]>,

    /// HMM node/state -> CM state mapping (reverse direction)
    /// hns2cs[k][ks][0]: first CM state, hns2cs[k][ks][1]: second or -1
    /// Index: [0..=hmm_m][0..3][0..2]
    pub hns2cs: Vec<[[i32; 2]; 3]>,

    /// Consensus length (number of HMM nodes)
    pub hmm_m: usize,

    /// Number of CM states
    pub cm_m: usize,

    /// Number of CM nodes
    pub cm_nodes: usize,
}

impl CP9Map {
    /// Allocate and initialize a new CP9Map for the given CM
    pub fn new(cm: &CM) -> Option<Self> {
        // Calculate consensus length (hmm_m) from clen
        let hmm_m = cm.clen as usize;
        if hmm_m == 0 {
            return None;
        }

        let cm_m = cm.m as usize;
        let cm_nodes = cm.nodes as usize;

        // Initialize arrays with -1
        let nd2lpos = vec![-1i32; cm_nodes];
        let nd2rpos = vec![-1i32; cm_nodes];
        let pos2nd = vec![-1i32; hmm_m + 1];
        let cs2hn = vec![[-1i32; 2]; cm_m];
        let cs2hs = vec![[-1i32; 2]; cm_m];
        let hns2cs = vec![[[-1i32; 2]; 3]; hmm_m + 1];

        Some(CP9Map {
            nd2lpos,
            nd2rpos,
            pos2nd,
            cs2hn,
            cs2hs,
            hns2cs,
            hmm_m,
            cm_m,
            cm_nodes,
        })
    }

    /// Build the complete CM to HMM mapping
    ///
    /// This is the main entry point, porting CP9_map_cm2hmm().
    pub fn build_mapping(&mut self, cm: &CM) -> Result<(), String> {
        // Phase 1: Create emit map and copy lpos/rpos
        let emap = create_emit_map(cm).ok_or("Failed to create emit map")?;
        self.copy_emit_map_positions(cm, &emap);

        // Phase 2: Map root node (k=0)
        self.map_root_node(cm);

        // Phase 3: Map all consensus positions
        for k in 1..=self.hmm_m {
            self.map_consensus_position(cm, k)?;
        }

        Ok(())
    }

    /// Copy emit map positions for MATP, MATL, MATR nodes
    fn copy_emit_map_positions(&mut self, cm: &CM, emap: &CMEmitMap) {
        for n in 0..cm.nodes as usize {
            let ndtype = cm.ndtype[n] as i32;

            // Copy lpos for MATP, MATL
            if ndtype == MATP_ND || ndtype == MATL_ND {
                self.nd2lpos[n] = emap.lpos[n];
                let pos = emap.lpos[n] as usize;
                if pos <= self.hmm_m {
                    self.pos2nd[pos] = n as i32;
                }
            }

            // Copy rpos for MATP, MATR
            if ndtype == MATP_ND || ndtype == MATR_ND {
                self.nd2rpos[n] = emap.rpos[n];
                let pos = emap.rpos[n] as usize;
                if pos <= self.hmm_m {
                    self.pos2nd[pos] = n as i32;
                }
            }
        }
    }

    /// Map root node states (k=0)
    fn map_root_node(&mut self, cm: &CM) {
        // ROOT_S -> k=0, ks=0 (Match)
        // State 0 is ROOT_S
        self.map_helper(cm, 0, HmmStateType::Match, 0);

        // ROOT_IL -> k=0, ks=1 (Insert)
        // State 1 is ROOT_IL
        self.map_helper(cm, 0, HmmStateType::Insert, 1);

        // ROOT_IR is handled specially - it maps to the last HMM node's insert
        // (handled in map_consensus_position for k == hmm_m)
    }

    /// Map a single consensus position
    fn map_consensus_position(&mut self, cm: &CM, k: usize) -> Result<(), String> {
        if self.pos2nd[k] == -1 {
            return Err(format!("No CM node mapped to consensus position {}", k));
        }

        let n = self.pos2nd[k] as usize;
        let is_left = self.nd2lpos[n] == k as i32;
        let is_right = self.nd2rpos[n] == k as i32;
        let ndtype = cm.ndtype[n] as i32;

        match ndtype {
            MATP_ND => {
                if is_left {
                    self.map_matp_left(cm, k, n);
                }
                if is_right {
                    self.map_matp_right(cm, k, n)?;
                }
            }
            MATL_ND => {
                self.map_matl(cm, k, n);
            }
            MATR_ND => {
                self.map_matr(cm, k, n)?;
            }
            _ => {
                return Err(format!(
                    "HMM node {} maps to invalid CM node type {}",
                    k, ndtype
                ));
            }
        }

        Ok(())
    }

    /// Map MATP left half
    ///
    /// MATP node state offsets:
    /// 0: MP, 1: ML, 2: MR, 3: D, 4: IL, 5: IR
    fn map_matp_left(&mut self, cm: &CM, k: usize, n: usize) {
        let v_base = cm.nodemap[n] as usize;

        // Match: MATP_MP, MATP_ML
        self.map_helper(cm, k, HmmStateType::Match, v_base); // MATP_MP
        self.map_helper(cm, k, HmmStateType::Match, v_base + 1); // MATP_ML

        // Insert: MATP_IL
        self.map_helper(cm, k, HmmStateType::Insert, v_base + 4); // MATP_IL

        // Delete: MATP_MR, MATP_D
        self.map_helper(cm, k, HmmStateType::Delete, v_base + 2); // MATP_MR
        self.map_helper(cm, k, HmmStateType::Delete, v_base + 3); // MATP_D
    }

    /// Map MATP right half (complex insert logic)
    fn map_matp_right(&mut self, cm: &CM, k: usize, n: usize) -> Result<(), String> {
        let v_base = cm.nodemap[n] as usize;

        // Match: MATP_MP, MATP_MR
        self.map_helper(cm, k, HmmStateType::Match, v_base); // MATP_MP
        self.map_helper(cm, k, HmmStateType::Match, v_base + 2); // MATP_MR

        // Insert: complex logic for finding correct insert state
        self.map_insert_right(cm, k, n, v_base)?;

        // Delete: MATP_ML, MATP_D
        self.map_helper(cm, k, HmmStateType::Delete, v_base + 1); // MATP_ML
        self.map_helper(cm, k, HmmStateType::Delete, v_base + 3); // MATP_D

        Ok(())
    }

    /// Map MATL node
    ///
    /// MATL node state offsets:
    /// 0: ML, 1: D, 2: IL
    fn map_matl(&mut self, cm: &CM, k: usize, n: usize) {
        let v_base = cm.nodemap[n] as usize;

        // Match: MATL_ML
        self.map_helper(cm, k, HmmStateType::Match, v_base); // MATL_ML

        // Insert: MATL_IL
        self.map_helper(cm, k, HmmStateType::Insert, v_base + 2); // MATL_IL

        // Delete: MATL_D
        self.map_helper(cm, k, HmmStateType::Delete, v_base + 1); // MATL_D

        // At last position, also map ROOT_IR
        if k == self.hmm_m {
            self.map_helper(cm, k, HmmStateType::Insert, 2); // ROOT_IR
        }
    }

    /// Map MATR node (similar complex insert logic as MATP right)
    ///
    /// MATR node state offsets:
    /// 0: MR, 1: D, 2: IR
    fn map_matr(&mut self, cm: &CM, k: usize, n: usize) -> Result<(), String> {
        let v_base = cm.nodemap[n] as usize;

        // Match: MATR_MR
        self.map_helper(cm, k, HmmStateType::Match, v_base); // MATR_MR

        // Insert: complex logic (same as MATP right half)
        self.map_insert_right(cm, k, n, v_base)?;

        // Delete: MATR_D
        self.map_helper(cm, k, HmmStateType::Delete, v_base + 1); // MATR_D

        Ok(())
    }

    /// Map insert states for right-side positions (complex logic)
    ///
    /// This handles the complex mapping of IR states which depends on
    /// what models the next consensus position.
    fn map_insert_right(&mut self, cm: &CM, k: usize, n: usize, v_base: usize) -> Result<(), String> {
        let ndtype = cm.ndtype[n] as i32;

        // Get the IR state offset for this node type
        let ir_offset = if ndtype == MATP_ND { 5 } else { 2 }; // MATP_IR or MATR_IR

        if k != self.hmm_m {
            // Not the last position - find what models k+1
            if self.pos2nd[k + 1] == -1 {
                return Err(format!("No CM node mapped to position {}", k + 1));
            }

            let nn = self.pos2nd[k + 1] as usize;
            let nn_type = cm.ndtype[nn] as i32;

            if self.nd2lpos[nn] == (k + 1) as i32 {
                // k+1 is modeled by left half of nn - find closest BEGR above nn
                if let Some(n_begr) = self.find_begr_above(cm, nn) {
                    let v = cm.nodemap[n_begr] as usize + 1; // BEGR_IL
                    self.map_helper(cm, k, HmmStateType::Insert, v);
                }
            } else if self.nd2rpos[nn] == (k + 1) as i32 {
                // k+1 is modeled by right half of nn
                match nn_type {
                    MATP_ND => {
                        let v = cm.nodemap[nn] as usize + 5; // MATP_IR
                        self.map_helper(cm, k, HmmStateType::Insert, v);
                    }
                    MATR_ND => {
                        let v = cm.nodemap[nn] as usize + 2; // MATR_IR
                        self.map_helper(cm, k, HmmStateType::Insert, v);
                    }
                    _ => {}
                }
            }

            // Special case: if k-1 is modeled by MATL, also map this node's IR to k-1
            if k > 1 && self.pos2nd[k - 1] != -1 {
                let prev_nd = self.pos2nd[k - 1] as usize;
                let prev_type = cm.ndtype[prev_nd] as i32;
                if prev_type == MATL_ND {
                    // Map this IR to position k-1
                    let v = v_base + ir_offset;
                    self.map_helper(cm, k - 1, HmmStateType::Insert, v);
                }
            }
        } else {
            // k == hmm_m: use ROOT_IR
            self.map_helper(cm, k, HmmStateType::Insert, 2); // ROOT_IR

            // Also map this node's IR to k
            let v = v_base + ir_offset;
            self.map_helper(cm, k, HmmStateType::Insert, v);
        }

        Ok(())
    }

    /// Find closest BEGR node above given node
    fn find_begr_above(&self, cm: &CM, start_node: usize) -> Option<usize> {
        let mut n = start_node as i32;
        while n >= 0 {
            if cm.ndtype[n as usize] as i32 == BEGR_ND {
                return Some(n as usize);
            }
            n -= 1;
        }
        None
    }

    /// Core mapping helper - updates both directions
    ///
    /// This function updates both:
    /// - cs2hn/cs2hs: CM state -> HMM node/state type
    /// - hns2cs: HMM node/state type -> CM state
    fn map_helper(&mut self, cm: &CM, k: usize, ks: HmmStateType, v: usize) {
        if v >= self.cm_m {
            return;
        }

        let ks_idx = ks as usize;

        // Skip detached insert states (followed by END_E)
        // These are insert states just before an END_E state
        if ks == HmmStateType::Insert {
            let sttype = cm.sttype[v] as i32;
            if sttype == IL_ST || sttype == IR_ST {
                if v + 1 < self.cm_m && cm.sttype[v + 1] as i32 == E_ST {
                    return;
                }
            }
        }

        // CM state -> HMM mapping
        if self.cs2hn[v][0] == -1 {
            self.cs2hn[v][0] = k as i32;
            self.cs2hs[v][0] = ks_idx as i32;
        } else if self.cs2hn[v][1] == -1 {
            self.cs2hn[v][1] = k as i32;
            self.cs2hs[v][1] = ks_idx as i32;
        }
        // If both slots filled, ignore (shouldn't happen in valid CM)

        // HMM -> CM state mapping
        if k <= self.hmm_m {
            if self.hns2cs[k][ks_idx][0] == -1 {
                self.hns2cs[k][ks_idx][0] = v as i32;
            } else if self.hns2cs[k][ks_idx][1] == -1 {
                self.hns2cs[k][ks_idx][1] = v as i32;
            }
        }
    }

    /// Get the HMM node that a CM state maps to (first mapping)
    pub fn get_hmm_node(&self, v: usize) -> Option<usize> {
        if v < self.cm_m && self.cs2hn[v][0] >= 0 {
            Some(self.cs2hn[v][0] as usize)
        } else {
            None
        }
    }

    /// Get the HMM state type that a CM state maps to (first mapping)
    pub fn get_hmm_state_type(&self, v: usize) -> Option<HmmStateType> {
        if v < self.cm_m && self.cs2hs[v][0] >= 0 {
            match self.cs2hs[v][0] {
                0 => Some(HmmStateType::Match),
                1 => Some(HmmStateType::Insert),
                2 => Some(HmmStateType::Delete),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Get the CM states that map to an HMM node/state
    pub fn get_cm_states(&self, k: usize, ks: HmmStateType) -> (Option<usize>, Option<usize>) {
        if k > self.hmm_m {
            return (None, None);
        }

        let ks_idx = ks as usize;
        let v1 = if self.hns2cs[k][ks_idx][0] >= 0 {
            Some(self.hns2cs[k][ks_idx][0] as usize)
        } else {
            None
        };
        let v2 = if self.hns2cs[k][ks_idx][1] >= 0 {
            Some(self.hns2cs[k][ks_idx][1] as usize)
        } else {
            None
        };

        (v1, v2)
    }
}

/// Create a CP9Map from a CM
///
/// This is the main entry point for creating the CM-HMM mapping.
pub fn cp9_map_cm2hmm(cm: &CM) -> Result<CP9Map, String> {
    let mut map = CP9Map::new(cm).ok_or("Failed to allocate CP9Map")?;
    map.build_mapping(cm)?;
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_state_type() {
        assert_eq!(HmmStateType::Match as u8, 0);
        assert_eq!(HmmStateType::Insert as u8, 1);
        assert_eq!(HmmStateType::Delete as u8, 2);
    }

    #[test]
    fn test_cp9map_allocation() {
        // Create a minimal CM for testing
        let mut cm = CM::new(10, 5);
        cm.clen = 3;

        let map = CP9Map::new(&cm);
        assert!(map.is_some());

        let map = map.unwrap();
        assert_eq!(map.hmm_m, 3);
        assert_eq!(map.cm_m, 10);
        assert_eq!(map.cm_nodes, 5);
        assert_eq!(map.nd2lpos.len(), 5);
        assert_eq!(map.nd2rpos.len(), 5);
        assert_eq!(map.pos2nd.len(), 4); // 0..=3
        assert_eq!(map.cs2hn.len(), 10);
        assert_eq!(map.cs2hs.len(), 10);
        assert_eq!(map.hns2cs.len(), 4);
    }

    #[test]
    fn test_zero_clen_returns_none() {
        let cm = CM::new(10, 5);
        // clen is 0 by default
        assert!(CP9Map::new(&cm).is_none());
    }
}
