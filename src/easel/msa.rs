//! Multiple sequence alignment object.
//!
//! Faithful port of `esl_msa.c` / `esl_msa.h` (the ESL_MSA struct + the
//! allocation / annotation-setter semantics needed for a Stockholm/Pfam
//! round-trip). Field names mirror the C source.
//!
//! C reference: original/easel/esl_msa.h (struct esl_msa_s), esl_msa.c.

use crate::easel::alphabet::EslAlphabet;

/* esl_msa.h:24-30 : cutoff array indices */
pub const ESL_MSA_TC1: usize = 0;
pub const ESL_MSA_TC2: usize = 1;
pub const ESL_MSA_GA1: usize = 2;
pub const ESL_MSA_GA2: usize = 3;
pub const ESL_MSA_NC1: usize = 4;
pub const ESL_MSA_NC2: usize = 5;
pub const ESL_MSA_NCUTS: usize = 6;

/* esl_msa.h:128-129 : flags for msa->flags */
pub const ESL_MSA_HASWGTS: i32 = 1 << 0; /* 1 if wgts were set, 0 if default 1.0's */
pub const ESL_MSA_DIGITAL: i32 = 1 << 1; /* if ax[][] is used instead of aseq[][]  */

/// ESL_MSA — a multiple sequence alignment.
///
/// C reference: `esl_msa.h`, `typedef struct esl_msa_s { ... } ESL_MSA;`
///
/// Text mode uses `aseq[i]`; digital mode uses `ax[i]` (with a sentinel at
/// index 0, so residues live at `ax[i][1..=alen]`, exactly like C).
#[derive(Debug, Default, Clone)]
pub struct EslMsa {
    /* ::cexcerpt::msa_mandatory:: */
    pub aseq: Vec<String>,   /* alignment itself [0..nseq-1], text mode           */
    pub sqname: Vec<String>, /* sequence names [0..nseq-1]                        */
    pub wgt: Vec<f64>,       /* sequence weights [0..nseq-1], default 1.0         */
    pub alen: i64,           /* length of alignment (columns); or -1 if growable  */
    pub nseq: usize,         /* number of seqs in alignment                       */
    pub flags: i32,          /* flags for what info has been set                  */

    /* digital alphabet storage (esl_msa.h : abc, ax) */
    pub is_digital: bool,
    pub ax: Vec<Vec<u8>>, /* digitized aseqs [0..nseq-1][0..=alen], ax[i][0]=sentinel */

    /* ::cexcerpt::msa_optional:: — stuff we understand and might have */
    pub name: Option<String>,    /* name of alignment (#=GF ID), or None    */
    pub desc: Option<String>,    /* description (#=GF DE), or None          */
    pub acc: Option<String>,     /* accession (#=GF AC), or None            */
    pub au: Option<String>,      /* author (#=GF AU), or None               */
    pub ss_cons: Option<String>, /* consensus sec structure (#=GC SS_cons)  */
    pub sa_cons: Option<String>, /* consensus surface access (#=GC SA_cons) */
    pub pp_cons: Option<String>, /* consensus posterior prob (#=GC PP_cons) */
    pub rf: Option<String>,      /* reference coord system  (#=GC RF)       */
    pub mm: Option<String>,      /* model mask              (#=GC MM)       */
    pub sqacc: Option<Vec<Option<String>>>,  /* per-seq accession  (#=GS AC) */
    pub sqdesc: Option<Vec<Option<String>>>, /* per-seq desc       (#=GS DE) */
    pub ss: Option<Vec<Option<String>>>,     /* per-seq SS  (#=GR SS)        */
    pub sa: Option<Vec<Option<String>>>,     /* per-seq SA  (#=GR SA)        */
    pub pp: Option<Vec<Option<String>>>,     /* per-seq PP  (#=GR PP)        */
    pub cutoff: [f32; ESL_MSA_NCUTS], /* NC/TC/GA cutoffs                    */
    pub cutset: [bool; ESL_MSA_NCUTS], /* TRUE if a cutoff is set           */

    /* Optional info we don't parse but can regurgitate (unparsed markup) */
    pub comment: Vec<String>, /* free text comments                              */

    pub gf_tag: Vec<String>, /* markup tags for unparsed #=GF lines               */
    pub gf: Vec<String>,     /* annotations for unparsed #=GF lines               */

    pub gs_tag: Vec<String>,           /* markup tags for unparsed #=GS lines     */
    pub gs: Vec<Vec<Option<String>>>,  /* [0..ngs-1][0..nseq-1] markup            */

    pub gc_tag: Vec<String>, /* markup tags for unparsed #=GC lines               */
    pub gc: Vec<String>,     /* [0..ngc-1] markup, each length alen               */

    pub gr_tag: Vec<String>,           /* markup tags for unparsed #=GR lines     */
    pub gr: Vec<Vec<Option<String>>>,  /* [0..ngr-1][0..nseq-1] markup            */
}

impl EslMsa {
    /* esl_msa.c:esl_msa_Create — a growable (text-mode) MSA. */
    pub fn new() -> Self {
        EslMsa {
            alen: -1,
            ..Default::default()
        }
    }

    #[inline]
    pub fn ngf(&self) -> usize {
        self.gf_tag.len()
    }
    #[inline]
    pub fn ngs(&self) -> usize {
        self.gs_tag.len()
    }
    #[inline]
    pub fn ngc(&self) -> usize {
        self.gc_tag.len()
    }
    #[inline]
    pub fn ngr(&self) -> usize {
        self.gr_tag.len()
    }

    /// esl_msa_CheckUniqueNames (esl_msa.c): returns false if any duplicate
    /// sequence name exists (the writer then makes names unique).
    pub fn check_unique_names(&self) -> bool {
        let mut seen = std::collections::HashSet::new();
        for nm in &self.sqname {
            if !seen.insert(nm.as_str()) {
                return false;
            }
        }
        true
    }

    /// Grow all per-sequence arrays to accommodate a new sequence, returning
    /// its index. Mirrors esl_msa_Expand + stockholm_get_seqidx storing a name.
    pub(crate) fn add_seq(&mut self, name: &str) -> usize {
        let idx = self.nseq;
        self.sqname.push(name.to_string());
        self.aseq.push(String::new());
        self.ax.push(vec![0u8]); /* sentinel at [0] */
        self.wgt.push(-1.0); /* -1 == "unset", per C parser convention */
        self.nseq += 1;
        self.grow_optional_seq_arrays();
        idx
    }

    fn grow_optional_seq_arrays(&mut self) {
        let n = self.nseq;
        for opt in [
            &mut self.sqacc,
            &mut self.sqdesc,
            &mut self.ss,
            &mut self.sa,
            &mut self.pp,
        ] {
            if let Some(v) = opt {
                while v.len() < n {
                    v.push(None);
                }
            }
        }
        for v in self.gs.iter_mut().chain(self.gr.iter_mut()) {
            while v.len() < n {
                v.push(None);
            }
        }
    }

    /// esl_msa_SetDefaultWeights (esl_msa.c): all weights = 1.0.
    pub fn set_default_weights(&mut self) {
        for w in &mut self.wgt {
            *w = 1.0;
        }
    }

    /// Fill in digital `ax[]` from `aseq[]` using the given alphabet.
    /// Mirrors esl_msa_Digitize: ax[i][0] is a sentinel, residues at [1..=alen].
    pub fn digitize(&mut self, abc: &EslAlphabet) {
        self.ax.clear();
        for s in &self.aseq {
            let mut row = Vec::with_capacity(s.len() + 2);
            row.push(crate::easel::constants::ESL_DSQ_SENTINEL);
            for b in s.bytes() {
                let c = if (b as usize) < 128 {
                    abc.inmap[b as usize]
                } else {
                    crate::easel::constants::ESL_DSQ_ILLEGAL
                };
                row.push(c);
            }
            row.push(crate::easel::constants::ESL_DSQ_SENTINEL);
            self.ax.push(row);
        }
        self.is_digital = true;
        self.flags |= ESL_MSA_DIGITAL;
    }

    /// C: esl_msa_SequenceSubset() (esl_msa.c:2276)
    ///
    /// Select the subset of sequences flagged TRUE in `useme` into a new,
    /// smaller MSA (same alignment length; columns are NOT minimized). Parsed
    /// per-seq and consensus annotation is transferred; unparsed GC/GF markup
    /// and free-text comments are dropped (potentially invalidated by subsetting).
    /// Weights are transferred exactly. Panics if the subset is empty.
    pub fn sequence_subset(&self, useme: &[bool]) -> EslMsa {
        let nnew = (0..self.nseq).filter(|&i| useme[i]).count();
        assert!(nnew > 0, "No sequences selected");

        let digital = self.flags & ESL_MSA_DIGITAL != 0;
        let mut new_msa = EslMsa {
            alen: self.alen,
            nseq: nnew,
            is_digital: digital,
            ..Default::default()
        };

        // Prepare optional per-seq array containers only if the source has them.
        let mut n_sqacc = self.sqacc.as_ref().map(|_| Vec::with_capacity(nnew));
        let mut n_sqdesc = self.sqdesc.as_ref().map(|_| Vec::with_capacity(nnew));
        let mut n_ss = self.ss.as_ref().map(|_| Vec::with_capacity(nnew));
        let mut n_sa = self.sa.as_ref().map(|_| Vec::with_capacity(nnew));
        let mut n_pp = self.pp.as_ref().map(|_| Vec::with_capacity(nnew));

        // GS / GR: same tags, subset of seqs.
        new_msa.gs_tag = self.gs_tag.clone();
        new_msa.gr_tag = self.gr_tag.clone();
        let mut n_gs: Vec<Vec<Option<String>>> = vec![Vec::with_capacity(nnew); self.gs_tag.len()];
        let mut n_gr: Vec<Vec<Option<String>>> = vec![Vec::with_capacity(nnew); self.gr_tag.len()];

        for oidx in 0..self.nseq {
            if !useme[oidx] {
                continue;
            }
            if digital {
                new_msa.ax.push(self.ax[oidx].clone());
                new_msa.aseq.push(String::new());
            } else {
                new_msa.aseq.push(self.aseq[oidx].clone());
            }
            new_msa.sqname.push(self.sqname[oidx].clone());
            new_msa.wgt.push(self.wgt[oidx]);

            if let Some(v) = n_sqacc.as_mut() {
                v.push(self.sqacc.as_ref().unwrap()[oidx].clone());
            }
            if let Some(v) = n_sqdesc.as_mut() {
                v.push(self.sqdesc.as_ref().unwrap()[oidx].clone());
            }
            if let Some(v) = n_ss.as_mut() {
                v.push(self.ss.as_ref().unwrap()[oidx].clone());
            }
            if let Some(v) = n_sa.as_mut() {
                v.push(self.sa.as_ref().unwrap()[oidx].clone());
            }
            if let Some(v) = n_pp.as_mut() {
                v.push(self.pp.as_ref().unwrap()[oidx].clone());
            }
            for (ti, col) in n_gs.iter_mut().enumerate() {
                col.push(self.gs[ti].get(oidx).cloned().flatten());
            }
            for (ti, col) in n_gr.iter_mut().enumerate() {
                col.push(self.gr[ti].get(oidx).cloned().flatten());
            }
        }

        new_msa.sqacc = n_sqacc;
        new_msa.sqdesc = n_sqdesc;
        new_msa.ss = n_ss;
        new_msa.sa = n_sa;
        new_msa.pp = n_pp;
        new_msa.gs = n_gs;
        new_msa.gr = n_gr;

        new_msa.flags = self.flags;

        new_msa.name = self.name.clone();
        new_msa.desc = self.desc.clone();
        new_msa.acc = self.acc.clone();
        new_msa.au = self.au.clone();
        new_msa.ss_cons = self.ss_cons.clone();
        new_msa.sa_cons = self.sa_cons.clone();
        new_msa.pp_cons = self.pp_cons.clone();
        new_msa.rf = self.rf.clone();
        new_msa.mm = self.mm.clone();

        new_msa.cutoff = self.cutoff;
        new_msa.cutset = self.cutset;

        new_msa
    }
}
