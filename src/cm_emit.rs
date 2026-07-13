//! cm_emit — faithful port of the sequence-emission core used by `cmemit`.
//!
//! This module ports the pieces of Infernal 1.1.5 that `cmemit` needs to
//! sample sequences from a covariance model, byte-for-byte identically to the
//! C tool for a fixed RNG seed:
//!
//!   * `esl_rnd_FChoose` (esl_random.c) — sample an index from a float prob vector.
//!   * `EmitParsetree`  (cm_parsetree.c:1889) — stochastic parse-tree/sequence
//!     generator. We port the exact PDA traversal and RNG draw order, but only
//!     accumulate the emitted residue sequence (the parse tree itself is only
//!     needed for `-a`/`--tfile`, which cmemit's `-u`/`-c` modes do not use).
//!
//! The RNG stream and draw order are what make the output deterministic and
//! byte-identical to C; the parse-tree bookkeeping (InsertTraceNode/emitr/tpos)
//! does not consume randomness and does not affect the emitted residues, so it
//! is omitted here.

use crate::cm::{CM, CM_LOCAL_BEGIN, CM_LOCAL_END, ALPHABET_SIZE};
use crate::constants::{MP_ST, ML_ST, MR_ST, IL_ST, IR_ST, E_ST, B_ST, EL_ST, MAXCONNECT};

/// C: easel.c:esl_mix3() — Bob Jenkins's `mix()`. Mixes three u32 inputs into a
/// quasirandom u32. All arithmetic is 32-bit wrapping (C unsigned overflow).
/// Used to disperse the seed for the FAST (LCG) RNG.
/// ```c
/// a -= b; a -= c; a ^= (c>>13);
/// b -= c; b -= a; b ^= (a<<8);
/// ... (9 rounds) ...  return c;
/// ```
pub fn esl_mix3(mut a: u32, mut b: u32, mut c: u32) -> u32 {
    a = a.wrapping_sub(b); a = a.wrapping_sub(c); a ^= c >> 13;
    b = b.wrapping_sub(c); b = b.wrapping_sub(a); b ^= a << 8;
    c = c.wrapping_sub(a); c = c.wrapping_sub(b); c ^= b >> 13;
    a = a.wrapping_sub(b); a = a.wrapping_sub(c); a ^= c >> 12;
    b = b.wrapping_sub(c); b = b.wrapping_sub(a); b ^= a << 16;
    c = c.wrapping_sub(a); c = c.wrapping_sub(b); c ^= b >> 5;
    a = a.wrapping_sub(b); a = a.wrapping_sub(c); a ^= c >> 3;
    b = b.wrapping_sub(c); b = b.wrapping_sub(a); b ^= a << 10;
    c = c.wrapping_sub(a); c = c.wrapping_sub(b); c ^= b >> 15;
    c
}

/// C: esl_random.c — the FAST ("CreateFast") RNG, a Knuth LCG with a=69069,
/// c=1, period 2^32. **cmemit uses this generator**, via
/// `esl_randomness_CreateFast(--seed)` (cmemit.c:207), NOT the default Mersenne
/// Twister. Matching this LCG stream exactly is what makes emit output
/// byte-identical to C for a fixed seed.
pub struct EslRandomFast {
    /// C: r->x — the LCG state.
    x: u32,
    /// C: r->seed.
    seed: u32,
}

impl EslRandomFast {
    /// C: esl_randomness_CreateFast(seed) + esl_randomness_Init().
    pub fn new(seed: u32) -> Self {
        let mut r = EslRandomFast { x: 0, seed: 0 };
        r.init(seed);
        r
    }

    /// C: esl_randomness_Init() for the FAST type.
    /// ```c
    /// if (seed == 0) seed = choose_arbitrary_seed();
    /// r->seed = seed;
    /// r->x    = esl_mix3(seed, 87654321, 12345678);
    /// if (r->x == 0) r->x = 42;
    /// ```
    pub fn init(&mut self, seed: u32) {
        let seed = if seed == 0 { choose_arbitrary_seed() } else { seed };
        self.seed = seed;
        self.x = esl_mix3(seed, 87654321, 12345678);
        if self.x == 0 {
            self.x = 42;
        }
    }

    pub fn get_seed(&self) -> u32 {
        self.seed
    }

    /// C: esl_random.c:knuth() — `r->x = r->x*69069 + 1; return r->x;`
    #[inline]
    fn knuth(&mut self) -> u32 {
        self.x = self.x.wrapping_mul(69069).wrapping_add(1);
        self.x
    }

    /// C: esl_random(r) for FAST — `(double) knuth(r) / 4294967296.0`, in [0,1).
    #[inline]
    pub fn random(&mut self) -> f64 {
        (self.knuth() as f64) / 4294967296.0
    }

    /// C: esl_random_uint32(r) for FAST — returns knuth(r).
    #[inline]
    pub fn random_uint32(&mut self) -> u32 {
        self.knuth()
    }

    /// C: esl_rnd_Roll(r, n) — uniform integer on 0..n-1 via rejection sampling.
    /// ```c
    /// uint32_t factor = UINT32_MAX / (uint32_t) n;
    /// do { u = esl_random_uint32(r) / factor; } while (u >= n);
    /// return (int) u;
    /// ```
    pub fn roll(&mut self, n: u32) -> u32 {
        let factor = u32::MAX / n;
        loop {
            let u = self.random_uint32() / factor;
            if u < n {
                return u;
            }
        }
    }
}

/// C: esl_random.c:choose_arbitrary_seed() — used only when `--seed 0`, in which
/// case both C and this port are non-deterministic (they will not match). We
/// derive a nonzero seed from wall-clock time; the exact value is irrelevant
/// because seed 0 is not reproducible in C either.
fn choose_arbitrary_seed() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(1);
    let mut seed = esl_mix3(t, std::process::id(), 0);
    if seed == 0 {
        seed = 42;
    }
    seed
}

/// C: esl_random.c:esl_rnd_FChoose()
/// ```c
/// int esl_rnd_FChoose(ESL_RANDOMNESS *r, const float *p, int N) {
///   double norm = 0.0;
///   double sum  = 0.0;
///   double roll = esl_random(r);   /* random fraction */
///   int    i;
///   for (i = 0; i < N; i++) norm += p[i];
///   for (i = 0; i < N; i++) { sum += (double) p[i]; if (roll < (sum / norm)) return i; }
///   esl_fatal("unreached code was reached. universe collapses.");
///   return 0;
/// }
/// ```
/// Computing in double precision is important (see C comment): casting `roll`
/// to float would give a [0,1] number instead of [0,1).
pub fn esl_rnd_fchoose(r: &mut EslRandomFast, p: &[f32], n: usize) -> usize {
    let roll: f64 = r.random(); // random fraction in [0,1)
    let mut norm: f64 = 0.0; // ~ 1.0
    let mut sum: f64 = 0.0; // integrated prob
    for i in 0..n {
        norm += p[i] as f64;
    }
    for i in 0..n {
        sum += p[i] as f64;
        if roll < (sum / norm) {
            return i;
        }
    }
    // "unreached code was reached. universe collapses." — return last index as
    // the C roundoff-safety fallthrough would after esl_fatal (never taken).
    n - 1
}

/// Stack frame for the parse-tree PDA. Mirrors the two frame kinds pushed onto
/// C's `pda` (ESL_STACK) in EmitParsetree: a deferred right-residue marker and a
/// state to expand. We keep only the fields that affect the emitted sequence
/// (rchar/lchar residue indices and the state index v); the parse-tree
/// bookkeeping fields (tpos/tparent/whichway) are dropped.
enum Frame {
    /// C: PDA_RESIDUE marker — carries the deferred right-emission char for a state.
    Residue { rchar: i32 },
    /// C: PDA_STATE marker — a state v to expand, with its (already-decided)
    /// left/right emission chars.
    State { rchar: i32, lchar: i32, v: i32 },
}

/// C: cm_parsetree.c:EmitParsetree() — sequence-only faithful port.
///
/// Samples a sequence from Prob(sequence, parsetree | CM), consuming RNG draws
/// in the exact same order as C so that, for a fixed seed, the emitted residue
/// stream is byte-identical. Returns the emitted residues as canonical indices
/// (0=A,1=C,2=G,3=U). The caller maps indices to output characters.
///
/// Only the global (default) and `-l` local transition branches are ported.
/// The CM_EMIT_NO_LOCAL_BEGINS / CM_EMIT_NO_LOCAL_ENDS special cases are never
/// set by cmemit, so they are omitted (they would take a different draw path).
pub fn emit_parsetree_seq(cm: &CM, r: &mut EslRandomFast) -> Vec<u8> {
    let k = ALPHABET_SIZE as i32; // cm->abc->K == 4 for RNA
    let mut pda: Vec<Frame> = Vec::new();
    let mut gsq: Vec<u8> = Vec::new(); // growing emitted sequence (residue indices)
    // tmp transition vector: enough room for max transitions plus a local-end
    // transition (C: sizeof(float) * (MAXCONNECT+1)).
    let mut tmp_tvec = [0f32; (MAXCONNECT as usize) + 1];

    // C init: push the root state's info (v=0). The rchar/lchar/tparent/whichway
    // integers pushed in C are all -1 / TRACE_LEFT_CHILD here; only v matters.
    pda.push(Frame::State { rchar: -1, lchar: -1, v: 0 });

    // Iterate until the pda is empty (C: while esl_stack_IPop(pda,&type) != eslEOD)
    while let Some(frame) = pda.pop() {
        match frame {
            Frame::Residue { rchar } => {
                // C: PDA_RESIDUE branch — emit the deferred right char, if any.
                if rchar != -1 {
                    gsq.push(rchar as u8);
                }
                // (tr->emitr[tpos] = N; — parse-tree only, omitted)
            }
            Frame::State { rchar, lchar, v } => {
                // C: PDA_STATE branch. InsertTraceNode(...) -> tpos (omitted).
                // If v emitted left, add that symbol to the growing seq.
                if lchar != -1 {
                    gsq.push(lchar as u8);
                }
                // Push the deferred right-emission marker for state v now.
                pda.push(Frame::Residue { rchar });

                if cm.sttype[v as usize] as i32 == B_ST {
                    // Bifurcation: push right start, then left start (left on top).
                    let y = cm.cfirst[v as usize]; // left child
                    let z = cm.cnum[v as usize]; // right child
                    pda.push(Frame::State { rchar: -1, lchar: -1, v: z });
                    pda.push(Frame::State { rchar: -1, lchar: -1, v: y });
                } else {
                    // Decide the next state y.
                    let y: i32;
                    if v == 0 && (cm.flags & CM_LOCAL_BEGIN) != 0 {
                        // ROOT_S with local begins (cmemit never sets NO_LOCAL_BEGINS)
                        // C: y = esl_rnd_FChoose(r, cm->begin, cm->M);
                        y = esl_rnd_fchoose(r, &cm.begin, cm.m as usize) as i32;
                    } else if (cm.flags & CM_LOCAL_END) != 0 {
                        // May choose a child of v, or a local end (cmemit never
                        // sets NO_LOCAL_ENDS).
                        // C: tmp_tvec = t[v][0..cnum]; tmp_tvec[cnum] = end[v];
                        //    y = FChoose(tmp_tvec, cnum+1); if y==cnum -> M else += cfirst
                        for x in tmp_tvec.iter_mut() {
                            *x = 0.0;
                        }
                        let cn = cm.cnum[v as usize] as usize;
                        tmp_tvec[..cn].copy_from_slice(&cm.t[v as usize][..cn]);
                        tmp_tvec[cn] = cm.end[v as usize];
                        let off = esl_rnd_fchoose(r, &tmp_tvec, cn + 1) as i32;
                        if off == cm.cnum[v as usize] {
                            y = cm.m; // local end (EL)
                        } else {
                            y = cm.cfirst[v as usize] + off;
                        }
                    } else {
                        // Global (default) path.
                        // C: y = cm->cfirst[v] + esl_rnd_FChoose(r, cm->t[v], cm->cnum[v]);
                        y = cm.cfirst[v as usize]
                            + esl_rnd_fchoose(r, &cm.t[v as usize], cm.cnum[v as usize] as usize) as i32;
                    }

                    // y == cm->M denotes the EL (End Local) state, which has no
                    // sttype[] entry in this port; treat it as EL_st.
                    let yst = if y == cm.m {
                        EL_ST
                    } else {
                        cm.sttype[y as usize] as i32
                    };

                    // Sample emission char(s) for y (C switch on cm->sttype[y]).
                    let mut ylchar: i32 = -1;
                    let mut yrchar: i32 = -1;
                    match yst {
                        MP_ST => {
                            // x = FChoose(cm->e[y], K*K); l = x/K; r = x%K
                            let x = esl_rnd_fchoose(r, &cm.e[y as usize], (k * k) as usize) as i32;
                            ylchar = x / k;
                            yrchar = x % k;
                        }
                        ML_ST | IL_ST => {
                            ylchar = esl_rnd_fchoose(r, &cm.e[y as usize], k as usize) as i32;
                            yrchar = -1;
                        }
                        MR_ST | IR_ST => {
                            ylchar = -1;
                            yrchar = esl_rnd_fchoose(r, &cm.e[y as usize], k as usize) as i32;
                        }
                        _ => {
                            // E_st, EL_st, D_st, S_st, B_st: no emission char here.
                        }
                    }

                    if yst == E_ST {
                        // C: InsertTraceNode(...E...) — parse-tree only, no push, no draw.
                    } else if yst == EL_ST {
                        // EL emits on transition; choose number of residues (>=0),
                        // each from the NULL distribution. Single trace node in C.
                        for x in tmp_tvec.iter_mut() {
                            *x = 0.0;
                        }
                        // sreEXP2(cm->el_selfsc) == 2^el_selfsc
                        tmp_tvec[0] = (cm.el_selfsc as f64).exp2() as f32; // EL self prob
                        tmp_tvec[1] = 1.0 - tmp_tvec[0]; // prob of implicit END
                        let mut yy = esl_rnd_fchoose(r, &tmp_tvec, 2);
                        while yy == 0 {
                            // self-transition: emit 1 res from NULL distro
                            let lc = esl_rnd_fchoose(r, &cm.null, k as usize);
                            gsq.push(lc as u8);
                            yy = esl_rnd_fchoose(r, &tmp_tvec, 2);
                        }
                    } else {
                        // Non-B, non-E, non-EL: defer expansion of y.
                        pda.push(Frame::State {
                            rchar: yrchar,
                            lchar: ylchar,
                            v: y,
                        });
                    }
                }
            }
        }
    }

    gsq
}

/// Map canonical residue indices (0=A,1=C,2=G,3=U) to output characters.
/// RNA sym order "ACGU"; DNA "ACGT" (U printed as T). This reproduces C's
/// digitize-with-CM-alphabet then textize-with-output-alphabet round trip in
/// emit_unaligned (cmemit.c:369-370, esl_sqio FASTA write).
pub fn residues_to_chars(seq: &[u8], dna: bool) -> Vec<u8> {
    let sym: &[u8; 4] = if dna { b"ACGT" } else { b"ACGU" };
    seq.iter().map(|&i| sym[i as usize]).collect()
}

/// C: esl_random.c:esl_rnd_DChoose() — random choice from a normalized discrete
/// distribution `p[0..n-1]` using a single `esl_random(r)` fraction and cumulative
/// sums, computed in double precision. Same algorithm as `esl_rnd_fchoose` but for
/// `f64` distributions (the genomic HMM parameters). One RNG draw per call.
pub fn esl_rnd_dchoose(r: &mut EslRandomFast, p: &[f64]) -> usize {
    let roll: f64 = r.random(); // [0,1)
    let mut norm: f64 = 0.0;
    for &pi in p {
        norm += pi;
    }
    let mut sum: f64 = 0.0;
    for (i, &pi) in p.iter().enumerate() {
        sum += pi;
        if roll < (sum / norm) {
            return i;
        }
    }
    p.len() - 1
}

/// C: stats.c:SampleGenomicSequenceFromHMM() — sample `l` residues from the
/// 5-state genomic HMM (`ghmm`). RNG draw order (identical to C): one DChoose on
/// the start vector, then for each of the `l` positions a DChoose on the current
/// state's emission vector followed by a DChoose on its transition vector. Returns
/// the `l` residue indices (0..K-1); the caller pads/embeds as needed.
pub fn sample_genomic_sequence_from_hmm(
    r: &mut EslRandomFast,
    ghmm: &crate::cm_calibrate::GenomicHmm,
    l: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(l);
    // C: si = esl_rnd_DChoose(r, sA, nstates);
    let mut si = esl_rnd_dchoose(r, &ghmm.s_a);
    for _ in 0..l {
        // C: dsq[x] = esl_rnd_DChoose(r, eAA[si], K);
        out.push(esl_rnd_dchoose(r, &ghmm.e_aa[si]) as u8);
        // C: si = esl_rnd_DChoose(r, tAA[si], nstates);
        si = esl_rnd_dchoose(r, &ghmm.t_aa[si]);
    }
    out
}

/// C: esl_sqio_ascii.c:esl_sqascii_WriteFasta() — FASTA writer.
/// Header line ">name" (emitted seqs have no acc/desc), then the sequence in
/// 60-character lines. An empty sequence writes just the header line.
pub fn write_fasta(out: &mut dyn std::io::Write, name: &str, seq_chars: &[u8]) -> std::io::Result<()> {
    writeln!(out, ">{}", name)?;
    let mut pos = 0usize;
    while pos < seq_chars.len() {
        let end = (pos + 60).min(seq_chars.len());
        out.write_all(&seq_chars[pos..end])?;
        out.write_all(b"\n")?;
        pos += 60;
    }
    Ok(())
}
