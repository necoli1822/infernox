// SPDX-License-Identifier: BSD-3-Clause
//! RSEARCH RIBOSUM matrix parameterization for cmbuild `--rsearch`.
//!
//! Faithful port of Infernal 1.1.5's `src/rnamat.c` (RIBOSUM matrix reader,
//! target-prob conversion, single-seq degeneracy resolution) and the
//! `rsearch_CMProbifyEmissions()` routine from `src/cm.c:665`.
//!
//! With `--rsearch <matrixfile>`, cmbuild builds a CM from a single-sequence
//! MSA whose emission probabilities are taken from a RIBOSUM substitution
//! matrix (converted to target frequencies) rather than the Dirichlet prior.
//! The CM null model is set to the RIBOSUM background `g` vector.

use crate::cm::CM;
use crate::easel::alphabet::EslAlphabet;
use crate::easel::msa::EslMsa;

// C rnamat.h:22-23
const RNAPAIR_ALPHABET: &[u8; 16] = b"AAAACCCCGGGGUUUU";
const RNAPAIR_ALPHABET2: &[u8; 16] = b"ACGUACGUACGUACGU";

/// C: infernal.h:154 `#define sreEXP2(x)  (exp((x) * 0.69314718 ))`
/// NOTE: C uses the *truncated* constant 0.69314718 (not full-precision ln2).
#[inline]
fn sre_exp2(x: f64) -> f64 {
    (x * 0.69314718).exp()
}

/// C rnamat.h:91 `#define matrix_index(X,Y) ((X>Y) ? X*(X+1)/2+Y: Y*(Y+1)/2+X)`
#[inline]
pub fn matrix_index(x: i32, y: i32) -> usize {
    (if x > y {
        x * (x + 1) / 2 + y
    } else {
        y * (y + 1) / 2 + x
    }) as usize
}

/// C rnamat.c:45 numbered_nucleotide(). A->0 C->1 G->2 T/U->3 else->-1.
#[inline]
pub fn numbered_nucleotide(c: u8) -> i32 {
    match c {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' | b'U' | b'u' => 3,
        _ => -1,
    }
}

/// C rnamat.c:75 numbered_basepair(). Returns (c_num<<2)|d_num or -1.
#[inline]
pub fn numbered_basepair(c: u8, d: u8) -> i32 {
    let c_num = numbered_nucleotide(c);
    let d_num = numbered_nucleotide(d);
    if c_num < 0 || d_num < 0 {
        -1
    } else {
        (c_num << 2) | d_num
    }
}

/// C rnamat.h:31 `matrix_t`.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub matrix: Vec<f64>,
    pub edge_size: i32,
    pub full_size: i32,
    pub h: f64,
    pub e: f64,
}

/// C rnamat.h:42 `fullmat_t` (minus the ESL_ALPHABET pointer; we pass abc explicitly).
#[derive(Debug, Clone)]
pub struct FullMat {
    pub unpaired: Matrix,
    pub paired: Matrix,
    pub name: String,
    /// C: `float *g` — the RIBOSUM background distribution.
    pub g: Vec<f32>,
    pub scores_flag: bool,
    pub probs_flag: bool,
}

/// C rnamat.c:156 setup_matrix(). Allocate a triangular matrix, init to 0.0.
fn setup_matrix(size: i32) -> Matrix {
    let full_size = matrix_index(size - 1, size - 1) + 1;
    Matrix {
        matrix: vec![0.0; full_size],
        edge_size: size,
        full_size: full_size as i32,
        h: 0.0,
        e: 0.0,
    }
}

/// C: esl_vec_DSum() — Kahan (compensated) summation for doubles.
fn kahan_dsum(vec: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for &v in vec {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// C: esl_vec_DNorm(vec, n) — divide each of first n elems by Kahan sum, or set 1/n.
fn dnorm(vec: &mut [f64], n: usize) {
    let sum = kahan_dsum(&vec[..n]);
    if sum != 0.0 {
        for x in &mut vec[..n] {
            *x /= sum;
        }
    } else {
        for x in &mut vec[..n] {
            *x = 1.0 / n as f64;
        }
    }
}

/// C: esl_vec_FSum(vec, n) — Kahan (compensated) summation for floats.
fn kahan_fsum(vec: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for i in 0..n {
        let y = vec[i] - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// C: esl_vec_FNorm(vec, n) — divide each of first n elems by Kahan sum, or set 1/n.
fn fnorm(vec: &mut [f32], n: usize) {
    let sum = kahan_fsum(vec, n);
    if sum != 0.0 {
        for x in &mut vec[..n] {
            *x /= sum;
        }
    } else {
        for x in &mut vec[..n] {
            *x = 1.0 / n as f32;
        }
    }
}

// ---- byte-scanning helpers replicating the C pointer walk in ReadMatrix ----

#[inline]
fn is_space(b: u8) -> bool {
    // C isspace(): ' ' \t \n \v \f \r
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

#[inline]
fn is_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

/// Parse a C atof()-style number starting at buf[pos]: an optional sign, digits
/// and a decimal point (no exponent — RIBOSUM files have none). Returns the f64.
/// `pos` is assumed to be at a digit/'-'/'.' (as the C skip-loop guarantees).
fn atof_at(buf: &[u8], pos: usize) -> f64 {
    let mut end = pos;
    while end < buf.len() && (is_digit(buf[end]) || buf[end] == b'-' || buf[end] == b'.') {
        end += 1;
    }
    // Safe: run is [0-9.-] only; parse mirrors atof for well-formed RIBOSUM tokens.
    std::str::from_utf8(&buf[pos..end])
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// find the next occurrence of byte `needle` in buf at or after `from`.
fn find_byte(buf: &[u8], from: usize, needle: u8) -> Option<usize> {
    (from..buf.len()).find(|&i| buf[i] == needle)
}

/// find the next occurrence of substring `needle` in buf at or after `from`.
fn find_str(buf: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || from > buf.len() {
        return None;
    }
    (from..=buf.len().saturating_sub(needle.len())).find(|&i| &buf[i..i + needle.len()] == needle)
}

/// C rnamat.c:478 ReadMatrix(). Reads a RIBOSUM matrix file.
///
/// `matbytes` is the entire matrix-file content (C reads it into `fullbuf`).
pub fn read_matrix(abc: &EslAlphabet, matbytes: &[u8]) -> FullMat {
    let k = abc.K; // eslRNA -> 4
    let mut unpaired = setup_matrix(k);
    let mut paired = setup_matrix(k * k);
    let mut g = vec![0.0f32; k as usize];

    let buf = matbytes;
    let used = buf.len(); // C fullbuf_used

    // First, find "RIBO", and copy matrix name to fullmat->name (rnamat.c:507)
    let ribo = find_str(buf, 0, b"RIBO").expect("matrix file has no RIBO name");
    let mut i = 0usize;
    while ribo + i < used && !is_space(buf[ribo + i]) {
        i += 1;
    }
    let name = String::from_utf8_lossy(&buf[ribo..ribo + i]).into_owned();
    let mut cp = ribo + i; // cp = cp + i
    let scores_flag;
    let probs_flag;
    // C rnamat.c:513: name has "SUM" => log-odds scores; "PROB" => target probs.
    if name.contains("SUM") {
        scores_flag = true;
        probs_flag = false;
    } else if name.contains("PROB") {
        scores_flag = false;
        probs_flag = true;
    } else {
        panic!("ERROR reading matrix, name does not include SUM or PROB.");
    }

    // Now, find the first A (rnamat.c:518) and count unpaired edge size
    cp = find_byte(buf, cp, b'A').expect("no A in matrix");
    unpaired.edge_size = 0;
    // while (*cp != '\n' && cp-fullbuf < fullbuf_used)
    while cp < used && buf[cp] != b'\n' {
        if !is_space(buf[cp]) && cp + 1 < used && is_space(buf[cp + 1]) {
            unpaired.edge_size += 1;
        }
        cp += 1;
    }

    // Read background freqs until we hit the next A (rnamat.c:530)
    let mut end_mat_pos = find_byte(buf, cp, b'A').expect("no A after unpaired header");
    let mut gi = 0usize;
    // for (i=0; cp-fullbuf < end_mat_pos-fullbuf; i++)
    while cp < end_mat_pos {
        while cp < used
            && cp != end_mat_pos
            && !is_digit(buf[cp])
            && buf[cp] != b'-'
            && buf[cp] != b'.'
        {
            cp += 1;
        }
        if cp == end_mat_pos {
            break;
        }
        if cp < used {
            g[gi] = atof_at(buf, cp) as f32;
            gi += 1;
            while cp < used && (is_digit(buf[cp]) || buf[cp] == b'-' || buf[cp] == b'.') {
                cp += 1;
            }
        }
    }
    // normalize the background (rnamat.c:547)
    fnorm(&mut g, unpaired.edge_size as usize);

    // We've already found the next A. Take numbers until "H:" (rnamat.c:553)
    end_mat_pos = find_str(buf, cp, b"H:").expect("no H: after unpaired matrix");
    let mut ui = 0usize;
    while cp < end_mat_pos {
        while cp < used
            && cp != end_mat_pos
            && !is_digit(buf[cp])
            && buf[cp] != b'-'
            && buf[cp] != b'.'
        {
            cp += 1;
        }
        if cp == end_mat_pos {
            break;
        }
        if cp < used {
            unpaired.matrix[ui] = atof_at(buf, cp);
            ui += 1;
            while cp < used && (is_digit(buf[cp]) || buf[cp] == b'-' || buf[cp] == b'.') {
                cp += 1;
            }
        }
    }
    unpaired.full_size = ui as i32;

    // Skip the H: (rnamat.c:572)
    cp += 2;
    unpaired.h = atof_at_loose(buf, cp);

    // Now, go past the E: (rnamat.c:576)
    cp = find_str(buf, cp, b"E:").expect("no E: after unpaired H:") + 2;
    unpaired.e = atof_at_loose(buf, cp);

    // ---- PAIRED MATRIX ----
    // Now, find the first A (rnamat.c:581)
    cp = find_byte(buf, cp, b'A').expect("no A for paired header");
    paired.edge_size = 0;
    // while (*cp != '\n')  (C has no bound here; we add one to be safe)
    while cp < used && buf[cp] != b'\n' {
        if !is_space(buf[cp]) && cp + 1 < used && is_space(buf[cp + 1]) {
            paired.edge_size += 1;
        }
        cp += 1;
    }

    // Find next A (rnamat.c:592)
    while cp < used && buf[cp] != b'A' {
        cp += 1;
    }

    // Take numbers until we hit the H: (rnamat.c:595)
    end_mat_pos = find_str(buf, cp, b"H:").expect("no H: after paired matrix");
    let mut pi = 0usize;
    while cp < end_mat_pos {
        while cp < used
            && cp != end_mat_pos
            && !is_digit(buf[cp])
            && buf[cp] != b'-'
            && buf[cp] != b'.'
        {
            cp += 1;
        }
        if cp == end_mat_pos {
            break;
        }
        if cp < used {
            paired.matrix[pi] = atof_at(buf, cp);
            pi += 1;
            while cp < used && (is_digit(buf[cp]) || buf[cp] == b'-' || buf[cp] == b'.') {
                cp += 1;
            }
        }
    }
    paired.full_size = pi as i32;

    // Skip the H: (rnamat.c:614)
    cp += 2;
    paired.h = atof_at_loose(buf, cp);

    // Now, go past the E: (rnamat.c:618)
    cp = find_str(buf, cp, b"E:").expect("no E: after paired H:") + 2;
    paired.e = atof_at_loose(buf, cp);

    FullMat {
        unpaired,
        paired,
        name,
        g,
        scores_flag,
        probs_flag,
    }
}

/// C atof() applied where the pointer may sit on whitespace before the number
/// (as after "H:" / "E:"): atof() skips leading whitespace, then parses.
fn atof_at_loose(buf: &[u8], mut pos: usize) -> f64 {
    while pos < buf.len() && is_space(buf[pos]) {
        pos += 1;
    }
    if pos < buf.len() {
        atof_at(buf, pos)
    } else {
        0.0
    }
}

/// C cm.c... actually rnamat.c:723 ribosum_calc_targets(). Convert log-odds
/// scores to target probs f_ij = g_i g_j 2^{s_ij}, symmetric-normalized.
pub fn ribosum_calc_targets(fullmat: &mut FullMat, abc: &EslAlphabet) {
    assert!(
        fullmat.scores_flag,
        "in ribosum_calc_targets(), matrix is not in log odds mode"
    );
    assert!(
        !fullmat.probs_flag,
        "in ribosum_calc_targets(), matrix is already in probs mode"
    );
    let k = abc.K;

    // unpaired (singlet): f_ij = g_i g_j 2^{s_ij}  (rnamat.c:741)
    // FAITHFUL C ARITHMETIC: in C `g[i] * g[j] * sreEXP2(..)`, `g` is float, so
    // `g[i]*g[j]` is a FLOAT multiply; only the final `* sreEXP2(..)` (double)
    // promotes to double. Compute the g-product in f32 first — casting to f64
    // early diverges by 1 ULP (surfaces in the p7 filter COMPO).
    let mut idx = 0usize;
    for i in 0..k {
        for j in 0..=i {
            let gp = fullmat.g[i as usize] * fullmat.g[j as usize]; // f32 * f32 -> f32
            fullmat.unpaired.matrix[idx] =
                gp as f64 * sre_exp2(fullmat.unpaired.matrix[idx]);
            idx += 1;
        }
    }
    // paired: f = g_i g_j g_k g_l 2^{s}  (rnamat.c:752)
    // C: `g[i]*g[j]*g[k]*g[l]*sreEXP2(..)` — the four float g's multiply in f32
    // (left-assoc), then promote to double for the final `* sreEXP2`.
    idx = 0;
    for a in 0..RNAPAIR_ALPHABET.len() {
        for b in 0..=a {
            let i = (a as i32) / k;
            let j = (a as i32) % k;
            let kk = (b as i32) / k;
            let l = (b as i32) % k;
            let gp = fullmat.g[i as usize] * fullmat.g[j as usize]
                * fullmat.g[kk as usize]
                * fullmat.g[l as usize]; // all f32 (left-assoc)
            fullmat.paired.matrix[idx] = gp as f64 * sre_exp2(fullmat.paired.matrix[idx]);
            idx += 1;
        }
    }

    // normalize the unpaired matrix, doubling off-diagonals (rnamat.c:772)
    idx = 0;
    for i in 0..k {
        for j in 0..=i {
            if i != j {
                fullmat.unpaired.matrix[idx] *= 2.0;
            }
            idx += 1;
        }
    }
    let usize_full = fullmat.unpaired.full_size as usize;
    dnorm(&mut fullmat.unpaired.matrix, usize_full);
    idx = 0;
    for i in 0..k {
        for j in 0..=i {
            if i != j {
                fullmat.unpaired.matrix[idx] *= 0.5;
            }
            idx += 1;
        }
    }

    // normalize the paired matrix, doubling off-diagonals (rnamat.c:789)
    idx = 0;
    for a in 0..RNAPAIR_ALPHABET.len() {
        for b in 0..=a {
            if a != b {
                fullmat.paired.matrix[idx] *= 2.0;
            }
            idx += 1;
        }
    }
    let psize_full = fullmat.paired.full_size as usize;
    dnorm(&mut fullmat.paired.matrix, psize_full);
    idx = 0;
    for a in 0..RNAPAIR_ALPHABET.len() {
        for b in 0..=a {
            if a != b {
                fullmat.paired.matrix[idx] *= 0.5;
            }
            idx += 1;
        }
    }

    fullmat.scores_flag = false;
    fullmat.probs_flag = true;
}

/// C cm.c:683 rsearch_CMProbifyEmissions(). Convert single-sequence counts-based
/// CM emissions to probabilities using the RIBOSUM target frequencies.
pub fn rsearch_cm_probify_emissions(cm: &mut CM, fullmat: &FullMat, abc: &EslAlphabet) {
    use crate::constants::{
        IL_ST, IR_ST, MATL_ML, MATP_ML, MATP_MP, MATP_MR, MATR_MR,
    };
    let thresh = 0.000001f32;
    let k = abc.K;
    let ku = k as usize;

    assert!(
        !fullmat.scores_flag,
        "in rsearch_CMProbifyEmissions(), matrix is in log odds mode, it should be in probs mode"
    );
    assert!(
        cm.flags & crate::cm::CM_RSEARCHEMIT != 0,
        "in rsearch_CMProbifyEmissions(), CM_RSEARCHEMIT flag is down"
    );

    // sym[x] for x in 0..K is 'A','C','G','U' for eslRNA.
    let sym = |x: i32| abc.sym[x as usize] as u8;

    for v in 0..cm.m as usize {
        let mut found_ct_flag = false;
        let mut cur_emission: i32 = 0;
        let stid = cm.stid[v] as i32;
        let sttype = cm.sttype[v] as i32;

        if stid == MATP_MP {
            // figure out which base pair was in the query (cm.c:704)
            for x in 0..k {
                for y in 0..k {
                    if (cm.e[v][(x * k + y) as usize] - 0.0).abs() > thresh {
                        if found_ct_flag {
                            panic!("cm->e[v:{}] a MATP_MP has > 1 non-zero count", v);
                        }
                        cur_emission = numbered_basepair(sym(x), sym(y));
                        found_ct_flag = true;
                    }
                }
            }
            // set emission probs from paired target matrix (cm.c:718)
            for x in 0..(k * k) {
                cm.e[v][x as usize] =
                    fullmat.paired.matrix[matrix_index(cur_emission, x)] as f32;
            }
            fnorm(&mut cm.e[v], (k * k) as usize);
        } else if stid == MATL_ML || stid == MATR_MR {
            for x in 0..k {
                if (cm.e[v][x as usize] - 0.0).abs() > thresh {
                    if found_ct_flag {
                        panic!("cm->e[v:{}] a MAT{{L,R}}_M{{L,R}} has > 1 non-zero count", v);
                    }
                    cur_emission = numbered_nucleotide(sym(x));
                    found_ct_flag = true;
                }
            }
            for x in 0..k {
                cm.e[v][x as usize] =
                    fullmat.unpaired.matrix[matrix_index(cur_emission, x)] as f32;
            }
            fnorm(&mut cm.e[v], ku);
        } else if stid == MATP_ML || stid == MATP_MR {
            // RSEARCH technique: determine residue emitted to left/right from the
            // MATP_MP emission (v-1 for ML, v-2 for MR), use unpaired target freqs.
            for x in 0..k {
                for y in 0..k {
                    if stid == MATP_ML
                        && (cm.e[v - 1][(x * k + y) as usize] - 0.0).abs() > thresh
                    {
                        cur_emission = numbered_nucleotide(sym(x));
                    } else if stid == MATP_MR
                        && (cm.e[v - 2][(x * k + y) as usize] - 0.0).abs() > thresh
                    {
                        cur_emission = numbered_nucleotide(sym(y));
                    }
                }
            }
            for x in 0..k {
                cm.e[v][x as usize] =
                    fullmat.unpaired.matrix[matrix_index(cur_emission, x)] as f32;
            }
            fnorm(&mut cm.e[v], ku);
        } else if sttype == IL_ST || sttype == IR_ST {
            // Insert states: no counts expected; renormalize (all were zero).
            for x in 0..k {
                if (cm.e[v][x as usize] - 0.0).abs() > thresh {
                    panic!("cm->e[v:{}] an I{{L,R}} has > 0 non-zero count", v);
                }
            }
            fnorm(&mut cm.e[v], ku);
        }
    }
}

/// C rnamat.c:844 ribosum_MSA_resolve_degeneracies(). Replaces ambiguous bases in
/// a single-sequence MSA with the most-likely compatible A/C/G/U (or base pair),
/// per the RIBOSUM target-prob marginals, so downstream emission counts are clean.
///
/// Operates on `msa.aseq[0]` (text); caller must re-digitize `ax` afterwards.
pub fn ribosum_msa_resolve_degeneracies(fullmat: &FullMat, msa: &mut EslMsa, abc: &EslAlphabet) {
    assert!(
        fullmat.probs_flag,
        "in ribosum_MSA_resolve_degeneracies(), matrix is not in probs mode"
    );
    assert!(
        !fullmat.scores_flag,
        "in ribosum_MSA_resolve_degeneracies(), matrix is in scores mode"
    );
    assert_eq!(msa.nseq, 1, "MSA does not have exactly 1 seq");
    let k = abc.K;
    let ku = k as usize;

    // degen_string / rna_string (rnamat.c:854)
    let degen_string: &[u8] = b"XRYMKSWHBVDN";
    let rna_string: &[u8] = b"ACGU";

    // degen_mx[12][K] (rnamat.c:886)
    let mut degen_mx = vec![[0i32; 4]; 12];
    degen_mx[0] = [1, 1, 1, 1]; // X = A|C|G|U
    degen_mx[1] = [1, 0, 1, 0]; // R = A|G
    degen_mx[2] = [0, 1, 0, 1]; // Y = C|U
    degen_mx[3] = [1, 1, 0, 0]; // M = A|C
    degen_mx[4] = [0, 0, 1, 1]; // K = G|U
    degen_mx[5] = [0, 1, 1, 0]; // S = C|G
    degen_mx[6] = [1, 0, 0, 1]; // W = A|U
    degen_mx[7] = [1, 1, 0, 1]; // H = A|C|U
    degen_mx[8] = [0, 1, 1, 1]; // B = C|G|U
    degen_mx[9] = [1, 1, 1, 0]; // V = A|C|G
    degen_mx[10] = [1, 0, 1, 1]; // D = A|G|U
    degen_mx[11] = [1, 1, 1, 1]; // N = A|C|G|U

    // marginals (rnamat.c:917)
    let mut unpaired_marginals = vec![0.0f32; ku];
    let mut paired_marginals = vec![0.0f32; ku * ku];
    for i in 0..k {
        for j in 0..k {
            unpaired_marginals[i as usize] +=
                fullmat.unpaired.matrix[matrix_index(i, j)] as f32;
        }
    }
    for i in 0..(k * k) {
        for j in 0..(k * k) {
            paired_marginals[i as usize] += fullmat.paired.matrix[matrix_index(i, j)] as f32;
        }
    }
    fnorm(&mut unpaired_marginals, ku);
    fnorm(&mut paired_marginals, ku * ku);

    // ct array 1..alen (rnamat.c:940). ss_cons already cleaned to WUSS by caller.
    let alen = msa.alen as usize;
    let ss = msa
        .ss_cons
        .as_ref()
        .expect("no SS_cons for resolve_degeneracies")
        .as_bytes()
        .to_vec();
    let ct = crate::cm_modelmaker::wuss2ct(&ss, alen).expect("wuss2ct failed");

    // work on text seq 0 (C esl_msa_Textize; aseq is always populated here).
    let mut aseq: Vec<u8> = msa.aseq[0].as_bytes().to_vec();

    let is_gap = |c: u8| abc.inmap[(c as usize) & 0x7f] == 4;
    let fargmax = |v: &[f32]| -> i32 {
        let mut best = 0usize;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best as i32
    };

    for apos in 0..alen {
        if is_gap(aseq[apos]) {
            continue;
        }
        let mut mate = ct[apos + 1]; // 1..alen
        if mate != 0 && is_gap(aseq[(mate - 1) as usize]) {
            mate = 0;
        } else if mate != 0 && ((mate - 1) as usize) < apos {
            continue;
        }

        let mut c = aseq[apos].to_ascii_uppercase();
        if c == b'T' {
            c = b'U';
        }
        if rna_string.iter().position(|&r| r == c).is_none() {
            // a degeneracy
            let dpos = degen_string
                .iter()
                .position(|&d| d == c)
                .expect("character is not ACGTU or a recognized ambiguity code");
            if mate == 0 {
                // single stranded (rnamat.c:968)
                let mut cur = vec![0.0f32; ku];
                for i in 0..ku {
                    cur[i] = degen_mx[dpos][i] as f32 * unpaired_marginals[i];
                }
                let argmax = fargmax(&cur);
                aseq[apos] = rna_string[argmax as usize];
            } else {
                // paired
                let mut c_m = aseq[(mate - 1) as usize].to_ascii_uppercase();
                if c_m == b'T' {
                    c_m = b'U';
                }
                if rna_string.iter().position(|&r| r == c_m).is_none() {
                    // mate is ambiguous (rnamat.c:988)
                    let dpos_m = degen_string
                        .iter()
                        .position(|&d| d == c_m)
                        .expect("mate character is not a recognized ambiguity code");
                    let mut cur = vec![0.0f32; ku * ku];
                    let mut idx = 0usize;
                    for i in 0..ku {
                        for j in 0..ku {
                            cur[idx] = degen_mx[dpos][i] as f32
                                * degen_mx[dpos_m][j] as f32
                                * paired_marginals[idx];
                            idx += 1;
                        }
                    }
                    let argmax = fargmax(&cur);
                    aseq[apos] = RNAPAIR_ALPHABET[argmax as usize];
                    aseq[(mate - 1) as usize] = RNAPAIR_ALPHABET2[argmax as usize];
                } else {
                    // mate is unambiguous (rnamat.c:1013)
                    let dpos_m = rna_string.iter().position(|&r| r == c_m).unwrap() as i32;
                    let mut cur = vec![0.0f32; ku * ku];
                    let mut idx = 0usize;
                    for i in 0..ku {
                        for j in 0..ku {
                            cur[idx] = degen_mx[dpos][i] as f32
                                * ((j as i32 == dpos_m) as i32 as f32)
                                * paired_marginals[idx];
                            idx += 1;
                        }
                    }
                    let argmax = fargmax(&cur);
                    aseq[apos] = RNAPAIR_ALPHABET[argmax as usize];
                    aseq[(mate - 1) as usize] = RNAPAIR_ALPHABET2[argmax as usize];
                }
            }
        }
        // unambiguous apos, but ambiguous mate (rnamat.c:1038)
        if mate != 0 {
            let mut c_m = aseq[(mate - 1) as usize].to_ascii_uppercase();
            // NOTE: faithful reproduction of C bug — it writes to `c`, not `c_m`.
            if c_m == b'T' {
                c = b'U';
            }
            if rna_string.iter().position(|&r| r == c_m).is_none() {
                let dpos_m = degen_string
                    .iter()
                    .position(|&d| d == c_m)
                    .expect("mate character is not a recognized ambiguity code") as i32;
                // C recomputes dpos from c (the possibly-mutated apos residue).
                let cc = aseq[apos].to_ascii_uppercase();
                let cc = if cc == b'T' { b'U' } else { cc };
                let dpos = rna_string
                    .iter()
                    .position(|&r| r == cc)
                    .expect("character is not ACGTU") as i32;
                let mut cur = vec![0.0f32; ku * ku];
                let mut idx = 0usize;
                for i in 0..ku {
                    for j in 0..ku {
                        cur[idx] = ((i as i32 == dpos) as i32 as f32)
                            * degen_mx[dpos_m as usize][j] as f32
                            * paired_marginals[idx];
                        idx += 1;
                    }
                }
                let argmax = fargmax(&cur);
                aseq[apos] = RNAPAIR_ALPHABET[argmax as usize];
                aseq[(mate - 1) as usize] = RNAPAIR_ALPHABET2[argmax as usize];
            }
            let _ = c;
        }
    }

    // write back text and re-digitize (C esl_msa_Digitize).
    msa.aseq[0] = String::from_utf8(aseq).expect("resolved aseq not utf8");
    msa.digitize(abc);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_index() {
        // C rnamat.h:91 matrix_index(X,Y) = X>Y ? X(X+1)/2+Y : Y(Y+1)/2+X
        assert_eq!(matrix_index(0, 0), 0);
        assert_eq!(matrix_index(1, 0), 1);
        assert_eq!(matrix_index(0, 1), 1); // symmetric
        assert_eq!(matrix_index(3, 3), 9); // unpaired last (10 entries)
        assert_eq!(matrix_index(15, 15), 135); // paired last (136 entries)
    }

    #[test]
    fn test_numbered_nucleotide_and_basepair() {
        assert_eq!(numbered_nucleotide(b'A'), 0);
        assert_eq!(numbered_nucleotide(b'c'), 1);
        assert_eq!(numbered_nucleotide(b'G'), 2);
        assert_eq!(numbered_nucleotide(b'T'), 3);
        assert_eq!(numbered_nucleotide(b'U'), 3);
        assert_eq!(numbered_nucleotide(b'N'), -1);
        // AA->0, AU->3, UG->14, UU->15
        assert_eq!(numbered_basepair(b'A', b'A'), 0);
        assert_eq!(numbered_basepair(b'A', b'U'), 3);
        assert_eq!(numbered_basepair(b'U', b'G'), 14);
        assert_eq!(numbered_basepair(b'U', b'U'), 15);
        assert_eq!(numbered_basepair(b'A', b'N'), -1);
    }

    // A minimal RIBOSUM-format matrix (same layout as the shipped files) with
    // simple values, to exercise the byte-scanning parser and calc_targets.
    const MINI: &str = "\
RIBOSUM-TEST-SUM

    A           C           G           U
    0.25        0.25        0.25        0.25

    A           C           G           U
A   1.0
C   -1.0        1.0
G   -1.0        -1.0        1.0
U   -1.0        -1.0        -1.0        1.0
H: 0.5000
E: -0.2500

    AA          AC          AG          AU          CA          CC          CG          CU          GA          GC          GG          GU          UA          UC          UG          UU
AA  1.0
AC  0.0         1.0
AG  0.0         0.0         1.0
AU  0.0         0.0         0.0         1.0
CA  0.0         0.0         0.0         0.0         1.0
CC  0.0         0.0         0.0         0.0         0.0         1.0
CG  0.0         0.0         0.0         0.0         0.0         0.0         1.0
CU  0.0         0.0         0.0         0.0         0.0         0.0         0.0         1.0
GA  0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         1.0
GC  0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         1.0
GG  0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         1.0
GU  0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         1.0
UA  0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         1.0
UC  0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         1.0
UG  0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         1.0
UU  0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         0.0         1.0
H: 1.0000
E: -0.5000
";

    #[test]
    fn test_read_matrix_headers_and_dims() {
        let abc = EslAlphabet::rna();
        let fm = read_matrix(&abc, MINI.as_bytes());
        assert_eq!(fm.name, "RIBOSUM-TEST-SUM");
        assert!(fm.scores_flag);
        assert!(!fm.probs_flag);
        assert_eq!(fm.unpaired.edge_size, 4);
        assert_eq!(fm.unpaired.full_size, 10);
        assert_eq!(fm.paired.edge_size, 16);
        assert_eq!(fm.paired.full_size, 136);
        for g in fm.g.iter() {
            assert!((g - 0.25).abs() < 1e-6);
        }
        assert!((fm.unpaired.h - 0.5).abs() < 1e-9);
        assert!((fm.unpaired.e - (-0.25)).abs() < 1e-9);
        assert!((fm.paired.h - 1.0).abs() < 1e-9);
        assert!((fm.paired.e - (-0.5)).abs() < 1e-9);
        assert_eq!(fm.unpaired.matrix[matrix_index(0, 0)], 1.0);
        assert_eq!(fm.unpaired.matrix[matrix_index(1, 0)], -1.0);
    }

    #[test]
    fn test_calc_targets_normalizes() {
        let abc = EslAlphabet::rna();
        let mut fm = read_matrix(&abc, MINI.as_bytes());
        ribosum_calc_targets(&mut fm, &abc);
        assert!(fm.probs_flag);
        assert!(!fm.scores_flag);
        // Reconstruct the full symmetric distribution (off-diagonals were halved)
        // and confirm it sums to 1.
        let k = abc.K;
        let mut usum = 0.0f64;
        let mut idx = 0usize;
        for i in 0..k {
            for j in 0..=i {
                usum += if i != j { 2.0 } else { 1.0 } * fm.unpaired.matrix[idx];
                idx += 1;
            }
        }
        assert!((usum - 1.0).abs() < 1e-9, "unpaired full sum = {usum}");
    }
}
