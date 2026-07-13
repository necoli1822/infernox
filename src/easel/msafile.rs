//! Stockholm / Pfam alignment file I/O.
//!
//! Faithful port of `esl_msafile_stockholm.c` (the Stockholm reader
//! `esl_msafile_stockholm_Read` and the writer `stockholm_write`, dispatched
//! by `esl_msafile_stockholm_Write`) plus the `esl_msafile_Write` cpl choice
//! in `esl_msafile.c:355-360`.
//!
//! Stockholm = multi-block, 200 aligned residues per line (cpl=200).
//! Pfam      = single block, one alignment line per sequence (cpl=alen).

use crate::easel::alphabet::EslAlphabet;
use crate::easel::msa::*;
use std::io::{self, Write};

/// Output/format selector. C: eslMSAFILE_* codes (esl_msafile.h:81-91).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsaFormat {
    Stockholm,   // eslMSAFILE_STOCKHOLM   101 (interleaved)
    Pfam,        // eslMSAFILE_PFAM        102 (one-line-per-seq Stockholm)
    A2m,         // eslMSAFILE_A2M         103 (UCSC SAM fasta-like a2m)
    Psiblast,    // eslMSAFILE_PSIBLAST    104 (NCBI PSI-BLAST)
    Selex,       // eslMSAFILE_SELEX       105 (old SELEX)
    Afa,         // eslMSAFILE_AFA         106 (aligned FASTA)
    Clustal,     // eslMSAFILE_CLUSTAL     107 (CLUSTAL)
    ClustalLike, // eslMSAFILE_CLUSTALLIKE 108 (CLUSTAL-like: MUSCLE/PROBCONS)
    Phylip,      // eslMSAFILE_PHYLIP      109 (interleaved PHYLIP)
    Phylips,     // eslMSAFILE_PHYLIPS     110 (sequential PHYLIP)
}

/// esl_msafile_EncodeFormat (esl_msafile.c:717): case-insensitive string ->
/// format code. Returns None for eslMSAFILE_UNKNOWN.
pub fn esl_msafile_encode_format(fmtstring: &str) -> Option<MsaFormat> {
    // C: strcasecmp() against each name.
    match fmtstring.to_ascii_lowercase().as_str() {
        "stockholm" => Some(MsaFormat::Stockholm),
        "pfam" => Some(MsaFormat::Pfam),
        "a2m" => Some(MsaFormat::A2m),
        "psiblast" => Some(MsaFormat::Psiblast),
        "selex" => Some(MsaFormat::Selex),
        "afa" => Some(MsaFormat::Afa),
        "clustal" => Some(MsaFormat::Clustal),
        "clustallike" => Some(MsaFormat::ClustalLike),
        "phylip" => Some(MsaFormat::Phylip),
        "phylips" => Some(MsaFormat::Phylips),
        _ => None,
    }
}

/// esl_msafile_DecodeFormat (esl_msafile.c:749): format code -> display string.
pub fn esl_msafile_decode_format(fmt: MsaFormat) -> &'static str {
    match fmt {
        MsaFormat::Stockholm => "Stockholm",
        MsaFormat::Pfam => "Pfam",
        MsaFormat::A2m => "UCSC A2M",
        MsaFormat::Psiblast => "PSI-BLAST",
        MsaFormat::Selex => "SELEX",
        MsaFormat::Afa => "aligned FASTA",
        MsaFormat::Clustal => "Clustal",
        MsaFormat::ClustalLike => "Clustal-like",
        MsaFormat::Phylip => "PHYLIP (interleaved)",
        MsaFormat::Phylips => "PHYLIP (sequential)",
    }
}

#[derive(Debug)]
pub enum MsaError {
    Format(String),
    Io(io::Error),
}

impl std::fmt::Display for MsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MsaError::Format(s) => write!(f, "stockholm format error: {s}"),
            MsaError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}
impl std::error::Error for MsaError {}
impl From<io::Error> for MsaError {
    fn from(e: io::Error) -> Self {
        MsaError::Io(e)
    }
}

type PResult<T> = Result<T, MsaError>;

/*****************************************************************
 * esl_str_GetMaxWidth (easel.c:1703) — longest string in a slice.
 *****************************************************************/
fn str_get_max_width<S: AsRef<str>>(v: &[S]) -> i32 {
    v.iter().map(|s| s.as_ref().len() as i32).max().unwrap_or(0)
}

/*****************************************************************
 * esl_memtok (esl_mem.c) — split on a delimiter set.
 * Returns (token, rest) where rest has leading delimiters after the token
 * already consumed, exactly like the C pointer/length update.
 *****************************************************************/
fn memtok<'a>(p: &'a str, delim: &[char]) -> Option<(&'a str, &'a str)> {
    let bytes = p.as_bytes();
    let is_delim = |b: u8| delim.iter().any(|&d| d as u8 == b);
    let n = bytes.len();
    let mut so = 0;
    while so < n && is_delim(bytes[so]) {
        so += 1;
    }
    let mut xo = so;
    while xo < n && !is_delim(bytes[xo]) {
        xo += 1;
    }
    let mut eo = xo;
    while eo < n && is_delim(bytes[eo]) {
        eo += 1;
    }
    if so == n {
        None
    } else {
        Some((&p[so..xo], &p[eo..]))
    }
}

/* trim trailing spaces/tabs (C: `while (n && strchr(" \t", p[n-1])) n--;`) */
fn rtrim_ws(s: &str) -> &str {
    s.trim_end_matches([' ', '\t'])
}

/*****************************************************************
 * Reader: esl_msafile_stockholm_Read
 *****************************************************************/

/// Read *all* MSA records from a Stockholm/Pfam text buffer (esl-reformat
/// loops `esl_msafile_Read` until EOF). If `abc` is Some, the resulting MSAs
/// are digitized (`ax[]` filled). Text `aseq[]` is always populated.
pub fn read_all(input: &str, abc: Option<&EslAlphabet>) -> PResult<Vec<EslMsa>> {
    let lines: Vec<&str> = input.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l)).collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    // Skip trailing empty last element produced by a final '\n'
    let nlines = if input.ends_with('\n') { lines.len() - 1 } else { lines.len() };
    while i < nlines {
        // Skip leading blank/comment lines, but stop on a Stockholm header.
        while i < nlines {
            let l = lines[i];
            let blank = l.chars().all(|c| c == ' ' || c == '\t');
            if blank || (l.starts_with('#') && !l.starts_with("# STOCKHOLM")) {
                i += 1;
            } else {
                break;
            }
        }
        if i >= nlines {
            break;
        }
        if !lines[i].starts_with("# STOCKHOLM 1.") {
            return Err(MsaError::Format("missing Stockholm header".into()));
        }
        i += 1;
        let (msa, next) = read_one(&lines, i, nlines, abc)?;
        out.push(msa);
        i = next;
    }
    Ok(out)
}

/// Read a single MSA record after its `# STOCKHOLM` header has been consumed;
/// `start` is the first line index after the header. Returns the MSA and the
/// index just past the `//` terminator.
fn read_one(
    lines: &[&str],
    start: usize,
    nlines: usize,
    abc: Option<&EslAlphabet>,
) -> PResult<(EslMsa, usize)> {
    let mut msa = EslMsa::new();
    let mut pd = ParseData::default();
    let mut i = start;
    let mut saw_terminator = false;

    while i < nlines {
        // skip leading whitespace on the line (C: while *p==' '||'\t')
        let raw = lines[i];
        i += 1;
        let p = raw.trim_start_matches([' ', '\t']);

        if p.is_empty() || p.starts_with("//") {
            end_of_block(&mut msa, &mut pd);
            if p.starts_with("//") {
                saw_terminator = true;
                break;
            }
            continue;
        }

        if p.starts_with('#') {
            if p.starts_with("#=GF") {
                parse_gf(&mut msa, p)?;
            } else if p.starts_with("#=GS") {
                parse_gs(&mut msa, &mut pd, p)?;
            } else if p.starts_with("#=GC") {
                parse_gc(&mut msa, &mut pd, p)?;
            } else if p.starts_with("#=GR") {
                parse_gr(&mut msa, &mut pd, p)?;
            } else if p == "# STOCKHOLM 1.0" {
                return Err(MsaError::Format("two # STOCKHOLM 1.0 headers in a row?".into()));
            } else {
                parse_comment(&mut msa, p);
            }
        } else {
            parse_sq(&mut msa, &mut pd, p)?;
        }
    }

    if !saw_terminator {
        return Err(MsaError::Format("missing // terminator after MSA".into()));
    }
    if pd.nblock == 0 {
        return Err(MsaError::Format("no alignment data followed Stockholm header".into()));
    }

    msa.alen = msa.aseq.first().map(|s| s.len() as i64).unwrap_or(0);

    if msa.flags & ESL_MSA_HASWGTS != 0 {
        for (idx, w) in msa.wgt.iter().enumerate() {
            if *w == -1.0 {
                return Err(MsaError::Format(format!(
                    "stockholm record ended without a weight for {}",
                    msa.sqname[idx]
                )));
            }
        }
    } else {
        msa.set_default_weights();
    }

    if let Some(abc) = abc {
        msa.digitize(abc);
    }

    Ok((msa, i))
}

/* Per-record parse bookkeeping (subset of ESL_STOCKHOLM_PARSEDATA that we
 * need: name->index map, and block accounting to reconstruct interleaved
 * alignments). */
#[derive(Default)]
struct ParseData {
    index: std::collections::HashMap<String, usize>,
    nblock: i32,
    in_block: bool,
    si: usize, // guess for next seq index
}

fn end_of_block(_msa: &mut EslMsa, pd: &mut ParseData) {
    if pd.in_block {
        pd.in_block = false;
        pd.nblock += 1;
        pd.si = 0;
    }
}

fn get_seqidx(msa: &mut EslMsa, pd: &mut ParseData, name: &str) -> usize {
    if let Some(&idx) = pd.index.get(name) {
        return idx;
    }
    let idx = msa.add_seq(name);
    pd.index.insert(name.to_string(), idx);
    idx
}

/* stockholm_parse_gf: `#=GF <tag> <text>` ; recognized {ID AC DE AU GA NC TC} */
fn parse_gf(msa: &mut EslMsa, p: &str) -> PResult<()> {
    let (_gf, rest) = memtok(p, &[' ', '\t']).ok_or_else(|| fe("EOL can't happen"))?;
    let (tag, rest) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("#=GF line is missing <tag>"))?;
    match tag {
        "ID" => {
            let (tok, more) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("No name on #=GF ID line"))?;
            if !more.is_empty() {
                return Err(fe("#=GF ID line should have only one name"));
            }
            msa.name = Some(tok.to_string());
        }
        "AC" => {
            let (tok, more) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("No accession on #=GF AC line"))?;
            if !more.is_empty() {
                return Err(fe("#=GF AC line should have only one accession"));
            }
            msa.acc = Some(tok.to_string());
        }
        "DE" => msa.desc = Some(rest.to_string()),
        "AU" => msa.au = Some(rest.to_string()),
        "GA" => parse_two_cutoffs(msa, rest, ESL_MSA_GA1, ESL_MSA_GA2, "GA")?,
        "NC" => parse_nc(msa, rest)?,
        "TC" => parse_two_cutoffs(msa, rest, ESL_MSA_TC1, ESL_MSA_TC2, "TC")?,
        _ => {
            msa.gf_tag.push(tag.to_string());
            msa.gf.push(rest.to_string());
        }
    }
    Ok(())
}

fn parse_two_cutoffs(msa: &mut EslMsa, rest: &str, i1: usize, i2: usize, name: &str) -> PResult<()> {
    let (tok, rest) = memtok(rest, &[' ', '\t'])
        .ok_or_else(|| fe(&format!("No {name} threshold value found on #=GF {name} line")))?;
    msa.cutoff[i1] = parse_real(tok, name)?;
    msa.cutset[i1] = true;
    if let Some((tok2, _)) = memtok(rest, &[' ', '\t']) {
        msa.cutoff[i2] = parse_real(tok2, name)?;
        msa.cutset[i2] = true;
    }
    Ok(())
}

/* NC has the Rfam10 "undefined" workaround. */
fn parse_nc(msa: &mut EslMsa, rest: &str) -> PResult<()> {
    let (tok, rest) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("No NC threshold value on #=GF NC line"))?;
    if tok != "undefined" {
        msa.cutoff[ESL_MSA_NC1] = parse_real(tok, "NC")?;
        msa.cutset[ESL_MSA_NC1] = true;
    }
    if let Some((tok2, _)) = memtok(rest, &[' ', '\t']) {
        msa.cutoff[ESL_MSA_NC2] = parse_real(tok2, "NC")?;
        msa.cutset[ESL_MSA_NC2] = true;
    }
    Ok(())
}

fn parse_real(tok: &str, name: &str) -> PResult<f32> {
    tok.parse::<f32>()
        .map_err(|_| fe(&format!("Expected a real number for {name} value")))
}

/* stockholm_parse_gs: `#=GS <seqname> <tag> <text>` ; recognized {WT AC DE} */
fn parse_gs(msa: &mut EslMsa, pd: &mut ParseData, p: &str) -> PResult<()> {
    let (_gs, rest) = memtok(p, &[' ', '\t']).ok_or_else(|| fe("EOL can't happen"))?;
    let (seqname, rest) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("#=GS line missing <seqname>"))?;
    let (tag, rest) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("#=GS line missing <tag>"))?;
    let seqidx = get_seqidx(msa, pd, seqname);

    match tag {
        "WT" => {
            let (tok, more) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("no weight value on #=GS WT line"))?;
            if !more.is_empty() {
                return Err(fe("#=GS WT line should have only one field"));
            }
            msa.wgt[seqidx] = tok.parse::<f64>().map_err(|_| fe("value on #=GS WT line isn't a real number"))?;
            msa.flags |= ESL_MSA_HASWGTS;
        }
        "AC" => {
            let (tok, more) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("no accession on #=GS AC line"))?;
            if !more.is_empty() {
                return Err(fe("#=GS AC line should have only one field"));
            }
            ensure_seq_opt(&mut msa.sqacc, msa.nseq)[seqidx] = Some(tok.to_string());
        }
        "DE" => {
            ensure_seq_opt(&mut msa.sqdesc, msa.nseq)[seqidx] = Some(rest.to_string());
        }
        _ => {
            let ti = gs_tagidx(msa, tag);
            let cell = &mut msa.gs[ti][seqidx];
            // Multiannotated GS: newline-join repeats (esl_msa_AddGS).
            match cell {
                Some(existing) => {
                    existing.push('\n');
                    existing.push_str(rest);
                }
                None => *cell = Some(rest.to_string()),
            }
        }
    }
    pd.si = seqidx + 1;
    Ok(())
}

/* stockholm_parse_gc: `#=GC <tag> <aligned text>` ; recognized {SS_cons SA_cons PP_cons RF MM} */
fn parse_gc(msa: &mut EslMsa, pd: &mut ParseData, p: &str) -> PResult<()> {
    let (_gc, rest) = memtok(p, &[' ', '\t']).ok_or_else(|| fe("EOL can't happen"))?;
    let (tag, rest) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("#=GC line missing <tag>"))?;
    let text = rtrim_ws(rest);
    if text.is_empty() {
        return Err(fe("#=GC line missing annotation?"));
    }
    match tag {
        "SS_cons" => append_opt(&mut msa.ss_cons, text),
        "SA_cons" => append_opt(&mut msa.sa_cons, text),
        "PP_cons" => append_opt(&mut msa.pp_cons, text),
        "RF" => append_opt(&mut msa.rf, text),
        "MM" => append_opt(&mut msa.mm, text),
        _ => {
            let ti = gc_tagidx(msa, tag);
            msa.gc[ti].push_str(text);
        }
    }
    pd.in_block = true;
    Ok(())
}

/* stockholm_parse_gr: `#=GR <seqname> <tag> <aligned text>` ; recognized {SS SA PP} */
fn parse_gr(msa: &mut EslMsa, pd: &mut ParseData, p: &str) -> PResult<()> {
    let (_gr, rest) = memtok(p, &[' ', '\t']).ok_or_else(|| fe("EOL can't happen"))?;
    let (name, rest) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("#=GR line missing <seqname>"))?;
    let (tag, rest) = memtok(rest, &[' ', '\t']).ok_or_else(|| fe("#=GR line missing <tag>"))?;
    let text = rtrim_ws(rest);
    if text.is_empty() {
        return Err(fe("#=GR line missing annotation?"));
    }
    let seqidx = get_seqidx(msa, pd, name);
    match tag {
        "SS" => append_seq_opt(&mut msa.ss, msa.nseq, seqidx, text),
        "SA" => append_seq_opt(&mut msa.sa, msa.nseq, seqidx, text),
        "PP" => append_seq_opt(&mut msa.pp, msa.nseq, seqidx, text),
        _ => {
            let ti = gr_tagidx(msa, tag);
            while msa.gr[ti].len() < msa.nseq {
                msa.gr[ti].push(None);
            }
            match &mut msa.gr[ti][seqidx] {
                Some(s) => s.push_str(text),
                None => msa.gr[ti][seqidx] = Some(text.to_string()),
            }
        }
    }
    pd.in_block = true;
    Ok(())
}

/* stockholm_parse_sq: `<seqname> <aligned text>` */
fn parse_sq(msa: &mut EslMsa, pd: &mut ParseData, p: &str) -> PResult<()> {
    let (seqname, rest) = memtok(p, &[' ', '\t']).ok_or_else(|| fe("EOL can't happen"))?;
    let text = rtrim_ws(rest);
    if text.is_empty() {
        return Err(fe("sequence line with no sequence?"));
    }
    let seqidx = get_seqidx(msa, pd, seqname);
    msa.aseq[seqidx].push_str(text);
    pd.in_block = true;
    pd.si = seqidx + 1;
    Ok(())
}

/* stockholm_parse_comment: strip one '#' and leading whitespace, store. */
fn parse_comment(msa: &mut EslMsa, p: &str) {
    let s = p.strip_prefix('#').unwrap_or(p);
    let s = s.trim_start();
    msa.comment.push(s.to_string());
}

/* ---- small helpers for optional/tagged storage ---- */

fn fe(s: &str) -> MsaError {
    MsaError::Format(s.to_string())
}

fn append_opt(opt: &mut Option<String>, text: &str) {
    match opt {
        Some(s) => s.push_str(text),
        None => *opt = Some(text.to_string()),
    }
}

fn ensure_seq_opt(opt: &mut Option<Vec<Option<String>>>, nseq: usize) -> &mut Vec<Option<String>> {
    if opt.is_none() {
        *opt = Some(vec![None; nseq]);
    }
    let v = opt.as_mut().unwrap();
    while v.len() < nseq {
        v.push(None);
    }
    v
}

fn append_seq_opt(opt: &mut Option<Vec<Option<String>>>, nseq: usize, seqidx: usize, text: &str) {
    let v = ensure_seq_opt(opt, nseq);
    match &mut v[seqidx] {
        Some(s) => s.push_str(text),
        None => v[seqidx] = Some(text.to_string()),
    }
}

fn gs_tagidx(msa: &mut EslMsa, tag: &str) -> usize {
    if let Some(i) = msa.gs_tag.iter().position(|t| t == tag) {
        return i;
    }
    msa.gs_tag.push(tag.to_string());
    msa.gs.push(vec![None; msa.nseq]);
    msa.gs.len() - 1
}

fn gc_tagidx(msa: &mut EslMsa, tag: &str) -> usize {
    if let Some(i) = msa.gc_tag.iter().position(|t| t == tag) {
        return i;
    }
    msa.gc_tag.push(tag.to_string());
    msa.gc.push(String::new());
    msa.gc.len() - 1
}

fn gr_tagidx(msa: &mut EslMsa, tag: &str) -> usize {
    if let Some(i) = msa.gr_tag.iter().position(|t| t == tag) {
        return i;
    }
    msa.gr_tag.push(tag.to_string());
    msa.gr.push(vec![None; msa.nseq]);
    msa.gr.len() - 1
}

/*****************************************************************
 * Writer: stockholm_write (esl_msafile_stockholm.c:1069)
 *****************************************************************/

/// esl_msafile_Write (esl_msafile.c:1100): dispatch `msa` to the requested
/// format writer.
pub fn esl_msafile_write<W: Write>(w: &mut W, msa: &EslMsa, fmt: MsaFormat) -> io::Result<()> {
    match fmt {
        MsaFormat::Stockholm => stockholm_write(w, msa, 200),
        MsaFormat::Pfam => stockholm_write(w, msa, msa.alen),
        MsaFormat::A2m => a2m_write(w, msa),
        MsaFormat::Psiblast => psiblast_write(w, msa),
        MsaFormat::Selex => selex_write(w, msa),
        MsaFormat::Afa => afa_write(w, msa),
        MsaFormat::Clustal => clustal_write(w, msa, false),
        MsaFormat::ClustalLike => clustal_write(w, msa, true),
        MsaFormat::Phylip => phylip_write(w, msa, false),
        MsaFormat::Phylips => phylip_write(w, msa, true),
    }
}

/* left-justify `s` in a field of width `wid` (>=0). C: "%-*s" */
fn ljust(s: &str, wid: i32) -> String {
    let wid = wid.max(0) as usize;
    if s.len() >= wid {
        s.to_string()
    } else {
        let mut out = String::with_capacity(wid);
        out.push_str(s);
        for _ in 0..(wid - s.len()) {
            out.push(' ');
        }
        out
    }
}

fn stockholm_write<W: Write>(fp: &mut W, msa: &EslMsa, cpl: i64) -> io::Result<()> {
    let nseq = msa.nseq;

    /* Unique names? Else we uniqize with a "<seq#>|" prefix. */
    let make_uniquenames = !msa.check_unique_names();
    let mut uniqwidth = 0i32;
    if make_uniquenames {
        let mut tmp = nseq;
        while tmp != 0 {
            uniqwidth += 1;
            tmp /= 10;
        }
        uniqwidth += 1; /* includes the '|' */
    }

    let maxname = str_get_max_width(&msa.sqname);

    let mut maxgf = str_get_max_width(&msa.gf_tag);
    if maxgf < 2 {
        maxgf = 2;
    }

    let mut maxgc = str_get_max_width(&msa.gc_tag);
    if msa.rf.is_some() && maxgc < 2 {
        maxgc = 2;
    }
    if msa.mm.is_some() && maxgc < 2 {
        maxgc = 2;
    }
    if msa.ss_cons.is_some() && maxgc < 7 {
        maxgc = 7;
    }
    if msa.sa_cons.is_some() && maxgc < 7 {
        maxgc = 7;
    }
    if msa.pp_cons.is_some() && maxgc < 7 {
        maxgc = 7;
    }

    let mut maxgr = str_get_max_width(&msa.gr_tag);
    if msa.ss.is_some() && maxgr < 2 {
        maxgr = 2;
    }
    if msa.sa.is_some() && maxgr < 2 {
        maxgr = 2;
    }
    if msa.pp.is_some() && maxgr < 2 {
        maxgr = 2;
    }

    let mut margin = uniqwidth + maxname + 1;
    if maxgc > 0 && maxgc + 6 > margin {
        margin = maxgc + 6;
    }
    if maxgr > 0 && uniqwidth + maxname + maxgr + 7 > margin {
        margin = uniqwidth + maxname + maxgr + 7;
    }

    /* Magic Stockholm header */
    write!(fp, "# STOCKHOLM 1.0\n")?;
    if make_uniquenames {
        write!(
            fp,
            "# WARNING: seq names have been made unique by adding a prefix of \"<seq#>|\"\n"
        )?;
    }

    /* Free text comment section */
    for c in &msa.comment {
        write!(fp, "#{c}\n")?;
    }
    if !msa.comment.is_empty() {
        write!(fp, "\n")?;
    }

    /* GF section */
    if let Some(s) = &msa.name {
        write!(fp, "#=GF {} {}\n", ljust("ID", maxgf), s)?;
    }
    if let Some(s) = &msa.acc {
        write!(fp, "#=GF {} {}\n", ljust("AC", maxgf), s)?;
    }
    if let Some(s) = &msa.desc {
        write!(fp, "#=GF {} {}\n", ljust("DE", maxgf), s)?;
    }
    if let Some(s) = &msa.au {
        write!(fp, "#=GF {} {}\n", ljust("AU", maxgf), s)?;
    }

    write_cutoff(fp, msa, maxgf, ESL_MSA_GA1, ESL_MSA_GA2, "GA")?;
    write_cutoff(fp, msa, maxgf, ESL_MSA_NC1, ESL_MSA_NC2, "NC")?;
    write_cutoff(fp, msa, maxgf, ESL_MSA_TC1, ESL_MSA_TC2, "TC")?;

    for i in 0..msa.ngf() {
        write!(fp, "#=GF {} {}\n", ljust(&msa.gf_tag[i], maxgf), msa.gf[i])?;
    }
    write!(fp, "\n")?;

    /* GS section */
    if msa.flags & ESL_MSA_HASWGTS != 0 {
        for i in 0..nseq {
            write_gs_prefix(fp, make_uniquenames, uniqwidth, maxname, i, &msa.sqname[i])?;
            write!(fp, "WT {:.2}\n", msa.wgt[i])?;
        }
        write!(fp, "\n")?;
    }

    if let Some(sqacc) = &msa.sqacc {
        for i in 0..nseq {
            if let Some(Some(acc)) = sqacc.get(i) {
                write_gs_prefix(fp, make_uniquenames, uniqwidth, maxname, i, &msa.sqname[i])?;
                write!(fp, "AC {acc}\n")?;
            }
        }
        write!(fp, "\n")?;
    }

    if let Some(sqdesc) = &msa.sqdesc {
        for i in 0..nseq {
            if let Some(Some(d)) = sqdesc.get(i) {
                write_gs_prefix(fp, make_uniquenames, uniqwidth, maxname, i, &msa.sqname[i])?;
                write!(fp, "DE {d}\n")?;
            }
        }
        write!(fp, "\n")?;
    }

    /* Multiannotated GS tags (esl_msa.c stores newline-joined) */
    for ti in 0..msa.ngs() {
        let gslen = msa.gs_tag[ti].len() as i32;
        for j in 0..nseq {
            if let Some(Some(val)) = msa.gs[ti].get(j) {
                for tok in val.split('\n') {
                    write_gs_prefix(fp, make_uniquenames, uniqwidth, maxname, j, &msa.sqname[j])?;
                    write!(fp, "{} {}\n", ljust(&msa.gs_tag[ti], gslen), tok)?;
                }
            }
        }
        write!(fp, "\n")?;
    }

    /* Alignment section: blocks of cpl columns */
    let alen = msa.alen.max(0);
    let mut currpos: i64 = 0;
    while currpos < alen {
        let acpl = std::cmp::min(cpl, alen - currpos);
        let (lo, hi) = (currpos as usize, (currpos + acpl) as usize);
        if currpos > 0 {
            write!(fp, "\n")?;
        }

        for i in 0..nseq {
            let buf = row_slice_text(msa, i, lo, hi);
            /* seq line: name field width margin-uniqwidth-1 */
            if make_uniquenames {
                write!(
                    fp,
                    "{}|{} {}\n",
                    zpad(i, uniqwidth - 1),
                    ljust(&msa.sqname[i], margin - uniqwidth - 1),
                    buf
                )?;
            } else {
                write!(fp, "{} {}\n", ljust(&msa.sqname[i], margin - 1), buf)?;
            }

            write_gr(fp, msa, &msa.ss, i, lo, hi, "SS", make_uniquenames, uniqwidth, maxname, margin)?;
            write_gr(fp, msa, &msa.sa, i, lo, hi, "SA", make_uniquenames, uniqwidth, maxname, margin)?;
            write_gr(fp, msa, &msa.pp, i, lo, hi, "PP", make_uniquenames, uniqwidth, maxname, margin)?;
            for ti in 0..msa.ngr() {
                if let Some(Some(s)) = msa.gr[ti].get(i) {
                    let sub = &s[lo..hi.min(s.len())];
                    write_gr_line(fp, make_uniquenames, uniqwidth, maxname, margin, i, &msa.sqname[i], &msa.gr_tag[ti], sub)?;
                }
            }
        }

        /* #=GC lines, fixed order then other gc[] */
        write_gc(fp, &msa.ss_cons, "SS_cons", margin, lo, hi)?;
        write_gc(fp, &msa.sa_cons, "SA_cons", margin, lo, hi)?;
        write_gc(fp, &msa.pp_cons, "PP_cons", margin, lo, hi)?;
        write_gc(fp, &msa.rf, "RF", margin, lo, hi)?;
        write_gc(fp, &msa.mm, "MM", margin, lo, hi)?;
        for ti in 0..msa.ngc() {
            let s = &msa.gc[ti];
            write!(fp, "#=GC {} {}\n", ljust(&msa.gc_tag[ti], margin - 6), &s[lo..hi.min(s.len())])?;
        }

        currpos += cpl;
    }

    write!(fp, "//\n")?;
    Ok(())
}

fn zpad(i: usize, width: i32) -> String {
    format!("{:0>1$}", i, width.max(0) as usize)
}

fn write_gs_prefix<W: Write>(
    fp: &mut W,
    make_uniquenames: bool,
    uniqwidth: i32,
    maxname: i32,
    i: usize,
    name: &str,
) -> io::Result<()> {
    if make_uniquenames {
        write!(fp, "#=GS {}|{} ", zpad(i, uniqwidth - 1), ljust(name, maxname))
    } else {
        write!(fp, "#=GS {} ", ljust(name, maxname))
    }
}

fn write_cutoff<W: Write>(
    fp: &mut W,
    msa: &EslMsa,
    maxgf: i32,
    i1: usize,
    i2: usize,
    name: &str,
) -> io::Result<()> {
    if msa.cutset[i1] && msa.cutset[i2] {
        write!(fp, "#=GF {} {:.1} {:.1}\n", ljust(name, maxgf), msa.cutoff[i1], msa.cutoff[i2])?;
    } else if msa.cutset[i1] {
        write!(fp, "#=GF {} {:.1}\n", ljust(name, maxgf), msa.cutoff[i1])?;
    }
    Ok(())
}

/* Textized column window for sequence i (text mode uses aseq; digital re-textizes ax). */
fn row_slice_text(msa: &EslMsa, i: usize, lo: usize, hi: usize) -> String {
    if msa.is_digital {
        // ax[i][1..=alen]; window [lo,hi) maps to ax indices [lo+1, hi+1)
        let abc = EslAlphabet::rna();
        let row = &msa.ax[i];
        (lo..hi)
            .map(|c| {
                let code = row[c + 1] as usize;
                if code < abc.sym.len() {
                    abc.sym[code]
                } else {
                    '?'
                }
            })
            .collect()
    } else {
        let s = &msa.aseq[i];
        s[lo..hi.min(s.len())].to_string()
    }
}

#[allow(clippy::too_many_arguments)]
fn write_gr<W: Write>(
    fp: &mut W,
    _msa: &EslMsa,
    opt: &Option<Vec<Option<String>>>,
    i: usize,
    lo: usize,
    hi: usize,
    tag: &str,
    make_uniquenames: bool,
    uniqwidth: i32,
    maxname: i32,
    margin: i32,
) -> io::Result<()> {
    if let Some(v) = opt {
        if let Some(Some(s)) = v.get(i) {
            let sub = &s[lo..hi.min(s.len())];
            write_gr_line(fp, make_uniquenames, uniqwidth, maxname, margin, i, &_msa.sqname[i], tag, sub)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_gr_line<W: Write>(
    fp: &mut W,
    make_uniquenames: bool,
    uniqwidth: i32,
    maxname: i32,
    margin: i32,
    i: usize,
    name: &str,
    tag: &str,
    data: &str,
) -> io::Result<()> {
    if make_uniquenames {
        write!(
            fp,
            "#=GR {}|{} {} {}\n",
            zpad(i, uniqwidth - 1),
            ljust(name, maxname),
            ljust(tag, margin - maxname - uniqwidth - 7),
            data
        )
    } else {
        write!(fp, "#=GR {} {} {}\n", ljust(name, maxname), ljust(tag, margin - maxname - 7), data)
    }
}

fn write_gc<W: Write>(
    fp: &mut W,
    opt: &Option<String>,
    tag: &str,
    margin: i32,
    lo: usize,
    hi: usize,
) -> io::Result<()> {
    if let Some(s) = opt {
        write!(fp, "#=GC {} {}\n", ljust(tag, margin - 6), &s[lo..hi.min(s.len())])?;
    }
    Ok(())
}

/*****************************************************************
 * Other output formats: afa, a2m, psiblast, selex, clustal(+like), phylip(+s)
 *
 * Faithful ports of the per-format `esl_msafile_*_Write` functions. infernox
 * builds text-mode MSAs (msa->abc == NULL) for cmemit/cmalign, so the text
 * branch (`! msa->abc`) is the one exercised; the digital branch is ported for
 * completeness and re-textizes through the RNA alphabet like `row_slice_text`.
 *****************************************************************/

/* esl_config.h:17 : #define EASEL_VERSION "0.49" (clustallike magic header) */
const EASEL_VERSION: &str = "0.49";

/* esl_abc_XIsResidue (esl_alphabet.h:103) for the RNA alphabet (K=4, Kp=18):
 * x < K || (x > K && x < Kp-2). */
fn xis_residue(code: u8) -> bool {
    let x = code as i32;
    x < 4 || (x > 4 && x < 16)
}

/* C: fprintf(fp, ">%s", name); optional " %s" acc; optional " %s" desc; '\n'.
 * Shared by esl_msafile_afa_Write and esl_msafile_a2m_Write. */
fn write_fasta_header<W: Write>(fp: &mut W, msa: &EslMsa, i: usize) -> io::Result<()> {
    write!(fp, ">{}", msa.sqname[i])?;
    if let Some(v) = &msa.sqacc {
        if let Some(Some(a)) = v.get(i) {
            write!(fp, " {a}")?;
        }
    }
    if let Some(v) = &msa.sqdesc {
        if let Some(Some(d)) = v.get(i) {
            write!(fp, " {d}")?;
        }
    }
    fp.write_all(b"\n")?;
    Ok(())
}

/* Substring [apos, apos+cpl) clamped to the string length; mirrors C `%.*s`
 * with precision cpl on `s+apos` (which stops at the NUL terminator). */
fn sub_cpl(s: &str, apos: usize, cpl: usize) -> &str {
    if apos >= s.len() {
        ""
    } else {
        &s[apos..(apos + cpl).min(s.len())]
    }
}

/* C: "%-*.*s" namewidth namewidth — left-justify AND truncate name to `wid`. */
fn ljust_trunc(s: &str, wid: usize) -> String {
    let b = s.as_bytes();
    if b.len() >= wid {
        String::from_utf8_lossy(&b[..wid]).into_owned()
    } else {
        let mut out = String::with_capacity(wid);
        out.push_str(s);
        for _ in 0..(wid - b.len()) {
            out.push(' ');
        }
        out
    }
}

/*****************************************************************
 * esl_msafile_afa_Write (esl_msafile_afa.c) — aligned FASTA, 60 res/line.
 *****************************************************************/
fn afa_write<W: Write>(fp: &mut W, msa: &EslMsa) -> io::Result<()> {
    let alen = msa.alen.max(0);
    for i in 0..msa.nseq {
        write_fasta_header(fp, msa, i)?;
        let mut pos: i64 = 0;
        while pos < alen {
            let acpl = if alen - pos > 60 { 60 } else { alen - pos };
            let buf = row_slice_text(msa, i, pos as usize, (pos + acpl) as usize);
            write!(fp, "{buf}\n")?;
            pos += 60;
        }
    }
    Ok(())
}

/*****************************************************************
 * esl_msafile_a2m_Write (esl_msafile_a2m.c) — UCSC a2m, dotless (do_dotless=TRUE).
 * Consensus columns (RF alnum, or seq0 residue if no RF) print upper residue or
 * '-'; insert columns print lower residue and drop insert-gaps (dotless).
 *****************************************************************/
fn a2m_write<W: Write>(fp: &mut W, msa: &EslMsa) -> io::Result<()> {
    let cpl = 60usize;
    let alen = msa.alen.max(0) as usize;
    let do_dotless = true;
    let abc = EslAlphabet::rna();
    let rf = msa.rf.as_ref().map(|s| s.as_bytes());
    for i in 0..msa.nseq {
        write_fasta_header(fp, msa, i)?;
        let mut pos = 0usize;
        while pos < alen {
            let mut buf: Vec<u8> = Vec::with_capacity(cpl);
            while pos < alen && buf.len() < cpl {
                // sym + is_residue for this seq at column pos
                let (sym, is_residue) = if msa.is_digital {
                    let code = msa.ax[i][pos + 1];
                    let mut s = abc.sym[code as usize] as u8;
                    if s == b'O' {
                        s = abc.sym[(abc.Kp - 3) as usize] as u8; // esl_abc_CGetUnknown
                    }
                    (s, xis_residue(code))
                } else {
                    let orig = msa.aseq[i].as_bytes()[pos];
                    let is_res = orig.is_ascii_alphabetic();
                    let s = if orig == b'O' { b'X' } else { orig };
                    (s, is_res)
                };
                // is_consensus for this column
                let is_consensus = match rf {
                    Some(rf) => rf[pos].is_ascii_alphanumeric(),
                    None => {
                        if msa.is_digital {
                            xis_residue(msa.ax[0][pos + 1])
                        } else {
                            msa.aseq[0].as_bytes()[pos].is_ascii_alphanumeric()
                        }
                    }
                };
                if is_consensus {
                    buf.push(if is_residue { sym.to_ascii_uppercase() } else { b'-' });
                } else if is_residue {
                    buf.push(sym.to_ascii_lowercase());
                } else if !do_dotless {
                    buf.push(b'.');
                }
                pos += 1;
            }
            if !buf.is_empty() {
                fp.write_all(&buf)?;
                fp.write_all(b"\n")?;
            }
        }
    }
    Ok(())
}

/*****************************************************************
 * esl_msafile_psiblast_Write (esl_msafile_psiblast.c) — 60 col blocks, seqs
 * interleaved, gaps -> '-', case set by consensus (RF alnum / seq0 residue).
 *****************************************************************/
fn psiblast_write<W: Write>(fp: &mut W, msa: &EslMsa) -> io::Result<()> {
    let cpl = 60usize;
    let alen = msa.alen.max(0) as usize;
    let maxnamewidth = str_get_max_width(&msa.sqname);
    let abc = EslAlphabet::rna();
    let rf = msa.rf.as_ref().map(|s| s.as_bytes());
    let mut pos = 0usize;
    while pos < alen {
        for i in 0..msa.nseq {
            let acpl = if alen - pos > cpl { cpl } else { alen - pos };
            let mut buf: Vec<u8> = Vec::with_capacity(acpl);
            for bpos in 0..acpl {
                let col = pos + bpos;
                let (sym, is_residue) = if msa.is_digital {
                    let code = msa.ax[i][col + 1];
                    (abc.sym[code as usize] as u8, xis_residue(code))
                } else {
                    let s = msa.aseq[i].as_bytes()[col];
                    (s, s.is_ascii_alphanumeric()) // psiblast: isalnum
                };
                let is_consensus = match rf {
                    Some(rf) => rf[col].is_ascii_alphanumeric(),
                    None => {
                        if msa.is_digital {
                            xis_residue(msa.ax[0][col + 1])
                        } else {
                            msa.aseq[0].as_bytes()[col].is_ascii_alphanumeric()
                        }
                    }
                };
                let c = if is_consensus {
                    if is_residue { sym.to_ascii_uppercase() } else { b'-' }
                } else if is_residue {
                    sym.to_ascii_lowercase()
                } else {
                    b'-'
                };
                buf.push(c);
            }
            // C: fprintf(fp, "%-*s  %s\n", maxnamewidth, name, buf) — two spaces.
            write!(fp, "{}", ljust(&msa.sqname[i], maxnamewidth))?;
            fp.write_all(b"  ")?;
            fp.write_all(&buf)?;
            fp.write_all(b"\n")?;
        }
        if pos + cpl < alen {
            fp.write_all(b"\n")?;
        }
        pos += cpl;
    }
    Ok(())
}

/*****************************************************************
 * esl_msafile_selex_Write (esl_msafile_selex.c) — 60 col blocks with #=CS/#=RF/
 * #=MM markup above each block and #=SS/#=SA below each seq. maxnamelen init 4.
 *****************************************************************/
fn selex_write<W: Write>(fp: &mut W, msa: &EslMsa) -> io::Result<()> {
    let cpl = 60usize;
    let alen = msa.alen.max(0) as usize;
    let mut maxnamelen = 4i32; // min field is "#=CS", etc.
    for nm in &msa.sqname {
        maxnamelen = maxnamelen.max(nm.len() as i32);
    }
    let mut apos = 0usize;
    while apos < alen {
        if apos > 0 {
            fp.write_all(b"\n")?;
        }
        if let Some(s) = &msa.ss_cons {
            writeln!(fp, "{} {}", ljust("#=CS", maxnamelen), sub_cpl(s, apos, cpl))?;
        }
        if let Some(s) = &msa.rf {
            writeln!(fp, "{} {}", ljust("#=RF", maxnamelen), sub_cpl(s, apos, cpl))?;
        }
        if let Some(s) = &msa.mm {
            writeln!(fp, "{} {}", ljust("#=MM", maxnamelen), sub_cpl(s, apos, cpl))?;
        }
        for i in 0..msa.nseq {
            let win = row_slice_text(msa, i, apos, (apos + cpl).min(alen));
            writeln!(fp, "{} {}", ljust(&msa.sqname[i], maxnamelen), win)?;
            if let Some(v) = &msa.ss {
                if let Some(Some(s)) = v.get(i) {
                    writeln!(fp, "{} {}", ljust("#=SS", maxnamelen), sub_cpl(s, apos, cpl))?;
                }
            }
            if let Some(v) = &msa.sa {
                if let Some(Some(s)) = v.get(i) {
                    writeln!(fp, "{} {}", ljust("#=SA", maxnamelen), sub_cpl(s, apos, cpl))?;
                }
            }
        }
        apos += cpl;
    }
    Ok(())
}

/*****************************************************************
 * esl_msafile_clustal_Write (esl_msafile_clustal.c) — CLUSTAL / CLUSTAL-like.
 *****************************************************************/
fn clustal_write<W: Write>(fp: &mut W, msa: &EslMsa, like: bool) -> io::Result<()> {
    let cpl = 60usize;
    let alen = msa.alen.max(0) as usize;
    let mut maxnamelen = 0i32;
    for nm in &msa.sqname {
        maxnamelen = maxnamelen.max(nm.len() as i32);
    }
    let consline = if msa.is_digital {
        make_digital_consensus_line(msa)
    } else {
        make_text_consensus_line(msa)
    };
    if like {
        write!(fp, "EASEL ({EASEL_VERSION}) multiple sequence alignment\n")?;
    } else {
        write!(fp, "CLUSTAL 2.1 multiple sequence alignment\n")?;
    }
    let mut apos = 0usize;
    while apos < alen {
        fp.write_all(b"\n")?;
        for i in 0..msa.nseq {
            let win = row_slice_text(msa, i, apos, (apos + cpl).min(alen));
            writeln!(fp, "{} {}", ljust(&msa.sqname[i], maxnamelen), win)?;
        }
        let cwin = &consline[apos..(apos + cpl).min(alen)];
        writeln!(fp, "{} {}", ljust("", maxnamelen), cwin)?;
        apos += cpl;
    }
    Ok(())
}

/* make_text_consensus_line (esl_msafile_clustal.c): '*' where a column is fully
 * conserved to a single letter (case-insensitive) with no gaps, else ' '. */
fn make_text_consensus_line(msa: &EslMsa) -> String {
    let alen = msa.alen.max(0) as usize;
    let maxv: u32 = (1u32 << 26) - 1;
    let mut v = vec![0u32; alen];
    for idx in 0..msa.nseq {
        let bytes = msa.aseq[idx].as_bytes();
        for (apos, vb) in v.iter_mut().enumerate() {
            let x = (bytes[apos].to_ascii_uppercase() as i32) - ('A' as i32);
            if (0..26).contains(&x) {
                *vb |= 1u32 << x;
            } else {
                *vb |= 1u32 << 26;
            }
        }
    }
    let mut consline = Vec::with_capacity(alen);
    for &vv in &v {
        consline.push(if vv.count_ones() == 1 && vv < maxv {
            b'*'
        } else {
            b' '
        });
    }
    String::from_utf8(consline).unwrap()
}

/* make_digital_consensus_line (esl_msafile_clustal.c). infernox CMs are
 * nucleic-acid only, so only the non-amino path is reachable ('*' for full
 * conservation of a canonical residue); the eslAMINO ':'/'.' similarity groups
 * are not ported. */
fn make_digital_consensus_line(msa: &EslMsa) -> String {
    let abc = EslAlphabet::rna();
    let alen = msa.alen.max(0) as usize;
    let maxv: u32 = (1u32 << abc.K) - 1;
    let mut v = vec![0u32; alen + 1];
    for idx in 0..msa.nseq {
        for apos in 1..=alen {
            v[apos] |= 1u32 << msa.ax[idx][apos];
        }
    }
    let mut consline = vec![b' '; alen];
    for apos in 1..=alen {
        let n = v[apos].count_ones();
        if n == 0 || n > 6 {
            continue;
        } else if v[apos] > maxv {
            continue;
        } else if n == 1 {
            consline[apos - 1] = b'*';
        }
    }
    String::from_utf8(consline).unwrap()
}

/*****************************************************************
 * esl_msafile_phylip_Write (esl_msafile_phylip.c) — interleaved / sequential.
 * Default rpl=60, namewidth=10. Output residues are "rectified":
 *  text mode  : uppercase, "._ " -> '-', '~' -> '?'.
 *  digital    : '~' -> '?'.
 *****************************************************************/
fn phylip_rectify(msa: &EslMsa, idx: usize, lo: usize, hi: usize) -> Vec<u8> {
    let mut b = row_slice_text(msa, idx, lo, hi).into_bytes();
    if msa.is_digital {
        for c in b.iter_mut() {
            if *c == b'~' {
                *c = b'?';
            }
        }
    } else {
        for c in b.iter_mut() {
            if c.is_ascii_lowercase() {
                *c = c.to_ascii_uppercase();
            }
            if *c == b'.' || *c == b'_' || *c == b' ' {
                *c = b'-';
            }
            if *c == b'~' {
                *c = b'?';
            }
        }
    }
    b
}

fn phylip_write<W: Write>(fp: &mut W, msa: &EslMsa, sequential: bool) -> io::Result<()> {
    let rpl = 60usize;
    let namewidth = 10usize;
    let alen = msa.alen.max(0) as usize;

    if sequential {
        // phylip_sequential_Write: header WITH newline.
        write!(fp, " {} {}\n", msa.nseq, msa.alen)?;
        for idx in 0..msa.nseq {
            let mut apos = 0usize;
            while apos < alen {
                let win = phylip_rectify(msa, idx, apos, (apos + rpl).min(alen));
                if apos == 0 {
                    write!(fp, "{} ", ljust_trunc(&msa.sqname[idx], namewidth))?;
                }
                fp.write_all(&win)?;
                fp.write_all(b"\n")?;
                apos += rpl;
            }
        }
    } else {
        // phylip_interleaved_Write: header WITHOUT newline; each block preceded
        // by a "\n" (so a blank line separates blocks; names on first block).
        write!(fp, " {} {}", msa.nseq, msa.alen)?;
        let mut apos = 0usize;
        while apos < alen {
            fp.write_all(b"\n")?;
            for idx in 0..msa.nseq {
                let win = phylip_rectify(msa, idx, apos, (apos + rpl).min(alen));
                if apos == 0 {
                    write!(fp, "{} ", ljust_trunc(&msa.sqname[idx], namewidth))?;
                }
                fp.write_all(&win)?;
                fp.write_all(b"\n")?;
            }
            apos += rpl;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::process::Command;

    const TRNA: &str =
        "/mnt/DAS/sunju/programme/bactars/infernal/original/easel/testsuite/trna-5.stk";
    const RF: &str = "/mnt/DAS/sunju/programme/bactars/infernal/original/testsuite/3.4.12.rf.stk";
    const REFORMAT: &str =
        "/mnt/DAS/sunju/programme/bactars/infernal/original/easel/miniapps/esl-reformat";

    /// Read `fixture`, write it back in `fmt`, and require byte-identity with
    /// the C golden binary (`esl-reformat <fmt> fixture`). Skips (passes) if
    /// the fixture or the C binary is unavailable in this environment.
    fn roundtrip_matches_c(fixture: &str, fmt: MsaFormat, fmt_arg: &str) {
        if !Path::new(fixture).exists() || !Path::new(REFORMAT).exists() {
            eprintln!("skipping: fixture or esl-reformat not present");
            return;
        }
        let golden = Command::new(REFORMAT)
            .arg(fmt_arg)
            .arg(fixture)
            .output()
            .expect("run esl-reformat")
            .stdout;

        let input = std::fs::read_to_string(fixture).unwrap();
        let msas = read_all(&input, None).unwrap();
        let mut mine = Vec::new();
        for m in &msas {
            esl_msafile_write(&mut mine, m, fmt).unwrap();
        }
        assert_eq!(
            mine, golden,
            "byte mismatch vs esl-reformat {fmt_arg} on {fixture}"
        );
    }

    #[test]
    fn trna_stockholm_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::Stockholm, "stockholm");
    }
    #[test]
    fn trna_pfam_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::Pfam, "pfam");
    }
    #[test]
    fn rf_stockholm_byte_parity() {
        roundtrip_matches_c(RF, MsaFormat::Stockholm, "stockholm");
    }
    #[test]
    fn rf_pfam_byte_parity() {
        roundtrip_matches_c(RF, MsaFormat::Pfam, "pfam");
    }

    /* New format writers vs esl-reformat golden (text mode). */
    #[test]
    fn trna_afa_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::Afa, "afa");
    }
    #[test]
    fn trna_a2m_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::A2m, "a2m");
    }
    #[test]
    fn trna_psiblast_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::Psiblast, "psiblast");
    }
    #[test]
    fn trna_selex_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::Selex, "selex");
    }
    #[test]
    fn trna_clustal_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::Clustal, "clustal");
    }
    #[test]
    fn trna_clustallike_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::ClustalLike, "clustallike");
    }
    #[test]
    fn trna_phylip_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::Phylip, "phylip");
    }
    #[test]
    fn trna_phylips_byte_parity() {
        roundtrip_matches_c(TRNA, MsaFormat::Phylips, "phylips");
    }
    // NOTE: afa/a2m/psiblast/selex/clustal are single-alignment formats. C's
    // esl-reformat refuses to write a multi-record input to them ("...output
    // file can only contain 1"), so the multi-record RF fixture is only a valid
    // golden for the formats that C does emit per-record: phylip/phylips (and
    // stockholm/pfam above). Single-record parity for all formats is covered by
    // the trna_* tests, and end-to-end by the cmemit/cmalign byte-diffs.
    #[test]
    fn rf_phylip_byte_parity() {
        roundtrip_matches_c(RF, MsaFormat::Phylip, "phylip");
    }
    #[test]
    fn rf_phylips_byte_parity() {
        roundtrip_matches_c(RF, MsaFormat::Phylips, "phylips");
    }

    #[test]
    fn encode_format_roundtrip() {
        assert_eq!(esl_msafile_encode_format("Stockholm"), Some(MsaFormat::Stockholm));
        assert_eq!(esl_msafile_encode_format("AFA"), Some(MsaFormat::Afa));
        assert_eq!(esl_msafile_encode_format("clustallike"), Some(MsaFormat::ClustalLike));
        assert_eq!(esl_msafile_encode_format("phylips"), Some(MsaFormat::Phylips));
        assert_eq!(esl_msafile_encode_format("bogus"), None);
    }

    #[test]
    fn parses_annotation_fields() {
        if !Path::new(TRNA).exists() {
            return;
        }
        let input = std::fs::read_to_string(TRNA).unwrap();
        let msas = read_all(&input, None).unwrap();
        assert_eq!(msas.len(), 1);
        let m = &msas[0];
        assert_eq!(m.nseq, 5);
        assert_eq!(m.au.as_deref(), Some("Infernal 0.1"));
        assert!(m.ss_cons.is_some());
        assert!(m.rf.is_some());
        assert!(m.pp.is_some()); // per-seq #=GR PP
        assert_eq!(m.alen, m.aseq[0].len() as i64);
    }

    #[test]
    fn digital_read_reconstructs_length() {
        if !Path::new(TRNA).exists() {
            return;
        }
        let input = std::fs::read_to_string(TRNA).unwrap();
        let abc = crate::easel::alphabet::EslAlphabet::rna();
        let msas = read_all(&input, Some(&abc)).unwrap();
        let m = &msas[0];
        assert!(m.is_digital);
        // ax[i] has sentinels at both ends: length = alen + 2.
        for row in &m.ax {
            assert_eq!(row.len() as i64, m.alen + 2);
            assert_eq!(row[0], crate::easel::constants::ESL_DSQ_SENTINEL);
            assert_eq!(*row.last().unwrap(), crate::easel::constants::ESL_DSQ_SENTINEL);
        }
    }

    #[test]
    fn rf_multi_record() {
        if !Path::new(RF).exists() {
            return;
        }
        let input = std::fs::read_to_string(RF).unwrap();
        let msas = read_all(&input, None).unwrap();
        assert_eq!(msas.len(), 3); // 3.4.12.rf.stk holds U1, U2, U4
        assert_eq!(msas[0].name.as_deref(), Some("U1"));
    }
}
