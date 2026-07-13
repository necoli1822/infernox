//! sqio - faithful Rust port of Easel's multi-format sequence input (`esl_sqio`).
//!
//! Supports the ASCII sequence formats that C Infernal's tool bins accept for
//! their target/query databases: FASTA, EMBL, GenBank, DDBJ (== GenBank parser),
//! UniProt (== EMBL parser), plus gzip (`.gz`) transparently. Format is either
//! forced via `--informat`/`--qformat`/`--tformat` (see [`esl_sqio_encode_format`])
//! or autodetected (see [`guess_file_format`]) exactly as C does.
//!
//! The reader returns `(name, desc, seq)` triples so it is a drop-in replacement
//! for the hand-rolled `read_fasta()` the bins previously used; `seq` is the raw
//! residue string (undigitized) — downstream digitization is unchanged, keeping
//! the FASTA path byte-identical.
//!
//! C references: `original/easel/esl_sqio.c`, `esl_sqio_ascii.c`, `esl_sqio.h`.

use std::io::Read;

/// Sequence file format codes.
///
/// C `esl_sqio.h:106-115`: eslSQFILE_UNKNOWN=0, FASTA=1, EMBL=2, GENBANK=3,
/// DDBJ=4, UNIPROT=5, NCBI=6, DAEMON=7, HMMPGMD=8, FMINDEX=9. We implement the
/// ASCII formats the CM tool bins actually accept (FASTA/EMBL/GENBANK/DDBJ/UNIPROT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqFormat {
    Unknown = 0,
    Fasta = 1,
    Embl = 2,
    Genbank = 3,
    Ddbj = 4,
    Uniprot = 5,
}

impl SqFormat {
    /// True for any format we can actually parse here.
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            SqFormat::Fasta
                | SqFormat::Embl
                | SqFormat::Genbank
                | SqFormat::Ddbj
                | SqFormat::Uniprot
        )
    }
}

/// C `esl_sqio.c:esl_sqio_EncodeFormat`: decode a `--informat`/`--qformat`/
/// `--tformat` string (case-insensitive) to a format code. Unknown strings would
/// in C fall through to `esl_msafile_EncodeFormat`; here we return `None` so the
/// caller can emit the same "not a recognized... file format" diagnostic.
///
/// C `esl_sqio.c`:
///   if (strcasecmp(fmtstring, "fasta")   == 0) return eslSQFILE_FASTA;
///   if (strcasecmp(fmtstring, "embl")    == 0) return eslSQFILE_EMBL;
///   if (strcasecmp(fmtstring, "genbank") == 0) return eslSQFILE_GENBANK;
///   if (strcasecmp(fmtstring, "ddbj")    == 0) return eslSQFILE_DDBJ;
///   if (strcasecmp(fmtstring, "uniprot") == 0) return eslSQFILE_UNIPROT;
pub fn esl_sqio_encode_format(fmtstring: &str) -> Option<SqFormat> {
    let f = fmtstring.to_ascii_lowercase();
    match f.as_str() {
        "fasta" => Some(SqFormat::Fasta),
        "embl" => Some(SqFormat::Embl),
        "genbank" => Some(SqFormat::Genbank),
        "ddbj" => Some(SqFormat::Ddbj),
        "uniprot" => Some(SqFormat::Uniprot),
        _ => None,
    }
}

/// C `esl_sqio_ascii.c:sqascii_GuessFileFormat`: guess format from filename suffix,
/// then (if inconclusive) from the first nonblank line of the file.
///
/// `first_nonblank` should be the first line of the file that is not all
/// whitespace (or empty if none). `filename` is used for suffix detection; a
/// trailing `.gz` is ignored when locating the format suffix.
pub fn guess_file_format(filename: &str, first_nonblank: &str) -> SqFormat {
    // C: "Is <filename> gzip'ed? Look at suffix." then locate the suffix that
    // might indicate format (ignoring .gz).
    let name_no_gz = filename.strip_suffix(".gz").unwrap_or(filename);

    // C: strcmp(sfx, ".fa") / ".gb" on the last dotted suffix.
    if let Some(dot) = name_no_gz.rfind('.') {
        let sfx = &name_no_gz[dot..];
        if sfx == ".fa" {
            return SqFormat::Fasta;
        } else if sfx == ".gb" {
            return SqFormat::Genbank;
        }
    }

    // C: peek at the first nonblank line of the stream.
    // if (*buf == '>')                                    FASTA
    // else if (strncmp(buf, "ID   ", 5)    == 0)          EMBL
    // else if (strncmp(buf, "LOCUS   ", 8) == 0)          GENBANK
    // else if (strstr(buf, "Genetic Sequence Data Bank")) GENBANK
    if first_nonblank.starts_with('>') {
        SqFormat::Fasta
    } else if first_nonblank.starts_with("ID   ") {
        SqFormat::Embl
    } else if first_nonblank.starts_with("LOCUS   ") {
        SqFormat::Genbank
    } else if first_nonblank.contains("Genetic Sequence Data Bank") {
        SqFormat::Genbank
    } else {
        SqFormat::Unknown
    }
}

/// C `easel.c:esl_str_IsBlank`: true if the string is empty or all whitespace.
fn is_blank(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_whitespace())
}

/// C `easel.c:esl_strtok`: return the first token of `s`, where a token is a
/// maximal run of characters not in `delims`, after skipping any leading `delims`.
/// Returns `None` on end-of-line (no token).
fn strtok<'a>(s: &'a str, delims: &str) -> Option<&'a str> {
    let is_delim = |c: char| delims.contains(c);
    let start = s.find(|c| !is_delim(c))?;
    let rest = &s[start..];
    let end = rest.find(is_delim).unwrap_or(rest.len());
    Some(&rest[..end])
}

/// C `easel.c:esl_strchop`: trim trailing whitespace.
fn strchop(s: &str) -> &str {
    s.trim_end_matches(|c: char| c.is_ascii_whitespace())
}

/// Read a sequence database file, returning `(name, desc, seq)` records.
///
/// `informat` forces a parser (from `--informat`/`--qformat`/`--tformat`); pass
/// [`SqFormat::Unknown`] to autodetect. A `.gz` file is transparently decompressed.
/// On any error a human-readable message is returned in `Err`.
pub fn read_seqfile(
    path: &str,
    informat: SqFormat,
) -> Result<Vec<(String, String, String)>, String> {
    // Read the whole file (decompressing gzip transparently). C uses esl_buffer /
    // gzip -dc; we read to a String because the CM bins load the entire DB anyway.
    let raw = std::fs::read(path).map_err(|e| format!("cannot read sequence file '{path}': {e}"))?;
    let text = if path.ends_with(".gz") {
        let mut dec = flate2::read::GzDecoder::new(&raw[..]);
        let mut s = String::new();
        dec.read_to_string(&mut s)
            .map_err(|e| format!("cannot decompress gzip file '{path}': {e}"))?;
        s
    } else {
        String::from_utf8_lossy(&raw).into_owned()
    };

    // Determine format: forced, or guess from suffix + first nonblank line.
    let fmt = if informat != SqFormat::Unknown {
        informat
    } else {
        let first_nonblank = text.lines().find(|l| !is_blank(l)).unwrap_or("");
        let g = guess_file_format(path, first_nonblank);
        if g == SqFormat::Unknown {
            return Err(format!(
                "Failed to determine format of sequence file '{path}'"
            ));
        }
        g
    };

    match fmt {
        // C `esl_sqio_ascii.c:272-276`: DDBJ shares the GenBank parser, UniProt
        // shares the EMBL parser.
        SqFormat::Fasta => Ok(parse_fasta(&text)),
        SqFormat::Genbank | SqFormat::Ddbj => parse_genbank(&text, path),
        SqFormat::Embl | SqFormat::Uniprot => parse_embl(&text, path),
        SqFormat::Unknown => Err(format!(
            "Failed to determine format of sequence file '{path}'"
        )),
    }
}

/// FASTA parser. Byte-for-byte preserves the behavior of the bins' previous
/// hand-rolled `read_fasta()` (name = first whitespace-delimited token after `>`,
/// desc = remainder trimmed, seq = concatenation of trimmed residue lines), so the
/// verified FASTA hit output does not regress.
///
/// C `esl_sqio_ascii.c:header_fasta` skips whitespace after `>`, takes the
/// space-delimited name, then the rest of the line (to EOL) as description.
fn parse_fasta(text: &str) -> Vec<(String, String, String)> {
    let mut recs = Vec::new();
    let mut name = String::new();
    let mut desc = String::new();
    let mut seq = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            if !name.is_empty() {
                recs.push((name.clone(), desc.clone(), seq.clone()));
            }
            // Skip leading whitespace after '>' (esl_sqio treats "> name" == ">name").
            let mut it = rest.trim_start().splitn(2, char::is_whitespace);
            name = it.next().unwrap_or("").to_string();
            desc = it.next().unwrap_or("").trim().to_string();
            seq.clear();
        } else {
            seq.push_str(line.trim());
        }
    }
    if !name.is_empty() {
        recs.push((name, desc, seq));
    }
    recs
}

/// Collect residue characters from a sequence body line, following the C `inmap`
/// for the ungapped GenBank/EMBL formats: letters `A-Za-z` and `*` are residues;
/// digits, spaces, tabs, CR/LF are IGNORED; everything else is ILLEGAL (skipped
/// here, matching the effect on clean records). C `esl_sqio_ascii.c:inmap_genbank`
/// / `inmap_embl`.
fn push_residues(dst: &mut String, line: &str) {
    for c in line.chars() {
        if c.is_ascii_alphabetic() || c == '*' {
            dst.push(c);
        }
        // digits, whitespace, and anything else are ignored (IGNORED in C inmap).
    }
}

/// GenBank / DDBJ parser. C `esl_sqio_ascii.c:header_genbank` + sequence body up
/// to the `//` terminator.
fn parse_genbank(text: &str, path: &str) -> Result<Vec<(String, String, String)>, String> {
    let mut recs = Vec::new();
    let mut lines = text.lines().peekable();

    loop {
        // C header_genbank: "Find LOCUS line, allowing for ignoration of a file
        // header." while (strncmp(buf, "LOCUS   ", 8) != 0) loadbuf; EOF -> done.
        let mut locus: Option<&str> = None;
        for line in lines.by_ref() {
            if line.starts_with("LOCUS   ") {
                locus = Some(line);
                break;
            }
        }
        let locus = match locus {
            Some(l) => l,
            None => break, // eslEOF: no more records
        };

        // C: s = buf+12; esl_strtok(&s, " ", &tok) -> name.
        let mut name = String::new();
        if locus.len() > 12 {
            if let Some(tok) = strtok(&locus[12..], " ") {
                name = tok.to_string();
            }
        }
        if name.is_empty() {
            return Err(format!(
                "sequence file '{path}': failed to parse name on LOCUS line"
            ));
        }

        // C: loop loadbuf until "ORIGIN", parsing VERSION (acc) and DEFINITION (desc).
        let mut acc = String::new();
        let mut desc = String::new();
        let mut found_origin = false;
        for line in lines.by_ref() {
            // C: strncmp(buf, "VERSION   ", 10) == 0 -> acc = strtok(buf+12, " \t\n").
            if line.starts_with("VERSION   ") && line.len() > 12 {
                if let Some(tok) = strtok(&line[12..], " \t") {
                    acc = tok.to_string();
                }
            }
            // C: strncmp(buf, "DEFINITION ", 11) == 0 -> AppendDesc(strchop(buf+12)).
            if line.starts_with("DEFINITION ") && line.len() > 12 {
                append_desc(&mut desc, strchop(&line[12..]));
            }
            // C: while (strncmp(buf, "ORIGIN", 6) != 0)
            if line.starts_with("ORIGIN") {
                found_origin = true;
                break;
            }
        }
        if !found_origin {
            return Err(format!("sequence file '{path}': failed to find ORIGIN line"));
        }
        let _ = acc; // accession parsed faithfully; bins use (name, desc, seq).

        // Sequence body: lines until "//" (C inmap '/' = eslDSQ_EOD).
        let mut seq = String::new();
        let mut found_end = false;
        for line in lines.by_ref() {
            if line.starts_with("//") {
                found_end = true;
                break;
            }
            push_residues(&mut seq, line);
        }
        if !found_end {
            return Err(format!(
                "sequence file '{path}': did not find // terminator at end of seq record"
            ));
        }
        recs.push((name, desc, seq));
    }
    Ok(recs)
}

/// EMBL / UniProt parser. C `esl_sqio_ascii.c:header_embl` + sequence body up to
/// the `//` terminator.
fn parse_embl(text: &str, path: &str) -> Result<Vec<(String, String, String)>, String> {
    let mut recs = Vec::new();
    let mut lines = text.lines().peekable();

    loop {
        // C header_embl: skip blank lines, then require "ID   " line. EOF -> done.
        let mut idline: Option<&str> = None;
        for line in lines.by_ref() {
            if is_blank(line) {
                continue;
            }
            if line.starts_with("ID   ") {
                idline = Some(line);
                break;
            }
            // Non-blank, non-ID line where an ID line was expected.
            return Err(format!(
                "sequence file '{path}': failed to find ID line"
            ));
        }
        let idline = match idline {
            Some(l) => l,
            None => break, // eslEOF
        };

        // C: s = buf+5; esl_strtok(&s, " ;", &tok) -> name.
        let name = match strtok(&idline[5..], " ;") {
            Some(tok) if !tok.is_empty() => tok.to_string(),
            _ => {
                return Err(format!(
                    "sequence file '{path}': failed to parse name on ID line"
                ))
            }
        };

        // C: loop loadbuf until "SQ   ", parsing AC (primary accession) and DE (desc).
        let mut acc = String::new();
        let mut desc = String::new();
        let mut found_sq = false;
        for line in lines.by_ref() {
            // C: strncmp(buf, "AC   ", 5)==0 && sq->acc[0]=='\0' -> acc = strtok(buf+5, ";").
            if line.starts_with("AC   ") && acc.is_empty() && line.len() > 5 {
                if let Some(tok) = strtok(&line[5..], ";") {
                    acc = tok.to_string();
                }
            }
            // C: strncmp(buf, "DE   ", 5)==0 -> AppendDesc(strchop(buf+5)).
            if line.starts_with("DE   ") && line.len() > 5 {
                append_desc(&mut desc, strchop(&line[5..]));
            }
            // C: while (strncmp(buf, "SQ   ", 5) != 0)
            if line.starts_with("SQ   ") {
                found_sq = true;
                break;
            }
        }
        if !found_sq {
            return Err(format!("sequence file '{path}': failed to find SQ line"));
        }
        let _ = acc;

        // Sequence body: lines until "//".
        let mut seq = String::new();
        let mut found_end = false;
        for line in lines.by_ref() {
            if line.starts_with("//") {
                found_end = true;
                break;
            }
            push_residues(&mut seq, line);
        }
        if !found_end {
            return Err(format!(
                "sequence file '{path}': did not find // terminator at end of seq record"
            ));
        }
        recs.push((name, desc, seq));
    }
    Ok(recs)
}

/// C `esl_sq.c:esl_sq_AppendDesc`: concatenate description fragments, inserting a
/// single space between successive (non-empty) fragments.
fn append_desc(desc: &mut String, frag: &str) {
    if !desc.is_empty() {
        desc.push(' ');
    }
    desc.push_str(frag);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_format_case_insensitive() {
        assert_eq!(esl_sqio_encode_format("fasta"), Some(SqFormat::Fasta));
        assert_eq!(esl_sqio_encode_format("FASTA"), Some(SqFormat::Fasta));
        assert_eq!(esl_sqio_encode_format("GenBank"), Some(SqFormat::Genbank));
        assert_eq!(esl_sqio_encode_format("embl"), Some(SqFormat::Embl));
        assert_eq!(esl_sqio_encode_format("ddbj"), Some(SqFormat::Ddbj));
        assert_eq!(esl_sqio_encode_format("uniprot"), Some(SqFormat::Uniprot));
        assert_eq!(esl_sqio_encode_format("bogus"), None);
    }

    #[test]
    fn guess_by_suffix_and_firstline() {
        assert_eq!(guess_file_format("x.fa", ""), SqFormat::Fasta);
        assert_eq!(guess_file_format("x.fa.gz", ""), SqFormat::Fasta);
        assert_eq!(guess_file_format("x.gb", ""), SqFormat::Genbank);
        assert_eq!(guess_file_format("x.seq", ">foo"), SqFormat::Fasta);
        assert_eq!(guess_file_format("x.seq", "ID   ABC"), SqFormat::Embl);
        assert_eq!(guess_file_format("x.seq", "LOCUS   ABC"), SqFormat::Genbank);
        assert_eq!(guess_file_format("x.seq", "???"), SqFormat::Unknown);
    }

    #[test]
    fn strtok_semantics() {
        assert_eq!(strtok("  X06347; SV 1", " ;"), Some("X06347"));
        assert_eq!(strtok("NAME rest", " "), Some("NAME"));
        assert_eq!(strtok("   ", " "), None);
    }

    #[test]
    fn genbank_matches_fasta() {
        let fasta = ">seq1 a description\nACGTACGTAC\nGGGGTTTTAA\n";
        let gb = "LOCUS       seq1          20 bp    DNA\n\
                  DEFINITION  a description.\n\
                  VERSION     seq1.1\n\
                  ORIGIN\n\
                  \x20\x20\x20\x20\x20\x20\x201 acgtacgtac gggg\n\
                  \x20\x20\x20\x20\x20\x20\x2011 ttttaa\n\
                  //\n";
        let fr = parse_fasta(fasta);
        let gr = parse_genbank(gb, "t.gb").unwrap();
        assert_eq!(fr.len(), 1);
        assert_eq!(gr.len(), 1);
        assert_eq!(gr[0].0, "seq1");
        // residues equal ignoring case
        assert_eq!(gr[0].2.to_uppercase(), fr[0].2);
    }

    #[test]
    fn embl_parses_name_and_seq() {
        let embl = "ID   X06347; SV 1; linear; mRNA; STD; HUM; 20 BP.\n\
                    AC   X06347;\n\
                    DE   test description\n\
                    SQ   Sequence 20 BP;\n\
                    \x20\x20\x20\x20 acgtacgtac gggg ttttaa                20\n\
                    //\n";
        let r = parse_embl(embl, "t.embl").unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, "X06347");
        assert_eq!(r[0].1, "test description");
        assert_eq!(r[0].2.to_uppercase(), "ACGTACGTACGGGGTTTTAA");
    }
}
