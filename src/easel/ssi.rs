//! Sequence/subsequence indices (SSI): fast lookup in large sequence files by keyword.
//!
//! Faithful 1:1 port of Easel's `esl_ssi.c` / `esl_ssi.h`.
//!
//! The on-disk `.ssi` / `.i1i` byte layout is reproduced exactly, byte-for-byte
//! identical to the C implementation. All multi-byte integers and file offsets
//! are stored in network (big-endian) byte order, exactly as C's
//! `esl_hton*`/`esl_fwrite_*` helpers write them.
//!
//! Porting notes are given as `esl_ssi.c:<function>:<line>` next to each
//! transcription.

use crate::easel::error::{InfernalError, Result};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

// esl_ssi.c:19  static uint32_t v30magic = 0xd3d3c9b3; /* SSI 3.0: "ssi3" + 0x80808080 */
const V30MAGIC: u32 = 0xd3d3_c9b3;
// esl_ssi.c:20  static uint32_t v30swap  = 0xb3c9d3d3; /* byteswapped */
const V30SWAP: u32 = 0xb3c9_d3d3;

// esl_ssi.h:58  #define eslSSI_FASTSUBSEQ (1<<0)
pub const ESL_SSI_FASTSUBSEQ: u32 = 1 << 0;

// esl_ssi.h:19  #define eslSSI_MAXFILES 32767  (2^15-1)
const ESL_SSI_MAXFILES: u32 = 32767;
// esl_ssi.h:20  #define eslSSI_MAXKEYS  (1ULL<<63)-1
const ESL_SSI_MAXKEYS: u64 = (1u64 << 63) - 1;

/// On this (and all modern 64-bit) platform, `sizeof(off_t) == 8`. The golden C
/// binaries write `offsz = 8` and store every offset as an 8-byte big-endian
/// integer. We reproduce that verbatim.
///
/// esl_ssi.c uses `sizeof(off_t)` throughout (e.g. header write at :1107, record
/// sizes at :1056-1064). We fix it at 8 to match the 64-bit golden output.
const OFFSET_SIZE: u32 = 8;

/* ****************************************************************
 * 1. Using (reading) an SSI index.  (esl_ssi.c section 1)
 * ****************************************************************/

/// ESL_SSI — an open, read-only SSI index. (esl_ssi.h:31 `typedef struct { ... } ESL_SSI;`)
pub struct EslSsi {
    fp: BufReader<File>, // esl_ssi.h:32  FILE *fp
    pub flags: u32,      // esl_ssi.h:33  optional behavior flags
    pub offsz: u32,      // esl_ssi.h:34  sizeof(off_t)'s in the SSI file
    pub nfiles: u16,     // esl_ssi.h:35  number of files
    pub nprimary: u64,   // esl_ssi.h:36  number of primary keys
    pub nsecondary: u64, // esl_ssi.h:37  number of secondary keys
    pub flen: u32,       // esl_ssi.h:38  length of filenames (inc '\0')
    pub plen: u32,       // esl_ssi.h:39  length of primary keys (inc '\0')
    pub slen: u32,       // esl_ssi.h:40  length of secondary keys (inc '\0')
    pub frecsize: u32,   // esl_ssi.h:41  # bytes in a file record
    pub precsize: u32,   // esl_ssi.h:42  # bytes in a primary key record
    pub srecsize: u32,   // esl_ssi.h:43  # bytes in a secondary key record
    pub foffset: u64,    // esl_ssi.h:44  disk offset, start of file records
    pub poffset: u64,    // esl_ssi.h:45  disk offset, start of pri key recs
    pub soffset: u64,    // esl_ssi.h:46  disk offset, start of sec key recs

    // File information (esl_ssi.h:49-54)
    pub filename: Vec<String>,
    pub fileformat: Vec<u32>,
    pub fileflags: Vec<u32>,
    pub bpl: Vec<u32>,
    pub rpl: Vec<u32>,
}

/// Result of a successful key lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsiEntry {
    pub fh: u16,     // handle on file that key is in
    pub roff: u64,   // offset of start of record
    pub doff: u64,   // data offset (may be 0 if unset)
    pub len: i64,    // length of data record (may be 0 if unset)
}

impl EslSsi {
    /// esl_ssi_Open() — esl_ssi.c:50
    ///
    /// Open the SSI index file `filename`.
    pub fn open<P: AsRef<Path>>(filename: P) -> Result<EslSsi> {
        // esl_ssi.c:72  if ((ssi->fp = fopen(filename, "rb")) == NULL) -> eslENOTFOUND
        let file = File::open(filename).map_err(|_| InfernalError::NotFound)?;
        let mut fp = BufReader::new(file);

        // esl_ssi.c:78-79  read magic; must be v30magic or v30swap.
        let magic = read_u32(&mut fp)?;
        if magic != V30MAGIC && magic != V30SWAP {
            return Err(InfernalError::Format);
        }
        // Note: SSI is *always* stored in network (big-endian) order. v30swap
        // would only be seen if a big-endian machine wrote the file and a
        // little-endian machine reads it via host-order fread; our reader always
        // reads big-endian, so we accept the magic and proceed. (esl_ssi.c:80-81)
        let flags = read_u32(&mut fp)?;
        let offsz = read_u32(&mut fp)?;

        // esl_ssi.c:83-85  offsz must be 4 or 8, and <= sizeof(off_t).
        if offsz != 4 && offsz != 8 {
            return Err(InfernalError::Range);
        }
        if offsz > OFFSET_SIZE {
            return Err(InfernalError::Range);
        }

        // esl_ssi.c:88-101  header data.
        let nfiles = read_u16(&mut fp)?;
        let nprimary = read_u64(&mut fp)?;
        let nsecondary = read_u64(&mut fp)?;
        let flen = read_u32(&mut fp)?;
        let plen = read_u32(&mut fp)?;
        let slen = read_u32(&mut fp)?;
        let frecsize = read_u32(&mut fp)?;
        let precsize = read_u32(&mut fp)?;
        let srecsize = read_u32(&mut fp)?;
        let foffset = read_offset(&mut fp, offsz)?;
        let poffset = read_offset(&mut fp, offsz)?;
        let soffset = read_offset(&mut fp, offsz)?;

        // esl_ssi.c:110  if (ssi->nfiles == 0) goto ERROR; (eslEFORMAT)
        if nfiles == 0 {
            return Err(InfernalError::Format);
        }

        let mut filename = Vec::with_capacity(nfiles as usize);
        let mut fileformat = Vec::with_capacity(nfiles as usize);
        let mut fileflags = Vec::with_capacity(nfiles as usize);
        let mut bpl = Vec::with_capacity(nfiles as usize);
        let mut rpl = Vec::with_capacity(nfiles as usize);

        // esl_ssi.c:120-134  read the file records, positioning explicitly.
        let mut namebuf = vec![0u8; flen as usize];
        for i in 0..nfiles as u64 {
            fp.seek(SeekFrom::Start(foffset + i * frecsize as u64))
                .map_err(|_| InfernalError::Format)?;
            fp.read_exact(&mut namebuf).map_err(|_| InfernalError::Format)?;
            filename.push(cstr_to_string(&namebuf));
            fileformat.push(read_u32(&mut fp)?);
            fileflags.push(read_u32(&mut fp)?);
            bpl.push(read_u32(&mut fp)?);
            rpl.push(read_u32(&mut fp)?);
        }

        Ok(EslSsi {
            fp,
            flags,
            offsz,
            nfiles,
            nprimary,
            nsecondary,
            flen,
            plen,
            slen,
            frecsize,
            precsize,
            srecsize,
            foffset,
            poffset,
            soffset,
            filename,
            fileformat,
            fileflags,
            bpl,
            rpl,
        })
    }

    /// esl_ssi_FindName() — esl_ssi.c:171
    ///
    /// Look up primary or secondary key `key`.
    pub fn find_name(&mut self, key: &str) -> Result<SsiEntry> {
        // esl_ssi.c:181  search the primary keys.
        let status = binary_search(
            &mut self.fp,
            key,
            self.plen,
            self.poffset,
            self.precsize,
            self.nprimary,
        );

        match status {
            Ok(()) => {
                // esl_ssi.c:186-190  found as primary key; read the data.
                let fh = read_u16(&mut self.fp)?;
                let roff = read_offset(&mut self.fp, self.offsz)?;
                let doff = read_offset(&mut self.fp, self.offsz)?;
                let len = read_i64(&mut self.fp)?;
                Ok(SsiEntry { fh, roff, doff, len })
            }
            Err(InfernalError::NotFound) => {
                // esl_ssi.c:192-202  try the secondary keys.
                if self.nsecondary > 0 {
                    binary_search(
                        &mut self.fp,
                        key,
                        self.slen,
                        self.soffset,
                        self.srecsize,
                        self.nsecondary,
                    )?;
                    // esl_ssi.c:198-201  flip to primary key, then look that up.
                    let mut pkeybuf = vec![0u8; self.plen as usize];
                    self.fp
                        .read_exact(&mut pkeybuf)
                        .map_err(|_| InfernalError::Format)?;
                    let pkey = cstr_to_string(&pkeybuf);
                    self.find_name(&pkey)
                } else {
                    Err(InfernalError::NotFound)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// esl_ssi_FindNumber() — esl_ssi.c:252
    ///
    /// Look up the n'th primary key (0..nprimary-1). Returns (entry, primary key name).
    pub fn find_number(&mut self, nkey: i64) -> Result<(SsiEntry, String)> {
        // esl_ssi.c:261  if (nkey >= ssi->nprimary) -> eslENOTFOUND
        if nkey < 0 || nkey as u64 >= self.nprimary {
            return Err(InfernalError::NotFound);
        }
        // esl_ssi.c:265-270  seek to record and read.
        self.fp
            .seek(SeekFrom::Start(
                self.poffset + self.precsize as u64 * nkey as u64,
            ))
            .map_err(|_| InfernalError::Format)?;
        let mut pkeybuf = vec![0u8; self.plen as usize];
        self.fp
            .read_exact(&mut pkeybuf)
            .map_err(|_| InfernalError::Format)?;
        let pkey = cstr_to_string(&pkeybuf);
        let fh = read_u16(&mut self.fp)?;
        let roff = read_offset(&mut self.fp, self.offsz)?;
        let doff = read_offset(&mut self.fp, self.offsz)?;
        // esl_ssi.c:270 reads with esl_fread_u64 (unsigned); reinterpret as i64.
        let len = read_u64(&mut self.fp)? as i64;
        Ok((SsiEntry { fh, roff, doff, len }, pkey))
    }

    /// esl_ssi_FindSubseq() — esl_ssi.c:406
    ///
    /// Returns (entry, actual_start). See C docs for the four outcomes.
    pub fn find_subseq(&mut self, key: &str, requested_start: i64) -> Result<(SsiEntry, i64)> {
        // esl_ssi.c:415  look up the key by name.
        let mut e = self.find_name(key)?;
        // esl_ssi.c:416  range check.
        if requested_start < 0 || requested_start > e.len {
            return Err(InfernalError::Range);
        }

        // esl_ssi.c:420  no data offset, or file can't do fast subseq -> case 4/3.
        if e.doff == 0 || (self.fileflags[e.fh as usize] & ESL_SSI_FASTSUBSEQ) == 0 {
            return Ok((e, 1));
        }

        // esl_ssi.c:429-432  set up tmp variables.
        let r = self.rpl[e.fh as usize] as i64; // residues per line
        let b = self.bpl[e.fh as usize] as i64; // bytes per line
        let i = requested_start; // 1..L
        // esl_ssi.c:433  if (r == 0 || b == 0) -> eslEINVAL
        if r == 0 || b == 0 {
            return Err(InfernalError::Inval);
        }
        let l = (i - 1) / r; // data line # (0..) that the residue is on

        let actual_start;
        if b == r + 1 {
            // esl_ssi.c:438-442  outcome #1: single-residue resolution.
            e.doff += (l * b + (i - 1) % r) as u64;
            actual_start = requested_start;
        } else {
            // esl_ssi.c:447-451  line resolution.
            e.doff += (l * b) as u64;
            actual_start = 1 + l * r;
        }
        Ok((e, actual_start))
    }

    /// esl_ssi_FileInfo() — esl_ssi.c:485
    pub fn file_info(&self, fh: u16) -> Result<(&str, u32)> {
        if fh >= self.nfiles {
            return Err(InfernalError::Inval);
        }
        Ok((&self.filename[fh as usize], self.fileformat[fh as usize]))
    }
}

/// binary_search() — esl_ssi.c:557
///
/// Find `key` in a sorted key section. On success leaves `fp` positioned to read
/// the rest of the record's data (immediately after the fixed-width key field).
fn binary_search(
    fp: &mut BufReader<File>,
    key: &str,
    klen: u32,
    base: u64,
    recsize: u32,
    maxidx: u64,
) -> Result<()> {
    // esl_ssi.c:566  empty index special case.
    if maxidx == 0 {
        return Err(InfernalError::NotFound);
    }

    let key_bytes = key.as_bytes();
    let mut name = vec![0u8; klen as usize];

    let mut left: u64 = 0;
    let mut right: u64 = maxidx - 1;
    loop {
        // esl_ssi.c:573  mid = (left+right)/2
        let mid = (left + right) / 2;
        fp.seek(SeekFrom::Start(base + recsize as u64 * mid))
            .map_err(|_| InfernalError::Format)?;
        fp.read_exact(&mut name).map_err(|_| InfernalError::Format)?;

        // esl_ssi.c:580  cmp = strcmp(name, key)
        let cmp = strcmp(&name, key_bytes);
        if cmp == 0 {
            // esl_ssi.c:581  found it; fp is positioned to read the record.
            return Ok(());
        } else if left >= right {
            // esl_ssi.c:582  no such key
            return Err(InfernalError::NotFound);
        } else if cmp < 0 {
            // esl_ssi.c:583  it's right of mid
            left = mid + 1;
        } else {
            // esl_ssi.c:584-586  cmp > 0: it's left of mid
            if mid == 0 {
                return Err(InfernalError::NotFound);
            }
            right = mid - 1;
        }
    }
}

/* ****************************************************************
 * 2. Creating (writing) new SSI files.  (esl_ssi.c section 2)
 * ****************************************************************/

// esl_ssi.h:64  ESL_PKEY — primary key data.
#[derive(Clone)]
struct EslPkey {
    key: String,
    fnum: u16,
    r_off: u64,
    d_off: u64,
    len: i64,
}

// esl_ssi.h:72  ESL_SKEY — secondary key data.
#[derive(Clone)]
struct EslSkey {
    key: String,
    pkey: String,
}

/// ESL_NEWSSI — a new SSI index under construction. (esl_ssi.h:77)
///
/// This port implements the in-memory ("internal") path faithfully, which is
/// what produces byte-identical output for the indices Infernal writes (a handful
/// of models). The external-sort path (esl_ssi.c switches to it above
/// `eslSSI_MAXRAM` = 2048 MB) is not needed here and is omitted; `write()` uses
/// an in-memory sort exactly matching the internal branches of `esl_newssi_Write`.
pub struct EslNewSsi {
    ssifile: String, // esl_ssi.h:78  name of the SSI file we're creating

    // File section (esl_ssi.h:83-88)
    filenames: Vec<String>,
    fileformat: Vec<u32>,
    bpl: Vec<u32>,
    rpl: Vec<u32>,
    flen: u32,   // length of longest filename, inc '\0'
    nfiles: u16, // up to 2^15-1 files

    // Primary keys (esl_ssi.h:90-92)
    pkeys: Vec<EslPkey>,
    plen: u32,     // length of longest pkey, inc '\0'
    nprimary: u64, // up to 2^63-1

    // Secondary keys (esl_ssi.h:96-98)
    skeys: Vec<EslSkey>,
    slen: u32,
    nsecondary: u64,

    written: bool,
}

impl EslNewSsi {
    /// esl_newssi_Open() — esl_ssi.c:626
    ///
    /// Create a new `EslNewSsi` to build an SSI index at `ssifile`.
    pub fn open<P: AsRef<Path>>(ssifile: P, allow_overwrite: bool) -> Result<EslNewSsi> {
        let ssifile = ssifile.as_ref();
        let ssifile_str = ssifile.to_string_lossy().into_owned();
        let ptmpfile = format!("{}.1", ssifile_str);
        let stmpfile = format!("{}.2", ssifile_str);

        // esl_ssi.c:662-668  refuse to overwrite unless allowed.
        if !allow_overwrite
            && (Path::new(&ssifile_str).exists()
                || Path::new(&ptmpfile).exists()
                || Path::new(&stmpfile).exists())
        {
            return Err(InfernalError::Overwrite);
        }

        // esl_ssi.c:670  make sure we can create the file (fopen "w").
        // We create/truncate now to mirror C semantics (and fail early), then
        // reopen for writing in write(). ENOTFOUND on failure.
        File::create(&ssifile_str).map_err(|_| InfernalError::NotFound)?;

        Ok(EslNewSsi {
            ssifile: ssifile_str,
            filenames: Vec::new(),
            fileformat: Vec::new(),
            bpl: Vec::new(),
            rpl: Vec::new(),
            flen: 0,
            nfiles: 0,
            pkeys: Vec::new(),
            plen: 0,
            nprimary: 0,
            skeys: Vec::new(),
            slen: 0,
            nsecondary: 0,
            written: false,
        })
    }

    /// esl_newssi_AddFile() — esl_ssi.c:717
    ///
    /// Register file `filename` (with format code `fmt`); returns its handle.
    pub fn add_file(&mut self, filename: &str, fmt: u32) -> Result<u16> {
        // esl_ssi.c:725  cap on number of files.
        if self.nfiles as u32 >= ESL_SSI_MAXFILES {
            return Err(InfernalError::Range);
        }
        // esl_ssi.c:727-728  flen is sized from the FULL filename string:
        //     n = strlen(filename);
        //     if ((n+1) > ns->flen) ns->flen = n+1;
        // NOTE: flen is computed from the full path passed in, NOT from the tail
        // that actually gets stored (esl_FileTail strips the directory below).
        // This is exactly C's behavior — the field width therefore depends on the
        // full path length, so byte-parity requires invoking with the same path.
        let n = filename.len() + 1;
        if n as u32 > self.flen {
            self.flen = n as u32;
        }
        // esl_ssi.c:730  esl_FileTail(filename, FALSE, ...): store the basename.
        let tail = file_tail(filename);
        self.filenames.push(tail);
        self.fileformat.push(fmt);
        self.bpl.push(0);
        self.rpl.push(0);
        let fh = self.nfiles;
        self.nfiles += 1;
        Ok(fh)
    }

    /// esl_newssi_SetSubseq() — esl_ssi.c:777
    pub fn set_subseq(&mut self, fh: u16, bpl: u32, rpl: u32) -> Result<()> {
        // esl_ssi.c:782-783  validate.
        if fh >= self.nfiles {
            return Err(InfernalError::Inval);
        }
        if bpl == 0 || rpl == 0 {
            return Err(InfernalError::Inval);
        }
        self.bpl[fh as usize] = bpl;
        self.rpl[fh as usize] = rpl;
        Ok(())
    }

    /// esl_newssi_AddKey() — esl_ssi.c:842
    ///
    /// Register primary key `key` in file `fh`, at record offset `r_off`, data
    /// offset `d_off`, data length `L`.
    pub fn add_key(&mut self, key: &str, fh: u16, r_off: u64, d_off: u64, l: i64) -> Result<()> {
        // esl_ssi.c:850-851  validation.
        if fh as u32 >= ESL_SSI_MAXFILES {
            return Err(InfernalError::Inval);
        }
        if self.nprimary >= ESL_SSI_MAXKEYS {
            return Err(InfernalError::Range);
        }
        // esl_ssi.c:861-862  plen = max(plen, strlen(key)+1)
        let n = key.len() + 1;
        if n as u32 > self.plen {
            self.plen = n as u32;
        }
        // esl_ssi.c:883-888  internal mode: store in memory.
        self.pkeys.push(EslPkey {
            key: key.to_string(),
            fnum: fh,
            r_off,
            d_off,
            len: l,
        });
        self.nprimary += 1;
        Ok(())
    }

    /// esl_newssi_AddAlias() — esl_ssi.c:924
    ///
    /// Register secondary key `alias`, mapping to already-registered primary `key`.
    pub fn add_alias(&mut self, alias: &str, key: &str) -> Result<()> {
        // esl_ssi.c:931  cap on secondary keys.
        if self.nsecondary >= ESL_SSI_MAXKEYS {
            return Err(InfernalError::Range);
        }
        // esl_ssi.c:940-941  slen = max(slen, strlen(alias)+1)
        let n = alias.len() + 1;
        if n as u32 > self.slen {
            self.slen = n as u32;
        }
        // esl_ssi.c:951-953  internal mode: store in memory.
        self.skeys.push(EslSkey {
            key: alias.to_string(),
            pkey: key.to_string(),
        });
        self.nsecondary += 1;
        Ok(())
    }

    /// esl_newssi_Write() — esl_ssi.c:1006
    ///
    /// Sort keys, write the complete index in SSI format, and close the file.
    pub fn write(&mut self) -> Result<()> {
        // esl_ssi.c:1027-1030  guards.
        if self.nsecondary > 0 && self.slen == 0 {
            return Err(InfernalError::Inval);
        }
        if self.written {
            return Err(InfernalError::Inval);
        }

        // esl_ssi.c:1056-1058  record sizes.
        let frecsize: u32 = 4 * 4 + self.flen;
        let precsize: u32 = 2 * OFFSET_SIZE + 2 + 8 + self.plen; // 2 off_t + u16 + u64 + plen
        let srecsize: u32 = self.slen + self.plen;
        let header_flags: u32 = 0; // esl_ssi.c:1059

        // esl_ssi.c:1064-1066  section offsets.
        // foffset = 9*u32 + 2*u64 + u16 + 3*off_t
        let foffset: u64 = (9 * 4 + 2 * 8 + 2 + 3 * OFFSET_SIZE) as u64;
        let poffset: u64 = foffset + frecsize as u64 * self.nfiles as u64;
        let soffset: u64 = poffset + precsize as u64 * self.nprimary;

        // esl_ssi.c:1099-1100  sort the keys (qsort by strcmp on key bytes).
        self.pkeys
            .sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
        self.skeys
            .sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));

        // Build the output image in memory, then write it out.
        let mut out: Vec<u8> = Vec::new();

        // esl_ssi.c:1105-1119  write the header.
        write_u32(&mut out, V30MAGIC);
        write_u32(&mut out, header_flags);
        write_u32(&mut out, OFFSET_SIZE); // sizeof(off_t)
        write_u16(&mut out, self.nfiles);
        write_u64(&mut out, self.nprimary);
        write_u64(&mut out, self.nsecondary);
        write_u32(&mut out, self.flen);
        write_u32(&mut out, self.plen);
        write_u32(&mut out, self.slen);
        write_u32(&mut out, frecsize);
        write_u32(&mut out, precsize);
        write_u32(&mut out, srecsize);
        write_offset(&mut out, foffset);
        write_offset(&mut out, poffset);
        write_offset(&mut out, soffset);

        // esl_ssi.c:1124-1136  write the file section.
        for i in 0..self.nfiles as usize {
            // esl_ssi.c:1126-1127  fast-subseq flag if bpl & rpl set.
            let mut file_flags: u32 = 0;
            if self.bpl[i] > 0 && self.rpl[i] > 0 {
                file_flags |= ESL_SSI_FASTSUBSEQ;
            }
            write_fixed(&mut out, self.filenames[i].as_bytes(), self.flen);
            write_u32(&mut out, self.fileformat[i]);
            write_u32(&mut out, file_flags);
            write_u32(&mut out, self.bpl[i]);
            write_u32(&mut out, self.rpl[i]);
        }

        // esl_ssi.c:1160-1172  write the primary key section (internal branch).
        let mut prev = String::new();
        for (i, pk) in self.pkeys.iter().enumerate() {
            // esl_ssi.c:1163  duplicate check (keys are sorted).
            if i > 0 && prev == pk.key {
                return self.fail(InfernalError::Dup);
            }
            prev = pk.key.clone();
            write_fixed(&mut out, pk.key.as_bytes(), self.plen);
            write_u16(&mut out, pk.fnum);
            write_offset(&mut out, pk.r_off);
            write_offset(&mut out, pk.d_off);
            write_i64(&mut out, pk.len);
        }

        // esl_ssi.c:1197-1207  write the secondary key section (internal branch).
        let mut prev_s = String::new();
        for (i, sk) in self.skeys.iter().enumerate() {
            // esl_ssi.c:1200  duplicate check.
            if i > 0 && prev_s == sk.key {
                return self.fail(InfernalError::Dup);
            }
            prev_s = sk.key.clone();
            write_fixed(&mut out, sk.key.as_bytes(), self.slen);
            write_fixed(&mut out, sk.pkey.as_bytes(), self.plen);
        }

        // esl_ssi.c:1210  write image and close.
        let mut f = File::create(&self.ssifile).map_err(|_| InfernalError::Write)?;
        f.write_all(&out).map_err(|_| InfernalError::Write)?;
        f.flush().map_err(|_| InfernalError::Write)?;
        self.written = true;
        Ok(())
    }

    /// esl_ssi.c:1221  ERROR path: delete the failed <ssifile>.
    fn fail(&self, e: InfernalError) -> Result<()> {
        let _ = std::fs::remove_file(&self.ssifile);
        Err(e)
    }
}

/* ****************************************************************
 * 3. Portable binary i/o.  (esl_ssi.c section 3)
 *
 * SSI stores every integer/offset in network (big-endian) order. On disk the
 * bytes are identical regardless of host endianness. We therefore read/write
 * big-endian directly (Rust `to_be_bytes`/`from_be_bytes`), which is exactly
 * what C's esl_hton / esl_ntoh helpers + fwrite/fread produce.
 * ****************************************************************/

fn read_u16(r: &mut impl Read) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|_| InfernalError::Fail)?;
    Ok(u16::from_be_bytes(b))
}
fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|_| InfernalError::Fail)?;
    Ok(u32::from_be_bytes(b))
}
fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|_| InfernalError::Fail)?;
    Ok(u64::from_be_bytes(b))
}
fn read_i64(r: &mut impl Read) -> Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|_| InfernalError::Fail)?;
    Ok(i64::from_be_bytes(b))
}

/// esl_fread_offset() — esl_ssi.c:1722. `sz` is 4 or 8.
fn read_offset(r: &mut impl Read, sz: u32) -> Result<u64> {
    match sz {
        8 => read_u64(r),
        4 => Ok(read_u32(r)? as u64),
        _ => Err(InfernalError::Inval),
    }
}

fn write_u16(out: &mut Vec<u8>, n: u16) {
    out.extend_from_slice(&n.to_be_bytes());
}
fn write_u32(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_be_bytes());
}
fn write_u64(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_be_bytes());
}
fn write_i64(out: &mut Vec<u8>, n: i64) {
    out.extend_from_slice(&n.to_be_bytes());
}
/// esl_fwrite_offset() — esl_ssi.c:1761. off_t is 8 bytes here.
fn write_offset(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_be_bytes());
}

/// Write `data` into a fixed-width field of `width` bytes, NUL-padded — exactly
/// C's `strncpy(field, s, width)` followed by `fwrite(field, 1, width, fp)`.
/// (esl_ssi.c:1128-1130, 1164-1166, 1201-1205)
fn write_fixed(out: &mut Vec<u8>, data: &[u8], width: u32) {
    let width = width as usize;
    let n = data.len().min(width);
    out.extend_from_slice(&data[..n]);
    // pad remaining bytes with NUL (strncpy behavior)
    out.extend(std::iter::repeat(0u8).take(width - n));
}

/// Interpret a fixed-width NUL-terminated (or -padded) byte field as a String.
fn cstr_to_string(buf: &[u8]) -> String {
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// strcmp() semantics on NUL-terminated C strings living in `name` (fixed-width,
/// NUL-padded) versus `key` (already the exact bytes, no NUL). Compares byte by
/// byte as unsigned char up to the first NUL, matching C `strcmp`.
fn strcmp(name: &[u8], key: &[u8]) -> i32 {
    let mut i = 0;
    loop {
        let a = if i < name.len() { name[i] } else { 0 };
        // name is NUL-terminated within its fixed field; stop at NUL.
        let a = if a == 0 { 0u8 } else { a };
        let b = if i < key.len() { key[i] } else { 0 };
        if a != b {
            return a as i32 - b as i32;
        }
        if a == 0 {
            return 0;
        }
        i += 1;
    }
}

/// esl_FileTail(path, nosuffix=FALSE, ...) — easel.c:1781. Strips the directory
/// prefix (keeps the suffix). Uses '/' as the directory separator.
fn file_tail(path: &str) -> String {
    match path.rfind('/') {
        Some(idx) => path[idx + 1..].to_string(),
        None => path.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> String {
        let dir = std::env::temp_dir();
        dir.join(format!("infernox_ssi_test_{}_{}", std::process::id(), name))
            .to_string_lossy()
            .into_owned()
    }

    /// Build an index from scratch, then read it back with our reader.
    #[test]
    fn test_writer_reader_roundtrip() {
        let ssipath = scratch("rt.ssi");
        let _ = std::fs::remove_file(&ssipath);

        {
            let mut ns = EslNewSsi::open(&ssipath, true).unwrap();
            let fh = ns.add_file("some/dir/multi.cm", 0).unwrap();
            assert_eq!(fh, 0);
            // Add in unsorted order; write() must sort.
            ns.add_key("tRNA", fh, 0, 0, 0).unwrap();
            ns.add_key("Vault", fh, 49711, 0, 0).unwrap();
            ns.add_key("tRNA-Sec", fh, 321100, 0, 0).unwrap();
            ns.add_key("rnaseP-eubact", fh, 115956, 0, 0).unwrap();
            ns.add_alias("RF00005", "tRNA").unwrap();
            ns.add_alias("RF00006", "Vault").unwrap();
            ns.add_alias("RF01852", "tRNA-Sec").unwrap();
            ns.write().unwrap();
        }

        let mut ssi = EslSsi::open(&ssipath).unwrap();
        assert_eq!(ssi.nfiles, 1);
        assert_eq!(ssi.nprimary, 4);
        assert_eq!(ssi.nsecondary, 3);
        assert_eq!(ssi.filename[0], "multi.cm"); // basename stored

        // Primary key lookups.
        let e = ssi.find_name("tRNA").unwrap();
        assert_eq!(e.roff, 0);
        let e = ssi.find_name("Vault").unwrap();
        assert_eq!(e.roff, 49711);
        let e = ssi.find_name("rnaseP-eubact").unwrap();
        assert_eq!(e.roff, 115956);
        let e = ssi.find_name("tRNA-Sec").unwrap();
        assert_eq!(e.roff, 321100);

        // Secondary key (alias) lookup flips to the primary.
        let e = ssi.find_name("RF00005").unwrap();
        assert_eq!(e.roff, 0); // -> tRNA
        let e = ssi.find_name("RF00006").unwrap();
        assert_eq!(e.roff, 49711); // -> Vault
        let e = ssi.find_name("RF01852").unwrap();
        assert_eq!(e.roff, 321100); // -> tRNA-Sec

        // Missing key.
        assert_eq!(ssi.find_name("nope"), Err(InfernalError::NotFound));

        // find_number returns keys in sorted order.
        let (_, k0) = ssi.find_number(0).unwrap();
        assert_eq!(k0, "Vault");
        let (_, k3) = ssi.find_number(3).unwrap();
        assert_eq!(k3, "tRNA-Sec");

        let _ = std::fs::remove_file(&ssipath);
    }

    /// Byte-identical reproduction of the C golden `.ssi` written by `cmfetch
    /// --index`, if that golden is present. The golden was produced from a
    /// 4-model concatenated CM (tRNA, Vault, rnaseP-eubact, tRNA-Sec) named
    /// "multi.cm", with accessions RF00005/RF00006/RF01852 as aliases.
    #[test]
    fn test_byte_identical_to_c_golden() {
        let golden = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/multi.cm.ssi"
        );
        if !Path::new(golden).exists() {
            eprintln!("golden {} not present; skipping byte-identical test", golden);
            return;
        }
        let golden_bytes = std::fs::read(golden).unwrap();

        let ssipath = scratch("golden.ssi");
        let _ = std::fs::remove_file(&ssipath);
        {
            let mut ns = EslNewSsi::open(&ssipath, true).unwrap();
            let fh = ns.add_file("multi.cm", 0).unwrap();
            // Feed the exact tuples cmfetch registered (order irrelevant; write sorts).
            ns.add_key("tRNA", fh, 0, 0, 0).unwrap();
            ns.add_key("Vault", fh, 49711, 0, 0).unwrap();
            ns.add_key("rnaseP-eubact", fh, 115956, 0, 0).unwrap();
            ns.add_key("tRNA-Sec", fh, 321100, 0, 0).unwrap();
            ns.add_alias("RF00005", "tRNA").unwrap();
            ns.add_alias("RF00006", "Vault").unwrap();
            ns.add_alias("RF01852", "tRNA-Sec").unwrap();
            ns.write().unwrap();
        }
        let ours = std::fs::read(&ssipath).unwrap();
        assert_eq!(
            ours, golden_bytes,
            "Rust SSI output is not byte-identical to the C golden"
        );

        // Also confirm our reader parses the C golden itself.
        let mut ssi = EslSsi::open(golden).unwrap();
        let e = ssi.find_name("tRNA-Sec").unwrap();
        assert_eq!(e.roff, 321100);
        let e = ssi.find_name("RF00006").unwrap();
        assert_eq!(e.roff, 49711);

        let _ = std::fs::remove_file(&ssipath);
    }

    #[test]
    fn test_duplicate_primary_key_detected() {
        let ssipath = scratch("dup.ssi");
        let _ = std::fs::remove_file(&ssipath);
        let mut ns = EslNewSsi::open(&ssipath, true).unwrap();
        let fh = ns.add_file("x.cm", 0).unwrap();
        ns.add_key("A", fh, 1, 0, 0).unwrap();
        ns.add_key("A", fh, 2, 0, 0).unwrap();
        assert_eq!(ns.write(), Err(InfernalError::Dup));
        let _ = std::fs::remove_file(&ssipath);
    }
}
