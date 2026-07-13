//! Faithful `esl_getopts`-style command-line parser for the cmsearch / cmscan
//! binaries.
//!
//! C Infernal parses its command line with Easel's `esl_opt_ProcessCmdline`
//! (easel/esl_getopts.c). This module reproduces the parts of that behaviour that
//! affect which options / positionals a run sees — which in turn is a *correctness*
//! matter, because silently swallowing an unknown flag (or leaking a value-taking
//! option's argument into the positional list) produces the wrong search.
//!
//! Reproduced esl_getopts semantics (verified against the 1.1.5 C binary):
//!   * Options must precede positional arguments. As soon as a token that does not
//!     look like an option is seen, *all* remaining tokens are positionals
//!     (`esl_opt_ProcessCmdline`; C errors "Incorrect number of command line
//!     arguments." when e.g. `-g cm fa -E 50` is given).
//!   * Long options (`--name`) accept unambiguous prefix abbreviation
//!     (`--topon` == `--toponly`); an ambiguous prefix or an unknown name is an
//!     error ("No such option ...").
//!   * `--name=value` and `--name value` are both accepted for value-taking
//!     options; single-char options take their value as the next token (`-E 50`).
//!   * A boolean (no-argument) option given an `=value` is an error.
//!   * A value-taking option with no available argument is an error.
//!
//! Errors are reported by returning `Err(String)`; the caller prints them and
//! exits non-zero (C exits 1). We do not reproduce C's exact error wording (it is
//! not part of the `--tblout` parity surface), only the accept/reject decision.

use std::collections::HashMap;

/// The argument class of an option, mirroring the eslARG_* types that matter for
/// parsing (whether a value token is consumed).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArgKind {
    /// eslARG_NONE — a boolean flag, takes no argument.
    None,
    /// eslARG_INT / eslARG_REAL / eslARG_STRING / eslARG_OUTFILE / eslARG_INFILE —
    /// consumes exactly one value token.
    Value,
}

/// One entry of the option table (subset of ESL_OPTIONS that we need: name + arity).
pub struct OptSpec {
    pub name: &'static str,
    pub kind: ArgKind,
}

/// Result of parsing a command line: the set options (name -> optional value) and
/// the positional arguments (in order).
pub struct Parsed {
    values: HashMap<&'static str, Option<String>>,
    pub positionals: Vec<String>,
}

impl Parsed {
    /// True if option `name` (canonical long/short name incl. leading dashes) was set.
    pub fn is_set(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }
    /// The raw string value of a value-taking option, if it was set.
    pub fn get_str(&self, name: &str) -> Option<&str> {
        self.values.get(name).and_then(|v| v.as_deref())
    }
    /// Parse a value-taking option as f64, erroring like C's esl_getopts
    /// verify_type_and_range (easel/esl_getopts.c:1651) on a malformed number:
    /// `Option <name> takes real-valued arg; got <val> on cmdline`.
    pub fn get_f64(&self, name: &str) -> Result<Option<f64>, String> {
        match self.get_str(name) {
            None => Ok(None),
            Some(s) => s.parse::<f64>().map(Some).map_err(|_| {
                format!(
                    "Option {} takes real-valued arg; got {} on cmdline",
                    esl_field(name),
                    esl_field(s)
                )
            }),
        }
    }
    /// Parse a value-taking option as f32.
    pub fn get_f32(&self, name: &str) -> Result<Option<f32>, String> {
        Ok(self.get_f64(name)?.map(|v| v as f32))
    }
    /// Parse a value-taking option as i64 (C eslARG_INT type check,
    /// esl_getopts.c:1639): `Option <name> takes integer arg; got <val> on cmdline`.
    pub fn get_i64(&self, name: &str) -> Result<Option<i64>, String> {
        match self.get_str(name) {
            None => Ok(None),
            Some(s) => s.parse::<i64>().map(Some).map_err(|_| {
                format!(
                    "Option {} takes integer arg; got {} on cmdline",
                    esl_field(name),
                    esl_field(s)
                )
            }),
        }
    }
    /// Parse a value-taking option as usize. C stores these as eslARG_INT (e.g.
    /// --cpu, range n>=0); a non-integer fails the type check with the same
    /// integer-arg message as [`get_i64`].
    pub fn get_usize(&self, name: &str) -> Result<Option<usize>, String> {
        match self.get_str(name) {
            None => Ok(None),
            Some(s) => s.parse::<usize>().map(Some).map_err(|_| {
                format!(
                    "Option {} takes integer arg; got {} on cmdline",
                    esl_field(name),
                    esl_field(s)
                )
            }),
        }
    }
}

/// Public wrapper over [`esl_field`] for bins (e.g. cmalign) that build their own
/// esl_getopts-style type/range error strings and need C's `%.24s` truncation.
pub fn esl_field24(s: &str) -> String {
    esl_field(s)
}

/// C esl_getopts formats option names and values in its error strings with the
/// `%.24s` conversion (a max field width of 24 bytes). Reproduce that truncation.
fn esl_field(s: &str) -> String {
    if s.len() <= 24 {
        s.to_string()
    } else {
        // %.24s truncates at 24 bytes; option names/values are ASCII, so byte==char.
        s.chars().scan(0usize, |n, c| {
            *n += c.len_utf8();
            if *n <= 24 { Some(c) } else { None }
        }).collect()
    }
}

// ---------------------------------------------------------------------------
// Option-constraint enforcement (shared by cmsearch / cmscan), faithful to
// Easel's esl_opt_VerifyConfig + the programs' manual process_commandline guards.
// ---------------------------------------------------------------------------

/// One `ESL_OPTIONS` row's constraint fields: (name, require_optlist, incompat_optlist).
pub type OptConstraint = (&'static str, Option<&'static str>, Option<&'static str>);
/// An accel-preset manual guard: (primary, incompatible-others in C's check order).
pub type AccelGuard = (&'static str, &'static [&'static str]);
/// A threshold manual guard: (primary, full comma-list for the message, trigger options).
pub type ThreshGuard = (&'static str, &'static str, &'static [&'static str]);

/// Truncation-mode mutual-exclusion guard: (primary, triggers, literal message).
/// The message is emitted verbatim (some C messages carry copy-paste quirks that
/// must be reproduced byte-for-byte), unlike the derived Accel/Thresh messages.
pub type TruncGuard = (&'static str, &'static [&'static str], &'static str);

/// C `esl_opt_IsUsed` (esl_getopts.c): TRUE iff the option was given AND its value
/// differs from the default (`!esl_opt_IsDefault`). `--default` (accel preset) has
/// default value "default", so IsUsed(--default) is ALWAYS FALSE — it neither fires
/// its own guard nor triggers another's. (The options read through this are booleans
/// / no-default values, for which "given" ⟺ "used".)
pub fn used(p: &Parsed, name: &str) -> bool {
    name != "--default" && p.is_set(name)
}

/// C `esl_opt_VerifyConfig` (esl_getopts.c:719): the require loop (all rows in table
/// order) then the incompat loop. Messages use the full optlist string. Returns the
/// errbuf message on the first violation (caller prefixes "Failed to parse command
/// line: ").
pub fn verify_config(p: &Parsed, table: &[OptConstraint]) -> Result<(), String> {
    for (name, req, _) in table {
        if used(p, name) {
            if let Some(r) = req {
                for t in r.split(',') {
                    if !used(p, t) {
                        return Err(format!(
                            "Option {name} requires (or has no effect without) option(s) {r}"
                        ));
                    }
                }
            }
        }
    }
    for (name, _, inc) in table {
        if used(p, name) {
            if let Some(ic) = inc {
                for t in ic.split(',') {
                    if t != *name && used(p, t) {
                        return Err(format!("Option {name} is incompatible with option(s) {ic}"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// C's manual `process_commandline` guards ("combinations I don't know how to
/// disallow with esl_getopts"): the accel-preset blocks (singular "Option X is
/// incompatible with option Y", first hit in C's order) then the threshold block
/// (comma-list form). Runs AFTER `verify_config`. Returns the full first line
/// (already includes the "Failed to parse command line: " prefix).
pub fn manual_guards(
    p: &Parsed,
    accel: &[AccelGuard],
    thresh: &[ThreshGuard],
    trunc: &[TruncGuard],
) -> Result<(), String> {
    for (primary, others) in accel {
        if used(p, primary) {
            for o in *others {
                if used(p, o) {
                    return Err(format!(
                        "Failed to parse command line: Option {primary} is incompatible with option {o}"
                    ));
                }
            }
        }
    }
    for (primary, list, triggers) in thresh {
        if used(p, primary) && triggers.iter().any(|t| used(p, t)) {
            return Err(format!(
                "Failed to parse command line: Option {primary} is incompatible with {list}"
            ));
        }
    }
    for (primary, triggers, msg) in trunc {
        if used(p, primary) && triggers.iter().any(|t| used(p, t)) {
            return Err((*msg).to_string());
        }
    }
    // C cmsearch.c:1817-1826 / cmscan equivalent: --beta only makes sense with
    // --qdb/--nohmm/--max, --fbeta only with --fqdb/--nohmm (the QDB tail-loss
    // overrides are no-ops unless that round actually uses QDBs).
    if used(p, "--beta") && !used(p, "--qdb") && !used(p, "--nohmm") && !used(p, "--max") {
        return Err(
            "Failed to parse command line: --beta only makes sense in combination with --qdb, --nohmm or --max"
                .to_string(),
        );
    }
    if used(p, "--fbeta") && !used(p, "--fqdb") && !used(p, "--nohmm") {
        return Err(
            "Failed to parse command line: --fbeta only makes sense in combination with --fqdb or --nohmm"
                .to_string(),
        );
    }
    Ok(())
}

/// C `process_commandline()` ERROR: block (cmsearch.c:2223 / cmscan.c:2523). Every
/// command-line user error routes here: print the offending first line, then
/// `esl_usage` (`Usage: <usage_line>`) + the basic-options `esl_opt_DisplayHelp`
/// (group 1) block, then exit(1). All to STDOUT. `usage_line` is the program's
/// `esl_usage` output (e.g. "cmsearch [options] <cmfile> <seqdb>"). The final
/// "do <argv0> -h" line embeds the binary path (the one path-dependent line).
pub fn cmdline_fail(first_line: &str, usage_line: &str) -> ! {
    let argv0 = std::env::args().next().unwrap_or_else(|| "cm".to_string());
    print!(
        "{first_line}\n\
Usage: {usage_line}\n\
\n\
where basic options are:\n  \
-h        : show brief help on version and usage\n  \
-g        : configure CM for glocal alignment [default: local]\n  \
-Z <x>    : set search space size in *Mb* to <x> for E-value calculations  (x>0)\n  \
--devhelp : show list of otherwise hidden developer/expert options\n\
\n\
To see more help on available options, do {argv0} -h\n\n"
    );
    std::process::exit(1);
}

/// Resolve a `--name` token (already stripped of any `=value`) to a canonical
/// option name, honouring unambiguous prefix abbreviation among the long options.
fn resolve_long<'a>(tok: &str, table: &'a [OptSpec]) -> Result<&'a OptSpec, String> {
    // Exact match first.
    if let Some(s) = table.iter().find(|s| s.name == tok) {
        return Ok(s);
    }
    // Unambiguous prefix among long ("--") options.
    let matches: Vec<&OptSpec> = table
        .iter()
        .filter(|s| s.name.starts_with("--") && s.name.starts_with(tok))
        .collect();
    match matches.len() {
        1 => Ok(matches[0]),
        0 => Err(format!("No such option \"{}\".", tok)),
        _ => {
            let names: Vec<&str> = matches.iter().map(|s| s.name).collect();
            Err(format!("Abbreviated option \"{}\" is ambiguous ({}).", tok, names.join(", ")))
        }
    }
}

/// Expand attached short-option values in `argv`, mirroring C esl_getopts'
/// `process_stdopt` (easel/esl_getopts.c:1580). A single-dash token is an
/// "optstring": each char is a single-char option; the last optchar that takes an
/// argument consumes the remainder of the token if non-empty (`-E50` -> `-E 50`,
/// `-Warg` -> `-W arg`), and flags bundle (`-gE50` -> `-g -E 50`). `value_shorts`
/// lists the bin's single-char options that take an argument.
///
/// This normalizes the raw argv so a bin whose hand-rolled parser matches on whole
/// tokens (and already handles the space-separated `-E 50` form) transparently
/// accepts the attached `-E50` form. It only ever splits single-dash short
/// optstrings; `--long`, `--long=val`, `-`, `--`, the program name, and every
/// non-option token pass through unchanged, so already-valid command lines are
/// byte-for-byte identical after expansion.
pub fn expand_short_opts(argv: &[String], value_shorts: &[char]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(argv.len() + 4);
    for (idx, tok) in argv.iter().enumerate() {
        // Program name (idx 0), long options / `--`, a lone `-`, and non-option
        // tokens are passed through verbatim.
        if idx == 0 || tok == "-" || !tok.starts_with('-') || tok.starts_with("--") {
            out.push(tok.clone());
            continue;
        }
        // Single-dash short optstring: walk chars, emitting `-c` for each. A value-
        // taking optchar terminates the string, taking any remainder as its argument.
        let chars: Vec<char> = tok.chars().skip(1).collect();
        let mut j = 0;
        while j < chars.len() {
            let c = chars[j];
            out.push(['-', c].iter().collect());
            if value_shorts.contains(&c) {
                let rest: String = chars[j + 1..].iter().collect();
                if !rest.is_empty() {
                    out.push(rest);
                }
                break;
            }
            j += 1;
        }
    }
    out
}

/// Parse `argv[1..]` against `table`. Returns set options + positionals, or an
/// error string suitable for printing before a non-zero exit.
pub fn parse(argv: &[String], table: &[OptSpec]) -> Result<Parsed, String> {
    let mut values: HashMap<&'static str, Option<String>> = HashMap::new();
    let mut positionals: Vec<String> = Vec::new();
    let mut i = 1;
    let mut in_positionals = false;
    while i < argv.len() {
        let tok = &argv[i];
        // Once positional parsing has begun, every remaining token is positional
        // (esl_opt_ProcessCmdline stops option processing at the first arg).
        if in_positionals {
            positionals.push(tok.clone());
            i += 1;
            continue;
        }
        // A token that isn't an option ("-" alone, or not starting with '-') starts
        // the positional list.
        if !tok.starts_with('-') || tok == "-" {
            in_positionals = true;
            positionals.push(tok.clone());
            i += 1;
            continue;
        }
        // "--" alone: conventional end-of-options marker (esl accepts it).
        if tok == "--" {
            in_positionals = true;
            i += 1;
            continue;
        }
        // Long option "--name" / "--name=value" (C esl_getopts.c:process_longopt,
        // easel/esl_getopts.c:1485): '=' separates an attached value; abbreviations
        // resolve via resolve_long (get_optidx_abbrev).
        if tok.starts_with("--") {
            let (name_part, inline_val): (&str, Option<String>) = match tok.find('=') {
                Some(eq) => (&tok[..eq], Some(tok[eq + 1..].to_string())),
                None => (tok.as_str(), None),
            };
            let spec = resolve_long(name_part, table)?;
            match spec.kind {
                ArgKind::None => {
                    if inline_val.is_some() {
                        return Err(format!("Option {} takes no argument.", spec.name));
                    }
                    values.insert(spec.name, None);
                    i += 1;
                }
                ArgKind::Value => {
                    let val = match inline_val {
                        Some(v) => {
                            i += 1;
                            v
                        }
                        None => {
                            let v = argv
                                .get(i + 1)
                                .ok_or_else(|| format!("Option {} requires an argument.", spec.name))?
                                .clone();
                            i += 2;
                            v
                        }
                    };
                    values.insert(spec.name, Some(val));
                }
            }
        } else {
            // Single-dash short-option "optstring" (C esl_getopts.c:process_stdopt,
            // easel/esl_getopts.c:1580). The chars after '-' form an optstring; each
            // char is a single-char option (all Infernal short opts are single-char).
            // Bundling is supported (`-gE50` == `-g -E 50`). Only the last optchar may
            // take an argument: its value is the remainder of the token if non-empty
            // (attached, `-E50`/`-Warg`), else the next argv element (`-E 50`). An
            // arg-taking optchar terminates the optstring. Note the process_stdopt
            // "requires an argument" message has NO trailing '.', unlike the long form.
            let optstring: Vec<char> = tok.chars().skip(1).collect();
            let mut k = 0;
            let mut advance = 1; // argv elements this token consumes (2 iff next-arg form)
            while k < optstring.len() {
                let c = optstring[k];
                // C matches *(optstring) against opt[opti].name[1] (char after '-').
                let name_buf: String = ['-', c].iter().collect();
                let spec = table
                    .iter()
                    .find(|s| s.name == name_buf)
                    .ok_or_else(|| format!("No such option \"-{}\".", c))?;
                match spec.kind {
                    ArgKind::None => {
                        values.insert(spec.name, None);
                        k += 1;
                    }
                    ArgKind::Value => {
                        if k + 1 < optstring.len() {
                            // attached: the rest of this token (`-E50`, `-Warg`)
                            let arg: String = optstring[k + 1..].iter().collect();
                            values.insert(spec.name, Some(arg));
                        } else {
                            // unattached: consume the next argv element (`-E 50`)
                            let v = argv
                                .get(i + 1)
                                .ok_or_else(|| format!("Option {} requires an argument", spec.name))?
                                .clone();
                            values.insert(spec.name, Some(v));
                            advance = 2;
                        }
                        break; // an arg-taking optchar terminates the optstring
                    }
                }
            }
            i += advance;
        }
    }
    Ok(Parsed { values, positionals })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Vec<OptSpec> {
        use ArgKind::{None as N, Value as V};
        vec![
            OptSpec { name: "-g", kind: N },
            OptSpec { name: "-E", kind: V },
            OptSpec { name: "--tblout", kind: V },
            OptSpec { name: "--toponly", kind: N },
            OptSpec { name: "--fmt", kind: V },
            OptSpec { name: "--incE", kind: V },
            OptSpec { name: "--incT", kind: V },
        ]
    }
    fn argv(a: &[&str]) -> Vec<String> {
        std::iter::once("prog").chain(a.iter().copied()).map(String::from).collect()
    }

    #[test]
    fn value_option_consumes_next_token_no_leak() {
        // The historical footgun: `--fmt 2 db seq` must NOT leak "2" into positionals.
        let p = parse(&argv(&["--fmt", "2", "db", "seq"]), &table()).unwrap();
        assert_eq!(p.get_str("--fmt"), Some("2"));
        assert_eq!(p.positionals, vec!["db", "seq"]);
    }

    #[test]
    fn unknown_flag_is_error() {
        assert!(parse(&argv(&["--bogus", "db"]), &table()).is_err());
    }

    #[test]
    fn equals_form_and_prefix_abbrev() {
        let p = parse(&argv(&["--incE=1e-3", "--topon", "db", "seq"]), &table()).unwrap();
        assert_eq!(p.get_f64("--incE").unwrap(), Some(1e-3));
        assert!(p.is_set("--toponly")); // --topon is an unambiguous prefix
    }

    #[test]
    fn ambiguous_prefix_is_error() {
        // "--inc" is a prefix of both --incE and --incT.
        assert!(parse(&argv(&["--inc", "1"]), &table()).is_err());
    }

    #[test]
    fn options_must_precede_positionals() {
        // After the first positional, remaining tokens are positionals (esl semantics),
        // so -E is NOT parsed as an option here.
        let p = parse(&argv(&["-g", "db", "seq", "-E", "50"]), &table()).unwrap();
        assert!(p.is_set("-g"));
        assert!(!p.is_set("-E"));
        assert_eq!(p.positionals, vec!["db", "seq", "-E", "50"]);
    }

    #[test]
    fn boolean_with_value_is_error() {
        assert!(parse(&argv(&["--toponly=1"]), &table()).is_err());
    }

    #[test]
    fn missing_value_is_error() {
        assert!(parse(&argv(&["-E"]), &table()).is_err());
    }

    #[test]
    fn attached_short_value() {
        // C esl_getopts: `-E50` == `-E 50` (attached short-opt argument).
        let p = parse(&argv(&["-E50", "db", "seq"]), &table()).unwrap();
        assert_eq!(p.get_str("-E"), Some("50"));
        assert_eq!(p.positionals, vec!["db", "seq"]);
        // and the spaced form still works identically.
        let q = parse(&argv(&["-E", "50", "db", "seq"]), &table()).unwrap();
        assert_eq!(q.get_str("-E"), Some("50"));
        assert_eq!(q.positionals, vec!["db", "seq"]);
    }

    #[test]
    fn attached_short_value_negative_and_decimal() {
        // `-E1e-9` — remainder (incl. '-') is the attached arg, not a new option.
        let p = parse(&argv(&["-E1e-9"]), &table()).unwrap();
        assert_eq!(p.get_f64("-E").unwrap(), Some(1e-9));
    }

    #[test]
    fn bundled_short_flags_then_attached_value() {
        // C: `-gE50` == `-g -E 50` (flag bundling; last optchar takes the remainder).
        let p = parse(&argv(&["-gE50", "db"]), &table()).unwrap();
        assert!(p.is_set("-g"));
        assert_eq!(p.get_str("-E"), Some("50"));
        assert_eq!(p.positionals, vec!["db"]);
    }

    #[test]
    fn unknown_attached_short_optchar_is_error() {
        // C reports just the bad char: `No such option "-Q".`
        let e = parse(&argv(&["-Q50"]), &table()).err().unwrap();
        assert_eq!(e, "No such option \"-Q\".");
    }
}
