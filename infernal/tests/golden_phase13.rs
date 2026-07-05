//! Phase 13 Golden Tests - Output Formats
//!
//! Tests for Stockholm and tabular output format generation.
//! Golden data generated from Infernal 1.1.5 tRNA model.

use infernal::output::{StockholmOutput, TbloutLine, tblout_header};

// =====================================================================
// Stockholm Output Tests
// =====================================================================

#[test]
fn test_stockholm_header() {
    let stockholm = StockholmOutput {
        id: "tRNA".to_string(),
        au: "Infernal Golden Test".to_string(),
        se: "Test sequence".to_string(),
        tc: 24.04,
        nc: 0.00,
        ga: 24.00,
        ss_cons: "(((((((,,<<<<_______>>>>,<<<<<_______>>>>>,,,,<<<<<_______>>>>>))))))):".to_string(),
        rf: "gccccugUAGcucAaUGGUAgagCauuggaCUuuuAAuccaaaggugugGGUUCgAaUCCcaccaggggcA".to_string(),
    };

    let output = stockholm.format(
        "test_tRNA_seq",
        "GCGGAUUUAGCUCAGUuGGGAGAGCGCCAGACUGAAGAUCUGGAGGuCCUGUGUUCGAUCCACAGAAUUCGCaccA",
        "(((((((,,<<<<___.____>>>>,<<<<<_______>>>>>,,,.,<<<<<_______>>>>>)))))))...:",
    );

    // Verify GF lines are present
    assert!(output.contains("#=GF ID   tRNA"));
    assert!(output.contains("#=GF AU   Infernal Golden Test"));
    assert!(output.contains("#=GF SE   Test sequence"));
    assert!(output.contains("#=GF TC   24.04"));
    assert!(output.contains("#=GF NC   0.00"));
    assert!(output.contains("#=GF GA   24.00"));
}

#[test]
fn test_stockholm_gc_lines() {
    let stockholm = StockholmOutput {
        id: "tRNA".to_string(),
        au: "Infernal Golden Test".to_string(),
        se: "Test sequence".to_string(),
        tc: 24.04,
        nc: 0.00,
        ga: 24.00,
        ss_cons: "(((((((,,<<<<_______>>>>,<<<<<_______>>>>>,,,,<<<<<_______>>>>>))))))):".to_string(),
        rf: "gccccugUAGcucAaUGGUAgagCauuggaCUuuuAAuccaaaggugugGGUUCgAaUCCcaccaggggcA".to_string(),
    };

    let output = stockholm.format(
        "test_tRNA_seq",
        "GCGGAUUUAGCUCAGUuGGGAGAGCGCCAGACUGAAGAUCUGGAGGuCCUGUGUUCGAUCCACAGAAUUCGCaccA",
        "(((((((,,<<<<___.____>>>>,<<<<<_______>>>>>,,,.,<<<<<_______>>>>>)))))))...:",
    );

    // Verify GC lines (SS_cons and RF)
    assert!(output.contains("#=GC SS_cons (((((((,,<<<<_______>>>>,<<<<<_______>>>>>,,,,<<<<<_______>>>>>))))))):"));
    assert!(output.contains("#=GC RF      gccccugUAGcucAaUGGUAgagCauuggaCUuuuAAuccaaaggugugGGUUCgAaUCCcaccaggggcA"));
}


// =====================================================================
// Tabular Output Tests
// =====================================================================

#[test]
fn test_tblout_columns() {
    let tblout = TbloutLine {
        target_name: "test_tRNA_seq".to_string(),
        target_acc: "-".to_string(),
        query_name: "tRNA".to_string(),
        query_acc: "-".to_string(),
        mdl_from: 1,
        mdl_to: 71,
        seq_from: 1,
        seq_to: 76,
        strand: '+',
        trunc: "no".to_string(),
        pass: 1,
        gc: 0.54,
        bias: 0.0,
        score: 48.0963,
        evalue: 1.00e-10,
        inc: '!',
        desc: "-".to_string(),
    };

    let output = tblout.format();

    // Verify all columns are present
    assert!(output.contains("test_tRNA_seq"));
    assert!(output.contains("tRNA"));
    assert!(output.contains("cm"));
    assert!(output.contains("+"));
    assert!(output.contains("no"));
}

#[test]
fn test_tblout_score() {
    let tblout = TbloutLine {
        target_name: "test_tRNA_seq".to_string(),
        target_acc: "-".to_string(),
        query_name: "tRNA".to_string(),
        query_acc: "-".to_string(),
        mdl_from: 1,
        mdl_to: 71,
        seq_from: 1,
        seq_to: 76,
        strand: '+',
        trunc: "no".to_string(),
        pass: 1,
        gc: 0.54,
        bias: 0.0,
        score: 48.0963,
        evalue: 1.00e-10,
        inc: '!',
        desc: "-".to_string(),
    };

    let output = tblout.format();

    // Verify score is correct (48.1 rounded)
    assert!(output.contains("48.1"));
}

#[test]
fn test_tblout_strand() {
    let tblout = TbloutLine {
        target_name: "test_tRNA_seq".to_string(),
        target_acc: "-".to_string(),
        query_name: "tRNA".to_string(),
        query_acc: "-".to_string(),
        mdl_from: 1,
        mdl_to: 71,
        seq_from: 1,
        seq_to: 76,
        strand: '+',
        trunc: "no".to_string(),
        pass: 1,
        gc: 0.54,
        bias: 0.0,
        score: 48.0963,
        evalue: 1.00e-10,
        inc: '!',
        desc: "-".to_string(),
    };

    let output = tblout.format();

    // Verify strand is +
    assert!(output.contains("+"));
}

#[test]
fn test_tblout_gc() {
    let tblout = TbloutLine {
        target_name: "test_tRNA_seq".to_string(),
        target_acc: "-".to_string(),
        query_name: "tRNA".to_string(),
        query_acc: "-".to_string(),
        mdl_from: 1,
        mdl_to: 71,
        seq_from: 1,
        seq_to: 76,
        strand: '+',
        trunc: "no".to_string(),
        pass: 1,
        gc: 0.54,
        bias: 0.0,
        score: 48.0963,
        evalue: 1.00e-10,
        inc: '!',
        desc: "-".to_string(),
    };

    let output = tblout.format();

    // Verify gc is 0.54
    assert!(output.contains("0.54"));
}

#[test]
fn test_tblout_header_format() {
    let header = tblout_header();

    // Verify header contains required columns
    assert!(header.contains("target name"));
    assert!(header.contains("accession"));
    assert!(header.contains("query name"));
    assert!(header.contains("mdl from"));
    assert!(header.contains("seq from"));
    assert!(header.contains("strand"));
    assert!(header.contains("score"));
    assert!(header.contains("E-value"));

    // Verify dashes separator line exists
    assert!(header.contains("---"));
}
