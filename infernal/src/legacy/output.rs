//! Output format generators for CM search results

/// Stockholm format writer
pub struct StockholmOutput {
    pub id: String,
    pub au: String,
    pub se: String,
    pub tc: f32,
    pub nc: f32,
    pub ga: f32,
    pub ss_cons: String,
    pub rf: String,
}

impl StockholmOutput {
    /// Generate Stockholm format output
    pub fn format(&self, seq_name: &str, aligned_seq: &str, ss: &str) -> String {
        let mut output = String::new();
        output.push_str("# STOCKHOLM 1.0\n\n");
        output.push_str(&format!("#=GF ID   {}\n", self.id));
        output.push_str(&format!("#=GF AU   {}\n", self.au));
        output.push_str(&format!("#=GF SE   {}\n", self.se));
        output.push_str(&format!("#=GF TC   {:.2}\n", self.tc));
        output.push_str(&format!("#=GF NC   {:.2}\n", self.nc));
        output.push_str(&format!("#=GF GA   {:.2}\n", self.ga));
        output.push_str("\n");
        output.push_str(&format!("#=GC SS_cons {}\n", self.ss_cons));
        output.push_str(&format!("#=GC RF      {}\n", self.rf));
        output.push_str("\n");
        output.push_str(&format!("{:<20} {}\n", seq_name, aligned_seq));
        output.push_str(&format!("#=GC SS_cons         {}\n", ss));
        output.push_str("//\n");
        output
    }
}

/// Tabular output (tblout format)
#[derive(Debug, Clone)]
pub struct TbloutLine {
    pub target_name: String,
    pub target_acc: String,
    pub query_name: String,
    pub query_acc: String,
    pub mdl_from: i32,
    pub mdl_to: i32,
    pub seq_from: i32,
    pub seq_to: i32,
    pub strand: char,  // + or -
    pub trunc: String,
    pub pass: i32,
    pub gc: f32,
    pub bias: f32,
    pub score: f32,
    pub evalue: f64,
    pub inc: char,  // ! or ?
    pub desc: String,
}

impl TbloutLine {
    /// Format as tblout line
    pub fn format(&self) -> String {
        format!(
            "{:<20} {:<9} {:<20} {:<9} cm {:>8} {:>8} {:>8} {:>8} {:>6} {:>5} {:>4} {:>.2} {:>.1} {:>6.1} {:>9.2e} {:>3} {}",
            self.target_name, self.target_acc, self.query_name, self.query_acc,
            self.mdl_from, self.mdl_to, self.seq_from, self.seq_to,
            self.strand, self.trunc, self.pass, self.gc, self.bias,
            self.score, self.evalue, self.inc, self.desc
        )
    }
}

/// Generate tblout header
pub fn tblout_header() -> String {
    let header = "#target name         accession query name           accession mdl mdl from   mdl to seq from   seq to strand trunc pass   gc  bias  score   E-value inc description of target\n";
    let dashes = "#------------------- --------- -------------------- --------- --- -------- -------- -------- -------- ------ ----- ---- ---- ----- ------ --------- --- ---------------------\n";
    format!("{}{}", header, dashes)
}
