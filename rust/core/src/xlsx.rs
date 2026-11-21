//! `ExcelExporter`'s workbook writer.
//!
//! What the C++ core did, found by reading it and by running 0.8.0: the exporter keeps
//! its `template` argument but never opens it. `XLDocument::create` starts every export
//! from OpenXLSX's built-in blank workbook (one sheet, "Sheet1"), and each mapped column
//! gets its label in row 1 and values from row 2 down. Numbers are numeric cells,
//! strings are shared strings, NaN and ±inf are `#NUM!` error cells, a field missing
//! from the inputs leaves its cell empty, and the cell address is the column string
//! followed by the row number, parsed by OpenXLSX's rules (1–3 upper-case letters up to
//! XFD, then digits; "a", "AAAA" or "A$" are refused with a `RuntimeError`).
//!
//! This module writes the same cells into a minimal SpreadsheetML package with the
//! `zip` crate. Nothing reads a template, because nothing ever did. The file is first
//! written beside the target and then renamed over it, so a failed export leaves an
//! existing file untouched (as OpenXLSX's temp-file-and-rename did).
//!
//! Deliberate differences from OpenXLSX: control characters are escaped (`_x0001_`),
//! where OpenXLSX wrote them raw and produced a file Excel and openpyxl cannot open;
//! an empty file name or a directory as the target is an error, where OpenXLSX
//! reported success (and replaced an empty directory with the file).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::value::{Inputs, Value};

const MAX_ROWS: u64 = 1_048_576;
const MAX_COLS: u32 = 16_384;

/// The field name that maps a column to the computed premium.
pub const PREMIUM_FIELD: &[u8] = b"premium_output";

/// Column string → (field name, header label), in column-string order like the C++
/// `std::map<std::string, std::pair<…>>`. Mapping a column again replaces it.
pub type ColumnMapping = BTreeMap<Vec<u8>, (Vec<u8>, Vec<u8>)>;

#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Number(f64),
    Text(Vec<u8>),
}

/// The cells of one worksheet, keyed (row, column) so they iterate in sheet order.
#[derive(Debug, Default)]
pub struct Sheet {
    cells: BTreeMap<(u32, u32), Cell>,
}

/// OpenXLSX's `XLCellReference::coordinatesFromAddress` (v0.5.1), including its
/// wrapping integer arithmetic, then the constructor's row/column range check.
pub fn coordinates_from_address(address: &[u8]) -> Result<(u32, u32)> {
    let mut letter_count = 0usize;
    let mut col_no: u32 = 0;
    for &b in address {
        if b.is_ascii_uppercase() {
            letter_count += 1;
            col_no = col_no.wrapping_mul(26).wrapping_add(u32::from(b - b'A' + 1));
        } else {
            break;
        }
    }
    let shown = String::from_utf8_lossy(address);
    if col_no > 0 && col_no <= MAX_COLS && address.len() > letter_count {
        let mut pos = letter_count;
        let mut row_no: u64 = 0;
        while pos < address.len() && address[pos].is_ascii_digit() {
            row_no = row_no.wrapping_mul(10).wrapping_add(u64::from(address[pos] - b'0'));
            pos += 1;
        }
        if pos == address.len() && row_no <= MAX_ROWS {
            if row_no == 0 {
                return Err(Error::Runtime(format!("Cell reference \"{shown}\" is invalid")));
            }
            return Ok((row_no as u32, col_no));
        }
    }
    Err(Error::Runtime(format!(
        "XLCellReference::coordinatesFromAddress - address \"{shown}\" is invalid"
    )))
}

impl Sheet {
    pub fn new() -> Self {
        Self::default()
    }

    /// `wks.cell(address).value() = cell`.
    pub fn set(&mut self, address: &[u8], cell: Cell) -> Result<()> {
        let rc = coordinates_from_address(address)?;
        self.cells.insert(rc, cell);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn get(&self, row: u32, col: u32) -> Option<&Cell> {
        self.cells.get(&(row, col))
    }
}

fn address_with_row(col: &[u8], row: usize) -> Vec<u8> {
    let mut a = col.to_vec();
    a.extend_from_slice(row.to_string().as_bytes());
    a
}

fn input_cell(inputs: &Inputs, field_name: &[u8]) -> Option<Cell> {
    match inputs.get(field_name)? {
        Value::Num(x) => Some(Cell::Number(*x)),
        Value::Str(s) => Some(Cell::Text(s.clone())),
    }
}

/// The sheet `ExcelExporter::write_data` builds: labels in row 1, one quote in row 2.
pub fn single_sheet(mapping: &ColumnMapping, inputs: &Inputs, premium: f64) -> Result<Sheet> {
    let mut sheet = Sheet::new();
    for (col, (field_name, label)) in mapping {
        sheet.set(&address_with_row(col, 1), Cell::Text(label.clone()))?;
        let value_address = address_with_row(col, 2);
        if field_name.as_slice() == PREMIUM_FIELD {
            sheet.set(&value_address, Cell::Number(premium))?;
        } else if let Some(cell) = input_cell(inputs, field_name) {
            sheet.set(&value_address, cell)?;
        }
    }
    Ok(sheet)
}

/// The sheet `ExcelExporter::write_batch_data` builds: labels in row 1, then one row
/// per quote from row 2. A quote without a premium gets 0.
pub fn batch_sheet(mapping: &ColumnMapping, batch_inputs: &[Inputs], batch_premiums: &[f64]) -> Result<Sheet> {
    let mut sheet = Sheet::new();
    for (col, (_, label)) in mapping {
        sheet.set(&address_with_row(col, 1), Cell::Text(label.clone()))?;
    }
    for (i, inputs) in batch_inputs.iter().enumerate() {
        let row = i + 2;
        let premium = batch_premiums.get(i).copied().unwrap_or(0.0);
        for (col, (field_name, _)) in mapping {
            let address = address_with_row(col, row);
            if field_name.as_slice() == PREMIUM_FIELD {
                sheet.set(&address, Cell::Number(premium))?;
            } else if let Some(cell) = input_cell(inputs, field_name) {
                sheet.set(&address, cell)?;
            }
        }
    }
    Ok(sheet)
}

fn column_letters(mut col: u32) -> String {
    let mut out = Vec::new();
    while col > 0 {
        let rem = (col - 1) % 26;
        out.push(b'A' + rem as u8);
        col = (col - 1) / 26;
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII")
}

/// A double as xsd:double text that reads back as the same double: the shortest
/// round-trip digits, with an exponent outside [1e-5, 1e16).
fn format_number(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let a = x.abs();
    if (1e-5..1e16).contains(&a) { format!("{x}") } else { format!("{x:e}") }
}

/// Text content of a `<t>` element: XML-escaped, CR as `&#13;`, and the control
/// characters XML 1.0 cannot carry at all written as OOXML `_xHHHH_` escapes (with a
/// literal `_xHHHH_` protected as `_x005F_xHHHH_`, as the standard requires), so the
/// file always parses; Excel turns the escapes back into the characters.
fn escape_text(s: &str, out: &mut String) {
    let bytes = s.as_bytes();
    for (i, ch) in s.char_indices() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\t' | '\n' => out.push(ch),
            // A character reference survives XML's newline normalisation, which would
            // turn a raw CR into LF.
            '\r' => out.push_str("&#13;"),
            c if (c as u32) < 0x20 || c == '\u{FFFE}' || c == '\u{FFFF}' => {
                let _ = write!(out, "_x{:04X}_", c as u32);
            }
            '_' if is_ooxml_escape(&bytes[i..]) => out.push_str("_x005F_"),
            c => out.push(c),
        }
    }
}

fn is_ooxml_escape(b: &[u8]) -> bool {
    b.len() >= 7 && b[0] == b'_' && b[1] == b'x' && b[2..6].iter().all(u8::is_ascii_hexdigit) && b[6] == b'_'
}

fn needs_space_preserve(s: &str) -> bool {
    let first = s.chars().next();
    let last = s.chars().last();
    matches!(first, Some(c) if c.is_whitespace()) || matches!(last, Some(c) if c.is_whitespace())
}

const CONTENT_TYPES: &str = concat!(
    r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#,
    "\n",
    r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">"#,
    r#"<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>"#,
    r#"<Default Extension="xml" ContentType="application/xml"/>"#,
    r#"<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>"#,
    r#"<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>"#,
    r#"<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>"#,
    r#"<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>"#,
    r#"<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>"#,
    r#"<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>"#,
    "</Types>"
);

const ROOT_RELS: &str = concat!(
    r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#,
    "\n",
    r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
    r#"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>"#,
    r#"<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>"#,
    r#"<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>"#,
    "</Relationships>"
);

const APP_PROPS: &str = concat!(
    r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#,
    "\n",
    r#"<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" "#,
    r#"xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">"#,
    "<Application>Microsoft Excel</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>",
    r#"<HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>"#,
    "<vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs>",
    r#"<TitlesOfParts><vt:vector size="1" baseType="lpstr"><vt:lpstr>Sheet1</vt:lpstr></vt:vector></TitlesOfParts>"#,
    "<LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged>",
    "<AppVersion>12.0000</AppVersion></Properties>"
);

const CORE_PROPS: &str = concat!(
    r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#,
    "\n",
    r#"<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" "#,
    r#"xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" "#,
    r#"xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">"#,
    "<dc:creator>RiskPY</dc:creator></cp:coreProperties>"
);

const WORKBOOK: &str = concat!(
    r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#,
    "\n",
    r#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" "#,
    r#"xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">"#,
    r#"<bookViews><workbookView/></bookViews>"#,
    r#"<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>"#,
    "</workbook>"
);

const WORKBOOK_RELS: &str = concat!(
    r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#,
    "\n",
    r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
    r#"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>"#,
    r#"<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>"#,
    r#"<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>"#,
    "</Relationships>"
);

const STYLES: &str = concat!(
    r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#,
    "\n",
    r#"<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">"#,
    r#"<fonts count="1"><font><sz val="11"/><name val="Calibri"/><family val="2"/></font></fonts>"#,
    r#"<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>"#,
    r#"<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"#,
    r#"<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>"#,
    r#"<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>"#,
    r#"<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>"#,
    r#"<dxfs count="0"/><tableStyles count="0" defaultTableStyle="TableStyleMedium9" defaultPivotStyle="PivotStyleLight16"/>"#,
    "</styleSheet>"
);

/// The worksheet and shared-string parts for a sheet.
fn sheet_parts(sheet: &Sheet) -> (String, String) {
    let mut strings: Vec<String> = Vec::new();
    let mut index: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut string_refs = 0usize;

    let mut data = String::new();
    let mut current_row = 0u32;
    let (mut min_row, mut max_row, mut min_col, mut max_col) = (u32::MAX, 0u32, u32::MAX, 0u32);
    for (&(row, col), cell) in &sheet.cells {
        min_row = min_row.min(row);
        max_row = max_row.max(row);
        min_col = min_col.min(col);
        max_col = max_col.max(col);
        if row != current_row {
            if current_row != 0 {
                data.push_str("</row>");
            }
            let _ = write!(data, r#"<row r="{row}">"#);
            current_row = row;
        }
        let r = format!("{}{}", column_letters(col), row);
        match cell {
            Cell::Number(x) if x.is_finite() => {
                let _ = write!(data, r#"<c r="{r}"><v>{}</v></c>"#, format_number(*x));
            }
            Cell::Number(_) => {
                let _ = write!(data, r#"<c r="{r}" t="e"><v>#NUM!</v></c>"#);
            }
            Cell::Text(bytes) => {
                let next = strings.len();
                let idx = *index.entry(bytes.clone()).or_insert_with(|| {
                    strings.push(String::from_utf8_lossy(bytes).into_owned());
                    next
                });
                string_refs += 1;
                let _ = write!(data, r#"<c r="{r}" t="s"><v>{idx}</v></c>"#);
            }
        }
    }
    if current_row != 0 {
        data.push_str("</row>");
    }
    let dimension = if sheet.cells.is_empty() {
        "A1".to_string()
    } else if (min_row, min_col) == (max_row, max_col) {
        format!("{}{}", column_letters(min_col), min_row)
    } else {
        format!("{}{}:{}{}", column_letters(min_col), min_row, column_letters(max_col), max_row)
    };

    let mut ws = String::with_capacity(data.len() + 600);
    ws.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    ws.push('\n');
    ws.push_str(r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" "#);
    ws.push_str(r#"xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">"#);
    let _ = write!(ws, r#"<dimension ref="{dimension}"/>"#);
    ws.push_str(r#"<sheetViews><sheetView tabSelected="1" workbookViewId="0"/></sheetViews>"#);
    ws.push_str(r#"<sheetFormatPr defaultRowHeight="15"/>"#);
    if data.is_empty() {
        ws.push_str("<sheetData/>");
    } else {
        ws.push_str("<sheetData>");
        ws.push_str(&data);
        ws.push_str("</sheetData>");
    }
    ws.push_str(r#"<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>"#);
    ws.push_str("</worksheet>");

    let mut sst = String::new();
    sst.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    sst.push('\n');
    let _ = write!(
        sst,
        r#"<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{string_refs}" uniqueCount="{}">"#,
        strings.len()
    );
    for s in &strings {
        if needs_space_preserve(s) {
            sst.push_str(r#"<si><t xml:space="preserve">"#);
        } else {
            sst.push_str("<si><t>");
        }
        escape_text(s, &mut sst);
        sst.push_str("</t></si>");
    }
    sst.push_str("</sst>");
    (ws, sst)
}

/// The workbook as .xlsx bytes.
pub fn workbook_bytes(sheet: &Sheet) -> Result<Vec<u8>> {
    use zip::CompressionMethod;
    use zip::write::SimpleFileOptions;

    let (worksheet, shared_strings) = sheet_parts(sheet);
    let parts: [(&str, &[u8]); 8] = [
        ("[Content_Types].xml", CONTENT_TYPES.as_bytes()),
        ("_rels/.rels", ROOT_RELS.as_bytes()),
        ("docProps/app.xml", APP_PROPS.as_bytes()),
        ("docProps/core.xml", CORE_PROPS.as_bytes()),
        ("xl/workbook.xml", WORKBOOK.as_bytes()),
        ("xl/_rels/workbook.xml.rels", WORKBOOK_RELS.as_bytes()),
        ("xl/styles.xml", STYLES.as_bytes()),
        ("xl/sharedStrings.xml", shared_strings.as_bytes()),
    ];
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut zw = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let zerr = |e: zip::result::ZipError| Error::Runtime(format!("cannot build the workbook: {e}"));
    let ioerr = |e: std::io::Error| Error::Runtime(format!("cannot build the workbook: {e}"));
    for (name, body) in parts {
        zw.start_file(name, options).map_err(zerr)?;
        zw.write_all(body).map_err(ioerr)?;
    }
    zw.start_file("xl/worksheets/sheet1.xml", options).map_err(zerr)?;
    zw.write_all(worksheet.as_bytes()).map_err(ioerr)?;
    Ok(zw.finish().map_err(zerr)?.into_inner())
}

#[cfg(unix)]
fn path_from_bytes(b: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;
    PathBuf::from(std::ffi::OsStr::from_bytes(b))
}

#[cfg(not(unix))]
fn path_from_bytes(b: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(b).into_owned())
}

fn temp_name(target: &Path) -> PathBuf {
    let mut rnd = [0u8; 8];
    let tag = if getrandom::fill(&mut rnd).is_ok() {
        u64::from_le_bytes(rnd)
    } else {
        u64::from(std::process::id()) ^ 0x9e37_79b9_7f4a_7c15
    };
    let file = target.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let name = format!(".{file}.{tag:016x}.tmp");
    match target.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(name),
        _ => PathBuf::from(name),
    }
}

/// Writes the workbook to `filename` (raw bytes, as the C++ `std::string` path),
/// replacing any existing file only once the new one is complete.
pub fn save(filename: &[u8], sheet: &Sheet) -> Result<()> {
    let shown = String::from_utf8_lossy(filename).into_owned();
    if filename.is_empty() {
        return Err(Error::Runtime("cannot export to Excel: the file name is empty".into()));
    }
    let target = path_from_bytes(filename);
    if target.is_dir() {
        return Err(Error::Runtime(format!("cannot export to Excel: {shown} is a directory")));
    }
    let bytes = workbook_bytes(sheet)?;
    let tmp = temp_name(&target);
    let fail = |what: &str, e: std::io::Error| Error::Runtime(format!("cannot export to Excel: {what} {shown}: {e}"));
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| fail("could not create a temporary file beside", e))?;
        if let Err(e) = f.write_all(&bytes).and_then(|()| f.sync_all()) {
            drop(f);
            let _ = std::fs::remove_file(&tmp);
            return Err(fail("could not write", e));
        }
    }
    if let Err(e) = std::fs::rename(&tmp, &target) {
        let _ = std::fs::remove_file(&tmp);
        return Err(fail("could not replace", e));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_follow_openxlsx() {
        assert_eq!(coordinates_from_address(b"A1").unwrap(), (1, 1));
        assert_eq!(coordinates_from_address(b"XFD1048576").unwrap(), (1_048_576, 16_384));
        assert_eq!(coordinates_from_address(b"A01").unwrap(), (1, 1));
        assert_eq!(coordinates_from_address(b"A11").unwrap(), (11, 1));
        for bad in [&b"a1"[..], b"XFE1", b"AAAA1", b"A$1", b"$A1", b"1", b"A", b"A10485761", b" A1", b"A-11"] {
            let err = coordinates_from_address(bad).unwrap_err();
            assert!(matches!(err, Error::Runtime(_)), "{bad:?}");
        }
        assert_eq!(
            coordinates_from_address(b"a1").unwrap_err().to_string(),
            "XLCellReference::coordinatesFromAddress - address \"a1\" is invalid"
        );
    }

    #[test]
    fn column_letters_round_trip() {
        for col in [1u32, 26, 27, 52, 702, 703, 16_384] {
            let letters = column_letters(col);
            let addr = format!("{letters}1");
            assert_eq!(coordinates_from_address(addr.as_bytes()).unwrap(), (1, col));
        }
    }

    #[test]
    fn numbers_read_back_exactly() {
        for x in [0.1 + 0.2, std::f64::consts::PI, 1e300, 5e-324, -1.5e-7, 1e16, 123_456.0, 30.0] {
            let s = format_number(x);
            assert_eq!(s.parse::<f64>().unwrap(), x, "{s}");
        }
        assert_eq!(format_number(30.0), "30");
        assert_eq!(format_number(-0.0), "0");
    }

    #[test]
    fn escaping() {
        let mut out = String::new();
        escape_text("a<b>&c\u{1}d_x0041_e\r", &mut out);
        assert_eq!(out, "a&lt;b&gt;&amp;c_x0001_d_x005F_x0041_e&#13;");
    }

    #[test]
    fn batch_sheet_layout() {
        let mut mapping = ColumnMapping::new();
        mapping.insert(b"B".to_vec(), (PREMIUM_FIELD.to_vec(), b"P".to_vec()));
        mapping.insert(b"A".to_vec(), (b"age".to_vec(), b"Age".to_vec()));
        let mut q1 = Inputs::new();
        q1.insert(b"age".to_vec(), Value::Num(1.0));
        let sheet = batch_sheet(&mapping, &[q1, Inputs::new()], &[10.0]).unwrap();
        assert_eq!(sheet.get(1, 1), Some(&Cell::Text(b"Age".to_vec())));
        assert_eq!(sheet.get(2, 1), Some(&Cell::Number(1.0)));
        assert_eq!(sheet.get(2, 2), Some(&Cell::Number(10.0)));
        assert_eq!(sheet.get(3, 1), None);
        assert_eq!(sheet.get(3, 2), Some(&Cell::Number(0.0)));
    }
}
