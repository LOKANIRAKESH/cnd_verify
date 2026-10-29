"""
src/ingestion/document_loader.py
Parses CND documents (PDF or plain text) and extracts:
  - Raw text with [PAGE N] markers for page tracking
  - Section structure with page numbers
  - ISA instrument tags
  - Setpoint table values with page numbers
"""

import re
import fitz
import pdfplumber
from pathlib import Path


ISA_TAG_PATTERN = re.compile(r"\b[A-Z]{1,3}-[0-9]{1,4}[A-Z]?\b")

HEADING_PATTERN = re.compile(
    r"^(?:SECTION\s+\d+|[0-9]+\.[0-9]*\s+[A-Z]|\b(?:INTERLOCKS?|ALARMS?|SETPOINTS?|"
    r"STARTUP|SHUTDOWN|OPERATING MODES?|EQUIPMENT LIST|GENERAL DESCRIPTION|"
    r"EMERGENCY SHUTDOWN|CONTROL LOOPS?|REVISION)\b)",
    re.IGNORECASE,
)

# Lines per estimated "page" for plain text files
LINES_PER_PAGE = 50

AMBIGUITY_PATTERNS = {
    "QUANTIFIER": re.compile(
        r"\b(sufficient|adequate|excessive|appropriate|acceptable|significant|"
        r"high(?!\s+high)|low(?!\s+low))\b",
        re.IGNORECASE,
    ),
    "TEMPORAL": re.compile(
        r"\b(shortly|soon|quickly|gradually|slowly|after some time|"
        r"within a reasonable time|periodically|allow to stabilize|stabilise)\b",
        re.IGNORECASE,
    ),
    "REFERENTIAL": re.compile(
        r"\b(it should|it will|it must|it shall|this valve|this pump|"
        r"the pump\b|the sensor\b|the system\b|the controller\b)\b",
        re.IGNORECASE,
    ),
    "MODAL": re.compile(r"\b(should|may\b|might\b|could\b)\b", re.IGNORECASE),
}


def load_document(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    if path.suffix.lower() == ".pdf":
        return _parse_pdf(str(path))
    return _parse_text(str(path))


def _parse_text(file_path: str) -> str:
    """
    Read plain text CND and inject [PAGE N] markers every LINES_PER_PAGE lines.
    This lets every downstream component track approximate page numbers.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = []
    for i, line in enumerate(lines):
        if i % LINES_PER_PAGE == 0:
            result.append(f"\n[PAGE {i // LINES_PER_PAGE + 1}]\n")
        result.append(line)
    return "".join(result)


def _parse_pdf(file_path: str) -> str:
    parts = []
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if len(text.strip()) < 50:
            text = _extract_page_with_pdfplumber(file_path, page_num)
        parts.append(f"\n[PAGE {page_num + 1}]\n{text}")
    doc.close()
    return "\n".join(parts)


def _extract_page_with_pdfplumber(file_path: str, page_num: int) -> str:
    with pdfplumber.open(file_path) as pdf:
        if page_num < len(pdf.pages):
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            parts = [page.extract_text() or ""]
            for table in tables:
                for row in table:
                    if row:
                        parts.append(" | ".join(str(c) for c in row if c))
            return "\n".join(parts)
    return ""


def extract_sections(document_text: str) -> tuple[dict[str, str], dict[str, int]]:
    """
    Split document text into named sections.
    Returns:
      sections_text  — {section_title: section_text}
      sections_pages — {section_title: page_number}
    """
    sections_text: dict[str, str] = {}
    sections_pages: dict[str, int] = {}
    current_title = "PREAMBLE"
    current_lines: list[str] = []
    current_page = 1
    section_start_page = 1

    for line in document_text.split("\n"):
        stripped = line.strip()
        page_match = re.match(r"\[PAGE (\d+)\]", stripped)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        if _is_heading(stripped):
            if current_lines:
                sections_text[current_title] = "\n".join(current_lines).strip()
                sections_pages[current_title] = section_start_page
            current_title = stripped.upper()
            section_start_page = current_page
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections_text[current_title] = "\n".join(current_lines).strip()
        sections_pages[current_title] = section_start_page

    return sections_text, sections_pages


# Lines that look ALL-CAPS but must never be treated as section headings
_NOT_HEADING_RE = re.compile(
    r"^("
    r"[A-Z]{1,3}-[0-9]{1,4}[A-Z]?(?:-[A-Z0-9]+)*\s*$"  # ISA tags: PT-101, XV-201, ESD-1, PT-101-HH
    r"|DN\d+(?:[,\s]+PN\d+)?\s*$"                         # Pipe specs: DN200, DN200, PN16
    r"|PN\d+\s*$"                                          # Pressure nominal: PN16
    r"|REV(?:ISION)?[\s:]+[A-Z0-9].*$"                    # Revision markers: REV C, REVISION: C
    r"|CONFIDENTIAL.*$"                                    # Document stamps
    r"|NOT TO SCALE.*$"                                    # Drawing notes
    r"|FIG(?:URE)?[.\s].*$"                                # Figure captions: FIG. 1 …
    r")",
    re.IGNORECASE,
)

# Diagram/label keywords that are NOT section headings even when ALL-CAPS
_DIAGRAM_LABEL_WORDS = frozenset({
    "SUCTION", "DISCHARGE", "INLET", "OUTLET", "VESSEL", "PIPELINE",
    "LEGEND", "DATUM", "HEADER", "MANIFOLD", "RETURN", "BYPASS",
    "PROCESS",  # "TO PROCESS I" type diagram flow labels
})


def _is_heading(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    stripped = line.strip()

    # Explicit exclusions — these look like headings but aren't
    if _NOT_HEADING_RE.match(stripped):
        return False

    # Lines ending with sentence-ending punctuation are text, not headings
    if re.search(r"[.!?'\"]\s*$", stripped):
        return False

    # HMI state completion messages ("STARTUP COMPLETE", "SHUTDOWN COMPLETE")
    if re.match(r"^(?:STARTUP|SHUTDOWN|ESD|EMERGENCY)\s+COMPLETE\s*$", stripped, re.IGNORECASE):
        return False

    # Named heading keywords (INTERLOCKS, ALARMS, SETPOINTS, STARTUP, SHUTDOWN, …)
    # BUT reject long mixed-case prose sentences that merely contain a keyword
    # e.g. "Interlock IL-101 activates when PT-101 discharge pressure exceeds 150 PSI"
    # Real headings are ALL-CAPS short strings or short title-cased labels.
    if HEADING_PATTERN.search(line):
        if len(stripped) > 60 and stripped != stripped.upper():
            pass  # prose sentence with a keyword — not a heading
        else:
            return True

    # ALL CAPS line — require at least 2 words with ≥2 letters to avoid
    # ISA tags, single states ("STANDBY"), diagram labels ("LEGEND")
    if line.isupper() and 4 < len(line) < 80:
        alpha_words = [w for w in re.split(r"[\s,;/|]+", stripped) if re.search(r"[A-Za-z]{2,}", w)]
        if len(alpha_words) < 2:
            return False
        # Exclude P&ID diagram labels (common location/topology words)
        upper_words = set(w.upper() for w in alpha_words)
        if upper_words & _DIAGRAM_LABEL_WORDS:
            return False
        return True

    # Numbered section: "3.1 Description" or "4. Setpoints"
    if re.match(r"^[0-9]+\.[0-9]*[\s:]", line):
        return True
    return False


def extract_tags(document_text: str) -> list[str]:
    found = ISA_TAG_PATTERN.findall(document_text)
    valid = [t for t in found if len(t) >= 4]
    return sorted(set(valid))


def extract_setpoints_from_table(document_text: str) -> list[dict]:
    """
    Parse the setpoints table from text with [PAGE N] markers.
    Supports both pipe-delimited (|) and space-delimited rows.
    """
    setpoints = []
    in_table = False
    current_page = 1
    table_page = 1

    for line in document_text.split("\n"):
        stripped = line.strip()
        page_match = re.match(r"\[PAGE (\d+)\]", stripped)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        if "TAG" in stripped.upper() and "PARAMETER" in stripped.upper() and "VALUE" in stripped.upper():
            in_table = True
            table_page = current_page
            continue

        if not in_table:
            continue

        if not stripped or _is_heading(stripped):
            in_table = False
            continue

        if re.match(r"^[-|=\s]+$", stripped):
            continue

        row = _parse_table_row(stripped)
        if row:
            row["page"] = table_page
            setpoints.append(row)

    return setpoints


def extract_setpoints_from_pdf(file_path: str) -> list[dict]:
    """
    Use pdfplumber to extract structured setpoints tables from PDFs.
    This is much more reliable than parsing raw PyMuPDF text for table data.
    Returns list of {tag, parameter, value, units, page} dicts.
    """
    setpoints = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = [str(c).strip().upper() if c else "" for c in table[0]]
                    # Only process setpoints-style tables (have TAG + VALUE columns)
                    if "TAG" not in " ".join(header) and "VALUE" not in " ".join(header):
                        continue
                    # Map column indices
                    tag_col = _col_index(header, ("TAG",))
                    param_col = _col_index(header, ("PARAMETER", "PARAM", "DESCRIPTION"))
                    val_col = _col_index(header, ("VALUE", "VAL", "SETPOINT"))
                    unit_col = _col_index(header, ("UNITS", "UNIT", "UOM", "ENG"))
                    if tag_col is None or val_col is None:
                        continue
                    for row in table[1:]:
                        if not row or len(row) <= max(c for c in [tag_col, val_col] if c is not None):
                            continue
                        tag_raw = str(row[tag_col]).strip() if row[tag_col] else ""
                        if not ISA_TAG_PATTERN.match(tag_raw):
                            continue
                        val_raw = str(row[val_col]).strip() if row[val_col] else ""
                        num_match = re.search(r"[\d]+(?:\.[0-9]+)?", val_raw)
                        if not num_match:
                            continue
                        param_raw = str(row[param_col]).strip() if (param_col is not None and row[param_col]) else ""
                        unit_raw = str(row[unit_col]).strip() if (unit_col is not None and row[unit_col]) else ""
                        setpoints.append({
                            "tag": tag_raw.upper(),
                            "parameter": param_raw.lower().replace(" ", "_"),
                            "value": float(num_match.group()),
                            "units": unit_raw.lower().strip(),
                            "page": page_num,
                            "raw_line": " | ".join(str(c) for c in row if c),
                        })
    except Exception:
        pass
    return setpoints


def _col_index(header: list[str], candidates: tuple) -> int | None:
    """Return the first column index whose header matches any candidate string."""
    for i, h in enumerate(header):
        for c in candidates:
            if c in h:
                return i
    return None


def _parse_table_row(line: str) -> dict | None:
    # Try pipe-delimited first
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 4:
        tag = parts[0]
        parameter = parts[1]
        raw_value = parts[2]
        units = parts[3]
        if ISA_TAG_PATTERN.match(tag):
            try:
                value = float(re.search(r"[\d.]+", raw_value).group())
                return {
                    "tag": tag,
                    "parameter": parameter.lower().replace(" ", "_"),
                    "value": value,
                    "units": units.lower().strip(),
                    "raw_line": line,
                }
            except (AttributeError, ValueError):
                pass
    # Try whitespace-separated: TAG  PARAMETER  VALUE  UNITS
    m = re.match(r"^([A-Z]{1,3}-[0-9]{1,4}[A-Z]?)\s+(.+?)\s+([\d]+(?:\.[0-9]+)?)\s+(\S+)\s*$", line)
    if m:
        return {
            "tag": m.group(1),
            "parameter": m.group(2).lower().replace(" ", "_"),
            "value": float(m.group(3)),
            "units": m.group(4).lower(),
            "raw_line": line,
        }
    return None


def get_page_of_text(document_text: str, search_text: str) -> int:
    """
    Find which page a piece of text appears on.
    Returns 1-indexed page number, or 0 if not found.
    """
    current_page = 1
    for line in document_text.split("\n"):
        page_match = re.match(r"\[PAGE (\d+)\]", line.strip())
        if page_match:
            current_page = int(page_match.group(1))
        if search_text.lower()[:60] in line.lower():
            return current_page
    return 0


def prepare_document(file_path: str) -> dict:
    """Full pipeline: load → extract all → return initial VerificationState."""
    path = Path(file_path)
    text = load_document(file_path)
    sections_text, sections_pages = extract_sections(text)
    tags = extract_tags(text)

    # PDFs: use pdfplumber for proper structured table extraction
    # Text files: fall back to [PAGE N] marker text parsing
    if path.suffix.lower() == ".pdf":
        setpoints = extract_setpoints_from_pdf(file_path)
        if not setpoints:
            setpoints = extract_setpoints_from_table(text)
    else:
        setpoints = extract_setpoints_from_table(text)

    return {
        "document_path": file_path,
        "document_text": text,
        "sections": sections_text,
        "sections_pages": sections_pages,
        "tag_inventory": tags,
        "setpoint_table": setpoints,
        # Pass 1 fields — filled by build_document_index in graph
        "doc_collection_name": "",
        "tag_index": {},
        "setpoint_index": {},
        "definition_map": {},
        "chunk_groups": [],
        # Agent reports
        "completeness_report": None,
        "consistency_report": None,
        "ambiguity_report": None,
        "image_report": None,
        "image_summary": [],
        "all_findings": [],
        "vqs_scores": None,
        # PLC generation
        "plc_code": {},
        "plc_summary": {},
        # Hard fail flags
        "has_circular_interlock": False,
        "missing_critical_sections": [],
        "error": None,
    }
