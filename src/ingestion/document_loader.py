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


def _is_heading(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    if HEADING_PATTERN.search(line):
        return True
    if line.isupper() and 4 < len(line) < 80:
        return True
    if re.match(r"^[0-9]+\.[0-9]*[\s:]", line):
        return True
    return False


def extract_tags(document_text: str) -> list[str]:
    found = ISA_TAG_PATTERN.findall(document_text)
    valid = [t for t in found if len(t) >= 4]
    return sorted(set(valid))


def extract_setpoints_from_table(document_text: str) -> list[dict]:
    """
    Parse the setpoints table.
    Returns list of {tag, parameter, value, units, page} dicts.
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


def _parse_table_row(line: str) -> dict | None:
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 4:
        tag = parts[0]
        parameter = parts[1]
        raw_value = parts[2]
        units = parts[3]

        if not ISA_TAG_PATTERN.match(tag):
            return None
        try:
            value = float(re.search(r"[\d.]+", raw_value).group())
        except (AttributeError, ValueError):
            return None

        return {
            "tag": tag,
            "parameter": parameter.lower().replace(" ", "_"),
            "value": value,
            "units": units.lower().strip(),
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
    text = load_document(file_path)
    sections_text, sections_pages = extract_sections(text)
    tags = extract_tags(text)
    setpoints = extract_setpoints_from_table(text)

    return {
        "document_path": file_path,
        "document_text": text,
        "sections": sections_text,
        "sections_pages": sections_pages,
        "tag_inventory": tags,
        "setpoint_table": setpoints,
        "completeness_report": None,
        "consistency_report": None,
        "ambiguity_report": None,
        "all_findings": [],
        "vqs_scores": None,
        "has_circular_interlock": False,
        "missing_critical_sections": [],
        "error": None,
    }
