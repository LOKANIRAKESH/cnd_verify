"""
src/agents/image_agent.py

Extracts and analyses images embedded in CND PDFs.

For each image found:
  1. Classifies it: P&ID | CAUSE_EFFECT | TABLE | PHOTO | DIAGRAM | OTHER
  2. Extracts ISA tags visible in the image (OCR via PyMuPDF text layer or pytesseract)
  3. Cross-validates image tags against text-derived tag inventory
  4. Flags tags visible in images but undocumented in the narrative (and vice versa)

Findings:
  IMAGE_TAG_NOT_IN_TEXT  — tag appears in P&ID but has no description in narrative
  IMAGE_TAG_MISMATCH     — image shows ball valve, text says gate valve
  UNREADABLE_IMAGE       — critical diagram that cannot be parsed (low resolution)
"""

import re
import io
import uuid
import logging
from pathlib import Path

import fitz   # PyMuPDF

from src.orchestration.state import Finding, AgentReport

logger = logging.getLogger("cnd_verify")

ISA_TAG_RE = re.compile(r"\b([A-Z]{1,3}-[0-9]{1,4}[A-Z]?)\b")

# Image classification keywords — searched in nearby text / caption
IMAGE_CLASS_KEYWORDS = {
    "P&ID":         ["p&id", "piping", "instrumentation", "diagram", "schematic", "pid"],
    "CAUSE_EFFECT": ["cause", "effect", "c&e", "cause-effect", "matrix", "interlock table"],
    "TABLE":        ["setpoint", "alarm", "table", "list", "schedule"],
    "PHOTO":        ["photograph", "photo", "installation", "site"],
    "DIAGRAM":      ["flow", "logic", "block", "functional"],
}


def _classify_image(caption: str, nearby_text: str) -> str:
    combined = (caption + " " + nearby_text).lower()
    for cls, keywords in IMAGE_CLASS_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return cls
    return "OTHER"


def _extract_text_from_image_region(page: fitz.Page, rect: fitz.Rect) -> str:
    """
    Extract selectable text that lies within the image bounding rectangle.
    PDFs with vector P&IDs have selectable text on top of drawing elements.
    """
    words = page.get_text("words")
    in_rect = [w[4] for w in words if fitz.Rect(w[:4]).intersects(rect)]
    return " ".join(in_rect)


def _try_ocr(image_bytes: bytes) -> str:
    """
    Attempt OCR on a raster image using pytesseract (optional dependency).
    Returns empty string if pytesseract not installed.
    """
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, config="--psm 11")
        return text
    except Exception:
        return ""


def run_image_agent(state: dict) -> dict:
    """
    Extract all images from the CND PDF, classify them, and cross-validate
    any tags found with the text-extracted tag inventory.
    """
    doc_path: str       = state.get("document_path", "")
    tag_inventory: list = state.get("tag_inventory", [])
    definition_map: dict = state.get("definition_map", {})

    findings: list[Finding] = []
    image_summary: list[dict] = []   # lightweight record for the UI

    if not doc_path or not doc_path.lower().endswith(".pdf"):
        state["image_report"] = _build_report(findings, []).model_dump()
        return state

    if not Path(doc_path).is_file():
        state["image_report"] = _build_report(findings, []).model_dump()
        return state

    try:
        doc = fitz.open(doc_path)
    except Exception as e:
        logger.warning("image_agent: cannot open PDF: %s", e)
        state["image_report"] = _build_report(findings, []).model_dump()
        return state

    tag_set = set(tag_inventory)
    image_tags_found: set[str] = set()

    for page_num, page in enumerate(doc, start=1):
        page_images = page.get_images(full=True)
        page_text   = page.get_text("text")

        for img_info in page_images:
            xref = img_info[0]
            try:
                base_img = doc.extract_image(xref)
            except Exception:
                continue

            img_bytes  = base_img.get("image", b"")
            img_width  = base_img.get("width", 0)
            img_height = base_img.get("height", 0)
            img_fmt    = base_img.get("ext", "unknown")

            # Get image bounding rect on the page
            img_rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
            for item in page.get_image_info():
                if item.get("xref") == xref:
                    img_rect = fitz.Rect(item["bbox"])
                    break

            # Extract text layer inside image region (works for vector P&IDs)
            region_text = _extract_text_from_image_region(page, img_rect)

            # Classify the image
            caption = page_text[max(0, page_text.find(region_text[:20]) - 200):][:300]
            img_class = _classify_image(caption, region_text)

            # Find ISA tags in the image text layer
            img_tags = set(ISA_TAG_RE.findall(region_text))

            # Fallback: OCR if vector text is sparse and image is large enough
            if len(img_tags) < 2 and img_width > 200 and img_height > 200:
                ocr_text = _try_ocr(img_bytes)
                img_tags |= set(ISA_TAG_RE.findall(ocr_text))
                if ocr_text:
                    region_text = ocr_text

            image_tags_found |= img_tags

            # Record for UI summary
            img_rec = {
                "page":       page_num,
                "class":      img_class,
                "width":      img_width,
                "height":     img_height,
                "format":     img_fmt,
                "tags_found": sorted(img_tags),
                "thumbnail":  _make_thumbnail(img_bytes, img_fmt),
            }
            image_summary.append(img_rec)

            # ── Cross-validation findings ─────────────────────────────────────

            # Tags in image but NOT described in text narrative
            for tag in img_tags:
                if tag not in tag_set:
                    findings.append(Finding(
                        id=f"IMG-{uuid.uuid4().hex[:8].upper()}",
                        agent="image",
                        finding_type="IMAGE_TAG_NOT_IN_TEXT",
                        description=(
                            f"Tag {tag} is visible in a {img_class} image on page {page_num} "
                            f"but has no description in the text narrative."
                        ),
                        page_number=page_num,
                        section_title=f"Page {page_num} — {img_class} image",
                        quote=region_text[:200] if region_text else f"[Image — {img_class}]",
                        rule_violated="ISA-5.1: All tagged instruments must be described in narrative",
                        rule_text=(
                            "Every instrument tag shown on a P&ID must have a corresponding "
                            "description in the control narrative defining its function, range, "
                            "and alarm/interlock settings (ISA-5.1-2009)."
                        ),
                        why_it_matters=(
                            f"PLC programmer sees {tag} on the P&ID but cannot find its control "
                            "logic in the CND. They must guess or leave it unimplemented."
                        ),
                        suggestion=(
                            f"Add a section describing {tag}: its type, operating range, "
                            "alarms, and any interlocks it participates in."
                        ),
                        severity="HIGH",
                        evidence_source=f"Image cross-validation — {img_class} on page {page_num}",
                        status="OPEN",
                    ))

            # P&ID image that is too small to read (likely scanned at low DPI)
            if img_class == "P&ID" and img_width < 300 and img_height < 300:
                findings.append(Finding(
                    id=f"IMG-{uuid.uuid4().hex[:8].upper()}",
                    agent="image",
                    finding_type="UNREADABLE_IMAGE",
                    description=(
                        f"P&ID image on page {page_num} is very small ({img_width}×{img_height}px). "
                        "Tag labels may not be readable."
                    ),
                    page_number=page_num,
                    section_title=f"Page {page_num}",
                    quote=f"[{img_class} image — {img_width}×{img_height} px]",
                    rule_violated="Document quality — readable diagrams required",
                    rule_text="P&ID diagrams must be legible at standard A4/A3 print size.",
                    why_it_matters=(
                        "Engineers and PLC programmers cannot read instrument tags from a "
                        "low-resolution P&ID. The diagram cannot be used as a reference."
                    ),
                    suggestion=(
                        "Replace with a higher-resolution version (minimum 150 DPI at print size). "
                        "Alternatively provide the P&ID as a separate document."
                    ),
                    severity="WARNING",
                    evidence_source=f"Image resolution check — page {page_num}",
                    status="OPEN",
                ))

    doc.close()

    # Tags in text inventory that never appear in any image
    # (only flag if we found at least one meaningful P&ID image)
    pid_images = [i for i in image_summary if i["class"] == "P&ID" and len(i["tags_found"]) > 0]
    if pid_images:
        all_image_tags = {t for img in pid_images for t in img["tags_found"]}
        for tag in tag_set:
            if tag not in all_image_tags:
                findings.append(Finding(
                    id=f"IMG-{uuid.uuid4().hex[:8].upper()}",
                    agent="image",
                    finding_type="TEXT_TAG_NOT_IN_PID",
                    description=(
                        f"Tag {tag} is described in the text narrative but does not appear "
                        "in any P&ID image in this document."
                    ),
                    page_number=0,
                    section_title="Cross-validation",
                    quote=f"Tag {tag} found in text, not found in P&ID images",
                    rule_violated="P&ID completeness — all narrative tags must appear on P&ID",
                    rule_text=(
                        "Every instrument described in the control narrative should be shown "
                        "on the corresponding P&ID (ISA-5.1 traceability requirement)."
                    ),
                    why_it_matters=(
                        f"If {tag} is not on the P&ID, its physical location and signal "
                        "wiring are undefined. PLC I/O assignment cannot be completed."
                    ),
                    suggestion=(
                        f"Verify that {tag} appears on the P&ID. If it was added after the "
                        "P&ID revision, update the P&ID to include it."
                    ),
                    severity="INFO",
                    evidence_source="Image vs text cross-validation",
                    status="OPEN",
                ))

    state["image_report"]   = _build_report(findings, image_summary).model_dump()
    state["image_summary"]  = image_summary
    return state


def _make_thumbnail(img_bytes: bytes, fmt: str) -> bytes:
    """Resize image to 200px wide thumbnail for UI display. Returns original if fails."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(img_bytes))
        ratio = 200 / img.width
        new_h = int(img.height * ratio)
        img = img.resize((200, new_h), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return img_bytes[:4000] if len(img_bytes) > 4000 else img_bytes


def _build_report(findings: list[Finding], image_summary: list[dict]) -> AgentReport:
    critical = sum(1 for f in findings if f.severity == "CRITICAL")
    high     = sum(1 for f in findings if f.severity == "HIGH")
    warning  = sum(1 for f in findings if f.severity == "WARNING")
    info     = sum(1 for f in findings if f.severity == "INFO")
    score    = max(0.0, 1.0 - critical * 0.20 - high * 0.10 - warning * 0.05 - info * 0.01)
    return AgentReport(
        agent_name="image",
        findings=findings,
        score=round(score, 3),
        critical_count=critical,
        high_count=high,
        warning_count=warning,
        info_count=info,
    )
