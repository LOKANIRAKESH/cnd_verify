"""
generate_synthetic_pdfs.py
Generates 3 realistic pump station CND PDF documents with:
  - Cover page (document number, revision, date)
  - P&ID schematic (vector drawing — real image in the PDF)
  - Equipment list table
  - Setpoints table
  - Alarms table
  - All CND sections with proper formatting and page numbers

Variants:
  pump_station_01_clean.pdf         — no errors
  pump_station_02_completeness.pdf  — 6 injected completeness errors
  pump_station_03_consistency.pdf   — 4 injected consistency errors

Run: python generate_synthetic_pdfs.py
"""

import os
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether, ListFlowable, ListItem,
)
from reportlab.graphics.shapes import (
    Drawing, Rect, Circle, Line, String, Polygon, Group,
)
from reportlab.graphics import renderPDF
from reportlab.platypus.flowables import Flowable


W, H = A4  # 595 x 842 points


# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=22, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1e293b"), spaceAfter=8, alignment=TA_CENTER,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=13, fontName="Helvetica",
        textColor=colors.HexColor("#475569"), spaceAfter=6, alignment=TA_CENTER,
    )
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta", fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#64748b"), spaceAfter=4, alignment=TA_CENTER,
    )
    styles["h1"] = ParagraphStyle(
        "h1", fontSize=14, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a"), spaceBefore=16, spaceAfter=6,
        borderPad=4,
    )
    styles["h2"] = ParagraphStyle(
        "h2", fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1e3a5f"), spaceBefore=10, spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body", fontSize=9.5, fontName="Helvetica",
        textColor=colors.HexColor("#1e293b"), spaceAfter=5, leading=14,
        alignment=TA_JUSTIFY,
    )
    styles["caption"] = ParagraphStyle(
        "caption", fontSize=8.5, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#64748b"), spaceAfter=4, alignment=TA_CENTER,
    )
    styles["table_header"] = ParagraphStyle(
        "table_header", fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    styles["table_cell"] = ParagraphStyle(
        "table_cell", fontSize=8.5, fontName="Helvetica",
        textColor=colors.HexColor("#1e293b"),
    )
    styles["note"] = ParagraphStyle(
        "note", fontSize=8.5, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#7c3aed"), spaceAfter=4,
        leftIndent=10, borderPad=3,
    )
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# HEADER / FOOTER on every page
# ─────────────────────────────────────────────────────────────────────────────

def make_page_template(doc_number, title, revision):
    def _draw(canvas, doc):
        canvas.saveState()
        # Header bar
        canvas.setFillColor(colors.HexColor("#1e3a5f"))
        canvas.rect(cm, H - 1.6 * cm, W - 2 * cm, 0.9 * cm, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.white)
        canvas.drawString(1.4 * cm, H - 1.2 * cm, title.upper())
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W - 1.4 * cm, H - 1.2 * cm,
                               f"{doc_number}  |  Rev {revision}")
        # Footer
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(cm, 0.7 * cm, "CONFIDENTIAL — FOR INTERNAL USE ONLY")
        canvas.drawRightString(W - cm, 0.7 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
        canvas.setLineWidth(0.5)
        canvas.line(cm, 1.1 * cm, W - cm, 1.1 * cm)
        canvas.restoreState()
    return _draw


# ─────────────────────────────────────────────────────────────────────────────
# P&ID SCHEMATIC  (vector drawing — real image embedded in PDF)
# ─────────────────────────────────────────────────────────────────────────────

class PIDSchematic(Flowable):
    """Draws a pump station P&ID as a vector graphic."""

    def __init__(self, width=460, height=280, show_standby=True):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.show_standby = show_standby

    def draw(self):
        c = self.canv
        c.saveState()

        BLUE  = colors.HexColor("#1e3a5f")
        GRAY  = colors.HexColor("#94a3b8")
        LBLUE = colors.HexColor("#bfdbfe")
        BLACK = colors.HexColor("#0f172a")
        RED   = colors.HexColor("#dc2626")
        GREEN = colors.HexColor("#166534")

        def tag(x, y, text, color=BLUE, size=7):
            c.setFont("Helvetica-Bold", size)
            c.setFillColor(color)
            c.drawCentredString(x, y, text)

        def label(x, y, text, size=7):
            c.setFont("Helvetica", size)
            c.setFillColor(GRAY)
            c.drawCentredString(x, y, text)

        def pipe(x1, y1, x2, y2):
            c.setStrokeColor(BLACK)
            c.setLineWidth(1.5)
            c.line(x1, y1, x2, y2)

        def instrument_bubble(x, y, tag_text, sub="", fill=LBLUE):
            """ISA instrument bubble: circle with tag inside."""
            c.setStrokeColor(BLUE)
            c.setFillColor(fill)
            c.setLineWidth(1)
            c.circle(x, y, 14, fill=1)
            c.setFont("Helvetica-Bold", 7)
            c.setFillColor(BLUE)
            c.drawCentredString(x, y + 2, tag_text[:6])
            if sub:
                c.setFont("Helvetica", 6)
                c.drawCentredString(x, y - 5, sub)

        def vessel(x, y, w, h, label_text):
            """Rectangle for vessel/tank."""
            c.setStrokeColor(BLUE)
            c.setFillColor(colors.HexColor("#f0f9ff"))
            c.setLineWidth(1.5)
            c.rect(x - w/2, y - h/2, w, h, fill=1)
            c.setFont("Helvetica-Bold", 8)
            c.setFillColor(BLUE)
            c.drawCentredString(x, y + 4, label_text)

        def pump(x, y, label_text):
            """ISA pump symbol: circle."""
            c.setStrokeColor(BLUE)
            c.setFillColor(colors.HexColor("#eff6ff"))
            c.setLineWidth(1.5)
            c.circle(x, y, 18, fill=1)
            c.setFont("Helvetica-Bold", 8)
            c.setFillColor(BLUE)
            c.drawCentredString(x, y + 3, label_text[:5])
            c.setFont("Helvetica", 7)
            c.setFillColor(GRAY)
            c.drawCentredString(x, y - 6, label_text[5:] if len(label_text) > 5 else "")

        def gate_valve(x, y, horizontal=False):
            """Gate valve symbol: two triangles."""
            c.setStrokeColor(BLUE)
            c.setFillColor(LBLUE)
            c.setLineWidth(1)
            size = 7
            if horizontal:
                pts = [x-size, y-size, x-size, y+size, x, y,
                       x+size, y+size, x+size, y-size, x, y]
            else:
                pts = [x-size, y+size, x+size, y+size, x, y,
                       x-size, y-size, x+size, y-size, x, y]
            c.setFillColor(LBLUE)
            path = c.beginPath()
            path.moveTo(pts[0], pts[1])
            for i in range(2, len(pts), 2):
                path.lineTo(pts[i], pts[i+1])
            path.close()
            c.drawPath(path, fill=1, stroke=1)

        def control_valve(x, y):
            """Control valve: circle with X."""
            c.setStrokeColor(BLUE)
            c.setFillColor(colors.HexColor("#fef9c3"))
            c.setLineWidth(1)
            c.circle(x, y, 8, fill=1)
            c.setStrokeColor(BLUE)
            c.line(x-5, y-5, x+5, y+5)
            c.line(x+5, y-5, x-5, y+5)

        # ── Layout coordinates ────────────────────────────────────────────────
        # Suction vessel top
        VES_X, VES_Y = 230, 250

        # Two pump positions
        P1_X, P1_Y = 170, 150
        P2_X, P2_Y = 290, 150

        # Discharge merge point
        DISC_X, DISC_Y = 230, 85

        # ── Draw background grid (faint) ──────────────────────────────────────
        c.setStrokeColor(colors.HexColor("#f1f5f9"))
        c.setLineWidth(0.3)
        for gx in range(0, int(self.width), 40):
            c.line(gx, 0, gx, self.height)
        for gy in range(0, int(self.height), 40):
            c.line(0, gy, self.width, gy)

        # ── Suction vessel T-100 ──────────────────────────────────────────────
        vessel(VES_X, VES_Y, 100, 50, "T-100")
        label(VES_X, VES_Y - 8, "SUCTION VESSEL")
        # LT-301 bubble on vessel
        pipe(VES_X + 50, VES_Y + 5, VES_X + 72, VES_Y + 5)
        instrument_bubble(VES_X + 86, VES_Y + 5, "LT", "301")

        # ── Suction header ────────────────────────────────────────────────────
        pipe(VES_X, VES_Y - 25, VES_X, VES_Y - 55)     # down from vessel
        gate_valve(VES_X, VES_Y - 57)                    # XV-101
        tag(VES_X + 15, VES_Y - 62, "XV-101", BLUE, 7)
        pipe(VES_X, VES_Y - 67, VES_X, VES_Y - 90)     # continue down

        # Split to P-101 and P-102
        pipe(VES_X, VES_Y - 90, P1_X, VES_Y - 90)      # to P-101 side
        pipe(VES_X, VES_Y - 90, P2_X, VES_Y - 90)      # to P-102 side
        pipe(P1_X, VES_Y - 90, P1_X, P1_Y + 18)        # down to P-101
        pipe(P2_X, VES_Y - 90, P2_X, P2_Y + 18)        # down to P-102

        # Suction valves
        gate_valve(P1_X, VES_Y - 90 + 3, horizontal=False)
        tag(P1_X - 20, VES_Y - 88, "XV-101A", GRAY, 6)
        gate_valve(P2_X, VES_Y - 90 + 3, horizontal=False)
        tag(P2_X + 22, VES_Y - 88, "XV-101B", GRAY, 6)

        # ── Pumps ─────────────────────────────────────────────────────────────
        pump(P1_X, P1_Y, "P-101")
        tag(P1_X, P1_Y - 25, "DUTY", GREEN, 7)

        if self.show_standby:
            pump(P2_X, P2_Y, "P-102")
            tag(P2_X, P2_Y - 25, "STANDBY", GRAY, 7)
        else:
            pump(P2_X, P2_Y, "P-102")
            tag(P2_X, P2_Y - 25, "STANDBY", GRAY, 7)

        # TT-401 on P-101 (temperature on bearing)
        pipe(P1_X + 18, P1_Y + 5, P1_X + 36, P1_Y + 12)
        instrument_bubble(P1_X + 50, P1_Y + 12, "TT", "401",
                          fill=colors.HexColor("#fef3c7"))

        # ── Discharge headers ─────────────────────────────────────────────────
        pipe(P1_X, P1_Y - 18, P1_X, DISC_Y + 30)
        pipe(P2_X, P2_Y - 18, P2_X, DISC_Y + 30)
        pipe(P1_X, DISC_Y + 30, DISC_X, DISC_Y + 30)
        pipe(P2_X, DISC_Y + 30, DISC_X, DISC_Y + 30)

        # Discharge isolation valves
        gate_valve(P1_X, P1_Y - 20)
        tag(P1_X - 22, P1_Y - 22, "XV-102A", GRAY, 6)
        gate_valve(P2_X, P2_Y - 20)
        tag(P2_X + 22, P2_Y - 22, "XV-102B", GRAY, 6)

        # Main discharge pipe down
        pipe(DISC_X, DISC_Y + 30, DISC_X, DISC_Y)

        # XV-201 (main discharge valve)
        gate_valve(DISC_X, DISC_Y)
        tag(DISC_X + 22, DISC_Y, "XV-201", BLUE, 7)

        # PT-101 on discharge
        pipe(DISC_X, DISC_Y - 8, DISC_X, DISC_Y - 25)
        pipe(DISC_X, DISC_Y - 25, DISC_X + 40, DISC_Y - 25)
        instrument_bubble(DISC_X + 54, DISC_Y - 25, "PT", "101",
                          fill=colors.HexColor("#fce7f3"))

        # FT-201 flow transmitter
        pipe(DISC_X, DISC_Y - 35, DISC_X, DISC_Y - 50)
        instrument_bubble(DISC_X + 40, DISC_Y - 45, "FT", "201",
                          fill=colors.HexColor("#dcfce7"))
        pipe(DISC_X + 14, DISC_Y - 45, DISC_X + 26, DISC_Y - 45)

        # FV-301 control valve
        pipe(DISC_X, DISC_Y - 60, DISC_X, DISC_Y - 72)
        control_valve(DISC_X, DISC_Y - 72)
        tag(DISC_X + 22, DISC_Y - 72, "FV-301", BLUE, 7)

        # FIC-201 controller (dashed line to FV-301)
        instrument_bubble(DISC_X + 54, DISC_Y - 72, "FIC", "201",
                          fill=colors.HexColor("#dcfce7"))
        c.setDash(3, 3)
        pipe(DISC_X + 8, DISC_Y - 72, DISC_X + 40, DISC_Y - 72)
        c.setDash()

        # Discharge to battery limit
        pipe(DISC_X, DISC_Y - 80, DISC_X, DISC_Y - 95)
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(BLACK)
        c.drawCentredString(DISC_X, DISC_Y - 108, "TO PROCESS ▶")

        # ── Legend box ────────────────────────────────────────────────────────
        c.setStrokeColor(GRAY)
        c.setFillColor(colors.white)
        c.setLineWidth(0.5)
        c.rect(360, 5, 98, 110, fill=1)
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(BLUE)
        c.drawString(363, 107, "LEGEND")
        c.setLineWidth(0.5)
        c.setStrokeColor(GRAY)
        c.line(360, 104, 458, 104)

        legend_items = [
            ("●", LBLUE, "Instrument bubble"),
            ("○", colors.HexColor("#eff6ff"), "Pump"),
            ("◇", LBLUE, "Isolation valve"),
            ("⊕", colors.HexColor("#fef9c3"), "Control valve"),
            ("- -", GRAY, "Signal line"),
        ]
        for i, (sym, col, txt) in enumerate(legend_items):
            y_pos = 93 - i * 16
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(col if col != GRAY else GRAY)
            c.drawString(366, y_pos, sym)
            c.setFont("Helvetica", 7)
            c.setFillColor(GRAY)
            c.drawString(381, y_pos, txt)

        # ── Title block ───────────────────────────────────────────────────────
        c.setStrokeColor(GRAY)
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setLineWidth(0.5)
        c.rect(0, 0, self.width, 18, fill=1)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(BLUE)
        c.drawString(4, 6, "FIG. 1 — PUMP STATION PS-01 P&ID SCHEMATIC (SIMPLIFIED)")
        c.setFont("Helvetica", 7)
        c.setFillColor(GRAY)
        c.drawRightString(self.width - 4, 6, "NOT TO SCALE — FOR NARRATIVE REFERENCE ONLY")

        c.restoreState()

    def wrap(self, availWidth, availHeight):
        return self.width, self.height


# ─────────────────────────────────────────────────────────────────────────────
# TABLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

HEADER_BG = colors.HexColor("#1e3a5f")
ROW_BG_A  = colors.HexColor("#f8fafc")
ROW_BG_B  = colors.white

def make_table(headers, rows, col_widths=None):
    styles_map = make_styles()
    h_cells = [Paragraph(h, styles_map["table_header"]) for h in headers]
    data = [h_cells]
    for i, row in enumerate(rows):
        cells = [Paragraph(str(c), styles_map["table_cell"]) for c in row]
        data.append(cells)

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [ROW_BG_A, ROW_BG_B]),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8.5),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT CONTENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_cnd_01_clean(styles):
    """Pump Station PS-01 — complete, no errors."""
    story = []
    S = styles

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("CONTROL NARRATIVE DOCUMENT", S["cover_title"]))
    story.append(Paragraph("Pump Station PS-01", S["cover_sub"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#1e3a5f"),
                             hAlign="CENTER"))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Document Number: CND-PS01-001", S["cover_meta"]))
    story.append(Paragraph("Revision: A", S["cover_meta"]))
    story.append(Paragraph(f"Date: {date.today().strftime('%d %B %Y')}", S["cover_meta"]))
    story.append(Paragraph("Status: APPROVED FOR DESIGN", S["cover_meta"]))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("Prepared by: Instrumentation &amp; Control Engineering", S["cover_meta"]))
    story.append(Paragraph("Reviewed by: Process Engineering", S["cover_meta"]))
    story.append(PageBreak())

    # ── SECTION 1: GENERAL DESCRIPTION ───────────────────────────────────────
    story.append(Paragraph("1. GENERAL DESCRIPTION", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Pump Station PS-01 is designed to transfer process fluid from the suction vessel "
        "T-100 to the downstream process unit at a controlled flow rate. The station "
        "comprises two centrifugal pumps operating in duty/standby configuration, "
        "with automatic changeover on fault detection. The flow is regulated by control "
        "valve FV-301 based on the flow controller FIC-201 setpoint.",
        S["body"]
    ))
    story.append(Paragraph(
        "The system is designed in accordance with ISA-88 batch control standards, "
        "ISA-18.2 alarm management requirements, and ISA-84 safety instrumented system "
        "guidelines. The Programmable Logic Controller (PLC) shall implement all control "
        "logic defined in this document.",
        S["body"]
    ))

    # ── P&ID SCHEMATIC ────────────────────────────────────────────────────────
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("1.1  P&amp;ID Schematic", S["h2"]))
    story.append(PIDSchematic(width=460, height=290))
    story.append(Spacer(1, 3 * mm))

    # ── SECTION 2: EQUIPMENT LIST ─────────────────────────────────────────────
    story.append(Paragraph("2. EQUIPMENT LIST", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))
    equip_headers = ["Tag", "Description", "Type", "Range / Rating", "Normal State"]
    equip_rows = [
        ["P-101",   "Duty pump",               "Centrifugal pump",      "0–150 m³/h, 11 kW",    "Running"],
        ["P-102",   "Standby pump",             "Centrifugal pump",      "0–150 m³/h, 11 kW",    "Stopped"],
        ["XV-101",  "Suction isolation valve",  "Pneumatic gate valve",  "DN150, PN16",           "Open"],
        ["XV-102",  "Discharge isolation valve","Pneumatic gate valve",  "DN150, PN16",           "Open"],
        ["XV-201",  "Main discharge valve",     "Pneumatic gate valve",  "DN200, PN16",           "Open"],
        ["FV-301",  "Flow control valve",       "Pneumatic globe valve", "Cv 120, fail-open",     "Auto"],
        ["PT-101",  "Discharge pressure",       "Pressure transmitter",  "0–300 PSI, 4–20 mA",   "Active"],
        ["FT-201",  "Process flow",             "Magnetic flow transmitter","0–200 L/min, 4–20 mA","Active"],
        ["LT-301",  "Suction vessel level",     "Radar level transmitter","0–5 m, 4–20 mA",       "Active"],
        ["TT-401",  "Pump bearing temperature", "RTD temperature transmitter","0–150 °C, 4–20 mA","Active"],
        ["FIC-201", "Flow controller",          "PID controller (in PLC)","SP: 80 L/min",         "Auto"],
    ]
    story.append(make_table(equip_headers, equip_rows,
                            col_widths=[1.5*cm, 4*cm, 4.5*cm, 4*cm, 2.5*cm]))
    story.append(PageBreak())

    # ── SECTION 3: OPERATING MODES ────────────────────────────────────────────
    story.append(Paragraph("3. OPERATING MODES", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))

    modes = [
        ("STOPPED", "Both pumps de-energised. XV-201 closed. FV-301 in fail-open position. "
         "All alarms monitored but no control action taken. Entry: operator command or ESD activation. "
         "Exit: operator initiates STARTUP sequence."),
        ("STARTING", "Startup sequence executing. XV-101 opening sequence in progress. "
         "P-101 starting. XV-201 opening. Duration: up to 120 seconds. "
         "Exit: all permissives confirmed → RUNNING."),
        ("RUNNING", "P-101 running (duty). FIC-201 controlling FV-301 to maintain 80 L/min setpoint. "
         "P-102 on standby auto-start. All interlocks active. "
         "Exit: operator initiates SHUTDOWN or fault triggers ESD."),
        ("STANDBY", "P-101 stopped due to fault. P-102 auto-started. FIC-201 continues control. "
         "Alarm generated for P-101 fault. "
         "Exit: operator acknowledges fault, manually restores P-101."),
        ("STOPPING", "Shutdown sequence executing. FV-301 closing. P-101 stopping. "
         "XV-201 closing. Duration: up to 60 seconds. Exit: sequence complete → STOPPED."),
        ("EMERGENCY SHUTDOWN", "All pumps stopped immediately. XV-201 closed. FV-301 de-energised "
         "to fail-open. All interlocks latched. Manual reset required. "
         "Entry: any ESD interlock condition."),
    ]
    for mode, desc in modes:
        story.append(Paragraph(f"3.{modes.index((mode,desc))+1}  {mode}", S["h2"]))
        story.append(Paragraph(desc, S["body"]))

    story.append(PageBreak())

    # ── SECTION 4: SETPOINTS ──────────────────────────────────────────────────
    story.append(Paragraph("4. SETPOINTS TABLE", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))
    sp_headers = ["Tag", "Parameter", "Value", "Units", "Page Ref"]
    sp_rows = [
        ["PT-101", "High High Trip (HH)",    "150",  "PSI",    "This page"],
        ["PT-101", "High Alarm (H)",          "130",  "PSI",    "This page"],
        ["PT-101", "Low Alarm (L)",           "20",   "PSI",    "This page"],
        ["FT-201", "Low Flow Trip (LL)",      "10",   "L/min",  "This page"],
        ["FT-201", "Low Alarm (L)",           "20",   "L/min",  "This page"],
        ["FT-201", "Flow Control SP",         "80",   "L/min",  "This page"],
        ["LT-301", "Low Low Trip (LL)",       "15",   "%",      "This page"],
        ["LT-301", "Low Alarm (L)",           "25",   "%",      "This page"],
        ["TT-401", "High High Trip (HH)",     "85",   "°C",     "This page"],
        ["TT-401", "High Alarm (H)",          "75",   "°C",     "This page"],
        ["FV-301", "Fail Position",           "Open", "—",      "This page"],
    ]
    story.append(make_table(sp_headers, sp_rows,
                            col_widths=[2*cm, 5.5*cm, 2*cm, 2*cm, 5*cm]))

    story.append(Spacer(1, 8 * mm))

    # ── SECTION 5: INTERLOCKS ─────────────────────────────────────────────────
    story.append(Paragraph("5. INTERLOCKS", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "The following safety interlocks shall be implemented in the PLC. "
        "All safety interlocks require manual reset by the operator after the initiating "
        "condition has cleared and the root cause has been investigated.", S["body"]
    ))
    story.append(Spacer(1, 3 * mm))

    interlocks = [
        {
            "id": "IL-101", "desc": "High High Discharge Pressure Trip",
            "cause": "PT-101 discharge pressure exceeds 150 PSI for more than 3 seconds",
            "action": "Stop pump P-101 AND stop pump P-102 AND close XV-201 AND generate alarm 'HIGH HIGH PRESSURE TRIP' on HMI",
            "reset_cond": "PT-101 pressure falls below 120 PSI AND operator has investigated root cause",
            "reset_method": "Manual reset — operator must press RESET on HMI Panel",
            "priority": "Safety Critical",
        },
        {
            "id": "IL-201", "desc": "Low Low Flow Trip",
            "cause": "FT-201 flow rate falls below 10 L/min for more than 5 seconds while pump running",
            "action": "Stop pump P-101 AND auto-start standby pump P-102 AND generate alarm 'LOW FLOW TRIP'",
            "reset_cond": "FT-201 flow rate above 20 L/min AND P-102 confirmed running",
            "reset_method": "Automatic reset after 30 seconds if P-102 restores flow",
            "priority": "High",
        },
        {
            "id": "IL-301", "desc": "Low Low Suction Level Trip",
            "cause": "LT-301 suction vessel level falls below 15% for more than 10 seconds",
            "action": "Stop pump P-101 AND stop pump P-102 AND close XV-101 AND generate alarm 'LOW SUCTION LEVEL'",
            "reset_cond": "LT-301 level above 25% AND operator confirms feed restored",
            "reset_method": "Manual reset — operator must press RESET on HMI Panel",
            "priority": "Safety Critical",
        },
    ]

    il_headers = ["Field", "Content"]
    for il in interlocks:
        story.append(Paragraph(f"Interlock {il['id']} — {il['desc']}", S["h2"]))
        il_rows = [
            ["Cause (Initiating Condition)", il["cause"]],
            ["Action",                       il["action"]],
            ["Reset Condition",              il["reset_cond"]],
            ["Reset Method",                 il["reset_method"]],
            ["Priority",                     il["priority"]],
        ]
        story.append(make_table(il_headers, il_rows, col_widths=[4.5*cm, 12*cm]))
        story.append(Spacer(1, 4 * mm))

    story.append(PageBreak())

    # ── SECTION 6: ALARMS ─────────────────────────────────────────────────────
    story.append(Paragraph("6. ALARMS", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))
    al_headers = ["Tag", "Type", "Setpoint", "Units", "Priority", "Operator Response"]
    al_rows = [
        ["PT-101-HH", "HH Trip",    "150", "PSI",   "Priority 1", "Stop all pumps. Close XV-201. Investigate overpressure cause. Do not restart until root cause confirmed."],
        ["PT-101-H",  "H Alarm",    "130", "PSI",   "Priority 2", "Reduce flow setpoint on FIC-201 by 20%. Monitor PT-101 trend. Prepare for potential trip."],
        ["PT-101-L",  "L Alarm",    "20",  "PSI",   "Priority 3", "Check pump operation. Verify XV-201 and XV-201 are fully open. Inspect for downstream blockage."],
        ["FT-201-L",  "L Alarm",    "20",  "L/min", "Priority 2", "Check FV-301 position. Verify pump P-101 is running at correct speed. Check for line blockage."],
        ["LT-301-L",  "L Alarm",    "25",  "%",     "Priority 2", "Contact feed operations to restore suction vessel level. Prepare for pump shutdown if level continues falling."],
        ["TT-401-H",  "H Alarm",    "75",  "°C",    "Priority 2", "Reduce pump load or switch to P-102. Check bearing lubrication. Schedule P-101 maintenance inspection."],
    ]
    story.append(make_table(al_headers, al_rows,
                            col_widths=[2.2*cm, 1.8*cm, 1.8*cm, 1.5*cm, 2.2*cm, 7*cm]))

    story.append(PageBreak())

    # ── SECTION 7: STARTUP SEQUENCE ───────────────────────────────────────────
    story.append(Paragraph("7. STARTUP SEQUENCE", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "The following sequence shall be executed by the PLC when the operator initiates "
        "STARTUP from the HMI. Each step includes a verification check and a maximum "
        "timeout. If a step times out without confirmation, the sequence shall abort "
        "and generate an alarm.", S["body"]
    ))
    story.append(Spacer(1, 3 * mm))

    startup_steps = [
        ("Step 1", "Check Permissives",
         "Verify: LT-301 level > 25%, PT-101 < 100 PSI, no active ESD conditions. "
         "Timeout: N/A (permissive check only). Abort if any condition fails."),
        ("Step 2", "Open Suction Valve XV-101",
         "Command XV-101 to OPEN. Verify: XV-101 open limit switch confirmed within 15 seconds. "
         "Timeout: 15 seconds. Abort sequence and alarm if not confirmed."),
        ("Step 3", "Prime Pump P-101",
         "Wait 5 seconds after XV-101 confirmed open to allow pump casing to fill."),
        ("Step 4", "Start Duty Pump P-101",
         "Command P-101 motor starter to START. Verify: P-101 running feedback confirmed within 10 seconds. "
         "Timeout: 10 seconds."),
        ("Step 5", "Open Discharge Valve XV-201",
         "Command XV-201 to OPEN. Verify: XV-201 open limit switch confirmed within 15 seconds. "
         "Timeout: 15 seconds."),
        ("Step 6", "Confirm Flow Established",
         "Verify: FT-201 > 20 L/min within 30 seconds of XV-201 open. "
         "Timeout: 30 seconds. If no flow — stop P-101, close XV-201, alarm 'NO FLOW ON STARTUP'."),
        ("Step 7", "Enable Flow Controller",
         "Enable FIC-201 in AUTO mode with setpoint 80 L/min. Ramp FV-301 from current position "
         "to target over 10 seconds to avoid water hammer."),
        ("Step 8", "Enable Standby Auto-Start",
         "Enable P-102 auto-start logic. P-102 will auto-start if P-101 trips and flow drops below 15 L/min."),
        ("Step 9", "Startup Complete",
         "Set operating mode to RUNNING. Log startup timestamp to audit trail. "
         "Generate event: 'STARTUP COMPLETE — PS-01 IN NORMAL OPERATION'."),
    ]
    ss_headers = ["Step", "Action", "Details / Verification"]
    ss_rows = [[s, a, d] for s, a, d in startup_steps]
    story.append(make_table(ss_headers, ss_rows, col_widths=[1.5*cm, 4*cm, 11*cm]))

    story.append(PageBreak())

    # ── SECTION 8: SHUTDOWN SEQUENCE ──────────────────────────────────────────
    story.append(Paragraph("8. NORMAL SHUTDOWN SEQUENCE", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))

    shutdown_steps = [
        ("Step 1", "Disable Auto-Start", "Disable P-102 auto-start logic."),
        ("Step 2", "Ramp Down Flow",     "Reduce FIC-201 setpoint from 80 to 20 L/min over 15 seconds."),
        ("Step 3", "Close FV-301",       "Switch FIC-201 to MANUAL. Close FV-301 fully over 10 seconds."),
        ("Step 4", "Close XV-201",       "Command XV-201 to CLOSE. Verify closed within 15 seconds."),
        ("Step 5", "Stop Pump P-101",    "Command P-101 to STOP after XV-201 confirmed CLOSED. "
                                         "Verify P-101 stop feedback within 5 seconds."),
        ("Step 6", "Close XV-101",       "Command XV-101 to CLOSE. Verify closed within 15 seconds."),
        ("Step 7", "Shutdown Complete",  "Set mode to STOPPED. Log shutdown timestamp."),
    ]
    sd_rows = [[s, a, d] for s, a, d in shutdown_steps]
    story.append(make_table(ss_headers, sd_rows, col_widths=[1.5*cm, 4*cm, 11*cm]))

    story.append(Spacer(1, 8 * mm))

    # ── SECTION 9: EMERGENCY SHUTDOWN ────────────────────────────────────────
    story.append(Paragraph("9. EMERGENCY SHUTDOWN (ESD)", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "The Emergency Shutdown sequence is triggered automatically by any ESD interlock condition "
        "or manually by the operator pressing the ESD pushbutton on the panel.", S["body"]
    ))
    story.append(Spacer(1, 3 * mm))
    esd_steps = [
        ("ESD-1", "Stop all pumps",      "De-energise P-101 AND P-102 motor starters simultaneously."),
        ("ESD-2", "Close XV-201",        "Command XV-201 to CLOSE immediately. Hardwired signal bypasses PLC."),
        ("ESD-3", "Close XV-101",        "Command XV-101 to CLOSE to isolate suction."),
        ("ESD-4", "De-energise FV-301",  "FV-301 de-energised — moves to fail-open position automatically."),
        ("ESD-5", "Latch ESD state",     "Set ESD latch bit. All normal control commands ignored until reset."),
        ("ESD-6", "Generate alarms",     "Activate all ESD alarms on HMI and initiate audible alarm."),
    ]
    esd_rows = [[s, a, d] for s, a, d in esd_steps]
    story.append(make_table(ss_headers, esd_rows, col_widths=[1.5*cm, 4*cm, 11*cm]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "NOTE: Manual reset of ESD latch requires operator to: (1) confirm all ESD conditions cleared, "
        "(2) investigate and document root cause, (3) press RESET on the PLC panel.", S["note"]
    ))

    return story


def build_cnd_02_completeness(styles):
    """Pump Station PS-01 — with 6 completeness errors injected."""
    story = build_cnd_01_clean(styles)

    # Remove ESD section (C1) — replace last section with note
    # This is done by generating with ESD missing from the mode list
    # and without Section 9 (we rebuild the story without ESD)
    # For simplicity: regenerate but skip ESD section and startup section

    # We'll rebuild from scratch with errors injected
    story = []
    S = styles

    # Cover page
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("CONTROL NARRATIVE DOCUMENT", S["cover_title"]))
    story.append(Paragraph("Pump Station PS-01 — Rev B (Working Draft)", S["cover_sub"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="60%", thickness=2,
                             color=colors.HexColor("#1e3a5f"), hAlign="CENTER"))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Document Number: CND-PS01-002", S["cover_meta"]))
    story.append(Paragraph("Revision: B (DRAFT — NOT APPROVED)", S["cover_meta"]))
    story.append(Paragraph(f"Date: {date.today().strftime('%d %B %Y')}", S["cover_meta"]))
    story.append(PageBreak())

    story.append(Paragraph("1. GENERAL DESCRIPTION", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        "Pump Station PS-01 transfers process fluid from suction vessel T-100 to the "
        "downstream process unit. Two centrifugal pumps operate in duty/standby configuration "
        "with automatic changeover. Flow is regulated by FV-301 based on FIC-201 setpoint.",
        S["body"]))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("1.1  P&amp;ID Schematic", S["h2"]))
    story.append(PIDSchematic(width=460, height=290))

    story.append(Paragraph("2. EQUIPMENT LIST", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    equip_headers = ["Tag", "Description", "Type", "Range / Rating", "Normal State"]
    equip_rows = [
        ["P-101",   "Duty pump",               "Centrifugal pump",      "0–150 m³/h, 11 kW",  "Running"],
        ["P-102",   "Standby pump",             "Centrifugal pump",      "0–150 m³/h, 11 kW",  "Stopped"],
        ["XV-101",  "Suction isolation valve",  "Pneumatic gate valve",  "DN150, PN16",         "Open"],
        ["XV-201",  "Main discharge valve",     "Pneumatic gate valve",  "DN200, PN16",         "Open"],
        ["FV-301",  "Flow control valve",       "Pneumatic globe valve", "Cv 120, fail-open",   "Auto"],
        ["PT-101",  "Discharge pressure",       "Pressure transmitter",  "0–300 PSI",           "Active"],
        ["FT-201",  "Process flow",             "Magnetic flow transmitter", "0–200 L/min",     "Active"],
        ["LT-301",  "Suction vessel level",     "Radar level transmitter","0–5 m",              "Active"],
        ["TT-401",  "Pump bearing temperature", "RTD temperature transmitter","0–150 °C",        "Active"],
    ]
    story.append(make_table(equip_headers, equip_rows,
                            col_widths=[1.5*cm, 4*cm, 4.5*cm, 4*cm, 2.5*cm]))
    story.append(PageBreak())

    story.append(Paragraph("3. OPERATING MODES", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    # C1: Emergency Shutdown mode is missing from this section
    modes = [
        ("3.1 STOPPED",  "Both pumps de-energised. XV-201 closed. Entry: operator or fault."),
        ("3.2 STARTING", "Startup sequence executing. Duration up to 120 seconds."),
        ("3.3 RUNNING",  "P-101 running. FIC-201 controlling FV-301 to 80 L/min setpoint."),
        ("3.4 STANDBY",  "P-101 stopped, P-102 auto-started. Alarm generated."),
        ("3.5 STOPPING", "Shutdown sequence executing. Duration up to 60 seconds."),
        # ERROR C1: EMERGENCY SHUTDOWN mode is completely missing
    ]
    for num_title, desc in modes:
        story.append(Paragraph(num_title, S["h2"]))
        story.append(Paragraph(desc, S["body"]))

    story.append(Paragraph("4. SETPOINTS TABLE", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    sp_headers = ["Tag", "Parameter", "Value", "Units", "Page Ref"]
    sp_rows = [
        ["PT-101", "High High Trip", "150", "PSI",   "p.3"],
        ["PT-101", "High Alarm",     "130", "PSI",   "p.3"],
        ["FT-201", "Low Flow Trip",  "10",  "L/min", "p.3"],
        ["FT-201", "Low Alarm",      "20",  "L/min", "p.3"],
        ["LT-301", "Low Low Trip",   "15",  "%",     "p.3"],
        ["TT-401", "High High Trip", "85",  "°C",    "p.3"],
    ]
    story.append(make_table(sp_headers, sp_rows,
                            col_widths=[2*cm, 5.5*cm, 2*cm, 2*cm, 5*cm]))
    story.append(PageBreak())

    story.append(Paragraph("5. INTERLOCKS", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))

    # IL-201: ERROR C2 — missing Reset Condition and Reset Method
    story.append(Paragraph("Interlock IL-101 — High High Discharge Pressure Trip", S["h2"]))
    il101_rows = [
        ["Cause",         "PT-101 discharge pressure exceeds 150 PSI for more than 3 seconds"],
        ["Action",        "Stop P-101 AND stop P-102 AND close XV-201 AND alarm on HMI"],
        ["Reset Condition","PT-101 falls below 120 PSI AND operator investigated root cause"],
        ["Reset Method",  "Manual reset — operator presses RESET on HMI Panel"],
        ["Priority",      "Safety Critical"],
    ]
    story.append(make_table(["Field","Content"], il101_rows, col_widths=[4.5*cm, 12*cm]))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Interlock IL-201 — Low Low Flow Trip", S["h2"]))
    # ERROR C3: Reset Condition missing; ERROR C4: Reset Method missing
    il201_rows = [
        ["Cause",  "FT-201 flow rate falls below 10 L/min for more than 5 seconds while pump running"],
        ["Action", "Stop pump P-101 AND auto-start standby pump P-102 AND alarm 'LOW FLOW TRIP'"],
        # C3: Reset Condition intentionally omitted
        # C4: Reset Method intentionally omitted
    ]
    story.append(make_table(["Field","Content"], il201_rows, col_widths=[4.5*cm, 12*cm]))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Interlock IL-301 — Low Low Suction Level Trip", S["h2"]))
    il301_rows = [
        ["Cause",          "LT-301 suction vessel level falls below 15% for more than 10 seconds"],
        # ERROR C5: Action is missing — intentionally omitted
        ["Reset Condition","LT-301 level above 25% AND operator confirms feed restored"],
        ["Reset Method",   "Manual reset — operator presses RESET on HMI Panel"],
        ["Priority",       "Safety Critical"],
    ]
    story.append(make_table(["Field","Content"], il301_rows, col_widths=[4.5*cm, 12*cm]))

    story.append(PageBreak())

    story.append(Paragraph("6. ALARMS", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    al_headers = ["Tag", "Type", "Setpoint", "Units", "Priority", "Operator Response"]
    # ERROR C6: PT-101-H alarm has no Priority
    al_rows = [
        ["PT-101-HH", "HH",   "150", "PSI",   "Priority 1", "Stop all pumps. Close XV-201. Investigate."],
        ["PT-101-H",  "H",    "130", "PSI",   "",           "Reduce FIC-201 setpoint. Monitor PT-101."],  # C6: no priority
        ["FT-201-L",  "L",    "20",  "L/min", "Priority 2", "Check FV-301 position. Verify P-101 running."],
        ["LT-301-L",  "L",    "25",  "%",     "Priority 2", "Contact feed ops. Prepare for pump shutdown."],
        ["TT-401-H",  "H",    "75",  "°C",    "Priority 2", "Reduce load or switch to P-102."],
    ]
    story.append(make_table(al_headers, al_rows,
                            col_widths=[2.2*cm, 1.5*cm, 1.8*cm, 1.5*cm, 2.2*cm, 7.3*cm]))

    # ERROR C1 (Section): Startup Sequence section is completely absent
    # ERROR C2 (Section): Emergency Shutdown section is completely absent
    # (Both sections simply not included in this document)

    return story


def build_cnd_03_consistency(styles):
    """Pump Station PS-01 — with 4 consistency errors injected."""
    story = []
    S = styles

    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("CONTROL NARRATIVE DOCUMENT", S["cover_title"]))
    story.append(Paragraph("Pump Station PS-01 — Rev C", S["cover_sub"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="60%", thickness=2,
                             color=colors.HexColor("#1e3a5f"), hAlign="CENTER"))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Document Number: CND-PS01-003", S["cover_meta"]))
    story.append(Paragraph("Revision: C", S["cover_meta"]))
    story.append(Paragraph(f"Date: {date.today().strftime('%d %B %Y')}", S["cover_meta"]))
    story.append(PageBreak())

    story.append(Paragraph("1. GENERAL DESCRIPTION", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        "Pump Station PS-01 transfers process fluid from suction vessel T-100. "
        "Two centrifugal pumps operate duty/standby. Flow regulated by FV-301.", S["body"]))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("1.1  P&amp;ID Schematic", S["h2"]))
    story.append(PIDSchematic(width=460, height=290))

    story.append(Paragraph("2. EQUIPMENT LIST", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    equip_rows = [
        ["P-101",  "Duty pump",               "Centrifugal pump",      "0–150 m³/h",    "Running"],
        ["P-102",  "Standby pump",             "Centrifugal pump",      "0–150 m³/h",    "Stopped"],
        # ERROR CON1: XV-201 described as "ball valve" here
        ["XV-201", "Main discharge valve",     "Ball valve",            "DN200, PN16",   "Open"],
        ["FV-301", "Flow control valve",       "Pneumatic globe valve", "Cv 120",        "Auto"],
        ["PT-101", "Discharge pressure",       "Pressure transmitter",  "0–300 PSI",     "Active"],
        ["FT-201", "Process flow",             "Magnetic flow meter",   "0–200 L/min",   "Active"],
        ["LT-301", "Suction vessel level",     "Radar level gauge",     "0–5 m",         "Active"],
    ]
    story.append(make_table(
        ["Tag","Description","Type","Range","Normal State"], equip_rows,
        col_widths=[1.5*cm, 4*cm, 4.5*cm, 4*cm, 2.5*cm]))
    story.append(PageBreak())

    story.append(Paragraph("3. OPERATING MODES", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    modes = [
        ("3.1 STOPPED",            "Both pumps stopped. XV-201 closed."),
        ("3.2 STARTING",           "Startup sequence in progress."),
        ("3.3 RUNNING",            "P-101 running, FIC-201 controlling to 80 L/min."),
        ("3.4 STOPPING",           "Shutdown sequence executing."),
        ("3.5 EMERGENCY SHUTDOWN", "All pumps stopped immediately. Manual reset required."),
    ]
    for num_title, desc in modes:
        story.append(Paragraph(num_title, S["h2"]))
        story.append(Paragraph(desc, S["body"]))

    story.append(Paragraph("4. SETPOINTS TABLE", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    # ERROR CON2: PT-101 HH trip shows 145 PSI — prose says 150 PSI
    # ERROR CON3: FT-201 Low Alarm shows 25 L/min — prose says 20 L/min
    sp_rows = [
        ["PT-101", "High High Trip",  "145",  "PSI",   "p.3"],   # CON2: should be 150
        ["PT-101", "High Alarm",      "130",  "PSI",   "p.3"],
        ["FT-201", "Low Flow Trip",   "10",   "L/min", "p.3"],
        ["FT-201", "Low Alarm",       "25",   "L/min", "p.3"],   # CON3: should be 20
        ["LT-301", "Low Low Trip",    "15",   "%",     "p.3"],
        ["TT-401", "High High Trip",  "85",   "°C",    "p.3"],
    ]
    story.append(make_table(["Tag","Parameter","Value","Units","Ref"], sp_rows,
                            col_widths=[2*cm, 5.5*cm, 2*cm, 2*cm, 5*cm]))
    story.append(PageBreak())

    story.append(Paragraph("5. INTERLOCKS", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        # ERROR CON2: prose says 150 PSI — setpoints table says 145 PSI
        "Interlock IL-101 activates when PT-101 discharge pressure exceeds 150 PSI for "
        "more than 3 seconds. Action: stop pump P-101 AND P-102 AND close XV-201. "
        "Reset condition: PT-101 below 120 PSI AND manual reset by operator.",
        S["body"]
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        # ERROR CON3: prose says 20 L/min — setpoints table says 25 L/min
        "Interlock IL-201 activates when FT-201 flow rate falls below 20 L/min for more "
        "than 5 seconds. Action: stop P-101 AND auto-start P-102. "
        "Reset: automatic after P-102 restores flow.",
        S["body"]
    ))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("6. ALARMS", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    al_rows = [
        ["PT-101-HH", "HH", "150", "PSI",   "Priority 1", "Stop all pumps. Close XV-201. Investigate."],
        ["PT-101-H",  "H",  "130", "PSI",   "Priority 2", "Reduce FIC-201 setpoint. Monitor."],
        ["FT-201-L",  "L",  "20",  "L/min", "Priority 2", "Check FV-301 and pump status."],
        ["LT-301-L",  "L",  "25",  "%",     "Priority 2", "Contact feed ops."],
    ]
    story.append(make_table(["Tag","Type","Setpoint","Units","Priority","Operator Response"],
                            al_rows, col_widths=[2.2*cm, 1.5*cm, 1.8*cm, 1.5*cm, 2.2*cm, 7.3*cm]))

    story.append(PageBreak())

    # ERROR CON4: Startup sequence contradicts shutdown sequence on step ordering
    story.append(Paragraph("7. STARTUP SEQUENCE", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    # CON1: XV-201 described as "gate valve" here — equipment list says "ball valve"
    startup_rows = [
        ["Step 1", "Check permissives", "Verify LT-301 > 25%, no active ESD."],
        ["Step 2", "Open XV-101",       "Open suction valve. Confirm open within 15 s."],
        ["Step 3", "Start P-101",       "Start duty pump. Confirm running within 10 s."],
        ["Step 4", "Open XV-201",       "Open the gate valve XV-201 to allow flow."],   # CON1: gate valve
        ["Step 5", "Enable FIC-201",    "Enable flow controller in AUTO at 80 L/min."],
    ]
    story.append(make_table(["Step","Action","Detail"], startup_rows,
                            col_widths=[1.5*cm, 4*cm, 11*cm]))
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("8. NORMAL SHUTDOWN SEQUENCE", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    # ERROR CON4: Step 5 says stop P-101 BEFORE closing XV-201
    # This contradicts ISA best practice AND contradicts the startup sequence order
    shutdown_rows = [
        ["Step 1", "Ramp down flow",  "Reduce FIC-201 to 20 L/min."],
        ["Step 2", "Close FV-301",    "Switch to MANUAL, close FV-301."],
        ["Step 3", "Stop P-101",      "Stop P-101 BEFORE closing XV-201."],  # CON4: pump stops before valve closes
        ["Step 4", "Close XV-201",    "Close XV-201 after P-101 has stopped."],
        ["Step 5", "Close XV-101",    "Close XV-101. Verify closed within 15 s."],
    ]
    story.append(make_table(["Step","Action","Detail"], shutdown_rows,
                            col_widths=[1.5*cm, 4*cm, 11*cm]))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("9. EMERGENCY SHUTDOWN (ESD)", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    esd_rows = [
        ["ESD-1", "Stop all pumps", "De-energise P-101 AND P-102 simultaneously."],
        ["ESD-2", "Close XV-201",   "Hardwired signal to XV-201. Closes immediately."],
        ["ESD-3", "Close XV-101",   "Close suction isolation."],
        ["ESD-4", "Latch ESD",      "Set ESD latch. Manual reset required."],
    ]
    story.append(make_table(["Step","Action","Detail"], esd_rows,
                            col_widths=[1.5*cm, 4*cm, 11*cm]))

    return story


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf(filename, title, doc_number, revision, build_fn):
    path = os.path.join("data", "synthetic_cnds", filename)
    styles = make_styles()
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=1.8 * cm,
        title=title,
        author="CND-Verify Research Project",
    )
    story = build_fn(styles)
    page_fn = make_page_template(doc_number, title, revision)
    doc.build(story, onFirstPage=page_fn, onLaterPages=page_fn)
    print(f"  Generated: {path}")


def main():
    os.makedirs(os.path.join("data", "synthetic_cnds"), exist_ok=True)
    print("Generating synthetic CND PDF documents...")

    generate_pdf(
        "pump_station_01_clean.pdf",
        "Pump Station PS-01 Control Narrative", "CND-PS01-001", "A",
        build_cnd_01_clean,
    )
    generate_pdf(
        "pump_station_02_completeness_errors.pdf",
        "Pump Station PS-01 Control Narrative (Draft B)", "CND-PS01-002", "B",
        build_cnd_02_completeness,
    )
    generate_pdf(
        "pump_station_03_consistency_errors.pdf",
        "Pump Station PS-01 Control Narrative (Rev C)", "CND-PS01-003", "C",
        build_cnd_03_consistency,
    )

    print("\nDone. PDFs saved to data/synthetic_cnds/")
    print("Test in the app: Upload & Verify -> upload one of these PDFs")


if __name__ == "__main__":
    main()
