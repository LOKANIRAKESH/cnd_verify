"""
src/agents/consistency_agent.py
Detects internal contradictions within a CND:
  1. Setpoint value mismatches between prose and setpoints table
  2. Equipment type contradictions (same tag described differently in two sections)
  3. Sequence ordering contradictions (do A before B in section 4, do B before A in section 6)

Every finding records: page number, section, quoted text, rule violated, why it matters.
"""

import re
import uuid
import itertools
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import yaml

from src.orchestration.state import Finding, AgentReport


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


ISA_TAG_RE = re.compile(r"\b[A-Z]{1,3}-[0-9]{1,4}[A-Z]?\b")
NUMBER_RE = re.compile(r"([\d]+(?:\.[0-9]+)?)")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Setpoint prose vs. table mismatch (deterministic + LLM extraction)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_prose_setpoints(section_text: str, llm) -> list[dict]:
    """
    Use LLM to extract all numerical setpoints mentioned in prose sections.
    Returns [{tag, parameter, value, units, sentence}]
    """
    prompt = PromptTemplate.from_template(
        """You are a control systems engineer. Extract every instrument setpoint, limit, or threshold from this text.

TEXT:
{text}

Return a JSON array. Each element:
{{
  "tag": "PT-101",
  "parameter": "high_trip",
  "value": 150.0,
  "units": "PSI",
  "sentence": "exact sentence from text that contains this value"
}}

Only include entries where both a tag (format: XX-NNN) AND a numeric value are present.
Return [] if nothing found. Return only JSON."""
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        return chain.invoke({"text": section_text[:3000]}) or []
    except Exception:
        return []


def _normalize_parameter(param: str) -> str:
    """Normalize parameter name for matching."""
    p = param.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "hh_trip": "high_high_trip",
        "hi_trip": "high_trip",
        "high_high": "high_high_trip",
        "hi_hi": "high_high_trip",
        "high_alarm": "high_trip",
        "low_low": "low_low_trip",
        "ll_trip": "low_low_trip",
        "lo_lo": "low_low_trip",
        "low_alarm": "low_trip",
    }
    return aliases.get(p, p)


def check_setpoint_consistency(
    sections: dict,
    setpoint_table: list[dict],
    document_text: str,
    llm,
    sections_pages: dict,
) -> list[Finding]:
    findings = []

    # Build table index: (tag, normalized_param) → {value, units, page}
    table_index: dict[tuple, dict] = {}
    for row in setpoint_table:
        key = (row["tag"].upper(), _normalize_parameter(row.get("parameter", "")))
        table_index[key] = row

    for section_title, section_text in sections.items():
        # Only scan sections likely to contain setpoint mentions
        if not any(kw in section_title.upper() for kw in
                   ("INTERLOCK", "ALARM", "CONTROL", "OPERATION", "STARTUP", "SHUTDOWN", "GENERAL")):
            continue

        section_page = sections_pages.get(section_title, 0)
        prose_vals = _extract_prose_setpoints(section_text, llm)

        for pv in prose_vals:
            tag = pv.get("tag", "").upper()
            param = _normalize_parameter(pv.get("parameter", ""))
            prose_value = float(pv.get("value", 0))
            prose_units = pv.get("units", "").lower().strip()
            sentence = pv.get("sentence", "")

            table_row = table_index.get((tag, param))
            if table_row is None:
                continue  # tag/param not in table — not a contradiction, could be legitimately prose-only

            table_value = float(table_row.get("value", 0))
            table_units = table_row.get("units", "").lower().strip()
            table_page = table_row.get("page", 0)

            # Unit mismatch
            if prose_units and table_units and prose_units != table_units:
                page = _find_text_page(document_text, sentence[:40], section_page)
                findings.append(_build_finding(
                    finding_type="UNIT_MISMATCH",
                    description=(
                        f"Tag {tag} setpoint units differ: "
                        f"prose says '{prose_units}', setpoints table says '{table_units}'."
                    ),
                    severity="CRITICAL",
                    section_title=section_title,
                    page_number=page,
                    quote=sentence,
                    rule_violated="Internal consistency — setpoint units must match across document",
                    rule_text=(
                        "A single instrument tag must use identical engineering units wherever "
                        "it appears in the document. Inconsistent units create ambiguity that "
                        "leads to misconfigured PLC alarm blocks."
                    ),
                    why_it_matters=(
                        f"The PLC programmer sees '{tag}' with units '{prose_units}' in the prose "
                        f"and '{table_units}' in the setpoints table. They must guess which is correct. "
                        f"A PSI vs. bar confusion is a 14.5× error in the alarm threshold."
                    ),
                    suggestion=(
                        f"Decide the correct unit for {tag} and update every occurrence to match. "
                        f"Prose ({section_title}, ~page {page}) says '{prose_units}'; "
                        f"Setpoints Table (page {table_page}) says '{table_units}'."
                    ),
                ))
                continue

            # Value mismatch (>5% relative deviation)
            if table_value != 0:
                deviation = abs(prose_value - table_value) / table_value
                if deviation > 0.05:
                    severity = "CRITICAL" if deviation > 0.10 else "HIGH"
                    page = _find_text_page(document_text, sentence[:40], section_page)
                    findings.append(_build_finding(
                        finding_type="SETPOINT_MISMATCH",
                        description=(
                            f"Tag {tag} '{param}' value contradicts itself: "
                            f"prose says {prose_value} {prose_units}, "
                            f"setpoints table says {table_value} {table_units} "
                            f"(deviation: {deviation*100:.1f}%)."
                        ),
                        severity=severity,
                        section_title=section_title,
                        page_number=page,
                        quote=sentence,
                        rule_violated="Internal consistency — setpoint values must match across document",
                        rule_text=(
                            "Every numerical setpoint must be identical wherever it appears: in "
                            "interlock prose, alarm descriptions, startup sequences, and the "
                            "setpoints table. A >5% discrepancy indicates a document error, not "
                            "a deliberate engineering choice."
                        ),
                        why_it_matters=(
                            f"The PLC programmer has two different values for the same trip: "
                            f"{prose_value} from the prose and {table_value} from the table. "
                            f"If they use the wrong value, the protective action triggers too "
                            f"early (nuisance trips) or too late (dangerous operation)."
                        ),
                        suggestion=(
                            f"Decide the correct value for {tag} {param} and update all occurrences. "
                            f"Prose ({section_title}, ~page {page}): {prose_value} {prose_units}. "
                            f"Setpoints Table (~page {table_page}): {table_value} {table_units}. "
                            f"Typical source of error: setpoints table was updated but prose was not (or vice versa)."
                        ),
                    ))

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Equipment type contradiction (same tag, different type in two sections)
# ─────────────────────────────────────────────────────────────────────────────

EQUIPMENT_TYPE_WORDS = re.compile(
    r"\b(ball valve|gate valve|globe valve|butterfly valve|check valve|"
    r"flow transmitter|pressure transmitter|level transmitter|temperature transmitter|"
    r"centrifugal pump|reciprocating pump|positive displacement pump|gear pump|"
    r"motor valve|solenoid valve|control valve|actuated valve|hand valve|"
    r"RTD|thermocouple|orifice plate|vortex meter|magnetic flowmeter|ultrasonic meter|"
    r"differential pressure transmitter|radar level gauge|float switch)\b",
    re.IGNORECASE,
)


def _find_tag_mentions(sections: dict, document_text: str) -> dict[str, list[dict]]:
    """
    Build {tag: [{section_title, text_around_tag, page}]} for tags appearing in >1 section.
    """
    tag_mentions: dict[str, list[dict]] = {}
    for section_title, section_text in sections.items():
        tags_in_section = set(ISA_TAG_RE.findall(section_text))
        for tag in tags_in_section:
            # Find the sentence containing this tag
            for sentence in re.split(r"[.\n]", section_text):
                if tag in sentence and len(sentence.strip()) > 10:
                    record = {
                        "section_title": section_title,
                        "sentence": sentence.strip(),
                        "equipment_types": EQUIPMENT_TYPE_WORDS.findall(sentence),
                    }
                    tag_mentions.setdefault(tag, []).append(record)
                    break  # one excerpt per section per tag

    return {t: v for t, v in tag_mentions.items() if len(v) > 1}


def check_equipment_type_consistency(
    sections: dict,
    document_text: str,
    llm,
    sections_pages: dict,
) -> list[Finding]:
    findings = []
    tag_mentions = _find_tag_mentions(sections, document_text)

    for tag, mentions in tag_mentions.items():
        # Collect all equipment type words across mentions
        all_types_per_mention = [
            (m["section_title"], m["sentence"], set(t.lower() for t in m["equipment_types"]))
            for m in mentions
            if m["equipment_types"]
        ]

        if len(all_types_per_mention) < 2:
            continue

        # Check for contradictions
        for (sec_a, sent_a, types_a), (sec_b, sent_b, types_b) in itertools.combinations(
            all_types_per_mention, 2
        ):
            if not types_a.isdisjoint(types_b):
                continue  # Same types — consistent
            if not types_a or not types_b:
                continue  # One side has no equipment type word — cannot compare

            page_a = _find_text_page(document_text, sent_a[:40], sections_pages.get(sec_a, 0))
            page_b = _find_text_page(document_text, sent_b[:40], sections_pages.get(sec_b, 0))
            type_a_str = ", ".join(sorted(types_a))
            type_b_str = ", ".join(sorted(types_b))

            findings.append(_build_finding(
                finding_type="EQUIPMENT_TYPE_CONTRADICTION",
                description=(
                    f"Tag {tag} is described as '{type_a_str}' in '{sec_a}' "
                    f"but as '{type_b_str}' in '{sec_b}'."
                ),
                severity="HIGH",
                section_title=sec_a,
                page_number=page_a,
                quote=f"[Page {page_a}] {sent_a}  ↔  [Page {page_b}] {sent_b}",
                rule_violated="Internal consistency — equipment type must be consistent",
                rule_text=(
                    "A given instrument tag must refer to the same physical equipment "
                    "type throughout the document. The equipment list, operating procedures, "
                    "interlock descriptions, and startup sequences must all be consistent."
                ),
                why_it_matters=(
                    f"The PLC programmer may configure wrong control parameters for {tag}: "
                    f"valve types affect fail-safe direction (fail-open vs. fail-closed), "
                    f"pump types affect control algorithm choice and startup sequence timing."
                ),
                suggestion=(
                    f"Review both occurrences. Determine the correct equipment type for {tag} "
                    f"from the P&ID or the hardware specification. "
                    f"Update all occurrences to use the same description."
                ),
            ))

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Sequence ordering contradiction (LLM-based)
# ─────────────────────────────────────────────────────────────────────────────

def check_sequence_contradictions(
    sections: dict,
    document_text: str,
    llm,
    sections_pages: dict,
) -> list[Finding]:
    findings = []

    # Collect all sequence-type sections
    sequence_sections = {
        t: s for t, s in sections.items()
        if any(kw in t.upper() for kw in ("STARTUP", "SHUTDOWN", "SEQUENCE", "EMERGENCY", "ESD"))
    }
    if len(sequence_sections) < 2:
        return []

    prompt = PromptTemplate.from_template(
        """You are a control systems engineer reviewing startup and shutdown sequences for contradictions.

IMPORTANT: A contradiction is when two sequences specify OPPOSITE ordering for the same pair of actions.
Example of a contradiction:
  - Startup says "Step 3: Open valve XV-201, Step 5: Start pump P-101"
  - Shutdown says "Step 1: Stop pump P-101, Step 3: Close valve XV-201 BEFORE stopping P-101"
  These contradict each other on the relative ordering of XV-201 and P-101.

SEQUENCE A — {title_a}:
{text_a}

SEQUENCE B — {title_b}:
{text_b}

List only REAL contradictions — opposite ordering of the same equipment actions.
Do NOT flag things that are just different (startup vs. shutdown have opposite directions by design).
A shutdown sequence reversing a startup sequence is EXPECTED and NOT a contradiction.
Flag it only if two sequences that should agree DISAGREE (e.g., two startup variants, or a sequence step contradicts a safety interlock).

Return JSON array. Each element:
{{
  "equipment_tags": ["P-101", "XV-201"],
  "description": "What the contradiction is",
  "sequence_a_quote": "exact step from Sequence A",
  "sequence_b_quote": "exact step from Sequence B",
  "severity": "HIGH or WARNING"
}}
Return [] if no contradictions. Return only JSON."""
    )

    section_pairs = list(itertools.combinations(sequence_sections.items(), 2))
    for (title_a, text_a), (title_b, text_b) in section_pairs:
        chain = prompt | llm | JsonOutputParser()
        try:
            results = chain.invoke({
                "title_a": title_a,
                "text_a": text_a[:2000],
                "title_b": title_b,
                "text_b": text_b[:2000],
            })
        except Exception:
            continue

        for item in (results or []):
            quote = f"[{title_a}] {item.get('sequence_a_quote', '')}  ↔  [{title_b}] {item.get('sequence_b_quote', '')}"
            page_a = _find_text_page(document_text, item.get("sequence_a_quote", "")[:40],
                                     sections_pages.get(title_a, 0))
            findings.append(_build_finding(
                finding_type="SEQUENCE_ORDERING_CONTRADICTION",
                description=item.get("description", "Sequence ordering contradiction detected."),
                severity=item.get("severity", "HIGH"),
                section_title=title_a,
                page_number=page_a,
                quote=quote,
                rule_violated="Internal consistency — sequence step ordering must be non-contradictory",
                rule_text=(
                    "When two sequences reference the same equipment, the relative ordering of "
                    "actions must be consistent unless a deliberate reversal is explicitly stated. "
                    "Contradictory ordering leads to incorrect startup/shutdown logic in the PLC."
                ),
                why_it_matters=(
                    "The PLC programmer must choose one ordering and implement it. If the correct "
                    "ordering is the one they did not choose, the equipment will be operated in a "
                    "physically incorrect sequence — risking water hammer, cavitation, or runaway conditions."
                ),
                suggestion=(
                    f"Compare '{title_a}' and '{title_b}' for the affected equipment. "
                    f"Decide the correct operational order based on the process P&ID and "
                    f"update both sections to be consistent."
                ),
            ))

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Main agent entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_consistency_agent(state: dict) -> dict:
    config = load_config()
    llm = OllamaLLM(model=config["model"]["name"], temperature=0.1)

    sections: dict = state["sections"]
    sections_pages: dict = state.get("sections_pages", {})
    setpoint_table: list = state.get("setpoint_table", [])
    document_text: str = state["document_text"]

    all_findings: list[Finding] = []

    all_findings.extend(check_setpoint_consistency(sections, setpoint_table, document_text, llm, sections_pages))
    all_findings.extend(check_equipment_type_consistency(sections, document_text, llm, sections_pages))
    all_findings.extend(check_sequence_contradictions(sections, document_text, llm, sections_pages))

    report = _build_report(all_findings, config)
    state["consistency_report"] = report.model_dump()
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_text_page(document_text: str, search_text: str, fallback: int) -> int:
    if not search_text:
        return fallback
    current_page = 1
    for line in document_text.split("\n"):
        m = re.match(r"\[PAGE (\d+)\]", line.strip())
        if m:
            current_page = int(m.group(1))
        if search_text.lower()[:40] in line.lower():
            return current_page
    return fallback


def _build_finding(
    finding_type: str,
    description: str,
    severity: str,
    suggestion: str,
    why_it_matters: str,
    rule_violated: str,
    rule_text: str,
    quote: str = "",
    page_number: int = 0,
    section_title: str = "",
    evidence_source: str = "Internal document cross-check",
) -> Finding:
    return Finding(
        id=f"CONS-{uuid.uuid4().hex[:8].upper()}",
        agent="consistency",
        finding_type=finding_type,
        description=description,
        page_number=page_number,
        section_title=section_title,
        quote=quote,
        rule_violated=rule_violated,
        rule_text=rule_text,
        why_it_matters=why_it_matters,
        severity=severity,
        suggestion=suggestion,
        evidence_source=evidence_source,
    )


def _build_report(findings: list[Finding], config: dict) -> AgentReport:
    critical = sum(1 for f in findings if f.severity == "CRITICAL")
    high = sum(1 for f in findings if f.severity == "HIGH")
    warning = sum(1 for f in findings if f.severity == "WARNING")
    info = sum(1 for f in findings if f.severity == "INFO")
    p = config["scoring"]["penalty_per_finding"]
    score = max(0.0, 1.0 - critical * p["CRITICAL"] - high * p["HIGH"] - warning * p["WARNING"] - info * p["INFO"])
    return AgentReport(
        agent_name="consistency",
        findings=findings,
        score=round(score, 3),
        critical_count=critical,
        high_count=high,
        warning_count=warning,
        info_count=info,
    )
