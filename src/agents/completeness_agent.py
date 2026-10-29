"""
src/agents/completeness_agent.py
Checks whether the CND contains all required sections, interlock fields, and alarm fields.
Every finding includes: page number, section title, rule violated, rule text, why it matters.
"""

import re
import uuid
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from difflib import SequenceMatcher
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import yaml

from src.orchestration.state import Finding, AgentReport


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fuzzy_section_match(doc_title: str, required_title: str, threshold: float = 0.60) -> bool:
    ratio = SequenceMatcher(None, doc_title.lower(), required_title.lower()).ratio()
    if ratio >= threshold:
        return True
    if required_title.lower() in doc_title.lower():
        return True
    return False


def find_missing_sections(sections: dict, required_sections: list[str]) -> list[str]:
    missing = []
    for required in required_sections:
        found = any(fuzzy_section_match(title, required) for title in sections)
        if not found:
            missing.append(required)
    return missing


def retrieve_kb(query: str, collection, top_k: int = 2) -> str:
    if collection is None:
        return ""
    results = collection.query(query_texts=[query], n_results=top_k)
    if results["documents"] and results["documents"][0]:
        return "\n\n".join(results["documents"][0])
    return ""


def build_finding(
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
    evidence_source: str = "",
) -> Finding:
    return Finding(
        id=f"COMP-{uuid.uuid4().hex[:8].upper()}",
        agent="completeness",
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


def _parse_llm_json(raw: str) -> list:
    """
    Robustly extract a JSON array from LLM output.
    Llama 3.1 often wraps JSON in prose or code fences — this handles that.
    """
    if not raw:
        return []
    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Try direct parse
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        pass
    # Find the first [...] block in the text
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            pass
    return []


def check_interlock_fields_llm(interlock_text: str, llm, domain_context: str) -> list[dict]:
    prompt = PromptTemplate.from_template(
        """You are a control systems engineer reviewing interlock definitions in a Control Narrative Document.

RULE (ISA-84 / industry standard): Every safety interlock must have ALL five fields:
1. cause — the specific tag and numerical condition that triggers the interlock (e.g., "PT-101 exceeds 150 PSI")
2. action — exactly what happens (specific equipment tags and actions, e.g., "Stop pump P-101 AND close XV-201")
3. reset_condition — the specific condition that must be true before the interlock can be cleared
4. reset_method — whether reset is manual (operator action) or automatic
5. priority — severity classification (Safety Critical / High / Medium)

DOMAIN KNOWLEDGE:
{domain_context}

INTERLOCK SECTION FROM DOCUMENT:
{interlock_text}

For each interlock, list only the fields that are MISSING or too vague to be useful.
A field is vague if: cause has no tag or no number, action has no specific equipment tag, reset_condition says only "condition clears" with no specifics.

Return a JSON array. Each element:
{{
  "interlock_name": "IL-001 or descriptive name",
  "missing_fields": ["field1", "field2"],
  "quotes": {{"field_if_vague": "the actual vague text"}},
  "severity": "CRITICAL or HIGH"
}}

Return [] if all interlocks are complete. Return ONLY the JSON array, no explanation."""
    )
    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"domain_context": domain_context, "interlock_text": interlock_text[:4000]})
        return _parse_llm_json(raw)
    except Exception:
        return []


def check_alarm_fields_llm(alarm_text: str, llm, domain_context: str) -> list[dict]:
    prompt = PromptTemplate.from_template(
        """You are a control systems engineer reviewing alarm definitions in a Control Narrative Document.

RULE (ISA-18.2): Every alarm must have ALL six fields:
1. tag — ISA-5.1 instrument tag (e.g., PT-101-HH)
2. alarm_type — HH, H, L, or LL
3. setpoint_value — a specific number (not "high" or "elevated")
4. units — engineering units (PSI, degrees C, L/min, %)
5. priority — Priority 1 (Critical), 2 (High), 3 (Medium), or 4 (Low). An empty or blank priority field counts as missing.
6. operator_response — specific action (not just "check the process" or "investigate")

DOMAIN KNOWLEDGE:
{domain_context}

ALARM SECTION FROM DOCUMENT:
{alarm_text}

For each alarm, list only the fields that are MISSING or empty.
A priority field with an empty value ("") counts as missing.
Operator response is vague only if it says literally "check" or "investigate" with NO specific action.

Return a JSON array. Each element:
{{
  "alarm_tag": "PT-101-HH",
  "missing_fields": ["priority", "operator_response"],
  "vague_quotes": {{"operator_response": "the actual vague text"}},
  "severity": "HIGH or WARNING"
}}

Return [] if all alarms are complete. Return ONLY the JSON array, no explanation."""
    )
    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"domain_context": domain_context, "alarm_text": alarm_text[:4000]})
        return _parse_llm_json(raw)
    except Exception:
        return []


def run_completeness_agent(state: dict) -> dict:
    """
    Main entry point — called by LangGraph as a node.
    Reads from state, writes completeness_report back.
    """
    config = load_config()
    findings: list[Finding] = []

    llm = OllamaLLM(model=config["model"]["name"], temperature=config["model"]["temperature"])

    chroma_client = chromadb.PersistentClient(path=config["chromadb"]["path"])
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=config["embeddings"]["model"])
    try:
        kb = chroma_client.get_collection(
            name=config["chromadb"]["collection_name"], embedding_function=embedding_fn
        )
    except Exception:
        kb = None

    sections: dict = state["sections"]
    sections_pages: dict = state.get("sections_pages", {})
    required_sections: list = config["required_sections"]
    document_text: str = state["document_text"]

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 1 — Required sections present
    # Deterministic checklist, no LLM needed
    # ══════════════════════════════════════════════════════════════════════════
    safety_critical = {"Emergency Shutdown", "Interlocks", "Alarms"}
    missing_sections = find_missing_sections(sections, required_sections)

    for section_name in missing_sections:
        is_critical = section_name in safety_critical
        severity = "CRITICAL" if is_critical else "HIGH"

        findings.append(build_finding(
            finding_type="MISSING_SECTION",
            description=f"Required section '{section_name}' is completely missing from the document.",
            severity=severity,
            section_title="Document Structure",
            page_number=0,
            quote="",
            rule_violated=f"ISA-88 Required Document Structure — '{section_name}' section",
            rule_text=(
                "A complete Control Narrative Document must contain sections for: "
                "General Description, Operating Modes, Startup Sequence, Shutdown Sequence, "
                "Emergency Shutdown, Interlocks, Alarms, and Setpoints."
            ),
            why_it_matters=(
                f"Without the '{section_name}' section, the PLC programmer has no instructions "
                f"for this aspect of the system. Code will either be missing entirely or written "
                f"based on the programmer's assumptions — leading to incorrect or unsafe behaviour."
            ),
            suggestion=(
                f"Add a '{section_name}' section to the document. "
                f"Refer to the domain knowledge base for the required content of this section."
            ),
            evidence_source="ISA-88 / CND required section checklist",
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 2 — Interlock completeness (LLM + RAG)
    # ══════════════════════════════════════════════════════════════════════════
    interlock_text, interlock_section_title = _find_section(sections, "INTERLOCK")
    interlock_page = sections_pages.get(interlock_section_title, 0)

    if interlock_text:
        domain_ctx = retrieve_kb("interlock required fields cause action reset condition method priority ISA-84", kb)
        llm_results = check_interlock_fields_llm(interlock_text, llm, domain_ctx)

        for item in llm_results:
            for missing_field in item.get("missing_fields", []):
                quote = item.get("quotes", {}).get(missing_field, "")
                # Find actual page of this interlock text
                page = _find_text_page(document_text, item.get("interlock_name", ""), interlock_page)

                findings.append(build_finding(
                    finding_type="MISSING_INTERLOCK_FIELD",
                    description=(
                        f"Interlock '{item.get('interlock_name', 'UNKNOWN')}' "
                        f"is missing the required field: '{missing_field}'."
                    ),
                    severity=item.get("severity", "HIGH"),
                    section_title=interlock_section_title,
                    page_number=page,
                    quote=quote if quote else f"[Field '{missing_field}' not found in {item.get('interlock_name', 'interlock')}]",
                    rule_violated=f"Interlock completeness — required field: {missing_field}",
                    rule_text=_interlock_rule_text(missing_field),
                    why_it_matters=_interlock_why_matters(missing_field),
                    suggestion=_interlock_suggestion(missing_field, item.get("interlock_name", "")),
                    evidence_source="ISA-84 / control narrative interlock standard",
                ))

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 3 — Alarm completeness (LLM + RAG)
    # ══════════════════════════════════════════════════════════════════════════
    alarm_text, alarm_section_title = _find_section(sections, "ALARM")
    alarm_page = sections_pages.get(alarm_section_title, 0)

    if alarm_text:
        domain_ctx = retrieve_kb("alarm required fields priority setpoint operator response ISA-18.2", kb)
        llm_results = check_alarm_fields_llm(alarm_text, llm, domain_ctx)

        for item in llm_results:
            for missing_field in item.get("missing_fields", []):
                quote = item.get("vague_quotes", {}).get(missing_field, "")
                page = _find_text_page(document_text, item.get("alarm_tag", ""), alarm_page)

                findings.append(build_finding(
                    finding_type="MISSING_ALARM_FIELD",
                    description=(
                        f"Alarm '{item.get('alarm_tag', 'UNKNOWN')}' "
                        f"is missing the required field: '{missing_field}'."
                    ),
                    severity=item.get("severity", "HIGH"),
                    section_title=alarm_section_title,
                    page_number=page,
                    quote=quote if quote else f"[Field '{missing_field}' absent from alarm {item.get('alarm_tag', '')}]",
                    rule_violated=f"ISA-18.2 Required Alarm Fields — '{missing_field}'",
                    rule_text=_alarm_rule_text(missing_field),
                    why_it_matters=_alarm_why_matters(missing_field),
                    suggestion=_alarm_suggestion(missing_field),
                    evidence_source="ISA-18.2 alarm management standard",
                ))

    # ══════════════════════════════════════════════════════════════════════════
    # CHECK 4 — Tags in interlocks defined in equipment list (deterministic)
    # ══════════════════════════════════════════════════════════════════════════
    equipment_text, _ = _find_section(sections, "EQUIPMENT")
    if interlock_text and equipment_text:
        interlock_tags = set(re.findall(r"\b[A-Z]{1,3}-[0-9]{1,4}[A-Z]?\b", interlock_text))
        equipment_tags = set(re.findall(r"\b[A-Z]{1,3}-[0-9]{1,4}[A-Z]?\b", equipment_text))
        for tag in sorted(interlock_tags - equipment_tags):
            page = _find_text_page(document_text, tag, interlock_page)
            findings.append(build_finding(
                finding_type="UNDEFINED_TAG_REFERENCE",
                description=f"Tag '{tag}' is used in the Interlocks section but not defined in the Equipment List.",
                severity="HIGH",
                section_title=interlock_section_title,
                page_number=page,
                quote=f"Tag {tag} referenced in interlocks but absent from equipment list",
                rule_violated="ISA-5.1 — All tags must be defined",
                rule_text="Every instrument tag used in a control narrative must appear in the equipment list with its type, range, and purpose.",
                why_it_matters=(
                    f"The PLC programmer cannot wire up '{tag}' because its instrument type, "
                    f"signal range, and I/O card assignment are unknown."
                ),
                suggestion=(
                    f"Add '{tag}' to the Equipment List with: instrument type, measurement range, "
                    f"output signal type (4-20mA / digital), and purpose description."
                ),
                evidence_source="ISA-5.1 tag naming and definition standard",
            ))

    # ── Build report ──────────────────────────────────────────────────────────
    report = _build_report(findings, config)
    state["completeness_report"] = report.model_dump()
    state["missing_critical_sections"] = [s for s in missing_sections if s in safety_critical]
    return state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_section(sections: dict, keyword: str) -> tuple[str, str]:
    """
    Return concatenated text and the first matching title for all sections whose
    title contains keyword.  This is critical when a single logical section (e.g.
    INTERLOCKS) is split by extract_sections() into a preamble entry plus one
    sub-entry per interlock (e.g. "INTERLOCK IL-101 — ...").
    Returning only the preamble would leave the LLM with nothing useful.
    """
    matched_texts: list[str] = []
    first_title = ""
    kw_upper = keyword.upper()
    for title, text in sections.items():
        if kw_upper in title.upper():
            if not first_title:
                first_title = title
            matched_texts.append(text)
    return "\n\n".join(matched_texts), first_title


def _find_text_page(document_text: str, search_text: str, fallback_page: int) -> int:
    """Find page of search_text in document; return fallback_page if not found."""
    if not search_text:
        return fallback_page
    current_page = 1
    for line in document_text.split("\n"):
        m = re.match(r"\[PAGE (\d+)\]", line.strip())
        if m:
            current_page = int(m.group(1))
        if search_text.lower()[:40] in line.lower():
            return current_page
    return fallback_page


def _build_report(findings: list[Finding], config: dict) -> AgentReport:
    from src.orchestration.state import AgentReport
    critical = sum(1 for f in findings if f.severity == "CRITICAL")
    high = sum(1 for f in findings if f.severity == "HIGH")
    warning = sum(1 for f in findings if f.severity == "WARNING")
    info = sum(1 for f in findings if f.severity == "INFO")
    p = config["scoring"]["penalty_per_finding"]
    score = max(0.0, 1.0 - critical * p["CRITICAL"] - high * p["HIGH"] - warning * p["WARNING"] - info * p["INFO"])
    return AgentReport(
        agent_name="completeness",
        findings=findings,
        score=round(score, 3),
        critical_count=critical,
        high_count=high,
        warning_count=warning,
        info_count=info,
    )


def _interlock_rule_text(field: str) -> str:
    rules = {
        "cause": "The cause (initiating condition) must specify: the instrument tag AND the numerical threshold that triggers the interlock. Example: 'PT-101 pressure exceeds 150 PSI'.",
        "action": "The action must specify: which specific equipment tags are affected AND exactly what happens to each. Example: 'Stop pump P-101 AND close discharge valve XV-201'.",
        "reset_condition": "A safety interlock must define when it can be reset — what specific condition must be true. Auto-resetting a safety interlock without defined conditions risks restarting into an unsafe state.",
        "reset_method": "The reset method must be explicitly stated: Manual (operator must deliberately press a reset button) or Automatic (clears when condition returns to normal). Safety interlocks should require manual reset.",
        "priority": "Every interlock must be classified by severity: Safety Critical (prevents injury or major damage), High, Medium, or Low.",
    }
    return rules.get(field.lower(), f"Field '{field}' is required per control narrative completeness standards.")


def _interlock_why_matters(field: str) -> str:
    matters = {
        "cause": "Without a specific tag and value, the PLC programmer cannot set the trip threshold. They must guess — and a wrong guess could allow the pump to run in a dangerous condition or trip unnecessarily.",
        "action": "Without specific equipment tags and actions, the PLC programmer does not know which outputs to wire to this interlock. The protective function cannot be programmed.",
        "reset_condition": "Without a reset condition, the PLC programmer does not know when the system is safe to restart. They may write logic that allows immediate restart after a trip — before the dangerous condition is confirmed clear.",
        "reset_method": "Without knowing if reset is manual or automatic, the programmer may write self-resetting logic for a safety interlock. This allows the system to restart without operator investigation — a safety hazard.",
        "priority": "Without priority, the HMI cannot display the interlock with the correct urgency. Operators may not respond in time to a safety-critical condition.",
    }
    return matters.get(field.lower(), f"The missing '{field}' field prevents correct PLC implementation of this interlock.")


def _interlock_suggestion(field: str, interlock_name: str) -> str:
    suggestions = {
        "cause": f"In {interlock_name}, add: 'Cause: [TAG] [measurement] exceeds/falls below [value] [units] for [time] seconds'. Example: 'PT-101 discharge pressure exceeds 150 PSI for more than 3 seconds'",
        "action": f"In {interlock_name}, add: 'Action: Stop [pump tag] AND close [valve tag] AND display alarm [alarm text] on HMI'",
        "reset_condition": f"In {interlock_name}, add: 'Reset Condition: [TAG] [measurement] returns below/above [safe value] AND operator has investigated and cleared the root cause'",
        "reset_method": f"In {interlock_name}, add: 'Reset Method: Manual reset required — operator must press RESET on HMI after confirming safe conditions'",
        "priority": f"In {interlock_name}, add: 'Priority: Safety Critical' (or High / Medium as appropriate for this interlock)",
    }
    return suggestions.get(field.lower(), f"Add the '{field}' field to {interlock_name} per ISA-84 requirements.")


def _alarm_rule_text(field: str) -> str:
    rules = {
        "priority": "ISA-18.2 requires every alarm to have a defined priority: Priority 1 (Critical — immediate action, <1 minute), Priority 2 (High — urgent, <10 minutes), Priority 3 (Medium — <30 minutes), Priority 4 (Low — informational).",
        "operator_response": "ISA-18.2 requires every alarm to have a specific operator response — the exact action the operator must take. Generic responses like 'investigate' or 'check the process' are not acceptable.",
        "setpoint_value": "The alarm setpoint must be a specific numerical value. Alarms described as activating at 'high pressure' or 'elevated temperature' without a number cannot be configured in the PLC or HMI.",
        "units": "Engineering units must be specified alongside the setpoint value. A setpoint of '150' is ambiguous — 150 PSI and 150 bar differ by a factor of 10.",
        "alarm_type": "Alarm type (HH, H, L, LL) determines the alarm priority and operator urgency. Missing type means the PLC alarm block cannot be configured.",
        "tag": "Every alarm must reference the instrument tag it is associated with. Without a tag, the alarm cannot be mapped to a physical sensor in the PLC I/O configuration.",
    }
    return rules.get(field.lower(), f"Field '{field}' is required per ISA-18.2 alarm management standard.")


def _alarm_why_matters(field: str) -> str:
    matters = {
        "priority": "Without priority, the HMI cannot sort alarms by urgency. During an emergency with many alarms active simultaneously, operators cannot identify which requires immediate action — a leading cause of alarm floods and accidents.",
        "operator_response": "Without a specific response, the operator does not know what to do when the alarm activates. They may take the wrong action, take no action, or spend valuable time consulting other documents — delays that could allow a dangerous condition to worsen.",
        "setpoint_value": "Without a numerical setpoint, the PLC alarm block cannot be programmed. The programmer must either leave the alarm unconfigured or guess a value — both are unacceptable for a safety-critical system.",
        "units": "A setpoint value without units can be interpreted in multiple ways. A programmer assuming PSI when the engineer meant bar would configure an alarm 14.5 times too sensitive or too insensitive.",
        "alarm_type": "Without alarm type, the programmer cannot determine whether this is a warning (H) or a trip-level alert (HH). This affects whether the alarm triggers automatic protective action.",
        "tag": "Without a tag, the alarm cannot be connected to a physical sensor in the PLC I/O database. It exists in the document but cannot be implemented.",
    }
    return matters.get(field.lower(), f"The missing '{field}' field prevents correct HMI/PLC alarm configuration.")


def _alarm_suggestion(field: str) -> str:
    suggestions = {
        "priority": "Add 'Priority: Priority 1 (Critical)' or Priority 2/3/4 as appropriate. Consider: does this alarm require immediate action to prevent injury or major damage? If yes, Priority 1.",
        "operator_response": "Add a specific 2-3 sentence operator response: (1) immediate action to take, (2) what to check/verify, (3) who to contact if condition persists. Example: 'Stop pump P-101 manually. Check for downstream blockage. Contact maintenance if pressure does not drop within 5 minutes.'",
        "setpoint_value": "Replace vague description with a specific number. Example: Replace 'high pressure' with '130 PSI' (the exact threshold where this alarm activates).",
        "units": "Add engineering units after the setpoint value. Example: '130 PSI', '85 degrees Celsius', '20 L/min', '30 %'",
        "alarm_type": "Specify alarm type: HH (highest severity, typically trips equipment), H (warning, operator attention needed), L (low condition warning), LL (critically low, typically trips equipment).",
        "tag": "Add ISA-5.1 format tag: [instrument tag]-[alarm type suffix]. Example: PT-101-HH, FT-201-L, LT-301-LL",
    }
    return suggestions.get(field.lower(), f"Add the '{field}' field per ISA-18.2 alarm management requirements.")
