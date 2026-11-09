"""
src/agents/logic_coherence_agent.py

Checks for logical coherence issues in the CND that cannot be caught by
completeness or consistency checks alone:

  1. CIRCULAR_INTERLOCK  — uses the networkx graph built in Pass 1;
                           A trips B trips C trips A = system can never recover
  2. MISSING_HYSTERESIS  — interlock trip threshold == reset threshold
                           = relay oscillation when process sits at that value
  3. MISSING_RESET       — interlock section describes a cause but no reset is defined
  4. UNSAFE_SEQUENCE     — pump stopped before discharge valve closes (runback / reverse flow)

Deterministic checks run first; LLM confirmation only for the UNSAFE_SEQUENCE
check where natural language reasoning is unavoidable.
"""

import re
import uuid
import logging

import yaml
from langchain_ollama import OllamaLLM

from src.orchestration.state import Finding, AgentReport

logger = logging.getLogger("cnd_verify")

ISA_TAG_RE = re.compile(r"\b([A-Z]{1,3}-\d{1,4}[A-Z]?)\b")

# Verbs that indicate a protective action in a sequence step
_STOP_VERBS  = re.compile(r"\b(stop|de-?energi[sz]e|shut\s+down|trip)\b", re.IGNORECASE)
_CLOSE_VERBS = re.compile(r"\b(close|clos[ei]ng|isolat[ei])\b", re.IGNORECASE)
_OPEN_VERBS  = re.compile(r"\b(open|start)\b", re.IGNORECASE)

# Section keywords that signal an interlock definition block
_IL_SECTION_KW = {"interlock", "esd", "emergency", "safety", "shutdown", "trip"}


def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_report(findings: list[Finding]) -> AgentReport:
    config = _load_config()
    p = config["scoring"]["penalty_per_finding"]
    critical = sum(1 for f in findings if f.severity == "CRITICAL")
    high     = sum(1 for f in findings if f.severity == "HIGH")
    warning  = sum(1 for f in findings if f.severity == "WARNING")
    info     = sum(1 for f in findings if f.severity == "INFO")
    score    = max(0.0, 1.0
                   - critical * p["CRITICAL"]
                   - high     * p["HIGH"]
                   - warning  * p["WARNING"]
                   - info     * p["INFO"])
    return AgentReport(
        agent_name="logic_coherence",
        findings=findings,
        score=round(score, 3),
        critical_count=critical,
        high_count=high,
        warning_count=warning,
        info_count=info,
    )


# ── Check 1: Circular Interlock ───────────────────────────────────────────────

def _check_circular_interlocks(state: dict) -> list[Finding]:
    graph_data = state.get("interlock_graph_data", {})
    if not graph_data.get("cycle_detected"):
        return []

    findings = []
    for cycle_path in graph_data.get("cycles", []):
        tags_in_cycle = ISA_TAG_RE.findall(cycle_path)
        tag_index = state.get("tag_index", {})
        page = 0
        for t in tags_in_cycle:
            entries = tag_index.get(t, [])
            if entries:
                page = entries[0].get("page", 0)
                break

        findings.append(Finding(
            id=f"LC-{uuid.uuid4().hex[:8].upper()}",
            agent="logic_coherence",
            finding_type="CIRCULAR_INTERLOCK",
            description=(
                f"Circular interlock detected: {cycle_path}. "
                "The system cannot reach a safe state because each element "
                "in the loop trips the next, creating an unresolvable deadlock."
            ),
            page_number=page,
            section_title="Interlocks",
            quote=cycle_path,
            rule_violated="ISA-84 / IEC 61511 §11 — Safety Instrumented System design",
            rule_text=(
                "Safety instrumented functions shall not be designed in a way "
                "that prevents the system from reaching and maintaining a safe "
                "state (IEC 61511-1 Clause 11.2)."
            ),
            why_it_matters=(
                "A circular interlock creates a logical deadlock: each protective "
                "action triggers the next interlock, preventing any manual or "
                "automatic recovery path. The system may be permanently blocked "
                "from restarting without disabling all safety interlocks."
            ),
            suggestion=(
                f"Break the cycle at the weakest link: {cycle_path}. "
                "Redesign so that at least one element in the loop has a "
                "manual reset that can be used to safely re-establish normal "
                "operation after investigating root cause."
            ),
            severity="CRITICAL",
            evidence_source="Interlock graph — cycle detection (networkx)",
            status="OPEN",
        ))
    return findings


# ── Check 2: Missing Hysteresis ───────────────────────────────────────────────

def _check_missing_hysteresis(state: dict) -> list[Finding]:
    """
    Detect interlocks where the trip setpoint equals the reset setpoint.
    A trip at 150 PSI that resets at 150 PSI will oscillate indefinitely.
    """
    findings = []
    sections  = state.get("sections", {})

    _TRIP_VAL_RE  = re.compile(
        r"(?:trip|shutdown|esd)[^.]{0,60}?([\d]+(?:\.\d+)?)\s*(PSI|BAR|°C|%|L/MIN|M3/H)?",
        re.IGNORECASE
    )
    _RESET_VAL_RE = re.compile(
        r"(?:reset|clear|recover)[^.]{0,60}?([\d]+(?:\.\d+)?)\s*(PSI|BAR|°C|%|L/MIN|M3/H)?",
        re.IGNORECASE
    )

    for section_title, section_text in sections.items():
        if not any(kw in section_title.lower() for kw in _IL_SECTION_KW):
            continue

        # Split on interlock boundary markers so we compare values per-interlock,
        # not across the whole section (avoids cross-interlock false positives).
        interlock_blocks = re.split(r"(?i)\bIL-\d{3}\b|\n\s*\n", section_text)
        if len(interlock_blocks) <= 1:
            interlock_blocks = [section_text]

        for block in interlock_blocks:
            trip_matches  = _TRIP_VAL_RE.findall(block)
            reset_matches = _RESET_VAL_RE.findall(block)

            if not trip_matches or not reset_matches:
                continue

            trip_vals  = {float(v) for v, _ in trip_matches if v}
            reset_vals = {float(v) for v, _ in reset_matches if v}
            shared = trip_vals & reset_vals

            for val in shared:
                page = state.get("sections_pages", {}).get(section_title, 0)
                tags = ISA_TAG_RE.findall(block[:500])
                tag_str = ", ".join(tags[:3]) if tags else "unknown tag"

                findings.append(Finding(
                    id=f"LC-{uuid.uuid4().hex[:8].upper()}",
                    agent="logic_coherence",
                    finding_type="MISSING_HYSTERESIS",
                    description=(
                        f"In section '{section_title}', the trip setpoint ({val}) equals the "
                        f"reset setpoint ({val}). Without a hysteresis band, the interlock "
                        f"will oscillate (trip → reset → trip → ...) when the process value "
                        f"sits at exactly this threshold."
                    ),
                    page_number=page,
                    section_title=section_title,
                    quote=(
                        f"Trip condition at {val} and reset condition also at {val} "
                        f"(tags: {tag_str})"
                    ),
                    rule_violated="ISA-84 §5.4.3 — Hysteresis requirement for SIF reset",
                    rule_text=(
                        "Safety instrumented functions should include appropriate hysteresis "
                        "between the trip initiation value and the reset value to prevent "
                        "process oscillation and chattering (ISA-84 §5.4.3)."
                    ),
                    why_it_matters=(
                        "Oscillating interlocks repeatedly start and stop equipment, causing "
                        "mechanical wear, false alarms, and operator alarm fatigue. In severe "
                        "cases, the oscillation can prevent the process from stabilising."
                    ),
                    suggestion=(
                        f"Define a hysteresis band: trip at {val}, reset at {val * 0.85:.1f} "
                        f"(15% below trip). The reset threshold should be documented explicitly "
                        f"in the interlock definition table."
                    ),
                    severity="HIGH",
                    evidence_source="Deterministic setpoint comparison — trip vs reset values",
                    status="OPEN",
                ))

    return findings


# ── Check 3: Missing Reset Condition ─────────────────────────────────────────

def _check_missing_reset_conditions(state: dict) -> list[Finding]:
    """
    Find interlock blocks that have a Cause defined but no Reset Condition.
    An interlock without a reset permanently locks the system.
    """
    findings = []
    sections = state.get("sections", {})

    _CAUSE_RE  = re.compile(r"\bcause\b|\binitiating\b|\bactivat",  re.IGNORECASE)
    _RESET_RE  = re.compile(r"\breset\b|\bclear\b|\brecovery\b|\bdeactivat", re.IGNORECASE)
    _IL_ID_RE  = re.compile(r"\b(IL-\d{3})\b", re.IGNORECASE)

    for section_title, section_text in sections.items():
        if not any(kw in section_title.lower() for kw in _IL_SECTION_KW):
            continue

        il_ids = _IL_ID_RE.findall(section_text)
        if not il_ids:
            has_cause = bool(_CAUSE_RE.search(section_text))
            has_reset = bool(_RESET_RE.search(section_text))
            if has_cause and not has_reset:
                page = state.get("sections_pages", {}).get(section_title, 0)
                tags = ISA_TAG_RE.findall(section_text[:300])
                tag_str = tags[0] if tags else "unknown"
                findings.append(Finding(
                    id=f"LC-{uuid.uuid4().hex[:8].upper()}",
                    agent="logic_coherence",
                    finding_type="MISSING_RESET_CONDITION",
                    description=(
                        f"Section '{section_title}' defines an interlock cause but "
                        f"contains no reset condition. Once tripped, the interlock "
                        f"cannot be cleared by any defined procedure."
                    ),
                    page_number=page,
                    section_title=section_title,
                    quote=section_text[:200].strip(),
                    rule_violated="ISA-84 §5.4.2 — Interlock reset must be defined",
                    rule_text=(
                        "Every safety instrumented function shall have a defined reset "
                        "condition and reset method. The reset logic shall be specified "
                        "in the Control Narrative before PLC programming begins (ISA-84)."
                    ),
                    why_it_matters=(
                        "An interlock without a defined reset may lock the system permanently "
                        "after first trip. The PLC programmer must guess the reset condition, "
                        "which may create an unsafe or unverifiable reset path."
                    ),
                    suggestion=(
                        "Add a Reset Condition field stating the process value threshold "
                        "that must be reached before reset is permitted. Add a Reset Method "
                        "field stating whether reset is manual (operator action) or automatic."
                    ),
                    severity="HIGH",
                    evidence_source="Deterministic keyword scan — cause present, reset absent",
                    status="OPEN",
                ))

    return findings


# ── Check 4: Unsafe Sequence Ordering ────────────────────────────────────────

def _check_sequence_ordering(state: dict, llm) -> list[Finding]:
    """
    Detect unsafe sequences using LLM analysis.
    Unsafe pattern: pump stopped BEFORE discharge valve closes (runback risk).
    Safe order: close discharge valve → stop pump.
    """
    findings = []
    sections = state.get("sections", {})

    seq_sections = {
        title: text for title, text in sections.items()
        if any(kw in title.lower() for kw in
               ("startup", "start-up", "shutdown", "shut-down", "sequence", "stopping"))
    }

    if not seq_sections:
        return []

    for section_title, section_text in seq_sections.items():
        tags = ISA_TAG_RE.findall(section_text)
        has_pump  = any(t.startswith("P-") for t in tags)
        has_valve = any(t.startswith(("XV", "FV", "PV", "LV")) for t in tags)
        if not has_pump or not has_valve:
            continue

        stop_pump_pos   = _find_pump_stop_position(section_text)
        close_valve_pos = _find_valve_close_position(section_text)

        if stop_pump_pos is not None and close_valve_pos is not None:
            if stop_pump_pos < close_valve_pos and "shutdown" in section_title.lower():
                try:
                    prompt = (
                        "You are a process safety engineer reviewing a pump station "
                        "shutdown sequence. Evaluate whether the sequence below is UNSAFE.\n\n"
                        "UNSAFE pattern: pump is stopped BEFORE the discharge valve is closed. "
                        "This can cause reverse flow (runback) through the stopped pump.\n"
                        "SAFE pattern: close discharge valve first, THEN stop pump.\n\n"
                        f"Section: {section_title}\n"
                        f"Text: {section_text[:600]}\n\n"
                        "Reply with JSON only: "
                        '{"unsafe": true/false, "reason": "one sentence"}'
                    )
                    response = llm.invoke(prompt)
                    import json as _json
                    m = re.search(r'\{[^}]+\}', response, re.DOTALL)
                    if m:
                        result = _json.loads(m.group())
                        if result.get("unsafe"):
                            page = state.get("sections_pages", {}).get(section_title, 0)
                            findings.append(Finding(
                                id=f"LC-{uuid.uuid4().hex[:8].upper()}",
                                agent="logic_coherence",
                                finding_type="UNSAFE_SEQUENCE",
                                description=(
                                    f"In '{section_title}': the shutdown sequence stops the pump "
                                    f"before closing the discharge valve. "
                                    f"{result.get('reason', '')}"
                                ),
                                page_number=page,
                                section_title=section_title,
                                quote=section_text[:300].strip(),
                                rule_violated="ISA-88 §4.7 — Safe equipment sequencing",
                                rule_text=(
                                    "Shutdown sequences shall be designed to prevent reverse "
                                    "flow conditions. Discharge isolation valves shall be "
                                    "closed before pumps are stopped (ISA-88 procedural control)."
                                ),
                                why_it_matters=(
                                    "Stopping a pump before closing its discharge valve allows "
                                    "downstream backpressure to drive reverse flow through the "
                                    "stopped pump. This can damage pump internals, contaminate "
                                    "the suction vessel, and void the pump warranty."
                                ),
                                suggestion=(
                                    "Reorder shutdown sequence: (1) close discharge valve and "
                                    "confirm closed, (2) then stop the pump motor. Verify "
                                    "startup sequence opens valve after pump starts."
                                ),
                                severity="HIGH",
                                evidence_source="Sequence ordering analysis — LLM confirmed",
                                status="OPEN",
                            ))
                except Exception as e:
                    logger.warning("LLM sequence check failed for '%s': %s", section_title, e)

    return findings


def _find_pump_stop_position(text: str) -> int | None:
    """Return char position of first 'stop <pump_tag>' pattern, or None."""
    for m in _STOP_VERBS.finditer(text):
        snippet = text[m.start():m.start() + 80]
        if re.search(r"\bP-\d{3}\b", snippet):
            return m.start()
    return None


def _find_valve_close_position(text: str) -> int | None:
    """Return char position of first 'close <valve_tag>' pattern, or None."""
    for m in _CLOSE_VERBS.finditer(text):
        snippet = text[m.start():m.start() + 80]
        if re.search(r"\b(?:XV|FV|PV|LV)-\d{3,4}\b", snippet):
            return m.start()
    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def run_logic_coherence_agent(state: dict) -> dict:
    """
    Run all four logic coherence checks and produce an AgentReport.
    Circular interlock check is deterministic (uses Pass 1 graph).
    Hysteresis and missing-reset checks are deterministic (regex/numeric).
    Unsafe sequence check uses LLM confirmation to reduce false positives.
    """
    config = _load_config()
    llm = OllamaLLM(
        model=config["model"]["name"],
        base_url="http://localhost:11434",
        temperature=config["model"]["temperature"],
    )

    findings: list[Finding] = []

    findings.extend(_check_circular_interlocks(state))
    findings.extend(_check_missing_hysteresis(state))
    findings.extend(_check_missing_reset_conditions(state))
    findings.extend(_check_sequence_ordering(state, llm))

    report = _build_report(findings)
    state["logic_coherence_report"] = report.model_dump()

    logger.info(
        "Logic coherence agent: %d findings (C=%d H=%d W=%d)",
        len(findings), report.critical_count, report.high_count, report.warning_count,
    )
    return state
