"""
evaluation/baselines.py

Baseline implementations for comparison with CND-Verify full pipeline.

B1: Rule-based checklist — keyword presence + section matching only.
    No LLM, no RAG, no setpoint comparison.
    Represents "industry current practice" (template-based review).

B2: Single-LLM — one LLM prompt covering all quality dimensions.
    No multi-agent separation, no RAG, no deterministic checks.
"""

import re
import time
import logging

logger = logging.getLogger("cnd_verify")


# ── Baseline B1: Rule-Based Keyword Checklist ─────────────────────────────────

_REQUIRED_SECTIONS = [
    "general description", "equipment list", "operating modes",
    "startup sequence", "normal operation", "shutdown sequence",
    "emergency shutdown", "interlocks", "alarms", "setpoints",
]

_INTERLOCK_FIELDS = ["cause", "action", "reset", "priority"]
_ALARM_FIELDS = ["priority", "setpoint", "operator response"]

ISA_TAG_RE = re.compile(r"\b([A-Z]{1,3}-\d{1,4}[A-Z]?)\b")


def _b1_run(doc_state: dict) -> list[dict]:
    """
    Baseline B1: rule-based checklist.
    Returns list of finding dicts (same schema as agent findings, but source=B1).
    """
    findings = []
    text_lower = " ".join(doc_state.get("sections", {}).values()).lower()
    sections_lower = {k.lower(): v for k, v in doc_state.get("sections", {}).items()}

    # Check 1: mandatory sections
    for req in _REQUIRED_SECTIONS:
        found = any(req in k for k in sections_lower)
        if not found:
            findings.append({
                "id": f"B1-SEC-{req[:8].upper()}",
                "agent": "completeness",
                "finding_type": "MISSING_SECTION",
                "description": f"Required section '{req.title()}' not found in document.",
                "severity": "CRITICAL" if req in ("interlocks", "emergency shutdown", "alarms") else "HIGH",
                "source": "B1",
            })

    # Check 2: interlock fields — keyword presence in interlock sections
    for title, text in doc_state.get("sections", {}).items():
        if not any(kw in title.lower() for kw in ("interlock", "esd", "emergency")):
            continue
        text_l = text.lower()
        for field in _INTERLOCK_FIELDS:
            if field not in text_l:
                findings.append({
                    "id": f"B1-IL-{field[:6].upper()}",
                    "agent": "completeness",
                    "finding_type": "MISSING_INTERLOCK_FIELD",
                    "description": f"Interlock section '{title}' appears to lack '{field}' field.",
                    "severity": "HIGH",
                    "source": "B1",
                })

    # Check 3: undefined tags (tags in text not in inventory)
    inventory = set(doc_state.get("tag_inventory", []))
    prose_tags = set(ISA_TAG_RE.findall(
        " ".join(doc_state.get("sections", {}).values())
    ))
    undefined = prose_tags - inventory
    for tag in sorted(undefined)[:5]:
        findings.append({
            "id": f"B1-TAG-{tag}",
            "agent": "consistency",
            "finding_type": "UNDEFINED_TAG",
            "description": f"Tag {tag} referenced in text but not in tag inventory.",
            "severity": "WARNING",
            "source": "B1",
        })

    return findings


# ── Baseline B2: Single-LLM ───────────────────────────────────────────────────

def _b2_run(doc_state: dict) -> list[dict]:
    """
    Baseline B2: single LLM prompt for all quality dimensions.
    Sends one prompt covering completeness, consistency, ambiguity, logic.
    """
    import json as _json
    import yaml

    try:
        from langchain_ollama import OllamaLLM
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        llm = OllamaLLM(
            model=cfg["model"]["name"],
            base_url="http://localhost:11434",
            temperature=0,
        )
    except Exception as e:
        logger.warning("B2 LLM init failed: %s", e)
        return []

    sections = doc_state.get("sections", {})
    full_text = "\n\n".join(
        f"=== {title} ===\n{text[:400]}"
        for title, text in list(sections.items())[:8]
    )

    prompt = (
        "You are a control systems engineer reviewing a Control Narrative Document (CND).\n"
        "Review the following CND excerpt for ALL quality issues across these dimensions:\n"
        "1. Completeness — missing sections, missing interlock fields, missing alarm definitions\n"
        "2. Consistency — numerical contradictions, equipment type contradictions\n"
        "3. Ambiguity — vague language, missing numerical values\n"
        "4. Logic coherence — unsafe sequences, missing reset conditions\n\n"
        f"CND EXCERPT:\n{full_text[:3000]}\n\n"
        "Return a JSON array of findings. Each finding must have:\n"
        '{"type": "...", "agent": "completeness|consistency|ambiguity|logic_coherence", '
        '"severity": "CRITICAL|HIGH|WARNING|INFO", "description": "..."}\n'
        "Return ONLY the JSON array, no other text."
    )

    findings = []
    try:
        t0 = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - t0

        # Parse JSON from response
        m = re.search(r"\[.*\]", response, re.DOTALL)
        if m:
            raw = _json.loads(m.group())
            for i, item in enumerate(raw[:20]):
                findings.append({
                    "id": f"B2-{i:03d}",
                    "agent": item.get("agent", "completeness"),
                    "finding_type": item.get("type", "UNKNOWN"),
                    "description": item.get("description", ""),
                    "severity": item.get("severity", "WARNING"),
                    "source": "B2",
                })
        logger.info("B2 baseline: %d findings in %.1fs", len(findings), elapsed)
    except Exception as e:
        logger.warning("B2 run failed: %s", e)

    return findings
