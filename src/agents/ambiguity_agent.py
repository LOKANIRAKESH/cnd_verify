"""
src/agents/ambiguity_agent.py
Detects language in a CND that is too vague to be converted into PLC code.

Six ambiguity types (from Gervasi et al. 2019, adapted to industrial control):
  QUANTIFIER  — "sufficient", "high pressure", "adequate" (no number)
  TEMPORAL    — "shortly", "soon", "after some time" (no duration)
  REFERENTIAL — "it should open", "the pump" with no tag (no specific identifier)
  MODAL       — "should" instead of "shall" or "must" (ambiguous obligation)
  CONDITIONAL — "if high" without a specific threshold (incomplete condition)
  PASSIVE     — "shall be reset" without identifying who resets it (no agent)
"""

import re
import uuid
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


# ─────────────────────────────────────────────────────────────────────────────
# Ambiguity regex patterns  (pre-filter before LLM confirmation)
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS: dict[str, re.Pattern] = {
    "QUANTIFIER": re.compile(
        r"\b(sufficient(?:ly)?|adequate(?:ly)?|appropriate(?:ly)?|significant(?:ly)?|"
        r"excessive(?:ly)?|acceptable|reasonable|proper(?:ly)?|"
        r"high pressure|high temperature|high flow|low pressure|low temperature|low flow|"
        r"elevated|increased|decreased|normal(?:ly)?)\b",
        re.IGNORECASE,
    ),
    "TEMPORAL": re.compile(
        r"\b(shortly|soon|quickly|gradually|slowly|"
        r"after some time|within a reasonable time|"
        r"periodically|regularly|occasionally|frequently|"
        r"allow(?:ing)? (?:it |the (?:pump|valve|system) )?to (?:stabilize|stabilise|settle)|"
        r"once (?:stable|settled|ready))\b",
        re.IGNORECASE,
    ),
    "REFERENTIAL": re.compile(
        r"\b(it (?:should|shall|must|will|may)|"
        r"the pump\b(?! P-| [A-Z]-[0-9])|"
        r"the valve\b(?! [A-Z]{1,3}-[0-9])|"
        r"the (?:system|controller|sensor|transmitter|unit)\b|"
        r"this (?:pump|valve|sensor|transmitter|interlock|alarm)|"
        r"they (?:should|shall|must)|"
        r"the above)\b",
        re.IGNORECASE,
    ),
    "MODAL": re.compile(r"\bshould\b|\bmight\b|\bcould\b|\bmay\b", re.IGNORECASE),
    "CONDITIONAL": re.compile(
        r"\bif (?:pressure|flow|temperature|level|speed|current) (?:is |becomes |goes |rises |falls )?(?:high|low|elevated|excessive|abnormal)\b|"
        r"\bwhen (?:the )?(?:pressure|flow|temperature|level) (?:is |becomes )?(?:high|low|abnormal)\b|"
        r"\bif (?:there is|there are) (?:a |an )?(?:problem|issue|fault|failure)\b",
        re.IGNORECASE,
    ),
    "PASSIVE": re.compile(
        r"\bshall be (?:reset|cleared|acknowledged|closed|opened|started|stopped|initiated)\b|"
        r"\bwill be (?:reset|cleared|acknowledged|closed|opened|started|stopped)\b|"
        r"\bmust be (?:reset|cleared|acknowledged)\b",
        re.IGNORECASE,
    ),
}

# Sections worth scanning (skip preamble and revision history)
SCAN_SECTIONS = {"INTERLOCK", "ALARM", "STARTUP", "SHUTDOWN", "OPERATION", "CONTROL", "EMERGENCY", "MODE"}

# ISA tag pattern — helps detect when "the pump" is actually specific
ISA_TAG_RE = re.compile(r"\b[A-Z]{1,3}-[0-9]{1,4}[A-Z]?\b")

# Phrases that look like vague quantifiers but are actually fine in context
FALSE_POSITIVE_GUARDS = re.compile(
    r"\b(normal operating|normal range|normally open|normally closed|"
    r"normal state|normally de-energized)\b",
    re.IGNORECASE,
)


def _extract_context(text: str, match: re.Match, window: int = 200) -> str:
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    return text[start:end].strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM confirmation — filters out false positives
# ─────────────────────────────────────────────────────────────────────────────

def _confirm_ambiguity_with_llm(phrase: str, context: str, ambiguity_type: str, llm) -> dict | None:
    prompt = PromptTemplate.from_template(
        """You are a control systems engineer reviewing a Control Narrative Document.

A pre-filter flagged the phrase '{phrase}' as a potential {ambiguity_type} ambiguity.

CONTEXT (surrounding text):
{context}

QUESTION: Is this phrase genuinely ambiguous in a way that a PLC programmer CANNOT resolve
without making assumptions?

Criteria for YES (genuine ambiguity):
- QUANTIFIER: No specific number is given that the PLC can compare against
- TEMPORAL: No specific duration in seconds/minutes that can be a timer
- REFERENTIAL: No ISA tag (format: PT-101, XV-201) that identifies the specific equipment
- MODAL: 'should' used for a safety-critical action where 'shall' or 'must' is required
- CONDITIONAL: 'if high/low' without specifying the threshold value
- PASSIVE: Passive voice for a reset/acknowledgement action with no identified operator/system

Criteria for NO (acceptable usage):
- The phrase appears in a non-operational context (title, description, historical note)
- The same sentence provides the specific value elsewhere
- The specific equipment tag is visible within 3 words

Return JSON:
{{
  "is_ambiguous": true or false,
  "severity": "CRITICAL or HIGH or WARNING or INFO",
  "explanation": "one sentence why it is or isn't ambiguous",
  "programmable_alternative": "what the text should say to be unambiguous (only if is_ambiguous=true)"
}}

Return only JSON."""
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        result = chain.invoke({"phrase": phrase, "context": context, "ambiguity_type": ambiguity_type})
        if isinstance(result, dict) and result.get("is_ambiguous"):
            return result
    except Exception:
        pass
    return None


def _retrieve_kb(query: str, collection) -> str:
    if collection is None:
        return ""
    try:
        results = collection.query(query_texts=[query], n_results=2)
        if results["documents"] and results["documents"][0]:
            return "\n\n".join(results["documents"][0])
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Rule text and why-it-matters per ambiguity type
# ─────────────────────────────────────────────────────────────────────────────

_RULE_TEXT = {
    "QUANTIFIER": (
        "ISA-5.1 and process control practice require all setpoints, limits, and thresholds "
        "to be expressed as specific numerical values with units. "
        "Phrases like 'high pressure' or 'sufficient flow' cannot be entered into a PLC configuration."
    ),
    "TEMPORAL": (
        "All time-based actions in a CND must specify the duration in seconds or minutes. "
        "The PLC timer block requires a numerical preset value. "
        "Phrases like 'shortly' or 'allow to stabilize' cannot be programmed."
    ),
    "REFERENTIAL": (
        "ISA-5.1 requires every instrument and equipment reference to use the official tag "
        "(e.g., P-101, XV-201). Pronouns and generic nouns ('it', 'the pump') are ambiguous "
        "in documents with multiple similar instruments."
    ),
    "MODAL": (
        "ISA standards use 'shall' for mandatory requirements and 'should' for recommendations. "
        "Using 'should' for a safety-critical action creates legal and engineering ambiguity: "
        "is this action mandatory or optional?"
    ),
    "CONDITIONAL": (
        "Every conditional in a CND must specify the complete condition including the threshold value "
        "and units. 'If pressure is high' without a value cannot be implemented as a PLC comparator."
    ),
    "PASSIVE": (
        "Passive voice for control actions (e.g., 'interlock shall be reset') omits who or what "
        "performs the action. The PLC cannot execute 'shall be reset' — it needs either "
        "'operator must press the RESET button on HMI' or 'system resets automatically when...'"
    ),
}

_WHY_IT_MATTERS = {
    "QUANTIFIER": (
        "The PLC programmer cannot configure a comparator block without a specific value. "
        "They must invent a value, guess from context, or leave the block unconfigured. "
        "Any of these creates a hidden defect that is not traceable to the CND."
    ),
    "TEMPORAL": (
        "Every PLC timer requires a preset time in milliseconds or seconds. "
        "Without a specific duration, the programmer guesses — and guesses on timing can cause "
        "premature trips, missed protective actions, or equipment damage."
    ),
    "REFERENTIAL": (
        "In a system with multiple pumps or valves, 'the pump' or 'it' could refer to any of them. "
        "Wrong equipment identification means the control action targets the wrong device."
    ),
    "MODAL": (
        "'Should' implies the action is a best practice, not a requirement. "
        "A programmer may skip a 'should' action to simplify code. "
        "If the action was actually safety-critical, the omission creates a safety hazard."
    ),
    "CONDITIONAL": (
        "A PLC comparator block requires a specific threshold: 'IF PT-101 > 150 PSI'. "
        "It cannot evaluate 'IF pressure is high'. The programmer must choose a threshold — "
        "and a wrong choice means the protective action fires too early or not at all."
    ),
    "PASSIVE": (
        "If the CND says 'interlock shall be reset' without specifying how, the programmer "
        "may implement automatic reset. For a safety interlock, automatic reset (without "
        "operator investigation) is dangerous and violates ISA-84 best practices."
    ),
}

_SEVERITY_MAP = {
    "QUANTIFIER": "WARNING",
    "TEMPORAL": "WARNING",
    "REFERENTIAL": "HIGH",
    "MODAL": "WARNING",
    "CONDITIONAL": "HIGH",
    "PASSIVE": "WARNING",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def run_ambiguity_agent(state: dict) -> dict:
    config = load_config()
    llm = OllamaLLM(model=config["model"]["name"], temperature=0.0)

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
    document_text: str = state["document_text"]

    findings: list[Finding] = []
    seen: set[str] = set()  # deduplicate by (section, phrase)

    for section_title, section_text in sections.items():
        # Only scan operational sections — skip preamble, revision history
        if not any(kw in section_title.upper() for kw in SCAN_SECTIONS):
            continue

        section_page = sections_pages.get(section_title, 0)

        for ambiguity_type, pattern in PATTERNS.items():
            for match in pattern.finditer(section_text):
                phrase = match.group(0).strip()

                # Skip false positives (e.g., "normally open" is NOT vague)
                if FALSE_POSITIVE_GUARDS.search(phrase):
                    continue

                # Deduplicate same phrase in same section
                dedup_key = f"{section_title}::{phrase.lower()}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Get context window
                context = _extract_context(section_text, match)

                # LLM confirmation — only keep genuine ambiguities
                confirmation = _confirm_ambiguity_with_llm(phrase, context, ambiguity_type, llm)
                if confirmation is None:
                    continue

                page = _find_text_page(document_text, context[:40], section_page)

                # Get relevant KB context for rule text
                kb_context = _retrieve_kb(
                    f"ambiguity {ambiguity_type.lower()} control narrative PLC programming vague", kb
                )

                # Build the full quote (sentence containing the phrase)
                quote_sentence = _extract_sentence(section_text, match)

                severity = confirmation.get("severity") or _SEVERITY_MAP.get(ambiguity_type, "WARNING")
                programmable_alternative = confirmation.get("programmable_alternative", "")

                findings.append(Finding(
                    id=f"AMB-{uuid.uuid4().hex[:8].upper()}",
                    agent="ambiguity",
                    finding_type=f"AMBIGUITY_{ambiguity_type}",
                    description=(
                        f"{ambiguity_type} ambiguity: '{phrase}' — "
                        f"{confirmation.get('explanation', 'vague phrase that cannot be programmed')}"
                    ),
                    page_number=page,
                    section_title=section_title,
                    quote=quote_sentence,
                    rule_violated=f"Unambiguous language requirement — {ambiguity_type}",
                    rule_text=_RULE_TEXT.get(ambiguity_type, ""),
                    why_it_matters=_WHY_IT_MATTERS.get(ambiguity_type, ""),
                    severity=severity,
                    suggestion=(
                        f"Replace '{phrase}' with a specific, programmable expression. "
                        + (f"Suggested replacement: '{programmable_alternative}'" if programmable_alternative else
                           f"Add specific {'value' if ambiguity_type in ('QUANTIFIER','CONDITIONAL') else 'duration' if ambiguity_type == 'TEMPORAL' else 'tag' if ambiguity_type == 'REFERENTIAL' else 'detail'}.")
                    ),
                    evidence_source=f"Regex detection confirmed by LLM — {ambiguity_type} ambiguity type",
                    status="OPEN",
                ))

    report = _build_report(findings, config)
    state["ambiguity_report"] = report.model_dump()
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


def _extract_sentence(text: str, match: re.Match) -> str:
    start = max(0, text.rfind(".", 0, match.start()) + 1)
    end = text.find(".", match.end())
    if end == -1:
        end = min(len(text), match.end() + 200)
    return text[start:end + 1].strip()


def _build_report(findings: list[Finding], config: dict) -> AgentReport:
    critical = sum(1 for f in findings if f.severity == "CRITICAL")
    high = sum(1 for f in findings if f.severity == "HIGH")
    warning = sum(1 for f in findings if f.severity == "WARNING")
    info = sum(1 for f in findings if f.severity == "INFO")
    p = config["scoring"]["penalty_per_finding"]
    score = max(0.0, 1.0 - critical * p["CRITICAL"] - high * p["HIGH"] - warning * p["WARNING"] - info * p["INFO"])
    return AgentReport(
        agent_name="ambiguity",
        findings=findings,
        score=round(score, 3),
        critical_count=critical,
        high_count=high,
        warning_count=warning,
        info_count=info,
    )
