"""
src/orchestration/state.py
Shared data contract between all agents in the CND-Verify pipeline.
Every agent reads from and writes to VerificationState.
"""

from typing import TypedDict, Optional
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """
    A single defect found by an agent.
    Every field maps to a visible section in the UI finding card.
    """

    id: str
    agent: str                  # completeness | consistency | ambiguity

    # WHAT is wrong
    finding_type: str           # MISSING_SECTION | MISSING_FIELD | SETPOINT_MISMATCH | AMBIGUITY | etc.
    description: str            # One clear sentence: what the problem is

    # WHERE it is
    page_number: int            # Page number in the document (estimated for text files)
    section_title: str = ""     # Section name (e.g., "Section 6.1 — Interlocks")

    # EVIDENCE — exact text from the document
    quote: str                  # The exact sentence or phrase that triggered this finding

    # BASIS — which rule was violated and what it says
    rule_violated: str = ""     # Short rule name (e.g., "ISA-18.2 Required Alarm Fields")
    rule_text: str = ""         # The actual rule text from the standard or KB

    # WHY IT MATTERS — consequence if not fixed
    why_it_matters: str = ""    # What goes wrong during PLC programming or operation

    # HOW TO FIX
    suggestion: str             # Specific, actionable correction

    severity: str               # CRITICAL | HIGH | WARNING | INFO
    evidence_source: str = ""   # ISA-18.2 | ISA-88 | ISA-5.1 | Domain KB | Document cross-check
    status: str = "OPEN"        # OPEN | ACCEPTED | REJECTED by engineer


class AgentReport(BaseModel):
    """Summary report produced by one agent."""

    agent_name: str
    findings: list[Finding] = Field(default_factory=list)
    score: float = 1.0          # 0.0 to 1.0, starts at 1.0 and decreases per finding
    critical_count: int = 0
    high_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    error_message: Optional[str] = None   # Set if agent failed to run


class VQSScores(BaseModel):
    """Verification Quality Scores — one per agent plus composite."""

    completeness: float = 1.0
    consistency: float = 1.0
    ambiguity: float = 1.0
    composite: float = 1.0
    routing_decision: str = "PENDING"  # AUTO_PASS | NEEDS_REVIEW | FAIL | MANDATORY_REVISION


class VerificationState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    Design rule: every agent reads what it needs and adds its own report.
    No agent modifies another agent's report.
    """

    # Input
    document_path: str
    document_text: str

    # Extracted from document by document_loader
    sections: dict                  # {section_title: section_text}
    sections_pages: dict            # {section_title: page_number}
    tag_inventory: list[str]        # All ISA tags found (e.g., ["PT-101", "XV-201"])
    setpoint_table: list[dict]      # [{tag, parameter, value, units, page}]

    # ── Pass 1: Global Knowledge Store (built by document_indexer) ────────────
    doc_collection_name: str        # ChromaDB collection name for this document
    tag_index: dict                 # {tag: [{page, section, text}]} — all mentions
    setpoint_index: dict            # {tag: {param: {value, unit, page}}}
    definition_map: dict            # {tag: {type, class, description, page}}
    chunk_groups: list              # [{group_id, section_titles, combined_text, pages}]

    # Agent outputs — each agent appends its report here
    completeness_report: Optional[dict]   # AgentReport serialised to dict
    consistency_report: Optional[dict]
    ambiguity_report: Optional[dict]
    image_report: Optional[dict]    # NEW — image cross-validation agent

    # Image metadata for UI display
    image_summary: list             # [{page, class, width, height, tags_found, thumbnail}]

    # Combined findings (populated by score_engine after all agents complete)
    all_findings: list[dict]

    # Scoring
    vqs_scores: Optional[dict]     # VQSScores serialised to dict

    # ── PLC Code Generation (built after verification) ────────────────────────
    plc_code: dict                  # {tag: {code, equipment_class, page, clean}}
    plc_summary: dict               # {clean_count, dirty_count, skipped_tags, ...}

    # Hard fail flags
    has_circular_interlock: bool
    missing_critical_sections: list[str]

    # Pipeline control
    error: Optional[str]           # Set if pipeline fails — stops further processing
