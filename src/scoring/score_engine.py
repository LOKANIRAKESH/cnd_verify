"""
src/scoring/score_engine.py
Computes the Verification Quality Score (VQS) from all agent reports.
Applies hard fail conditions that override the composite score.
Routes the document to AUTO_PASS, NEEDS_REVIEW, or FAIL.
"""

import yaml
from src.orchestration.state import VQSScores, VerificationState


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_agent_score(report: dict, penalties: dict) -> float:
    """
    Recompute agent score from findings list.
    Returns 0.5 when an agent report is absent (agent failed to run)
    so the composite score is degraded rather than inflated to 1.0.
    """
    if not report:
        return 0.5   # agent did not run — treat as uncertain, not perfect

    score = 1.0
    for finding in report.get("findings", []):
        severity = finding.get("severity", "INFO")
        score -= penalties.get(severity, 0.0)

    return round(max(0.0, score), 3)


def apply_hard_fail_conditions(state: VerificationState) -> str | None:
    """
    Check for conditions that force MANDATORY_REVISION regardless of score.
    Returns the reason string if hard fail, or None if no hard fail.
    Three conditions checked (per design spec):
      1. Circular interlock
      2. Missing critical sections (Emergency Shutdown, Interlocks, Alarms)
      3. >20% of prose-referenced tags absent from tag inventory
    NOTE: called before all_findings is written to state, so we derive tags
    directly from the agent reports rather than reading state["all_findings"].
    """
    import re
    reasons = []

    if state.get("has_circular_interlock"):
        reasons.append("Circular interlock detected — system cannot enter a safe state")

    missing_critical = state.get("missing_critical_sections", [])
    if missing_critical:
        reasons.append(
            f"Critical section(s) missing: {', '.join(missing_critical)}"
        )

    # Condition 3 — unresolved tag references (>20% of all prose tags not in inventory)
    tag_inventory: list = state.get("tag_inventory", [])
    if tag_inventory:
        inventory_set = set(tag_inventory)
        prose_tags: set[str] = set(tag_inventory)
        for report_key in ("completeness_report", "consistency_report",
                           "ambiguity_report", "image_report"):
            report = state.get(report_key) or {}
            for f in report.get("findings", []):
                for t in re.findall(
                    r"\b[A-Z]{1,3}-[0-9]{1,4}[A-Z]?\b",
                    f.get("description", "") + " " + f.get("quote", ""),
                ):
                    prose_tags.add(t)
        undefined = prose_tags - inventory_set
        ratio = len(undefined) / len(prose_tags)
        if ratio > 0.20:
            reasons.append(
                f"{len(undefined)}/{len(prose_tags)} prose-referenced tags "
                f"({ratio*100:.0f}%) absent from tag inventory — "
                f"undefined: {', '.join(sorted(undefined)[:10])}"
                + (" ..." if len(undefined) > 10 else "")
            )

    if reasons:
        return " | ".join(reasons)
    return None


def collect_all_findings(state: VerificationState) -> list[dict]:
    """Merge findings from all agent reports into one flat list."""
    all_findings = []
    for report_key in ["completeness_report", "consistency_report",
                       "ambiguity_report", "logic_coherence_report", "image_report"]:
        report = state.get(report_key)
        if report and "findings" in report:
            all_findings.extend(report["findings"])
    return all_findings


def deduplicate_findings(findings: list[dict]) -> list[dict]:
    """
    Remove near-duplicate findings from different agents.
    Key includes agent + finding_type + first 60 chars of description
    so that two MISSING_SECTION findings for different sections are NOT collapsed.
    """
    seen = set()
    unique = []
    for f in findings:
        key = (
            f.get("agent", ""),
            f.get("finding_type", ""),
            f.get("description", "")[:60],
        )
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def route_document(composite_score: float, hard_fail_reason: str | None, thresholds: dict) -> str:
    """
    Determine routing decision based on composite score and hard fail conditions.
    """
    if hard_fail_reason:
        return "MANDATORY_REVISION"

    if composite_score >= thresholds["pass"]:
        return "AUTO_PASS"
    elif composite_score >= thresholds["review"]:
        return "NEEDS_REVIEW"
    else:
        return "FAIL"


def run_score_engine(state: VerificationState) -> VerificationState:
    """
    Main entry point for the Score Engine.
    Reads all agent reports, computes VQS, routes the document.
    """
    config = load_config()
    weights = config["scoring"]["weights"]
    penalties = config["scoring"]["penalty_per_finding"]
    thresholds = config["scoring"]["thresholds"]

    # ── Compute per-agent scores ──────────────────────────────────────────────
    scores = {
        "completeness":    compute_agent_score(state.get("completeness_report"),    penalties),
        "consistency":     compute_agent_score(state.get("consistency_report"),     penalties),
        "ambiguity":       compute_agent_score(state.get("ambiguity_report"),       penalties),
        "logic_coherence": compute_agent_score(state.get("logic_coherence_report"), penalties),
        "image":           compute_agent_score(state.get("image_report"),           penalties),
    }

    # ── Weighted composite ────────────────────────────────────────────────────
    # image_agent contributes a small share (0.05) drawn proportionally from
    # the other weights so the total remains 1.0
    composite = round(
        weights["completeness"]    * 0.95 * scores["completeness"]
        + weights["consistency"]   * 0.95 * scores["consistency"]
        + weights["ambiguity"]     * 0.95 * scores["ambiguity"]
        + weights.get("logic_coherence", 0.15) * 0.95 * scores["logic_coherence"]
        + 0.05                                        * scores["image"],
        3,
    )

    # ── Hard fail check ───────────────────────────────────────────────────────
    hard_fail_reason = apply_hard_fail_conditions(state)

    # ── Routing decision ──────────────────────────────────────────────────────
    routing = route_document(composite, hard_fail_reason, thresholds)

    # ── Collect and deduplicate all findings ──────────────────────────────────
    all_findings = deduplicate_findings(collect_all_findings(state))

    vqs = VQSScores(
        completeness=scores["completeness"],
        consistency=scores["consistency"],
        ambiguity=scores["ambiguity"],
        logic_coherence=scores["logic_coherence"],
        image=scores["image"],
        composite=composite,
        routing_decision=routing,
    )

    state["vqs_scores"] = vqs.model_dump()
    state["all_findings"] = all_findings

    return state


def get_routing_message(routing_decision: str, composite_score: float, hard_fail_reason: str = None) -> dict:
    """Return display message and colour for the routing decision."""
    messages = {
        "AUTO_PASS": {
            "message": "Document passed verification. Ready for PLC code extraction.",
            "color": "green",
            "icon": "✓",
        },
        "NEEDS_REVIEW": {
            "message": "Document requires engineer review before proceeding. Address flagged findings.",
            "color": "orange",
            "icon": "⚠",
        },
        "FAIL": {
            "message": "Document failed verification. Return to authors for significant revision.",
            "color": "red",
            "icon": "✗",
        },
        "MANDATORY_REVISION": {
            "message": f"BLOCKED — Critical issue detected: {hard_fail_reason or 'see findings'}",
            "color": "red",
            "icon": "🚫",
        },
    }
    return messages.get(routing_decision, messages["NEEDS_REVIEW"])
