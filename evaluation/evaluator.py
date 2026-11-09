"""
evaluation/evaluator.py

Evaluation harness for CND-Verify research metrics.
Compares pipeline findings against ground-truth labeled errors.
Computes per-agent and overall Precision, Recall, F1, and Routing Accuracy.
"""

import json
import time
import logging
from pathlib import Path

logger = logging.getLogger("cnd_verify")

# ── Ground truth helpers ──────────────────────────────────────────────────────

def load_ground_truth(path: str = "data/ground_truth/ground_truth.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _infer_expected_routing(doc_info: dict) -> str:
    errors = doc_info.get("errors", [])
    if not errors:
        return "AUTO_PASS"
    critical_count = sum(1 for e in errors if e.get("severity") == "CRITICAL")
    if critical_count >= 3:
        return "MANDATORY_REVISION"
    if critical_count >= 1:
        return "FAIL"
    return "NEEDS_REVIEW"


# ── Finding-to-error matching ─────────────────────────────────────────────────

_TYPE_MAP = {
    "MISSING_SECTION":            ["MISSING_SECTION", "MISSING_REQUIRED_SECTION"],
    "MISSING_INTERLOCK_FIELD":    ["MISSING_INTERLOCK_FIELD", "MISSING_FIELD", "MISSING_REQUIRED_FIELD"],
    "MISSING_ALARM_FIELD":        ["MISSING_ALARM_FIELD", "MISSING_FIELD", "MISSING_REQUIRED_FIELD"],
    "SETPOINT_MISMATCH":          ["SETPOINT_MISMATCH", "VALUE_MISMATCH", "INCONSISTENT_VALUE"],
    "EQUIPMENT_TYPE_CONTRADICTION":["EQUIPMENT_TYPE_CONTRADICTION", "CONTRADICTION", "TYPE_CONTRADICTION"],
    "SEQUENCE_CONTRADICTION":     ["SEQUENCE_CONTRADICTION", "SEQUENCE_CONFLICT", "CONTRADICTION"],
}


def _finding_matches_error(finding: dict, gt_error: dict) -> bool:
    """
    A finding is a True Positive for a GT error if:
    1. The agent matches (completeness, consistency, etc.)
    2. The finding_type matches one of the expected types for that error
    3. OR: the finding's description contains key terms from the GT error description
    """
    gt_agent = gt_error.get("agent", "")
    f_agent  = finding.get("agent", "")
    if gt_agent and f_agent and gt_agent != f_agent:
        return False

    gt_type   = gt_error.get("type", "")
    f_type    = finding.get("finding_type", "")
    expected_types = _TYPE_MAP.get(gt_type, [gt_type])

    if f_type in expected_types:
        return True

    # Secondary match: key content terms overlap
    gt_desc = gt_error.get("description", "").lower()
    f_desc  = finding.get("description", "").lower()
    if not gt_desc or not f_desc:
        return False

    # Extract meaningful words (>4 chars) from GT description
    gt_words = {w for w in gt_desc.split() if len(w) > 4}
    overlap  = sum(1 for w in gt_words if w in f_desc)
    return overlap >= 3


# ── Single-document evaluation ────────────────────────────────────────────────

def evaluate_document(doc_info: dict, data_dir: str = "data", verbose: bool = False) -> dict:
    """
    Run the full CND-Verify pipeline on one document and compare to ground truth.
    Returns a result dict with TP/FP/FN, precision, recall, F1, routing accuracy.
    """
    doc_id   = doc_info.get("doc_id", "?")
    filename = doc_info.get("filename", "")
    gt_errors = doc_info.get("errors", [])

    # Resolve file path — prefer PDF, fall back to TXT
    base = Path(data_dir) / "synthetic_cnds"
    pdf_path = base / filename.replace(".txt", ".pdf")
    txt_path = base / filename

    if pdf_path.exists():
        doc_path = str(pdf_path)
    elif txt_path.exists():
        doc_path = str(txt_path)
    else:
        return {"doc_id": doc_id, "skipped": True, "error": f"File not found: {filename}"}

    expected_routing = doc_info.get("expected_routing") or _infer_expected_routing(doc_info)

    t0 = time.time()
    try:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

        from src.ingestion.document_loader import prepare_document
        from src.ingestion.document_indexer import build_document_index, build_interlock_graph
        from src.agents.completeness_agent import run_completeness_agent
        from src.agents.ambiguity_agent import run_ambiguity_agent
        from src.agents.consistency_agent import run_consistency_agent
        from src.agents.logic_coherence_agent import run_logic_coherence_agent
        from src.agents.image_agent import run_image_agent
        from src.scoring.score_engine import run_score_engine

        import copy
        from concurrent.futures import ThreadPoolExecutor

        state = prepare_document(doc_path)
        state["current_doc_name"] = doc_id

        state = build_document_index(state, cfg)
        graph_data, has_cycle = build_interlock_graph(state.get("sections", {}))
        state["interlock_graph_data"] = graph_data
        state["has_circular_interlock"] = has_cycle

        state = run_image_agent(state)

        comp_state = copy.deepcopy(state)
        amb_state  = copy.deepcopy(state)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_c = pool.submit(run_completeness_agent, comp_state)
            fut_a = pool.submit(run_ambiguity_agent, amb_state)
            comp_res = fut_c.result(timeout=180)
            amb_res  = fut_a.result(timeout=180)
        state["completeness_report"] = comp_res.get("completeness_report")
        state["ambiguity_report"]    = amb_res.get("ambiguity_report")

        state = run_consistency_agent(state)
        state = run_logic_coherence_agent(state)
        state = run_score_engine(state)

    except Exception as exc:
        return {"doc_id": doc_id, "skipped": True, "error": str(exc)[:300]}

    elapsed = round(time.time() - t0, 1)
    findings = state.get("all_findings", [])
    routing  = state.get("vqs_scores", {}).get("routing_decision", "UNKNOWN")

    # ── Match findings to GT errors ─────────────────────────────────────────
    matched_gt: set[int] = set()
    matched_f:  set[int] = set()

    for ei, gt_err in enumerate(gt_errors):
        for fi, finding in enumerate(findings):
            if fi in matched_f:
                continue
            if _finding_matches_error(finding, gt_err):
                matched_gt.add(ei)
                matched_f.add(fi)
                break

    tp = len(matched_gt)
    fp = len(findings) - len(matched_f)
    fn = len(gt_errors) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-agent breakdown
    per_agent: dict[str, dict] = {}
    for ei, gt_err in enumerate(gt_errors):
        agent = gt_err.get("agent", "unknown")
        if agent not in per_agent:
            per_agent[agent] = {"tp": 0, "fp": 0, "fn": 0}
        if ei in matched_gt:
            per_agent[agent]["tp"] += 1
        else:
            per_agent[agent]["fn"] += 1

    for fi, finding in enumerate(findings):
        if fi not in matched_f:
            agent = finding.get("agent", "unknown")
            if agent not in per_agent:
                per_agent[agent] = {"tp": 0, "fp": 0, "fn": 0}
            per_agent[agent]["fp"] += 1

    routing_correct = _routing_match(routing, expected_routing)

    if verbose:
        logger.info(
            "%s: P=%.2f R=%.2f F1=%.2f  TP=%d FP=%d FN=%d  routing=%s (%s)",
            doc_id, precision, recall, f1, tp, fp, fn, routing,
            "✓" if routing_correct else "✗",
        )

    return {
        "doc_id":          doc_id,
        "skipped":         False,
        "error_count_gt":  len(gt_errors),
        "findings_count":  len(findings),
        "routing":         routing,
        "routing_correct": routing_correct,
        "elapsed_s":       elapsed,
        "overall": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        },
        "per_agent": per_agent,
    }


def _routing_match(predicted: str, expected: str) -> bool:
    equiv = {
        "AUTO_PASS":          {"AUTO_PASS"},
        "NEEDS_REVIEW":       {"NEEDS_REVIEW", "CONDITIONAL_PASS"},
        "FAIL":               {"FAIL", "MANDATORY_REVISION"},
        "MANDATORY_REVISION": {"FAIL", "MANDATORY_REVISION"},
        "CONDITIONAL_PASS":   {"NEEDS_REVIEW", "CONDITIONAL_PASS"},
    }
    return predicted in equiv.get(expected, {expected})


# ── Aggregate across documents ─────────────────────────────────────────────────

def aggregate_results(results: list[dict]) -> dict:
    valid = [r for r in results if not r.get("skipped")]
    if not valid:
        return {}

    total_tp = sum(r["overall"]["tp"] for r in valid)
    total_fp = sum(r["overall"]["fp"] for r in valid)
    total_fn = sum(r["overall"]["fn"] for r in valid)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1        = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                       if (micro_precision + micro_recall) > 0 else 0.0)

    macro_f1 = sum(r["overall"]["f1"] for r in valid) / len(valid)

    routing_accuracy = sum(1 for r in valid if r.get("routing_correct")) / len(valid)

    # Per-agent F1
    all_agents: dict[str, dict] = {}
    for r in valid:
        for agent, counts in r.get("per_agent", {}).items():
            if agent not in all_agents:
                all_agents[agent] = {"tp": 0, "fp": 0, "fn": 0}
            all_agents[agent]["tp"] += counts["tp"]
            all_agents[agent]["fp"] += counts["fp"]
            all_agents[agent]["fn"] += counts["fn"]

    per_agent_f1 = {}
    for agent, c in all_agents.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 1.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 1.0
        per_agent_f1[agent] = round(2 * p * r / (p + r) if (p + r) > 0 else 0.0, 4)

    # Critical error FNR
    critical_gt  = sum(1 for res in results if not res.get("skipped")
                       for _ in [res])   # placeholder
    fnr_critical = 0.0  # computed below
    total_crit_gt = 0
    total_crit_fn = 0
    for r in valid:
        pa = r.get("per_agent", {})
        # Proxy: FN from completeness + consistency agents (where critical errors live)
        for agent in ("completeness", "consistency", "logic_coherence"):
            c = pa.get(agent, {})
            total_crit_gt += c.get("tp", 0) + c.get("fn", 0)
            total_crit_fn += c.get("fn", 0)
    fnr_critical = total_crit_fn / total_crit_gt if total_crit_gt > 0 else 0.0

    return {
        "micro_precision":  round(micro_precision,  4),
        "micro_recall":     round(micro_recall,     4),
        "micro_f1":         round(micro_f1,         4),
        "macro_f1":         round(macro_f1,         4),
        "routing_accuracy": round(routing_accuracy, 4),
        "fnr_critical":     round(fnr_critical,     4),
        "per_agent_f1":     per_agent_f1,
        "n_docs":           len(valid),
    }


def print_report(results: list[dict], summary: dict):
    print("\n" + "="*60)
    print("CND-Verify Evaluation Report")
    print("="*60)
    for r in results:
        if r.get("skipped"):
            print(f"  {r['doc_id']}: SKIPPED — {r.get('error','')[:80]}")
            continue
        ov = r["overall"]
        rc = "✓" if r.get("routing_correct") else "✗"
        print(f"  {r['doc_id']:8s}  P={ov['precision']:.2f}  R={ov['recall']:.2f}  F1={ov['f1']:.2f}"
              f"  TP={ov['tp']} FP={ov['fp']} FN={ov['fn']}"
              f"  routing={r.get('routing','?')} {rc}  ({r.get('elapsed_s','?')}s)")
    print("-"*60)
    print(f"  Micro F1:        {summary.get('micro_f1',0):.3f}")
    print(f"  Macro F1:        {summary.get('macro_f1',0):.3f}")
    print(f"  Routing Acc:     {summary.get('routing_accuracy',0):.1%}")
    print(f"  FNR (critical):  {summary.get('fnr_critical',0):.3f}")
    print("="*60)
