"""
evaluation/run_eval.py
======================
Run the full CND-Verify evaluation:
  - B1  (rule-based checklist)
  - B2  (single-LLM)
  - Full pipeline (CND-Verify)

against the ground truth in data/ground_truth/ground_truth.json.

Usage:
    cd cnd-verify
    python -m evaluation.run_eval            # all three systems
    python -m evaluation.run_eval --b1-only  # just B1 (fast, no LLM)
    python -m evaluation.run_eval --skip-b2  # skip slow single-LLM baseline
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cnd_eval")


# ── helpers ───────────────────────────────────────────────────────────────────

def _match(finding: dict, gt_error: dict) -> bool:
    """True if finding is a TP for gt_error."""
    gt_agent = gt_error.get("agent", "")
    f_agent  = finding.get("agent", "")
    if gt_agent and f_agent and gt_agent != f_agent:
        return False

    gt_type = gt_error.get("type", "")
    f_type  = finding.get("finding_type", "")

    _TYPE_MAP = {
        "MISSING_SECTION":              ["MISSING_SECTION", "MISSING_REQUIRED_SECTION"],
        "MISSING_INTERLOCK_FIELD":      ["MISSING_INTERLOCK_FIELD", "MISSING_FIELD"],
        "MISSING_ALARM_FIELD":          ["MISSING_ALARM_FIELD", "MISSING_FIELD"],
        "SETPOINT_MISMATCH":            ["SETPOINT_MISMATCH", "VALUE_MISMATCH"],
        "EQUIPMENT_TYPE_CONTRADICTION": ["EQUIPMENT_TYPE_CONTRADICTION", "CONTRADICTION"],
        "SEQUENCE_CONTRADICTION":       ["SEQUENCE_ORDERING_CONTRADICTION",
                                         "UNSAFE_SEQUENCE_ORDER", "SEQUENCE_CONTRADICTION"],
    }
    if f_type in _TYPE_MAP.get(gt_type, [gt_type]):
        return True

    # Fallback: word-overlap check
    gt_desc = gt_error.get("description", "").lower()
    f_desc  = finding.get("description", "").lower()
    words   = {w for w in gt_desc.split() if len(w) > 4}
    return sum(1 for w in words if w in f_desc) >= 3


def _score(findings: list[dict], gt_errors: list[dict]) -> dict:
    """Compute TP/FP/FN/P/R/F1 for one document."""
    matched_gt: set[int] = set()
    matched_f:  set[int] = set()
    for ei, gt in enumerate(gt_errors):
        for fi, f in enumerate(findings):
            if fi in matched_f:
                continue
            if _match(f, gt):
                matched_gt.add(ei)
                matched_f.add(fi)
                break
    tp = len(matched_gt)
    fp = len(findings) - len(matched_f)
    fn = len(gt_errors) - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall    = tp / (tp + fn) if (tp + fn) else 1.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall":    round(recall,    3),
            "f1":        round(f1,        3)}


def _infer_routing(doc_info: dict) -> str:
    errors = doc_info.get("errors", [])
    if not errors:
        return "AUTO_PASS"
    criticals = sum(1 for e in errors if e.get("severity") == "CRITICAL")
    if criticals >= 2:
        return "MANDATORY_REVISION"
    if criticals >= 1:
        return "NEEDS_REVIEW"
    return "NEEDS_REVIEW"


def _routing_match(predicted: str, expected: str) -> bool:
    equiv = {
        "AUTO_PASS":          {"AUTO_PASS"},
        "NEEDS_REVIEW":       {"NEEDS_REVIEW", "CONDITIONAL_PASS"},
        "FAIL":               {"FAIL", "MANDATORY_REVISION"},
        "MANDATORY_REVISION": {"FAIL", "MANDATORY_REVISION"},
    }
    return predicted in equiv.get(expected, {expected})


# ── B1 evaluation ─────────────────────────────────────────────────────────────

def _run_b1_on_doc(doc_info: dict, data_dir: str) -> dict:
    from evaluation.baselines import _b1_run
    from src.ingestion.document_loader import prepare_document

    base = Path(data_dir) / "synthetic_cnds"
    filename = doc_info["filename"]
    doc_path = str(base / filename.replace(".txt", ".pdf"))
    if not Path(doc_path).exists():
        doc_path = str(base / filename)
    if not Path(doc_path).exists():
        return {"doc_id": doc_info["doc_id"], "skipped": True,
                "error": f"File not found: {filename}"}

    try:
        t0 = time.time()
        state = prepare_document(doc_path)
        findings = _b1_run(state)
        elapsed = round(time.time() - t0, 1)
    except Exception as e:
        return {"doc_id": doc_info["doc_id"], "skipped": True, "error": str(e)[:200]}

    scores = _score(findings, doc_info.get("errors", []))
    expected_routing = _infer_routing(doc_info)
    return {
        "doc_id": doc_info["doc_id"],
        "skipped": False,
        "findings_count": len(findings),
        "elapsed_s": elapsed,
        "routing": "N/A",
        "routing_correct": False,
        "overall": scores,
    }


# ── B2 evaluation ─────────────────────────────────────────────────────────────

def _run_b2_on_doc(doc_info: dict, data_dir: str) -> dict:
    from evaluation.baselines import _b2_run
    from src.ingestion.document_loader import prepare_document

    base = Path(data_dir) / "synthetic_cnds"
    filename = doc_info["filename"]
    doc_path = str(base / filename.replace(".txt", ".pdf"))
    if not Path(doc_path).exists():
        doc_path = str(base / filename)
    if not Path(doc_path).exists():
        return {"doc_id": doc_info["doc_id"], "skipped": True,
                "error": f"File not found: {filename}"}

    try:
        t0 = time.time()
        state = prepare_document(doc_path)
        findings = _b2_run(state)
        elapsed = round(time.time() - t0, 1)
    except Exception as e:
        return {"doc_id": doc_info["doc_id"], "skipped": True, "error": str(e)[:200]}

    scores = _score(findings, doc_info.get("errors", []))
    return {
        "doc_id": doc_info["doc_id"],
        "skipped": False,
        "findings_count": len(findings),
        "elapsed_s": elapsed,
        "routing": "N/A",
        "routing_correct": False,
        "overall": scores,
    }


# ── Full pipeline evaluation ───────────────────────────────────────────────────

def _run_pipeline_on_doc(doc_info: dict, data_dir: str) -> dict:
    from evaluation.evaluator import evaluate_document
    return evaluate_document(doc_info, data_dir=data_dir, verbose=True)


# ── Aggregate ─────────────────────────────────────────────────────────────────

def _aggregate(results: list[dict]) -> dict:
    valid = [r for r in results if not r.get("skipped")]
    if not valid:
        return {}
    tp = sum(r["overall"]["tp"] for r in valid)
    fp = sum(r["overall"]["fp"] for r in valid)
    fn = sum(r["overall"]["fn"] for r in valid)
    p  = tp / (tp + fp) if (tp + fp) else 1.0
    r_ = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
    routing_acc = sum(1 for r in valid if r.get("routing_correct", False)) / len(valid)
    return {
        "n_docs": len(valid),
        "micro_precision": round(p,  3),
        "micro_recall":    round(r_, 3),
        "micro_f1":        round(f1, 3),
        "routing_accuracy": round(routing_acc, 3),
        "total_tp": tp, "total_fp": fp, "total_fn": fn,
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def _print_table(system_name: str, results: list[dict], summary: dict):
    w = 60
    print("\n" + "-" * w)
    print(f"  {system_name}")
    print("-" * w)
    for r in results:
        if r.get("skipped"):
            print(f"  {r['doc_id']:8s}  SKIPPED -- {r.get('error','')[:50]}")
            continue
        ov = r["overall"]
        rc = "OK" if r.get("routing_correct") else "WRONG"
        print(
            f"  {r['doc_id']:8s}  "
            f"TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}  "
            f"P={ov['precision']:.2f}  R={ov['recall']:.2f}  F1={ov['f1']:.2f}  "
            f"routing={r.get('routing','N/A')} [{rc}]  ({r.get('elapsed_s','?')}s)"
        )
    if summary:
        print("-" * w)
        print(
            f"  TOTAL  TP={summary['total_tp']}  FP={summary['total_fp']}  "
            f"FN={summary['total_fn']}  "
            f"Micro-P={summary['micro_precision']:.3f}  "
            f"Micro-R={summary['micro_recall']:.3f}  "
            f"Micro-F1={summary['micro_f1']:.3f}  "
            f"Routing-Acc={summary['routing_accuracy']:.2f}"
        )
    print("-" * w)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CND-Verify Evaluation Runner")
    parser.add_argument("--b1-only",  action="store_true",
                        help="Run only B1 (fast, no LLM)")
    parser.add_argument("--skip-b2",  action="store_true",
                        help="Skip B2 single-LLM baseline (saves time)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip full pipeline (just run baselines)")
    parser.add_argument("--data-dir", default="data",
                        help="Path to data directory (default: data)")
    parser.add_argument("--gt",       default="data/ground_truth/ground_truth.json",
                        help="Path to ground_truth.json")
    parser.add_argument("--out",      default=None,
                        help="Save results JSON to this file")
    args = parser.parse_args()

    # ── Load ground truth ─────────────────────────────────────────────────────
    logger.info("Loading ground truth from %s", args.gt)
    with open(args.gt) as f:
        gt = json.load(f)
    docs = gt["documents"]
    logger.info("Ground truth: %d documents, %d total errors",
                len(docs), sum(d.get("error_count", 0) for d in docs))

    all_results = {}

    # ── B1 ────────────────────────────────────────────────────────────────────
    logger.info("Running B1 (rule-based checklist)...")
    b1_results = []
    for doc in docs:
        logger.info("  B1 → %s", doc["doc_id"])
        b1_results.append(_run_b1_on_doc(doc, args.data_dir))
    b1_summary = _aggregate(b1_results)
    _print_table("B1: Rule-Based Checklist", b1_results, b1_summary)
    all_results["B1"] = {"results": b1_results, "summary": b1_summary}

    if args.b1_only:
        logger.info("--b1-only flag set, stopping here.")
        _save(all_results, args.out)
        return

    # ── B2 ────────────────────────────────────────────────────────────────────
    if not args.skip_b2:
        logger.info("Running B2 (single-LLM)... this will take several minutes")
        b2_results = []
        for doc in docs:
            logger.info("  B2 → %s", doc["doc_id"])
            b2_results.append(_run_b2_on_doc(doc, args.data_dir))
        b2_summary = _aggregate(b2_results)
        _print_table("B2: Single-LLM", b2_results, b2_summary)
        all_results["B2"] = {"results": b2_results, "summary": b2_summary}
    else:
        logger.info("Skipping B2 (--skip-b2).")

    # ── Full pipeline ─────────────────────────────────────────────────────────
    if not args.skip_pipeline:
        logger.info("Running full CND-Verify pipeline... this will take several minutes per document")
        pipe_results = []
        for doc in docs:
            logger.info("  Pipeline → %s", doc["doc_id"])
            pipe_results.append(_run_pipeline_on_doc(doc, args.data_dir))
        pipe_summary = _aggregate(pipe_results)
        _print_table("CND-Verify (Full Pipeline)", pipe_results, pipe_summary)
        all_results["CND-Verify"] = {"results": pipe_results, "summary": pipe_summary}

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'System':<22}  {'Micro-P':>8}  {'Micro-R':>8}  {'Micro-F1':>9}  {'Routing':>8}")
    print("  " + "-" * 61)
    for name, data in all_results.items():
        s = data["summary"]
        print(
            f"  {name:<22}  "
            f"{s.get('micro_precision',0):>8.3f}  "
            f"{s.get('micro_recall',0):>8.3f}  "
            f"{s.get('micro_f1',0):>9.3f}  "
            f"{s.get('routing_accuracy',0):>8.2f}"
        )
    print("=" * 65)

    _save(all_results, args.out)


def _save(all_results: dict, out_path: str | None):
    if out_path:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to %s", out_path)
    else:
        # Always save to evaluation/results.json by default
        default = Path("evaluation") / "results.json"
        with open(default, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to %s", default)


if __name__ == "__main__":
    main()
