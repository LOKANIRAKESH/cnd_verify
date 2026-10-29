"""
src/orchestration/graph.py

Full CND-Verify LangGraph pipeline — two-pass architecture:

  PASS 1 (build_document_index):
    Index all sentences into ChromaDB so agents can query cross-chunk context.
    Builds: tag_index, setpoint_index, definition_map, chunk_groups.

  PASS 2 (parallel-friendly verification):
    START
      → build_document_index          ← Pass 1: global knowledge store
      → image_agent                   ← image extraction (runs before text agents)
      → completeness_agent            ← section/field completeness
      → consistency_agent             ← cross-section contradictions
      → ambiguity_agent               ← vague language detection
      → score_engine                  ← VQS scoring + routing decision
      → plc_generator                 ← IEC 61131-3 ST code for clean equipment
    END
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END

from src.orchestration.state import VerificationState
from src.ingestion.document_indexer import build_document_index
from src.agents.completeness_agent import run_completeness_agent
from src.agents.consistency_agent import run_consistency_agent
from src.agents.ambiguity_agent import run_ambiguity_agent
from src.agents.image_agent import run_image_agent
from src.scoring.score_engine import run_score_engine
from src.extraction.plc_generator import run_plc_generator

import yaml

logger = logging.getLogger("cnd_verify")


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _safe(agent_fn):
    """
    Wraps an agent so that:
    - Prior errors skip the agent gracefully.
    - Runtime exceptions are logged and stored in state["error"].
    """
    def wrapper(state: dict) -> dict:
        if state.get("error"):
            logger.warning("Skipping %s — prior error: %s",
                           agent_fn.__name__, state["error"])
            return state
        logger.info("Starting: %s", agent_fn.__name__)
        try:
            result = agent_fn(state)
            # Log finding count for verification agents
            rk = agent_fn.__name__.replace("run_", "").replace("_agent", "_report")
            report = result.get(rk, {}) or {}
            if "findings" in report:
                logger.info("%s completed — %d findings",
                            agent_fn.__name__, len(report["findings"]))
            return result
        except Exception as exc:
            logger.error("%s FAILED: %s", agent_fn.__name__, exc, exc_info=True)
            state["error"] = f"{agent_fn.__name__} failed: {exc}"
            return state
    wrapper.__name__ = agent_fn.__name__
    return wrapper


def _pass1_index(state: dict) -> dict:
    """
    Pass 1 node: build the per-document ChromaDB index and global lookup tables.
    Must run before any verification agent.
    """
    if state.get("error"):
        return state
    logger.info("Pass 1 — building document index...")
    try:
        config = _load_config()
        return build_document_index(state, config)
    except Exception as exc:
        logger.error("document_indexer FAILED: %s", exc, exc_info=True)
        # Non-fatal — agents can still run without cross-chunk context
        state.setdefault("doc_collection_name", "")
        state.setdefault("tag_index", {})
        state.setdefault("setpoint_index", {})
        state.setdefault("definition_map", {})
        state.setdefault("chunk_groups", [])
        return state


def _parallel_completeness_ambiguity(state: dict) -> dict:
    """
    Run completeness and ambiguity agents in parallel using ThreadPoolExecutor.

    On RTX 3050 (single GPU): LLM calls queue at the GPU — but parsing,
    regex scanning, and ChromaDB queries run truly in parallel on CPU.
    This gives ~30% speedup vs fully sequential on large documents.

    NOTE: Both agents are STATELESS per section and do not write to each
    other's report keys, so merging is safe.
    """
    if state.get("error"):
        return state

    results = {}

    def run_completeness(s):
        return run_completeness_agent(dict(s))

    def run_ambiguity(s):
        return run_ambiguity_agent(dict(s))

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_comp = pool.submit(run_completeness, state)
        fut_amb  = pool.submit(run_ambiguity,    state)

        for fut in as_completed([fut_comp, fut_amb]):
            try:
                partial = fut.result()
                # Merge only the agent's own output keys back into results
                for key in ("completeness_report", "ambiguity_report"):
                    if key in partial and partial[key] is not None:
                        results[key] = partial[key]
                if partial.get("error"):
                    results["error"] = partial["error"]
            except Exception as exc:
                logger.error("Parallel agent failed: %s", exc, exc_info=True)

    state.update(results)
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(VerificationState)

    # Pass 1 — global knowledge store
    graph.add_node("build_document_index",   _pass1_index)

    # Image agent — runs after index, before text agents
    graph.add_node("image_agent",            _safe(run_image_agent))

    # Completeness + Ambiguity in parallel, then Consistency
    graph.add_node("completeness_ambiguity", _parallel_completeness_ambiguity)
    graph.add_node("consistency_agent",      _safe(run_consistency_agent))

    # Score engine — aggregates all findings
    graph.add_node("score_engine",           _safe(run_score_engine))

    # PLC generator — runs last, after score_engine populates all_findings
    graph.add_node("plc_generator",          _safe(run_plc_generator))

    # Wire the pipeline
    graph.set_entry_point("build_document_index")
    graph.add_edge("build_document_index",   "image_agent")
    graph.add_edge("image_agent",            "completeness_ambiguity")
    graph.add_edge("completeness_ambiguity", "consistency_agent")
    graph.add_edge("consistency_agent",      "score_engine")
    graph.add_edge("score_engine",           "plc_generator")
    graph.add_edge("plc_generator",          END)

    return graph


def compile_graph():
    """Returns a compiled, runnable LangGraph app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )
    return build_graph().compile()
