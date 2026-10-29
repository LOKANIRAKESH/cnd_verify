"""
src/orchestration/graph.py
LangGraph StateGraph wiring all three verification agents and the score engine.

Graph topology (sequential):
  START → completeness_agent → consistency_agent → ambiguity_agent → score_engine → END

Each node reads VerificationState and appends its report.
If any node sets state["error"], downstream nodes skip gracefully.
"""

import logging
from langgraph.graph import StateGraph, END

from src.orchestration.state import VerificationState
from src.agents.completeness_agent import run_completeness_agent
from src.agents.consistency_agent import run_consistency_agent
from src.agents.ambiguity_agent import run_ambiguity_agent
from src.scoring.score_engine import run_score_engine

logger = logging.getLogger("cnd_verify")


def _safe(agent_fn):
    """
    Wraps an agent function so that:
    - If state["error"] is already set, the agent is skipped.
    - Runtime exceptions are caught, logged, and stored in state["error"].
    - Agent name is preserved for debugging.
    """
    def wrapper(state: dict) -> dict:
        if state.get("error"):
            logger.warning("Skipping %s — prior error: %s", agent_fn.__name__, state["error"])
            return state
        logger.info("Starting agent: %s", agent_fn.__name__)
        try:
            result = agent_fn(state)
            # Log finding counts for each agent after it runs
            report_key = agent_fn.__name__.replace("run_", "").replace("_agent", "_report")
            report = result.get(report_key)
            if report and "findings" in report:
                count = len(report["findings"])
                logger.info("Agent %s completed: %d findings", agent_fn.__name__, count)
            return result
        except Exception as exc:
            logger.error("Agent %s FAILED: %s", agent_fn.__name__, exc, exc_info=True)
            state["error"] = f"{agent_fn.__name__} failed: {exc}"
            return state
    wrapper.__name__ = agent_fn.__name__
    return wrapper


def build_graph() -> StateGraph:
    graph = StateGraph(VerificationState)

    graph.add_node("completeness_agent", _safe(run_completeness_agent))
    graph.add_node("consistency_agent", _safe(run_consistency_agent))
    graph.add_node("ambiguity_agent", _safe(run_ambiguity_agent))
    graph.add_node("score_engine", _safe(run_score_engine))

    graph.set_entry_point("completeness_agent")
    graph.add_edge("completeness_agent", "consistency_agent")
    graph.add_edge("consistency_agent", "ambiguity_agent")
    graph.add_edge("ambiguity_agent", "score_engine")
    graph.add_edge("score_engine", END)

    return graph


def compile_graph():
    """Returns a compiled, runnable LangGraph app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )
    return build_graph().compile()
