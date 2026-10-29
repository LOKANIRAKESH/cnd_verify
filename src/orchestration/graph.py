"""
src/orchestration/graph.py
LangGraph StateGraph wiring all three verification agents and the score engine.

Graph topology (sequential):
  START → completeness_agent → consistency_agent → ambiguity_agent → score_engine → END

Each node reads VerificationState and appends its report.
If any node sets state["error"], downstream nodes skip gracefully.
"""

from langgraph.graph import StateGraph, END

from src.orchestration.state import VerificationState
from src.agents.completeness_agent import run_completeness_agent
from src.agents.consistency_agent import run_consistency_agent
from src.agents.ambiguity_agent import run_ambiguity_agent
from src.scoring.score_engine import run_score_engine


def _safe(agent_fn):
    """
    Wraps an agent function so that if state["error"] is already set,
    the agent is skipped and state is returned unchanged.
    Also catches runtime exceptions and records them in state["error"].
    """
    def wrapper(state: dict) -> dict:
        if state.get("error"):
            return state
        try:
            return agent_fn(state)
        except Exception as exc:
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
    return build_graph().compile()
