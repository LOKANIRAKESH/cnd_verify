"""
src/ui/app.py
CND-Verify — Enterprise Frontend
AI-Based Safety Verification and Inconsistency Detection in Control Narrative Documents.
11-page professional industrial safety platform.
"""

import os
import re
import sys
import json
import time
import logging
import requests
import yaml
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CND-Verify",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — Light Professional Theme (Streamlit 1.65) ────────────────────
st.markdown("""
<style>
  /* ═══ 1. PAGE BACKGROUND ═══════════════════════════════════════════════════ */
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewContainer"] > .main,
  [data-testid="stAppViewContainer"] > .main > .block-container {
    background-color: #f1f5f9 !important;
  }
  .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
  }

  /* ═══ 2. SIDEBAR ════════════════════════════════════════════════════════════ */
  [data-testid="stSidebar"],
  [data-testid="stSidebar"] > div:first-child {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b !important;
  }
  /* Specific text elements — no wildcard, but force color on all text nodes */
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown { color: #cbd5e1 !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #f8fafc !important; }
  [data-testid="stSidebar"] hr { border-color: #1e293b; }

  /* Radio nav — Streamlit 1.65 puts the label text inside label > div > p */
  [data-testid="stSidebar"] [data-testid="stRadio"] label,
  [data-testid="stSidebar"] [data-testid="stRadio"] label p,
  [data-testid="stSidebar"] [data-testid="stRadio"] label div,
  [data-testid="stSidebar"] [data-testid="stRadio"] label span {
    color: #94a3b8 !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover,
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover p,
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover span {
    color: #e2e8f0 !important;
  }
  /* Selected radio item */
  [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] ~ div p,
  [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] ~ div {
    color: #f8fafc !important; font-weight: 600;
  }
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #cbd5e1 !important; }

  /* ═══ 3. EXPANDER (Streamlit 1.65 uses <details>) ═══════════════════════════ */
  details[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    margin-bottom: 8px;
  }
  details[data-testid="stExpander"] > summary {
    background-color: #f8fafc !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    color: #0f172a !important;
    font-weight: 500;
  }
  details[data-testid="stExpander"][open] > summary {
    border-radius: 6px 6px 0 0 !important;
    border-bottom: 1px solid #e2e8f0 !important;
  }
  details[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    background-color: #ffffff !important;
    border-radius: 0 0 6px 6px !important;
    padding: 12px 16px !important;
    color: #0f172a;
  }

  /* ═══ 4. BUTTONS ════════════════════════════════════════════════════════════ */
  button[data-testid="baseButton-primary"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
  }
  button[data-testid="baseButton-primary"]:hover {
    background-color: #1d4ed8 !important;
  }
  button[data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
  }

  /* ═══ 5. TEXT INPUTS ════════════════════════════════════════════════════════ */
  [data-testid="stTextInput"] input,
  [data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    color: #0f172a !important;
    padding: 8px 12px !important;
  }
  [data-testid="stTextInput"] input:focus,
  [data-testid="stTextArea"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.15) !important;
  }
  [data-testid="stTextInput"] label,
  [data-testid="stTextArea"] label { color: #374151 !important; font-weight: 500; }

  /* ═══ 6. SELECTBOX & MULTISELECT ════════════════════════════════════════════ */
  [data-testid="stSelectbox"] > div > div,
  [data-testid="stMultiSelect"] > div > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    color: #0f172a !important;
  }
  [data-testid="stSelectbox"] label,
  [data-testid="stMultiSelect"] label { color: #374151 !important; font-weight: 500; }
  /* Multiselect selected tags */
  [data-baseweb="tag"] {
    background-color: #eff6ff !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 4px !important;
  }

  /* ═══ 7. TABS ═══════════════════════════════════════════════════════════════ */
  [data-baseweb="tab-list"] {
    background-color: #ffffff !important;
    border-bottom: 2px solid #e2e8f0 !important;
    gap: 4px;
    padding: 0 4px;
  }
  [data-baseweb="tab"] {
    background-color: #f8fafc !important;
    color: #64748b !important;
    border-radius: 6px 6px 0 0 !important;
    border: 1px solid #e2e8f0 !important;
    border-bottom: none !important;
    font-weight: 500;
    padding: 8px 16px !important;
  }
  [data-baseweb="tab"][aria-selected="true"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
  }
  [data-baseweb="tab-panel"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 16px !important;
  }

  /* ═══ 8. FILE UPLOADER ══════════════════════════════════════════════════════ */
  [data-testid="stFileUploaderDropzone"] {
    background-color: #f8fafc !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 8px !important;
    color: #475569 !important;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2563eb !important;
    background-color: #eff6ff !important;
  }
  [data-testid="stFileUploaderDropzone"] p,
  [data-testid="stFileUploaderDropzone"] span { color: #475569 !important; }

  /* ═══ 9. CHECKBOX & RADIO ════════════════════════════════════════════════════ */
  [data-testid="stCheckbox"] label,
  [data-testid="stRadio"] label { color: #0f172a !important; }
  [data-testid="stWidgetLabel"] p { color: #374151 !important; font-weight: 500; }

  /* ═══ 10. ALERTS / NOTIFICATIONS ════════════════════════════════════════════ */
  [data-testid="stAlert"],
  [data-testid="stNotification"] { border-radius: 6px !important; }

  /* ═══ 11. PROGRESS BAR ══════════════════════════════════════════════════════ */
  [data-testid="stProgress"] > div > div {
    background-color: #e2e8f0 !important;
    border-radius: 4px;
  }
  [data-testid="stProgress"] > div > div > div {
    background-color: #2563eb !important;
    border-radius: 4px;
  }

  /* ═══ 12. SPINNER ════════════════════════════════════════════════════════════ */
  [data-testid="stSpinner"] p { color: #64748b !important; }

  /* ═══ 13. CAPTION / SMALL TEXT ══════════════════════════════════════════════ */
  [data-testid="stCaptionContainer"] p,
  .stCaption p { color: #64748b !important; font-size: 0.80rem; }

  /* ═══ 14. MARKDOWN BODY TEXT ════════════════════════════════════════════════ */
  [data-testid="stMarkdownContainer"] p { color: #0f172a; }

  /* ═══ 15. CUSTOM COMPONENT CLASSES ══════════════════════════════════════════ */
  /* Cards */
  .card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
  }
  .card-sm {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }

  /* Page header */
  .page-header {
    background: #ffffff;
    border-bottom: 2px solid #e2e8f0;
    padding: 18px 0 14px 0;
    margin-bottom: 20px;
  }
  .page-title { font-size: 1.45rem; font-weight: 700; color: #0f172a; margin: 0; }
  .page-sub   { font-size: 0.85rem; color: #64748b; margin-top: 3px; }

  /* KPI tiles */
  .kpi {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }
  .kpi-val       { font-size: 2rem; font-weight: 700; color: #0f172a; line-height: 1.1; }
  .kpi-lbl       { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; margin-top: 4px; }
  .kpi-val.red   { color: #dc2626; }
  .kpi-val.amber { color: #d97706; }
  .kpi-val.green { color: #16a34a; }
  .kpi-val.blue  { color: #2563eb; }

  /* Severity badges */
  .badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.70rem; font-weight: 700; letter-spacing: .05em; }
  .badge-CRITICAL { background: #fee2e2; color: #991b1b; }
  .badge-HIGH     { background: #ffedd5; color: #9a3412; }
  .badge-WARNING  { background: #fef9c3; color: #854d0e; }
  .badge-INFO     { background: #dbeafe; color: #1e40af; }
  .badge-PASS     { background: #dcfce7; color: #166534; }
  .badge-FAIL     { background: #fee2e2; color: #991b1b; }
  .badge-WARN     { background: #fef9c3; color: #854d0e; }
  .badge-NA       { background: #f1f5f9; color: #475569; }

  /* Status dots */
  .dot-green { display:inline-block; width:8px; height:8px; border-radius:50%; background:#16a34a; margin-right:6px; }
  .dot-red   { display:inline-block; width:8px; height:8px; border-radius:50%; background:#dc2626; margin-right:6px; }
  .dot-amber { display:inline-block; width:8px; height:8px; border-radius:50%; background:#d97706; margin-right:6px; }
  .dot-grey  { display:inline-block; width:8px; height:8px; border-radius:50%; background:#94a3b8; margin-right:6px; }

  /* Tables */
  .tbl { width:100%; border-collapse:collapse; font-size:0.82rem; }
  .tbl th { background:#f8fafc; color:#475569; font-weight:600; padding:8px 12px; text-align:left; border-bottom:2px solid #e2e8f0; }
  .tbl td { padding:8px 12px; border-bottom:1px solid #f1f5f9; color:#0f172a; vertical-align:top; }
  .tbl tr:hover td { background:#f8fafc; }

  /* Finding card (severity-colored left border) */
  .finding {
    background: #ffffff;
    border-left: 4px solid #e2e8f0;
    border-top: 1px solid #f1f5f9;
    border-right: 1px solid #f1f5f9;
    border-bottom: 1px solid #f1f5f9;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
  }
  .finding.CRITICAL { border-left-color: #dc2626; }
  .finding.HIGH     { border-left-color: #ea580c; }
  .finding.WARNING  { border-left-color: #ca8a04; }
  .finding.INFO     { border-left-color: #2563eb; }

  /* Stage rows (pipeline progress) */
  .stage-row {
    display:flex; align-items:center; padding:10px 16px;
    background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; margin-bottom:6px;
  }
  .stage-icon { width:22px; text-align:center; margin-right:12px; font-size:1rem; }
  .stage-name { flex:1; font-size:0.85rem; color:#0f172a; font-weight:500; }
  .stage-time { font-size:0.75rem; color:#64748b; margin-right:12px; }

  /* Evidence chunk */
  .evidence-chunk {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius:6px;
    padding: 12px 16px; font-size: 0.80rem; color: #334155;
    margin-bottom: 10px; white-space: pre-wrap; word-break: break-word;
  }

  /* Rule row */
  .rule-row {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:6px;
    padding:10px 16px; margin-bottom:6px; font-size:0.82rem; color:#0f172a;
  }

  /* Custom progress bar */
  .prog-bar-bg   { background:#e2e8f0; border-radius:4px; height:6px; width:100%; }
  .prog-bar-fill { background:#2563eb; border-radius:4px; height:6px; }

  /* Section label */
  .section-lbl { font-size:0.75rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:.07em; margin-bottom:10px; display:block; }

  /* Empty state */
  .empty-state { text-align:center; padding:40px 20px; color:#94a3b8; background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; }
  .empty-state .icon { font-size:2.5rem; margin-bottom:10px; }
  .empty-state p { font-size:0.88rem; margin:0; }

  /* Thin divider */
  .divider { border:none; border-top:1px solid #e2e8f0; margin:16px 0; }

  /* ISA tag chip */
  .tag { display:inline-block; background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;
         border-radius:4px; padding:2px 7px; font-size:0.72rem; font-family:monospace; margin:2px; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "verification_state": None,       # last full pipeline result
    "analyzed_docs": {},              # {filename: state} — multi-doc store
    "uploaded_docs": {},              # {filename: prepare_document() state}
    "finding_statuses": {},           # {finding_id: status}
    "finding_notes": {},
    "exec_log": [],                   # [{stage, status, elapsed_s, detail}]
    "pdf_views_open": set(),
    "eval_results": None,
    "analysis_running": False,
    "current_doc_name": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Config helper ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def _load_config():
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# ── Service health ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _check_services() -> dict:
    """Check real service availability. Cached for 60 s to avoid blocking page load."""
    results = {}

    # Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            has_llama = any("llama3" in m for m in models)
            results["LLM (Ollama)"] = (has_llama, f"llama3.1 {'ready' if has_llama else 'NOT pulled'}")
        else:
            results["LLM (Ollama)"] = (False, f"HTTP {r.status_code}")
    except Exception as e:
        results["LLM (Ollama)"] = (False, "Not running — start with: ollama serve")

    # ChromaDB (try to import and connect)
    try:
        import chromadb
        cfg = _load_config()
        client = chromadb.PersistentClient(path=cfg.get("chromadb", {}).get("path", "./chroma_db"))
        n = len(client.list_collections())
        results["Vector DB (ChromaDB)"] = (True, f"{n} collection(s)")
    except Exception as e:
        results["Vector DB (ChromaDB)"] = (False, str(e)[:60])

    # Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        results["Embedding Service"] = (True, "all-MiniLM-L6-v2 available")
    except Exception:
        results["Embedding Service"] = (False, "sentence-transformers not installed")

    # NetworkX (Knowledge Graph)
    try:
        import networkx
        results["Knowledge Graph (NetworkX)"] = (True, f"v{networkx.__version__}")
    except Exception:
        results["Knowledge Graph (NetworkX)"] = (False, "networkx not installed")

    # Document processing
    try:
        import fitz
        import pdfplumber
        results["Document Processing"] = (True, f"PyMuPDF + pdfplumber")
    except Exception:
        results["Document Processing"] = (False, "pymupdf/pdfplumber missing")

    # Rule engine (embedded in agents — always present if agents importable)
    try:
        from src.agents.completeness_agent import run_completeness_agent
        results["Rule Engine"] = (True, "Embedded in agents")
    except Exception:
        results["Rule Engine"] = (False, "Agent import failed")

    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡 CND-Verify")
    st.markdown('<p style="font-size:0.75rem;color:#64748b;margin-top:-10px">Safety Verification Platform</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Quick-start workflow guide
    _vstate = st.session_state.verification_state
    _has_docs = bool(st.session_state.uploaded_docs)
    _has_analysis = bool(_vstate)

    step1_color = "#16a34a" if _has_docs else "#2563eb"
    step2_color = "#16a34a" if _has_analysis else ("#2563eb" if _has_docs else "#475569")
    step3_color = "#16a34a" if _has_analysis else "#475569"

    step1_icon = "✓" if _has_docs else "①"
    step2_icon = "✓" if _has_analysis else ("②" if _has_docs else "②")
    step3_icon = "✓" if _has_analysis else "③"

    st.markdown(
        f'<div style="background:#1e293b;border-radius:6px;padding:10px 12px;margin-bottom:10px">'
        f'<div style="font-size:0.70rem;color:#94a3b8;font-weight:600;letter-spacing:.06em;margin-bottom:8px">QUICK START</div>'
        f'<div style="font-size:0.78rem;color:{step1_color};margin-bottom:5px">{step1_icon} Upload documents</div>'
        f'<div style="font-size:0.78rem;color:{step2_color};margin-bottom:5px">{step2_icon} Run analysis</div>'
        f'<div style="font-size:0.78rem;color:{step3_color}">{step3_icon} Review findings</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    NAV_ITEMS = [
        "1. Overview Dashboard",
        "2. Document Management",
        "3. CND Analysis",
        "4. Safety Findings",
        "5. Knowledge Graph",
        "6. RAG Evidence Explorer",
        "7. Cross-Document Verification",
        "8. Rule Validation",
        "9. Test & Evaluation",
        "10. Execution Logs",
        "11. Settings",
    ]
    # Mark pages that need analysis
    _needs_analysis = {
        "4. Safety Findings", "5. Knowledge Graph", "6. RAG Evidence Explorer",
        "7. Cross-Document Verification", "8. Rule Validation", "10. Execution Logs",
    }
    _nav_display = []
    for item in NAV_ITEMS:
        if item in _needs_analysis and not _has_analysis:
            _nav_display.append(f"{item} ·")  # dot suffix = needs analysis first
        else:
            _nav_display.append(item)

    page_display = st.radio("Navigation", _nav_display, label_visibility="collapsed")
    # Map display name back to canonical name
    page = page_display.rstrip(" ·")

    st.markdown("---")

    # Quick status
    if _vstate:
        vqs = _vstate.get("vqs_scores", {})
        score = vqs.get("composite", 0) * 100
        routing = vqs.get("routing_decision", "")
        dot_map = {"AUTO_PASS": "dot-green", "NEEDS_REVIEW": "dot-amber", "FAIL": "dot-red", "MANDATORY_REVISION": "dot-red"}
        dot = dot_map.get(routing, "dot-grey")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#94a3b8">Last analysis</div>'
            f'<div style="font-size:0.85rem;color:#e2e8f0;margin-top:2px">'
            f'<span class="{dot}"></span>{_vstate.get("current_doc_name","document")}</div>'
            f'<div style="font-size:0.80rem;color:#94a3b8">Score: <b style="color:#f1f5f9">{score:.0f}%</b> — {routing.replace("_"," ")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

    st.markdown('<p style="font-size:0.70rem;color:#475569">M.Tech Research Project<br>Mentor: Padmaja Bodanapu<br>GE Vernova Innovation Labs</p>', unsafe_allow_html=True)


# ── Helper: render header ─────────────────────────────────────────────────────
def _header(title: str, sub: str = ""):
    st.markdown(
        f'<div class="page-header"><div class="page-title">{title}</div>'
        + (f'<div class="page-sub">{sub}</div>' if sub else "")
        + '</div>',
        unsafe_allow_html=True,
    )


def _kpi(col, val, label, color=""):
    with col:
        st.markdown(
            f'<div class="kpi"><div class="kpi-val {color}">{val}</div><div class="kpi-lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )


def _badge(sev: str) -> str:
    return f'<span class="badge badge-{sev}">{sev}</span>'


def _dot(ok: bool) -> str:
    return f'<span class="{"dot-green" if ok else "dot-red"}"></span>'


def _empty(msg: str, icon: str = "📭"):
    st.markdown(f'<div class="empty-state"><div class="icon">{icon}</div><p>{msg}</p></div>', unsafe_allow_html=True)


def _no_analysis():
    _empty("No analysis results yet. Go to <b>3. CND Analysis</b> to run verification.", "🔍")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def page_overview():
    _header("Overview Dashboard", "Real-time summary of CND-Verify system and analysis results")

    vstate = st.session_state.verification_state
    findings = (vstate or {}).get("all_findings", [])
    vqs = (vstate or {}).get("vqs_scores", {})

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k0, k1, k2, k3 = st.columns(4)
    docs_uploaded = len(st.session_state.uploaded_docs)
    docs_analyzed = len(st.session_state.analyzed_docs)
    critical_n = sum(1 for f in findings if f.get("severity") == "CRITICAL")
    high_n     = sum(1 for f in findings if f.get("severity") == "HIGH")
    warn_n     = sum(1 for f in findings if f.get("severity") in ("WARNING", "INFO"))
    verified_n = sum(1 for f in findings if f.get("status") == "ACCEPTED")
    _kpi(k0, docs_uploaded,         "Uploaded Docs")
    _kpi(k1, docs_analyzed,         "Docs Analyzed", "blue")
    _kpi(k2, len(st.session_state.exec_log), "Pipeline Runs")
    _kpi(k3, len(findings),         "Total Findings")

    k4, k5, k6, k7 = st.columns(4)
    _kpi(k4, critical_n,            "Critical",   "red"   if critical_n else "")
    _kpi(k5, high_n,                "High",       "amber" if high_n else "")
    _kpi(k6, verified_n,            "Accepted",   "green" if verified_n else "")
    score_val = f"{vqs.get('composite',0)*100:.0f}%" if vqs else "—"
    _kpi(k7, score_val,             "Last VQS Score", "blue" if vqs else "")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        # ── Findings distribution charts ───────────────────────────────────
        st.markdown('<div class="section-lbl">Safety Findings Distribution</div>', unsafe_allow_html=True)

        if findings:
            c1, c2 = st.columns(2)
            # Severity pie
            sev_counts = {}
            for f in findings:
                s = f.get("severity", "INFO")
                sev_counts[s] = sev_counts.get(s, 0) + 1
            with c1:
                sev_colors = {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "WARNING": "#ca8a04", "INFO": "#2563eb"}
                fig = go.Figure(go.Pie(
                    labels=list(sev_counts.keys()), values=list(sev_counts.values()),
                    marker_colors=[sev_colors.get(k, "#94a3b8") for k in sev_counts],
                    hole=0.4, textinfo="label+value",
                    textfont_size=11,
                ))
                fig.update_layout(
                    height=200, margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                    font={"color": "#0f172a", "size": 11},
                )
                st.plotly_chart(fig, width='stretch')

            # Agent bar
            agent_counts = {}
            for f in findings:
                a = f.get("agent", "unknown")
                agent_counts[a] = agent_counts.get(a, 0) + 1
            with c2:
                agent_colors = {"completeness": "#2563eb", "consistency": "#7c3aed",
                                "ambiguity": "#ca8a04", "logic_coherence": "#dc2626", "image": "#0891b2"}
                fig2 = go.Figure(go.Bar(
                    x=list(agent_counts.values()), y=list(agent_counts.keys()),
                    orientation="h",
                    marker_color=[agent_colors.get(k, "#94a3b8") for k in agent_counts],
                    text=list(agent_counts.values()), textposition="outside",
                ))
                fig2.update_layout(
                    height=200, margin=dict(t=10, b=10, l=0, r=20),
                    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(tickfont={"size": 10, "color": "#0f172a"}),
                    font={"color": "#0f172a"},
                )
                st.plotly_chart(fig2, width='stretch')
        else:
            _empty("No findings yet. Run CND Analysis to see results.", "📊")

    with col_right:
        # ── System health ─────────────────────────────────────────────────
        st.markdown('<div class="section-lbl">System Component Health</div>', unsafe_allow_html=True)
        services = _check_services()

        rows = ""
        for name, (ok, detail) in services.items():
            icon = "✓" if ok else "✗"
            color = "#16a34a" if ok else "#dc2626"
            rows += (
                f'<div class="card-sm" style="padding:9px 14px;margin-bottom:5px">'
                f'<span style="color:{color};font-weight:700;margin-right:8px">{icon}</span>'
                f'<span style="font-size:0.82rem;color:#0f172a;font-weight:500">{name}</span>'
                f'<div style="font-size:0.72rem;color:#64748b;margin-top:2px;margin-left:22px">{detail}</div>'
                f'</div>'
            )
        st.markdown(rows, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Recent activity ────────────────────────────────────────────────────────
    st.markdown('<div class="section-lbl">Recent Analysis Activity</div>', unsafe_allow_html=True)
    if st.session_state.analyzed_docs:
        rows_html = "".join(
            f'<tr><td><code style="font-size:0.78rem">{fname}</code></td>'
            f'<td>{s.get("vqs_scores",{}).get("routing_decision","—")}</td>'
            f'<td>{len(s.get("all_findings",[]))}</td>'
            f'<td>{s.get("vqs_scores",{}).get("composite",0)*100:.0f}%</td></tr>'
            for fname, s in st.session_state.analyzed_docs.items()
        )
        st.markdown(
            f'<div class="card" style="padding:0;overflow:hidden">'
            f'<table class="tbl"><thead><tr><th>Document</th><th>Routing</th><th>Findings</th><th>VQS Score</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True,
        )
    else:
        _empty("No analysis history. Upload a document and run analysis.", "📋")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DOCUMENT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def page_documents():
    _header("Document Management", "Upload, parse, and review control narrative documents")

    from src.ingestion.document_loader import prepare_document

    # ── Upload section ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-lbl">Upload Documents</div>', unsafe_allow_html=True)

    u1, u2 = st.columns([2, 1])
    with u1:
        uploaded_files = st.file_uploader(
            "Upload CND documents (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    with u2:
        st.markdown("**Supported document types**")
        for t in ["Control Narrative Document (CND)", "P&ID reference", "Equipment specification", "Instrument datasheet"]:
            st.caption(f"• {t}")

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_docs:
                tmp_path = f"temp_upload_{uf.name}"
                with open(tmp_path, "wb") as fp:
                    fp.write(uf.getbuffer())
                with st.spinner(f"Parsing {uf.name}…"):
                    try:
                        state = prepare_document(tmp_path)
                        state["current_doc_name"] = uf.name
                        state["upload_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                        state["file_size_kb"] = round(uf.size / 1024, 1)
                        st.session_state.uploaded_docs[uf.name] = state
                        st.success(f"✓ {uf.name} parsed — {len(state.get('sections',{}))} sections, {len(state.get('tag_inventory',[]))} ISA tags")
                    except Exception as e:
                        st.error(f"Failed to parse {uf.name}: {e}")

    # ── Sample documents ───────────────────────────────────────────────────────
    st.markdown('<div class="section-lbl" style="margin-top:16px">Sample Test Documents</div>', unsafe_allow_html=True)
    SAMPLES = {
        "pump_station_01_clean.pdf":               "Pump Station 01 — Clean (no errors)",
        "pump_station_02_completeness_errors.pdf": "Pump Station 02 — Completeness errors",
        "pump_station_03_consistency_errors.pdf":  "Pump Station 03 — Consistency errors",
        "pump_station_04_ambiguity_errors.pdf":    "Pump Station 04 — Ambiguity errors",
    }
    s_cols = st.columns(4)
    for i, (fname, label) in enumerate(SAMPLES.items()):
        path = f"data/synthetic_cnds/{fname}"
        with s_cols[i]:
            exists = Path(path).exists()
            status = "✓ Available" if exists else "✗ Not found"
            color  = "#16a34a" if exists else "#dc2626"
            st.markdown(
                f'<div class="card-sm"><div style="font-size:0.78rem;font-weight:600;color:#0f172a">{label}</div>'
                f'<div style="font-size:0.70rem;color:{color};margin-top:4px">{status}</div></div>',
                unsafe_allow_html=True,
            )
            if exists and fname not in st.session_state.uploaded_docs:
                if st.button(f"Load", key=f"load_{fname}"):
                    with st.spinner(f"Parsing {fname}…"):
                        try:
                            state = prepare_document(path)
                            state["current_doc_name"] = fname
                            state["upload_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                            state["file_size_kb"] = round(Path(path).stat().st_size / 1024, 1)
                            st.session_state.uploaded_docs[fname] = state
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Document list ──────────────────────────────────────────────────────────
    if not st.session_state.uploaded_docs:
        _empty("No documents loaded yet. Upload a PDF or load a sample document above.", "📂")
        return

    st.markdown(f'<div class="section-lbl">{len(st.session_state.uploaded_docs)} Document(s) Loaded</div>', unsafe_allow_html=True)

    for fname, doc_state in st.session_state.uploaded_docs.items():
        tags = doc_state.get("tag_inventory", [])
        sections = doc_state.get("sections", {})
        setpoints = doc_state.get("setpoint_table", [])
        analyzed = fname in st.session_state.analyzed_docs
        status_txt = "Analyzed ✓" if analyzed else "Parsed — not yet analyzed"
        status_col = "#16a34a" if analyzed else "#ca8a04"

        with st.expander(f"📄 {fname}  —  {len(sections)} sections · {len(tags)} ISA tags · {len(setpoints)} setpoints", expanded=False):
            meta1, meta2, meta3 = st.columns(3)
            with meta1:
                st.markdown(f"**Uploaded:** {doc_state.get('upload_time','—')}")
                st.markdown(f"**Size:** {doc_state.get('file_size_kb','—')} KB")
                st.markdown(f'**Status:** <span style="color:{status_col}">{status_txt}</span>', unsafe_allow_html=True)
            with meta2:
                st.markdown(f"**Sections detected:** {len(sections)}")
                st.markdown(f"**ISA instrument tags:** {len(tags)}")
                st.markdown(f"**Setpoints extracted:** {len(setpoints)}")

            with meta3:
                if st.button("Remove document", key=f"del_{fname}"):
                    del st.session_state.uploaded_docs[fname]
                    if fname in st.session_state.analyzed_docs:
                        del st.session_state.analyzed_docs[fname]
                    st.rerun()

            t1, t2, t3 = st.tabs(["ISA Tags", "Sections", "Setpoints"])

            with t1:
                if tags:
                    chips = " ".join(f'<span class="tag">{t}</span>' for t in sorted(tags))
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No ISA tags detected (expected format: XX-NNN)")

            with t2:
                cfg = _load_config()
                req = set(cfg.get("required_sections", []))
                rows_html = ""
                for title in sections:
                    found_req = any(r.lower() in title.lower() or title.lower() in r.lower() for r in req)
                    icon = "✓" if found_req else "·"
                    color = "#16a34a" if found_req else "#94a3b8"
                    page = doc_state.get("sections_pages", {}).get(title, "—")
                    rows_html += (
                        f'<tr><td style="color:{color};width:20px">{icon}</td>'
                        f'<td>{title}</td><td>p.{page}</td>'
                        f'<td style="color:#64748b">{len(sections[title])} chars</td></tr>'
                    )
                st.markdown(
                    f'<table class="tbl"><thead><tr><th></th><th>Section</th><th>Page</th><th>Size</th></tr></thead>'
                    f'<tbody>{rows_html}</tbody></table>',
                    unsafe_allow_html=True,
                )

            with t3:
                if setpoints:
                    rows_html = "".join(
                        f'<tr><td><span class="tag">{sp.get("tag","")}</span></td>'
                        f'<td>{sp.get("parameter","")}</td>'
                        f'<td><b>{sp.get("value","")}</b></td>'
                        f'<td>{sp.get("units","")}</td>'
                        f'<td style="color:#64748b">p.{sp.get("page","—")}</td></tr>'
                        for sp in setpoints[:50]
                    )
                    st.markdown(
                        f'<table class="tbl"><thead><tr><th>Tag</th><th>Parameter</th><th>Value</th><th>Units</th><th>Page</th></tr></thead>'
                        f'<tbody>{rows_html}</tbody></table>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No structured setpoints table found. Consistency setpoint checks will use prose extraction.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_analysis():
    _header("CND Analysis", "Run AI-powered safety verification on control narrative documents")

    # ── Document selector ──────────────────────────────────────────────────────
    doc_choices = list(st.session_state.uploaded_docs.keys())
    SAMPLE_PATHS = {
        "pump_station_01_clean.pdf":               "data/synthetic_cnds/pump_station_01_clean.pdf",
        "pump_station_02_completeness_errors.pdf": "data/synthetic_cnds/pump_station_02_completeness_errors.pdf",
        "pump_station_03_consistency_errors.pdf":  "data/synthetic_cnds/pump_station_03_consistency_errors.pdf",
        "pump_station_04_ambiguity_errors.pdf":    "data/synthetic_cnds/pump_station_04_ambiguity_errors.pdf",
    }
    quick_samples = [k for k, v in SAMPLE_PATHS.items() if Path(v).exists() and k not in doc_choices]

    a1, a2 = st.columns([2, 1])
    with a1:
        st.markdown('<div class="section-lbl">Select Document</div>', unsafe_allow_html=True)
        all_options = doc_choices + (["— Sample: " + k for k in quick_samples] if quick_samples else [])
        if not all_options:
            st.warning("No documents available. Go to Document Management to upload or load a sample.")
            return
        selected = st.selectbox("Document", all_options, label_visibility="collapsed")

    with a2:
        st.markdown('<div class="section-lbl">Analysis Options</div>', unsafe_allow_html=True)
        run_image = st.checkbox("Image agent (P&ID extraction)", value=True)
        run_plc   = st.checkbox("PLC code generation (AUTO_PASS only)", value=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Pipeline stages display ────────────────────────────────────────────────
    st.markdown('<div class="section-lbl">Verification Pipeline</div>', unsafe_allow_html=True)

    STAGES = [
        ("Document Parsing",              "📄"),
        ("Pass 1: Knowledge Index",       "🗂"),
        ("Image Agent",                   "🖼"),
        ("Completeness Agent",            "☑"),
        ("Ambiguity Agent",               "🔤"),
        ("Consistency Agent",             "⚖"),
        ("Logic Coherence Agent",         "🔗"),
        ("Score Engine",                  "📊"),
        ("PLC Generator",                 "⚙"),
    ]

    stage_slots = []
    for stage_name, icon in STAGES:
        slot = st.empty()
        stage_slots.append(slot)
        slot.markdown(
            f'<div class="stage-row">'
            f'<span class="stage-icon">{icon}</span>'
            f'<span class="stage-name">{stage_name}</span>'
            f'<span class="stage-time">—</span>'
            f'<span style="font-size:0.75rem;color:#94a3b8">Pending</span></div>',
            unsafe_allow_html=True,
        )

    def _update_stage(idx, status, elapsed=None):
        icon_m = {"Running": "⟳", "Completed": "✓", "Failed": "✗", "Skipped": "—", "Pending": "·"}
        color_m = {"Running": "#2563eb", "Completed": "#16a34a", "Failed": "#dc2626", "Skipped": "#94a3b8", "Pending": "#94a3b8"}
        sname, sicon = STAGES[idx]
        t_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
        stage_slots[idx].markdown(
            f'<div class="stage-row">'
            f'<span class="stage-icon">{sicon}</span>'
            f'<span class="stage-name">{sname}</span>'
            f'<span class="stage-time">{t_str}</span>'
            f'<span style="font-size:0.75rem;color:{color_m[status]};font-weight:600">{status}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Run button ─────────────────────────────────────────────────────────────
    st.markdown("")
    if not st.button("▶  Run Safety Verification", type="primary"):
        if st.session_state.verification_state:
            vqs = st.session_state.verification_state.get("vqs_scores", {})
            routing = vqs.get("routing_decision", "")
            score = vqs.get("composite", 0) * 100
            color_map = {"AUTO_PASS": "success", "NEEDS_REVIEW": "warning", "FAIL": "error", "MANDATORY_REVISION": "error"}
            fn = getattr(st, color_map.get(routing, "info"))
            fn(f"Last result: **{routing.replace('_',' ')}** — VQS {score:.1f}% — {len(st.session_state.verification_state.get('all_findings',[]))} findings. Go to **Safety Findings** to review.")
        return

    # ── Resolve document path ──────────────────────────────────────────────────
    if selected.startswith("— Sample: "):
        sample_fname = selected.replace("— Sample: ", "")
        doc_path = SAMPLE_PATHS.get(sample_fname, "")
        doc_name = sample_fname
    else:
        doc_state_pre = st.session_state.uploaded_docs.get(selected, {})
        doc_path = doc_state_pre.get("document_path", "")
        doc_name = selected

    if not doc_path or not Path(doc_path).exists():
        st.error(f"Document not found: {doc_path}")
        return

    exec_log = []
    state = {}

    def _run_stage(idx, fn, *args, **kwargs):
        _update_stage(idx, "Running")
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            _update_stage(idx, "Completed", elapsed)
            exec_log.append({"stage": STAGES[idx][0], "status": "Completed", "elapsed_s": round(elapsed, 2), "detail": ""})
            return result, True
        except Exception as e:
            elapsed = time.time() - t0
            _update_stage(idx, "Failed", elapsed)
            exec_log.append({"stage": STAGES[idx][0], "status": "Failed", "elapsed_s": round(elapsed, 2), "detail": str(e)[:200]})
            st.error(f"Stage '{STAGES[idx][0]}' failed: {e}")
            return None, False

    # Stage 0: Document Parsing
    from src.ingestion.document_loader import prepare_document
    state_result, ok = _run_stage(0, prepare_document, doc_path)
    if not ok:
        return
    state = state_result
    state["current_doc_name"] = doc_name

    # Stage 1: Pass 1 Index
    from src.ingestion.document_indexer import build_document_index, build_interlock_graph
    cfg = _load_config()
    state_result, ok = _run_stage(1, build_document_index, state, cfg)
    if ok:
        state = state_result
        # Build interlock graph
        try:
            graph_data, has_cycle = build_interlock_graph(state.get("sections", {}))
            state["interlock_graph_data"] = graph_data
            state["has_circular_interlock"] = has_cycle
        except Exception:
            state["interlock_graph_data"] = {"nodes": [], "edges": [], "cycle_detected": False, "cycles": []}
            state["has_circular_interlock"] = False

    # Stage 2: Image Agent
    if run_image:
        from src.agents.image_agent import run_image_agent
        state_result, ok = _run_stage(2, run_image_agent, state)
        if ok:
            state = state_result
    else:
        _update_stage(2, "Skipped")
        exec_log.append({"stage": STAGES[2][0], "status": "Skipped", "elapsed_s": 0, "detail": "Disabled by user"})

    # Stages 3+4: Completeness + Ambiguity (parallel in backend; sequential here for stage display)
    import copy
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    from src.agents.completeness_agent import run_completeness_agent
    from src.agents.ambiguity_agent import run_ambiguity_agent

    _update_stage(3, "Running")
    _update_stage(4, "Running")
    t0 = time.time()
    comp_state = copy.deepcopy(state)
    amb_state  = copy.deepcopy(state)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_c = pool.submit(run_completeness_agent, comp_state)
            fut_a = pool.submit(run_ambiguity_agent, amb_state)
            comp_res = fut_c.result(timeout=180)
            amb_res  = fut_a.result(timeout=180)
        state["completeness_report"] = comp_res.get("completeness_report")
        state["ambiguity_report"]    = amb_res.get("ambiguity_report")
        elapsed34 = time.time() - t0
        _update_stage(3, "Completed", elapsed34)
        _update_stage(4, "Completed", elapsed34)
        exec_log.append({"stage": "Completeness Agent", "status": "Completed", "elapsed_s": round(elapsed34, 2), "detail": ""})
        exec_log.append({"stage": "Ambiguity Agent",    "status": "Completed", "elapsed_s": round(elapsed34, 2), "detail": "ran in parallel"})
    except Exception as e:
        _update_stage(3, "Failed")
        _update_stage(4, "Failed")
        st.warning(f"Completeness/Ambiguity agents failed: {e}")

    # Stage 5: Consistency
    from src.agents.consistency_agent import run_consistency_agent
    state_result, ok = _run_stage(5, run_consistency_agent, state)
    if ok:
        state = state_result

    # Stage 6: Logic Coherence
    from src.agents.logic_coherence_agent import run_logic_coherence_agent
    state_result, ok = _run_stage(6, run_logic_coherence_agent, state)
    if ok:
        state = state_result

    # Stage 7: Score Engine
    from src.scoring.score_engine import run_score_engine
    state_result, ok = _run_stage(7, run_score_engine, state)
    if ok:
        state = state_result

    # Stage 8: PLC Generator
    vqs = state.get("vqs_scores", {})
    if run_plc and vqs.get("routing_decision") == "AUTO_PASS":
        from src.extraction.plc_generator import run_plc_generator
        state_result, ok = _run_stage(8, run_plc_generator, state)
        if ok:
            state = state_result
    else:
        _update_stage(8, "Skipped")
        reason = "Disabled by user" if not run_plc else f"Routing: {vqs.get('routing_decision','—')} (PLC only for AUTO_PASS)"
        exec_log.append({"stage": STAGES[8][0], "status": "Skipped", "elapsed_s": 0, "detail": reason})

    # ── Save results ───────────────────────────────────────────────────────────
    st.session_state.verification_state = state
    st.session_state.analyzed_docs[doc_name] = state
    st.session_state.exec_log = exec_log
    st.session_state.current_doc_name = doc_name
    # Reset finding statuses for new run
    st.session_state.finding_statuses = {f["id"]: "OPEN" for f in state.get("all_findings", [])}
    st.session_state.finding_notes = {}

    # ── Result banner ──────────────────────────────────────────────────────────
    routing = vqs.get("routing_decision", "")
    score = vqs.get("composite", 0) * 100
    n_findings = len(state.get("all_findings", []))
    critical = sum(1 for f in state.get("all_findings", []) if f.get("severity") == "CRITICAL")

    banner_map = {
        "AUTO_PASS":         ("success", f"✓ AUTO PASS — Score {score:.1f}%. {n_findings} finding(s). PLC code generation enabled."),
        "NEEDS_REVIEW":      ("warning", f"⚠ NEEDS REVIEW — Score {score:.1f}%. {n_findings} finding(s), {critical} critical. Engineer review required."),
        "FAIL":              ("error",   f"✗ FAIL — Score {score:.1f}%. {n_findings} finding(s). Significant revision required."),
        "MANDATORY_REVISION":("error",  f"🚫 MANDATORY REVISION — Blocking issue detected. Document cannot proceed to PLC generation."),
    }
    if routing in banner_map:
        level, msg = banner_map[routing]
        getattr(st, level)(msg)

    st.success("✓ Analysis complete. Navigate to **Safety Findings** to review all findings.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SAFETY FINDINGS
# ══════════════════════════════════════════════════════════════════════════════

def page_findings():
    _header("Safety Findings", "Review and disposition all detected safety issues")

    vstate = st.session_state.verification_state
    if not vstate:
        _no_analysis()
        return

    findings = vstate.get("all_findings", [])
    if not findings:
        st.success("✓ No findings detected. Document passed all verification checks.")
        return

    # ── Summary strip ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    _kpi(k1, len(findings),                                                  "Total")
    _kpi(k2, sum(1 for f in findings if f.get("severity") == "CRITICAL"),   "Critical",  "red")
    _kpi(k3, sum(1 for f in findings if f.get("severity") == "HIGH"),        "High",      "amber")
    _kpi(k4, sum(1 for f in findings if f.get("severity") == "WARNING"),    "Warning")
    _kpi(k5, sum(1 for f in findings if st.session_state.finding_statuses.get(f.get("id",""), "OPEN") != "OPEN"), "Reviewed", "green")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    f_sev   = f1.multiselect("Severity",  ["CRITICAL","HIGH","WARNING","INFO"], default=["CRITICAL","HIGH","WARNING","INFO"], key="ff_sev")
    f_agent = f2.multiselect("Agent",     ["completeness","consistency","ambiguity","logic_coherence","image"],
                              default=["completeness","consistency","ambiguity","logic_coherence","image"], key="ff_agent")
    f_stat  = f3.multiselect("Status",    ["OPEN","ACCEPTED","REJECTED"], default=["OPEN","ACCEPTED","REJECTED"], key="ff_stat")
    f_text  = f4.text_input("Search description", key="ff_text", placeholder="keyword…")

    sev_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}
    visible = [
        f for f in sorted(findings, key=lambda x: sev_order.get(x.get("severity","INFO"), 3))
        if f.get("severity") in f_sev
        and f.get("agent") in f_agent
        and st.session_state.finding_statuses.get(f.get("id",""), "OPEN") in f_stat
        and (not f_text or f_text.lower() in (f.get("description","") + f.get("quote","")).lower())
    ]
    st.caption(f"Showing {len(visible)} of {len(findings)} findings")

    first_critical_seen = False
    for f in visible:
        is_first_crit = (f.get("severity") == "CRITICAL") and (not first_critical_seen)
        _finding_card_detailed(f, vstate, expand_as_first_critical=is_first_crit)
        if f.get("severity") == "CRITICAL":
            first_critical_seen = True


def _finding_card_detailed(f: dict, vstate: dict, expand_as_first_critical: bool = False):
    fid  = f.get("id", "")
    sev  = f.get("severity", "INFO")
    pg   = f.get("page_number", 0)
    cur  = st.session_state.finding_statuses.get(fid, "OPEN")
    status_icon = {"OPEN": "⬜", "ACCEPTED": "✅", "REJECTED": "❌"}.get(cur, "⬜")
    border = {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "WARNING": "#ca8a04", "INFO": "#2563eb"}.get(sev, "#e2e8f0")

    sev_label = {"CRITICAL": "🔴 CRITICAL", "HIGH": "🟠 HIGH", "WARNING": "🟡 WARNING", "INFO": "🔵 INFO"}.get(sev, sev)
    with st.expander(
        f"{status_icon} [{sev_label}]  {f.get('description','')[:90]}",
        expanded=expand_as_first_critical,
    ):
        # ── Location bar ──────────────────────────────────────────────────────
        loc1, loc2, loc3 = st.columns(3)
        with loc1:
            st.markdown(f'**Page:** {pg if pg else "unknown"}  |  **Agent:** {f.get("agent","—").replace("_"," ").title()}')
        with loc2:
            st.markdown(f'**Type:** `{f.get("finding_type","—")}`  |  **Section:** {f.get("section_title","—")[:40]}')
        with loc3:
            st.markdown(f'**Evidence:** {f.get("evidence_source","—")}')

        st.markdown("---")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Problem**")
            st.markdown(f.get("description", "—"))
            if f.get("quote"):
                st.markdown("**Exact text from document**")
                st.code(f.get("quote"), language=None)

        with d2:
            if f.get("rule_violated"):
                st.markdown(f'**Rule violated:** `{f.get("rule_violated")}`')
                if f.get("rule_text"):
                    st.info(f.get("rule_text"))
            if f.get("why_it_matters"):
                st.warning(f'**Safety consequence:** {f.get("why_it_matters")}')

        st.success(f'**Recommended fix:** {f.get("suggestion","No suggestion.")}')

        # ── PDF inline view ────────────────────────────────────────────────────
        pdf_path = vstate.get("document_path", "")
        if pdf_path and pdf_path.lower().endswith(".pdf") and Path(pdf_path).exists() and pg:
            view_open = fid in st.session_state.pdf_views_open
            label = "🔒 Close document view" if view_open else "📄 Open document at this page"
            if st.button(label, key=f"pdfbtn_{fid}"):
                if view_open:
                    st.session_state.pdf_views_open.discard(fid)
                else:
                    st.session_state.pdf_views_open.add(fid)
            if fid in st.session_state.pdf_views_open:
                _render_pdf_inline(pdf_path, pg, f.get("quote",""), sev, border)

        # ── Engineer decision ─────────────────────────────────────────────────
        st.markdown("---")
        e1, e2 = st.columns([1, 2])
        with e1:
            new_status = st.radio(
                "Engineer decision",
                ["OPEN", "ACCEPTED", "REJECTED"],
                index=["OPEN", "ACCEPTED", "REJECTED"].index(cur),
                key=f"r_{fid}", horizontal=True,
            )
            st.session_state.finding_statuses[fid] = new_status
        with e2:
            note = st.text_input(
                "Note (optional)", value=st.session_state.finding_notes.get(fid, ""),
                key=f"n_{fid}", placeholder="e.g. Confirmed with P&ID rev 3, or not applicable",
            )
            st.session_state.finding_notes[fid] = note


_SEV_HIGHLIGHT = {"CRITICAL": (0.86, 0.15, 0.15), "HIGH": (0.92, 0.35, 0.04), "WARNING": (0.79, 0.54, 0.00), "INFO": (0.15, 0.39, 0.92)}

def _render_pdf_inline(pdf_path: str, page_num: int, search_text: str, severity: str, border_color: str):
    try:
        doc = fitz.open(pdf_path)
        idx = max(0, min(page_num - 1, len(doc) - 1))
        page = doc[idx]
        hl_color = _SEV_HIGHLIGHT.get(severity, (0.79, 0.54, 0.00))
        if search_text:
            clean = re.sub(r"\[Page \d+\]|↔|→|←|\[.*?\]", "", search_text)
            clean = re.sub(r"\s+", " ", clean).strip()
            for candidate in [clean[:80], clean[:50], clean[:30]]:
                if len(candidate.strip()) < 5:
                    continue
                rects = page.search_for(candidate.strip())
                if rects:
                    for rect in rects:
                        a = page.add_highlight_annot(rect)
                        a.set_colors(stroke=hl_color)
                        a.update()
                    break
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        png = pix.tobytes("png")
        doc.close()
        st.image(png, caption=f"Page {page_num} — highlighted text shows finding location", width='stretch')
    except Exception as e:
        st.caption(f"Could not render PDF page: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def page_knowledge_graph():
    _header("Knowledge Graph Explorer", "Visualize equipment relationships and interlock dependencies extracted from the CND")

    vstate = st.session_state.verification_state
    if not vstate:
        _no_analysis()
        return

    graph_data = vstate.get("interlock_graph_data", {})
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    has_cycle = graph_data.get("cycle_detected", False)
    cycles = graph_data.get("cycles", [])

    # ── Stats ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    _kpi(k1, len(nodes), "Equipment Nodes")
    _kpi(k2, len(edges), "Interlock Relationships")
    _kpi(k3, len(cycles), "Circular Interlocks", "red" if cycles else "")
    _kpi(k4, len(vstate.get("tag_inventory", [])), "Total ISA Tags")

    if has_cycle:
        st.error(f"🚫 **CRITICAL: Circular interlock(s) detected.** The following interlock chains form cycles that prevent the system from reaching a safe state:")
        for cyc in cycles:
            st.code(cyc)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not nodes:
        _empty(
            "No interlock relationships extracted from this document.\n"
            "The graph builder scans Interlocks / ESD / Safety sections for "
            "'TAG-A trips/stops/closes TAG-B' patterns.",
            "🕸"
        )
        # Still show the full tag inventory as a fallback list
        tags = vstate.get("tag_inventory", [])
        if tags:
            st.markdown('<div class="section-lbl">All ISA Tags in Document</div>', unsafe_allow_html=True)
            def_map = vstate.get("definition_map", {})
            rows_html = "".join(
                f'<tr><td><span class="tag">{t}</span></td>'
                f'<td style="color:#64748b">{def_map.get(t,{}).get("type","—")}</td>'
                f'<td>{def_map.get(t,{}).get("class","—")}</td>'
                f'<td style="color:#64748b">p.{def_map.get(t,{}).get("page","—")}</td></tr>'
                for t in sorted(tags)
            )
            st.markdown(
                f'<table class="tbl"><thead><tr><th>Tag</th><th>Equipment Type</th><th>Class</th><th>Page</th></tr></thead>'
                f'<tbody>{rows_html}</tbody></table>',
                unsafe_allow_html=True,
            )
        return

    # ── Network visualization ──────────────────────────────────────────────────
    try:
        import networkx as nx

        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n)
        for e in edges:
            G.add_edge(e["from"], e["to"])

        pos = nx.spring_layout(G, seed=42, k=1.5)

        # Nodes in cycles
        cycle_nodes = set()
        for cyc_str in cycles:
            cycle_nodes.update(re.findall(r"[A-Z]{1,3}-\d{1,4}[A-Z]?", cyc_str))

        def_map = vstate.get("definition_map", {})

        # Edge traces
        edge_traces = []
        for e in edges:
            x0, y0 = pos.get(e["from"], (0, 0))
            x1, y1 = pos.get(e["to"],   (0, 0))
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color="#cbd5e1" if e["from"] not in cycle_nodes else "#dc2626", width=1.5),
                hoverinfo="none", showlegend=False,
            ))

        # Node trace
        nx_vals = list(pos.values())
        node_x = [v[0] for v in nx_vals]
        node_y = [v[1] for v in nx_vals]
        node_labels = list(pos.keys())
        node_colors = [
            "#dc2626" if n in cycle_nodes else
            "#2563eb" if n in {e["from"] for e in edges} else
            "#64748b"
            for n in node_labels
        ]
        node_hover = [
            f"<b>{n}</b><br>Type: {def_map.get(n,{}).get('type','unknown')}<br>"
            f"Class: {def_map.get(n,{}).get('class','—')}<br>"
            f"Page: {def_map.get(n,{}).get('page','—')}<br>"
            f"{'⚠ In circular interlock' if n in cycle_nodes else ''}"
            for n in node_labels
        ]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(size=14, color=node_colors, line=dict(width=1.5, color="#ffffff")),
            text=node_labels,
            textposition="top center",
            textfont=dict(size=9, color="#0f172a"),
            hovertext=node_hover,
            hoverinfo="text",
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            height=500,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(t=20, b=20, l=20, r=20),
            font=dict(color="#0f172a"),
            annotations=[dict(
                text="🔴 Red = circular interlock  |  🔵 Blue = trigger node  |  ⬤ Grey = target node",
                showarrow=False, xref="paper", yref="paper", x=0.5, y=-0.02,
                font=dict(size=10, color="#64748b"), xanchor="center",
            )],
        )
        st.plotly_chart(fig, width='stretch')

    except ImportError:
        st.warning("networkx not installed — cannot render graph visualization.")

    # ── Edge table ─────────────────────────────────────────────────────────────
    if edges:
        st.markdown('<div class="section-lbl">Extracted Interlock Relationships</div>', unsafe_allow_html=True)
        rows_html = "".join(
            f'<tr>'
            f'<td><span class="tag">{e["from"]}</span></td>'
            f'<td style="color:#64748b;text-align:center">→ trips/stops →</td>'
            f'<td><span class="tag">{e["to"]}</span></td>'
            f'<td>{"⚠ In cycle" if e["from"] in cycle_nodes else ""}</td></tr>'
            for e in edges
        )
        st.markdown(
            f'<table class="tbl"><thead><tr><th>From (trigger)</th><th>Relationship</th><th>To (action)</th><th>Notes</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RAG EVIDENCE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def page_rag_explorer():
    _header("RAG Evidence Explorer", "Inspect how the system retrieves evidence from the document to support verification findings")

    vstate = st.session_state.verification_state
    if not vstate:
        _no_analysis()
        return

    collection_name = vstate.get("doc_collection_name", "")
    if not collection_name:
        st.warning("Document index not available. The knowledge index is built during Pass 1 of the analysis. Re-run analysis to enable RAG queries.")
        return

    cfg = _load_config()

    st.markdown('<div class="section-lbl">Enter a Verification Question</div>', unsafe_allow_html=True)
    col_q, col_n = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Verification query",
            placeholder="e.g. What is the high pressure trip setpoint for PT-101?",
            label_visibility="collapsed",
        )
    with col_n:
        top_k = st.selectbox("Top K results", [3, 5, 8], label_visibility="collapsed")

    if not query:
        st.markdown('<div class="card-sm"><p style="color:#64748b;font-size:0.85rem">Enter a question above to retrieve relevant document chunks. This demonstrates how the RAG pipeline grounds AI findings in actual document evidence.</p></div>', unsafe_allow_html=True)
        return

    with st.spinner("Querying document vector index…"):
        try:
            from src.ingestion.document_indexer import query_doc_index
            results = query_doc_index(collection_name, query, cfg, n=top_k)
        except Exception as e:
            st.error(f"Query failed: {e}")
            return

    if not results:
        st.warning("No relevant document chunks retrieved. The query may not match any content in this document, or the similarity threshold was not met.")
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    q1, q2 = st.columns([3, 1])
    with q1:
        st.markdown(f'<div class="section-lbl">Query</div><div class="card-sm" style="font-size:0.88rem;color:#0f172a">"{query}"</div>', unsafe_allow_html=True)
    with q2:
        st.markdown(f'<div class="section-lbl">Retrieved</div><div class="card-sm" style="font-size:1.2rem;font-weight:700;color:#2563eb;text-align:center">{len(results)} chunks</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-lbl" style="margin-top:16px">Retrieved Document Chunks</div>', unsafe_allow_html=True)

    for i, chunk in enumerate(results, 1):
        score = chunk.get("score", 0)
        bar_w = int(score * 100)
        bar_color = "#16a34a" if score > 0.7 else ("#ca8a04" if score > 0.5 else "#94a3b8")
        st.markdown(
            f'<div class="card" style="padding:14px 18px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            f'<span style="font-size:0.75rem;font-weight:600;color:#64748b">CHUNK {i}  ·  Page {chunk.get("page","—")}  ·  {chunk.get("section","")[:50]}</span>'
            f'<span style="font-size:0.78rem;font-weight:700;color:{bar_color}">Relevance: {score:.2f}</span>'
            f'</div>'
            f'<div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{bar_w}%;background:{bar_color}"></div></div>'
            f'<div class="evidence-chunk" style="margin-top:10px">{chunk.get("text","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Show related findings ──────────────────────────────────────────────────
    st.markdown('<div class="section-lbl" style="margin-top:16px">Findings Referencing Similar Content</div>', unsafe_allow_html=True)
    all_findings = vstate.get("all_findings", [])
    query_words = set(query.lower().split())
    related = [
        f for f in all_findings
        if any(w in (f.get("description","") + f.get("quote","")).lower() for w in query_words if len(w) > 4)
    ][:5]

    if related:
        for f in related:
            sev = f.get("severity", "INFO")
            st.markdown(
                f'<div class="finding {sev}">'
                f'{_badge(sev)} <b>p.{f.get("page_number","?")}</b>  [{f.get("agent","").title()}]  {f.get("description","")[:120]}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No existing findings match this query.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — CROSS-DOCUMENT VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def page_cross_doc():
    _header("Cross-Document Verification", "Compare parameter values across documents to detect inconsistencies")

    if len(st.session_state.analyzed_docs) < 1:
        _empty(
            "No analyzed documents yet. Run CND Analysis on at least one document to enable cross-document comparison.",
            "📊"
        )
        return

    analyzed = st.session_state.analyzed_docs

    if len(analyzed) == 1:
        st.info("**Single document mode.** Only one document has been analyzed. Upload and analyze additional documents (P&ID, plant spec, etc.) for full cross-document comparison. Showing internal consistency data below.")

    # ── Internal: prose setpoints vs table ────────────────────────────────────
    st.markdown('<div class="section-lbl">Setpoint Consistency — Prose vs. Structured Table</div>', unsafe_allow_html=True)

    for doc_name, state in analyzed.items():
        setpoint_index = state.get("setpoint_index", {})
        setpoint_table = {sp["tag"]: sp for sp in state.get("setpoint_table", [])}

        rows_html = ""
        mismatches = 0

        for tag in sorted(set(list(setpoint_index.keys()) + list(setpoint_table.keys()))):
            prose = setpoint_index.get(tag, {})
            table_sp = setpoint_table.get(tag, {})

            if prose and table_sp:
                for param, pdata in prose.items():
                    p_val = pdata.get("value")
                    t_val = table_sp.get("value")
                    if p_val is not None and t_val is not None:
                        match = abs(p_val - t_val) / max(abs(t_val), 0.001) < 0.10
                        icon = "✓" if match else "⚠"
                        color = "#16a34a" if match else "#ea580c"
                        if not match:
                            mismatches += 1
                        rows_html += (
                            f'<tr><td><span class="tag">{tag}</span></td>'
                            f'<td>{param}</td>'
                            f'<td>{p_val} {pdata.get("unit","")}</td>'
                            f'<td>{t_val} {table_sp.get("units","")}</td>'
                            f'<td style="color:{color};font-weight:600">{icon}</td></tr>'
                        )

        if rows_html:
            st.markdown(
                f'<div class="card" style="padding:0;overflow:hidden;margin-bottom:16px">'
                f'<div style="padding:10px 16px;background:#f8fafc;border-bottom:1px solid #e2e8f0;font-size:0.82rem;font-weight:600;color:#0f172a">'
                f'{doc_name} — {mismatches} mismatch(es)</div>'
                f'<table class="tbl"><thead><tr><th>Tag</th><th>Parameter</th><th>Prose Value</th><th>Table Value</th><th>Match</th></tr></thead>'
                f'<tbody>{rows_html}</tbody></table></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="card-sm">**{doc_name}:** No overlapping setpoint data between prose and table extraction.</div>', unsafe_allow_html=True)

    # ── Multi-doc: tag inventory comparison ───────────────────────────────────
    if len(analyzed) > 1:
        st.markdown('<div class="section-lbl" style="margin-top:16px">Tag Inventory Comparison Across Documents</div>', unsafe_allow_html=True)
        all_tags = {}
        for doc_name, state in analyzed.items():
            for t in state.get("tag_inventory", []):
                if t not in all_tags:
                    all_tags[t] = []
                all_tags[t].append(doc_name)

        doc_names = list(analyzed.keys())
        header_html = "<th>Tag</th>" + "".join(f"<th>{d[:30]}</th>" for d in doc_names) + "<th>Status</th>"
        rows_html = ""
        for tag in sorted(all_tags):
            presence = [("✓" if d in all_tags[tag] else "—") for d in doc_names]
            all_present = all(p == "✓" for p in presence)
            status = "✓ Consistent" if all_present else "⚠ Missing in some documents"
            color  = "#16a34a" if all_present else "#ea580c"
            cells  = "".join(
                f'<td style="text-align:center;color:{"#16a34a" if p=="✓" else "#94a3b8"}">{p}</td>'
                for p in presence
            )
            rows_html += f'<tr><td><span class="tag">{tag}</span></td>{cells}<td style="color:{color}">{status}</td></tr>'

        st.markdown(
            f'<div class="card" style="padding:0;overflow:hidden">'
            f'<table class="tbl"><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="card-sm" style="margin-top:16px">'
            '<b>Multi-document tag comparison unavailable</b> — '
            'Analyze additional documents (e.g., P&ID, plant specification) to compare tag inventories across documents.'
            '</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — RULE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def page_rules():
    _header("Rule Validation", "Engineering rule validation results from the verification pipeline")

    vstate = st.session_state.verification_state
    if not vstate:
        _no_analysis()
        return

    findings = vstate.get("all_findings", [])
    cfg = _load_config()

    # ── Build rule registry from config + findings ─────────────────────────────
    # Rules defined in config (structural completeness)
    BUILTIN_RULES = []
    for sec in cfg.get("required_sections", []):
        BUILTIN_RULES.append({
            "id":           f"RULE-SEC-{sec[:8].upper().replace(' ','-')}",
            "name":         f"Required Section: {sec}",
            "description":  f"CND must contain a '{sec}' section (ISA-88)",
            "source":       "ISA-88",
            "agent":        "completeness",
            "check_fn":     lambda s=sec: not any(
                s.lower() in t.lower() or t.lower() in s.lower()
                for t in vstate.get("sections", {})
            ),
        })
    for fld in cfg.get("required_interlock_fields", []):
        BUILTIN_RULES.append({
            "id":          f"RULE-IL-{fld[:8].upper()}",
            "name":        f"Interlock Field: {fld}",
            "description": f"Each interlock definition must contain field '{fld}' (ISA-84)",
            "source":      "ISA-84",
            "agent":       "completeness",
            "check_fn":    None,  # evaluated via findings
        })

    # VQS score per agent
    vqs = vstate.get("vqs_scores", {})

    st.markdown('<div class="section-lbl">Score-Based Rule Outcomes</div>', unsafe_allow_html=True)
    score_rules = [
        ("Completeness rules (ISA-88 / ISA-18.2)",  "completeness",    vqs.get("completeness", 1)),
        ("Consistency rules (internal cross-check)", "consistency",     vqs.get("consistency", 1)),
        ("Ambiguity rules (programmability)",        "ambiguity",       vqs.get("ambiguity", 1)),
        ("Logic coherence rules (ISA-84 / IEC-61511)","logic_coherence",vqs.get("logic_coherence", 1)),
    ]
    for rule_name, agent, score in score_rules:
        score_pct = score * 100
        result = "PASS" if score >= 0.85 else ("WARN" if score >= 0.65 else "FAIL")
        color = "#16a34a" if result == "PASS" else ("#ca8a04" if result == "WARN" else "#dc2626")
        n_agent = sum(1 for f in findings if f.get("agent") == agent)
        st.markdown(
            f'<div class="rule-row">'
            f'<span class="badge badge-{result}">{result}</span> &nbsp;'
            f'<b style="color:#0f172a">{rule_name}</b> &nbsp;'
            f'<span style="color:{color};font-weight:600">{score_pct:.0f}%</span> &nbsp;'
            f'<span style="color:#64748b;font-size:0.78rem">{n_agent} finding(s)</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-lbl" style="margin-top:20px">Individual Rule Findings</div>', unsafe_allow_html=True)

    # Group findings by rule_violated
    rules_map: dict[str, list] = {}
    for f in findings:
        key = f.get("rule_violated", "No rule cited") or "No rule cited"
        rules_map.setdefault(key, []).append(f)

    # Also add PASS for required sections that ARE present
    sections = vstate.get("sections", {})
    for sec in cfg.get("required_sections", []):
        found = any(sec.lower() in t.lower() or t.lower() in sec.lower() for t in sections)
        if found:
            rules_map.setdefault(f"ISA-88 Required Section: {sec}", [])

    for rule_name, rule_findings in sorted(rules_map.items()):
        n = len(rule_findings)
        result_label = "FAIL" if any(f.get("severity") in ("CRITICAL","HIGH") for f in rule_findings) else \
                 "WARN" if n > 0 else "PASS"
        result_icon = {"FAIL": "✗", "WARN": "⚠", "PASS": "✓"}.get(result_label, "·")
        badge = f'<span class="badge badge-{result_label}">{result_label}</span>'
        with st.expander(f"{result_icon} {rule_name} — {n} finding(s)", expanded=(result_label == "FAIL" and n > 0)):
            if not rule_findings:
                st.markdown('<span style="color:#16a34a">✓ No violations found — rule passes.</span>', unsafe_allow_html=True)
            for f in rule_findings:
                sev = f.get("severity", "INFO")
                st.markdown(
                    f'{_badge(sev)} p.{f.get("page_number","?")} &nbsp;'
                    f'`{f.get("finding_type","—")}` &nbsp; {f.get("description","")[:120]}',
                    unsafe_allow_html=True,
                )
                if f.get("rule_text"):
                    st.caption(f"Standard: {f.get('rule_text')[:200]}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — TEST & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def page_evaluation():
    _header("Test & Evaluation", "Research evaluation metrics — precision, recall, F1 against labeled ground truth")

    try:
        from evaluation.evaluator import load_ground_truth, evaluate_document, aggregate_results, print_report
        from evaluation.baselines import _b1_run, _b2_run
    except ImportError as e:
        st.error(f"Evaluation module not available: {e}")
        st.caption("Make sure you are running from the cnd-verify/ directory.")
        return

    # ── Ground truth info ──────────────────────────────────────────────────────
    st.markdown('<div class="section-lbl">Ground Truth Dataset</div>', unsafe_allow_html=True)
    try:
        gt = load_ground_truth()
        info = gt.get("dataset_info", {})
        docs = gt.get("documents", [])
    except Exception as e:
        st.error(f"Could not load ground truth: {e}")
        return

    g1, g2, g3, g4 = st.columns(4)
    _kpi(g1, info.get("total_documents", len(docs)), "Test Documents")
    _kpi(g2, info.get("total_errors", "—"), "Labeled Errors")
    _kpi(g3, info.get("version", "—"), "Dataset Version")
    _kpi(g4, info.get("created", "—"), "Created")

    # Show doc list
    rows_html = "".join(
        f'<tr><td>{d["doc_id"]}</td>'
        f'<td style="font-size:0.78rem;color:#64748b"><code>{d["filename"]}</code></td>'
        f'<td>{len(d.get("errors",[]))}</td>'
        f'<td>{d.get("expected_routing","—")}</td>'
        f'<td>{"✓ Found" if Path("data/synthetic_cnds/"+d["filename"]).exists() else "✗ Missing"}</td></tr>'
        for d in docs
    )
    st.markdown(
        f'<table class="tbl"><thead><tr><th>Doc ID</th><th>File</th><th>GT Errors</th><th>Expected Routing</th><th>File Status</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Run controls ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-lbl">Run Evaluation</div>', unsafe_allow_html=True)
    ev1, ev2, ev3 = st.columns(3)

    available_docs = [d["doc_id"] for d in docs if Path(f"data/synthetic_cnds/{d['filename']}").exists()]
    selected_docs = ev1.multiselect("Select documents", available_docs, default=available_docs[:1] if available_docs else [])
    run_b1 = ev2.checkbox("Run Baseline B1 (rule-based)", value=False)
    run_b2 = ev3.checkbox("Run Baseline B2 (single LLM)", value=False)

    if not selected_docs:
        st.caption("Select at least one document to run evaluation.")
        return

    if st.button("▶  Run Evaluation", type="primary"):
        results_list = []
        data_dir = "data"
        progress = st.progress(0)

        for i, doc_info in enumerate([d for d in docs if d["doc_id"] in selected_docs]):
            progress.progress((i + 1) / len(selected_docs))
            with st.spinner(f"Evaluating {doc_info['doc_id']}…"):
                try:
                    r = evaluate_document(doc_info, data_dir, verbose=True)
                    results_list.append(r)
                    if not r.get("skipped"):
                        ov = r["overall"]
                        st.info(
                            f"**{doc_info['doc_id']}** — "
                            f"P={ov['precision']:.2f}  R={ov['recall']:.2f}  F1={ov['f1']:.2f}  "
                            f"Routing: {r.get('routing','?')} {'✓' if r.get('routing_correct') else '✗'}"
                        )
                    else:
                        st.warning(f"{doc_info['doc_id']} skipped: {r.get('error','')}")
                except Exception as e:
                    st.error(f"Evaluation failed for {doc_info['doc_id']}: {e}")

        if results_list:
            summary = aggregate_results(results_list)
            st.session_state.eval_results = {"results": results_list, "summary": summary}

    # ── Display stored results ─────────────────────────────────────────────────
    if not st.session_state.eval_results:
        _empty("No evaluation results yet. Select documents and click Run Evaluation.", "📐")
        return

    stored = st.session_state.eval_results
    results_list = stored["results"]
    summary = stored["summary"]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-lbl">System Performance — CND-Verify Full Pipeline</div>', unsafe_allow_html=True)

    e1, e2, e3, e4, e5 = st.columns(5)
    _kpi(e1, f"{summary.get('micro_precision',0):.3f}", "Micro Precision",  "blue")
    _kpi(e2, f"{summary.get('micro_recall',0):.3f}",   "Micro Recall",     "blue")
    _kpi(e3, f"{summary.get('micro_f1',0):.3f}",       "Micro F1",         "blue")
    _kpi(e4, f"{summary.get('macro_f1',0):.3f}",       "Macro F1",         "blue")
    _kpi(e5, f"{summary.get('routing_accuracy',0):.0%}", "Routing Accuracy", "green")

    st.markdown('<div class="section-lbl" style="margin-top:16px">Critical Error Detection</div>', unsafe_allow_html=True)
    fnr = summary.get("fnr_critical", 0)
    fnr_ok = fnr < 0.10
    fnr_color = "#16a34a" if fnr_ok else "#dc2626"
    st.markdown(
        f'<div class="card">'
        f'<span style="font-size:1.4rem;font-weight:700;color:{fnr_color}">{fnr:.3f}</span>'
        f'<span style="font-size:0.85rem;color:#64748b;margin-left:12px">FNR (Critical errors missed)</span>'
        f'<span style="font-size:0.80rem;color:{fnr_color};margin-left:16px">{"✓ Target <0.10 met" if fnr_ok else "✗ Target <0.10 NOT met"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Per-agent F1 bar chart
    per_agent = summary.get("per_agent_f1", {})
    if per_agent:
        agents = list(per_agent.keys())
        f1s    = [per_agent[a] for a in agents]
        fig = go.Figure(go.Bar(
            x=agents, y=f1s,
            marker_color=["#dc2626" if v < 0.6 else ("#ca8a04" if v < 0.75 else "#16a34a") for v in f1s],
            text=[f"{v:.3f}" for v in f1s], textposition="outside",
        ))
        fig.add_hline(y=0.75, line_dash="dot", line_color="#64748b", annotation_text="Target F1=0.75")
        fig.update_layout(
            title="Per-Agent F1 Score", height=280,
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            yaxis=dict(range=[0, 1.1], gridcolor="#f1f5f9"),
            xaxis=dict(tickfont={"color":"#0f172a"}),
            font=dict(color="#0f172a"),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, width='stretch')

    # Per-document table
    st.markdown('<div class="section-lbl">Per-Document Results</div>', unsafe_allow_html=True)
    rows_html = ""
    for r in results_list:
        if r.get("skipped"):
            continue
        ov = r["overall"]
        rc = "✓" if r.get("routing_correct") else "✗"
        rows_html += (
            f'<tr>'
            f'<td><b>{r["doc_id"]}</b></td>'
            f'<td>{r.get("error_count_gt",0)}</td>'
            f'<td>{r.get("findings_count",0)}</td>'
            f'<td>{ov["tp"]} / {ov["fp"]} / {ov["fn"]}</td>'
            f'<td>{ov["precision"]:.2f}</td>'
            f'<td>{ov["recall"]:.2f}</td>'
            f'<td>{ov["f1"]:.2f}</td>'
            f'<td>{r.get("routing","?")} {rc}</td>'
            f'<td style="color:#64748b">{r.get("elapsed_s","?")}s</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="tbl"><thead><tr>'
        f'<th>Document</th><th>GT Errors</th><th>Findings</th><th>TP/FP/FN</th>'
        f'<th>Precision</th><th>Recall</th><th>F1</th><th>Routing</th><th>Time</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — EXECUTION LOGS
# ══════════════════════════════════════════════════════════════════════════════

def page_logs():
    _header("Execution Logs", "Detailed pipeline execution log from the last analysis run")

    logs = st.session_state.exec_log
    if not logs:
        _empty("No execution logs yet. Run CND Analysis to populate this log.", "📋")
        return

    # ── Summary ────────────────────────────────────────────────────────────────
    total_time = sum(l.get("elapsed_s", 0) for l in logs)
    failed = sum(1 for l in logs if l.get("status") == "Failed")
    skipped = sum(1 for l in logs if l.get("status") == "Skipped")

    k1, k2, k3, k4 = st.columns(4)
    _kpi(k1, len(logs),                 "Total Stages")
    _kpi(k2, f"{total_time:.1f}s",      "Total Time", "blue")
    _kpi(k3, failed,                    "Failed",  "red"  if failed  else "")
    _kpi(k4, skipped,                   "Skipped", "amber" if skipped else "")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Log table ──────────────────────────────────────────────────────────────
    rows_html = ""
    for i, entry in enumerate(logs, 1):
        status = entry.get("status", "")
        s_color = {"Completed": "#16a34a", "Failed": "#dc2626", "Skipped": "#94a3b8", "Running": "#2563eb"}.get(status, "#0f172a")
        detail = entry.get("detail", "")
        rows_html += (
            f'<tr>'
            f'<td style="color:#64748b">{i}</td>'
            f'<td><b>{entry.get("stage","—")}</b></td>'
            f'<td style="color:{s_color};font-weight:600">{status}</td>'
            f'<td>{entry.get("elapsed_s","—")}s</td>'
            f'<td style="color:#64748b;font-size:0.75rem">{detail[:80]}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="tbl"><thead><tr><th>#</th><th>Stage</th><th>Status</th><th>Duration</th><th>Detail</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Per-agent findings count ───────────────────────────────────────────────
    vstate = st.session_state.verification_state
    if vstate:
        st.markdown('<div class="section-lbl">Agent Output Summary</div>', unsafe_allow_html=True)
        AGENT_REPORTS = [
            ("completeness_report",    "Completeness Agent"),
            ("consistency_report",     "Consistency Agent"),
            ("ambiguity_report",       "Ambiguity Agent"),
            ("logic_coherence_report", "Logic Coherence Agent"),
            ("image_report",           "Image Agent"),
        ]
        rows_html = ""
        for key, name in AGENT_REPORTS:
            report = vstate.get(key) or {}
            if report:
                n = len(report.get("findings", []))
                score = report.get("score", 1.0) * 100
                critical = report.get("critical_count", 0)
                high = report.get("high_count", 0)
                err_msg = report.get("error_message", "")
                rows_html += (
                    f'<tr><td><b>{name}</b></td>'
                    f'<td>{n}</td>'
                    f'<td style="color:#dc2626">{critical}</td>'
                    f'<td style="color:#ea580c">{high}</td>'
                    f'<td>{score:.0f}%</td>'
                    f'<td style="color:#64748b">{err_msg[:60] if err_msg else "—"}</td></tr>'
                )
            else:
                rows_html += f'<tr><td><b>{name}</b></td><td colspan="5" style="color:#94a3b8">Did not run or no output</td></tr>'

        st.markdown(
            f'<table class="tbl"><thead><tr>'
            f'<th>Agent</th><th>Findings</th><th>Critical</th><th>High</th><th>Score</th><th>Error</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def page_settings():
    _header("Settings", "System configuration — view active settings from config.yaml")

    cfg = _load_config()
    if not cfg:
        st.error("config.yaml not found. Run from the cnd-verify/ directory.")
        return

    st.info("Settings are read from **config.yaml** in the project root. Edit that file to change system behaviour, then restart the application.")

    s1, s2 = st.columns(2)

    with s1:
        st.markdown('<div class="section-lbl">LLM Configuration</div>', unsafe_allow_html=True)
        model_cfg = cfg.get("model", {})
        st.markdown(
            f'<div class="card">'
            f'<div style="margin-bottom:8px"><b>Provider:</b> Ollama (local)</div>'
            f'<div style="margin-bottom:8px"><b>Model:</b> <code>{model_cfg.get("name","—")}</code></div>'
            f'<div style="margin-bottom:8px"><b>Temperature:</b> {model_cfg.get("temperature","—")}</div>'
            f'<div><b>Timeout:</b> {model_cfg.get("timeout_seconds","—")}s per call</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-lbl" style="margin-top:16px">Embeddings</div>', unsafe_allow_html=True)
        emb_cfg = cfg.get("embeddings", {})
        st.markdown(
            f'<div class="card">'
            f'<div style="margin-bottom:8px"><b>Model:</b> <code>{emb_cfg.get("model","—")}</code></div>'
            f'<div style="margin-bottom:8px"><b>Chunk size:</b> {emb_cfg.get("chunk_size","—")} tokens</div>'
            f'<div style="margin-bottom:8px"><b>Overlap:</b> {emb_cfg.get("chunk_overlap","—")} tokens</div>'
            f'<div style="margin-bottom:8px"><b>Top-K retrieval:</b> {emb_cfg.get("retrieval_top_k","—")}</div>'
            f'<div><b>Min similarity threshold:</b> {emb_cfg.get("min_similarity","—")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-lbl" style="margin-top:16px">Vector Database</div>', unsafe_allow_html=True)
        chroma_cfg = cfg.get("chromadb", {})
        st.markdown(
            f'<div class="card">'
            f'<div style="margin-bottom:8px"><b>Provider:</b> ChromaDB (local persistent)</div>'
            f'<div style="margin-bottom:8px"><b>Path:</b> <code>{chroma_cfg.get("path","—")}</code></div>'
            f'<div><b>Domain KB collection:</b> <code>{chroma_cfg.get("collection_name","—")}</code></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown('<div class="section-lbl">Scoring Weights</div>', unsafe_allow_html=True)
        weights = cfg.get("scoring", {}).get("weights", {})
        thresholds = cfg.get("scoring", {}).get("thresholds", {})
        penalties = cfg.get("scoring", {}).get("penalty_per_finding", {})

        weight_items = [(k, v) for k, v in weights.items()]
        fig = go.Figure(go.Pie(
            labels=[k.replace("_"," ").title() for k, _ in weight_items],
            values=[v for _, v in weight_items],
            marker_colors=["#2563eb","#7c3aed","#ca8a04","#dc2626"],
            hole=0.4, textinfo="label+percent",
            textfont_size=11,
        ))
        fig.update_layout(
            height=200, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            showlegend=False, font={"color":"#0f172a"},
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            f'<div class="card">'
            f'<b>Routing thresholds</b><br>'
            f'<span style="color:#16a34a">AUTO PASS:</span> VQS ≥ {thresholds.get("pass",0.85)*100:.0f}%<br>'
            f'<span style="color:#ca8a04">NEEDS REVIEW:</span> VQS ≥ {thresholds.get("review",0.65)*100:.0f}%<br>'
            f'<span style="color:#dc2626">FAIL:</span> VQS &lt; {thresholds.get("review",0.65)*100:.0f}%'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-lbl" style="margin-top:8px">Penalty per Finding</div>', unsafe_allow_html=True)
        rows_html = "".join(
            f'<tr><td>{_badge(sev)}</td><td>−{pen*100:.0f}% VQS</td></tr>'
            for sev, pen in penalties.items()
        )
        st.markdown(
            f'<table class="tbl"><thead><tr><th>Severity</th><th>Penalty</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-lbl" style="margin-top:16px">Required Sections Checklist</div>', unsafe_allow_html=True)
        for sec in cfg.get("required_sections", []):
            st.markdown(f'<div style="font-size:0.82rem;color:#0f172a;padding:3px 0">• {sec}</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-lbl">Raw config.yaml</div>', unsafe_allow_html=True)
    with st.expander("View raw configuration", expanded=False):
        try:
            with open("config.yaml") as f:
                st.code(f.read(), language="yaml")
        except Exception:
            st.caption("config.yaml not found in working directory.")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

page_num = page.split(".")[0].strip()
dispatch = {
    "1": page_overview,
    "2": page_documents,
    "3": page_analysis,
    "4": page_findings,
    "5": page_knowledge_graph,
    "6": page_rag_explorer,
    "7": page_cross_doc,
    "8": page_rules,
    "9": page_evaluation,
    "10": page_logs,
    "11": page_settings,
}
dispatch.get(page_num, page_overview)()
