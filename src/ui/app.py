"""
src/ui/app.py  —  CND-Verify professional dashboard
Three pages: Upload & Verify | Score Dashboard | Review Findings
"""

import streamlit as st
import plotly.graph_objects as go
import json, os, sys, time, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.document_loader import prepare_document
from src.scoring.score_engine import run_score_engine, get_routing_message


def _check_ollama() -> tuple[bool, str]:
    """Returns (is_running, model_status)."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            llama_found = any("llama3.1" in m or "llama3" in m for m in models)
            if llama_found:
                return True, "Ollama running ✓  |  llama3.1 ready ✓"
            elif models:
                return True, f"Ollama running ✓  |  llama3.1 NOT found (available: {', '.join(models[:2])})"
            else:
                return True, "Ollama running ✓  |  No models pulled yet — run: ollama pull llama3.1"
    except Exception:
        pass
    return False, "Ollama NOT running — LLM checks disabled"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CND-Verify",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Agent status card */
    .agent-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
        border-left: 4px solid #3b82f6;
    }
    .agent-card.ok  { border-left-color: #22c55e; }
    .agent-card.warn{ border-left-color: #f59e0b; }
    .agent-card.bad { border-left-color: #ef4444; }
    .agent-card h4  { margin: 0 0 4px 0; font-size: 0.95rem; color: #94a3b8; }
    .agent-card .big{ font-size: 2rem; font-weight: 700; color: #f1f5f9; }
    .agent-card .sub{ font-size: 0.78rem; color: #64748b; margin-top: 2px; }

    /* Severity badge */
    .badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 999px; font-size: 0.72rem;
        font-weight: 700; letter-spacing: .04em;
    }
    .badge-CRITICAL { background:#fecaca; color:#991b1b; }
    .badge-HIGH     { background:#fed7aa; color:#9a3412; }
    .badge-WARNING  { background:#fef08a; color:#854d0e; }
    .badge-INFO     { background:#bfdbfe; color:#1e40af; }

    /* Finding card */
    .finding-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 14px;
        border-left: 4px solid #64748b;
    }
    .finding-card.CRITICAL { border-left-color: #ef4444; }
    .finding-card.HIGH     { border-left-color: #f97316; }
    .finding-card.WARNING  { border-left-color: #eab308; }
    .finding-card.INFO     { border-left-color: #3b82f6; }

    /* Page badge */
    .page-badge {
        display: inline-block;
        background: #334155; color: #93c5fd;
        border-radius: 6px; padding: 3px 10px;
        font-size: 0.8rem; font-weight: 600;
    }
    /* Metric card */
    .metric-box {
        background: #1e293b; border-radius: 10px;
        padding: 14px 18px; text-align: center;
    }
    .metric-box .val { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
    .metric-box .lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

    /* Tag chip */
    .tag-chip {
        display: inline-block; background: #0f172a;
        color: #7dd3fc; border: 1px solid #1e3a5f;
        border-radius: 5px; padding: 2px 8px;
        font-size: 0.75rem; font-family: monospace;
        margin: 2px;
    }
    /* Section row */
    .sec-row {
        display: flex; align-items: center;
        padding: 5px 0; border-bottom: 1px solid #1e293b;
        font-size: 0.85rem;
    }
    .sec-found { color: #4ade80; margin-right: 8px; }
    .sec-miss  { color: #f87171; margin-right: 8px; }

    div[data-testid="stExpander"] > div { background: #1e293b; }
    .stProgress > div > div { background: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 CND-Verify")
    st.markdown("*Control Narrative Pre-Verification*")
    st.divider()
    page = st.radio("Navigation", ["Upload & Verify", "Score Dashboard", "Review Findings"])
    st.divider()

    # Ollama health check
    ollama_ok, ollama_msg = _check_ollama()
    if ollama_ok:
        st.success(ollama_msg)
    else:
        st.error(ollama_msg)
        st.caption("Start with: `ollama serve`  then  `ollama pull llama3.1`")
        st.caption("Without Ollama: only deterministic checks run (missing sections, tag patterns)")
    st.divider()

    if st.session_state.get("verification_state"):
        vqs = st.session_state.verification_state.get("vqs_scores", {})
        score = vqs.get("composite", 0) * 100
        routing = vqs.get("routing_decision", "")
        color = {"AUTO_PASS": "🟢", "NEEDS_REVIEW": "🟡", "FAIL": "🔴", "MANDATORY_REVISION": "🔴"}.get(routing, "⚪")
        st.markdown(f"**Last result:** {color} {score:.0f}%")
        st.caption(routing.replace("_", " "))
        st.divider()

    st.caption("M.Tech Research Project")
    st.caption("Mentor: Padmaja Bodanapu")
    st.caption("GE Vernova Innovation Labs")


# ── Session defaults ──────────────────────────────────────────────────────────
for k, v in {
    "verification_state": None,
    "finding_statuses": {},
    "finding_notes": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD & VERIFY
# ═════════════════════════════════════════════════════════════════════════════

def page_upload():
    st.title("Upload Control Narrative Document")
    st.markdown(
        "Upload a CND (PDF or TXT). Three AI agents analyse it for "
        "**completeness**, **consistency**, and **ambiguity** defects "
        "before any PLC code is generated."
    )

    # ── What each agent checks ────────────────────────────────────────────────
    st.markdown("### What will be checked")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class="agent-card ok">
<h4>Completeness Agent</h4>
<div class="sub">ISA-88 / ISA-84 / ISA-18.2</div>
<ul style="font-size:0.82rem;color:#94a3b8;margin-top:8px;padding-left:16px">
<li>Required sections present</li>
<li>All interlock fields defined</li>
<li>All alarm fields defined</li>
<li>All referenced tags declared</li>
</ul>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="agent-card warn">
<h4>Consistency Agent</h4>
<div class="sub">Internal cross-check</div>
<ul style="font-size:0.82rem;color:#94a3b8;margin-top:8px;padding-left:16px">
<li>Setpoint prose vs. table match</li>
<li>Equipment type consistent</li>
<li>Sequence ordering non-contradictory</li>
<li>Units identical across document</li>
</ul>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class="agent-card bad">
<h4>Ambiguity Agent</h4>
<div class="sub">Programmability check</div>
<ul style="font-size:0.82rem;color:#94a3b8;margin-top:8px;padding-left:16px">
<li>No vague quantities ("high")</li>
<li>No vague timing ("shortly")</li>
<li>No unresolved references ("it")</li>
<li>No incomplete conditions</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Upload / sample select ────────────────────────────────────────────────
    col_up, col_sample = st.columns(2)
    with col_up:
        st.markdown("### Upload your document")
        uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")
    with col_sample:
        st.markdown("### Or use a test document")
        samples = {
            "— Select —": None,
            "Pump Station 01 — Clean (no errors)":
                "data/synthetic_cnds/pump_station_01_clean.txt",
            "Pump Station 02 — 6 completeness errors":
                "data/synthetic_cnds/pump_station_02_completeness_errors.txt",
            "Pump Station 03 — 4 consistency errors":
                "data/synthetic_cnds/pump_station_03_consistency_errors.txt",
        }
        chosen = st.selectbox("Sample", list(samples.keys()), label_visibility="collapsed")

    st.divider()
    run_btn = st.button("▶  Run Verification", type="primary", use_container_width=False)

    if run_btn:
        fp = _resolve(uploaded, samples.get(chosen))
        if not fp:
            st.warning("Please upload a file or select a sample document.")
            return

        # ── Step progress ─────────────────────────────────────────────────────
        prog = st.progress(0)
        status_box = st.empty()

        def step(n, msg):
            prog.progress(n)
            status_box.info(f"**Step {n//20}/5 — {msg}**")

        step(10, "Parsing document and extracting structure...")
        state = prepare_document(fp)
        _show_parse_summary(state)

        step(25, "Running Completeness Agent...")
        from src.agents.completeness_agent import run_completeness_agent
        state = _run_safe(run_completeness_agent, state, "Completeness Agent")

        step(50, "Running Consistency Agent...")
        from src.agents.consistency_agent import run_consistency_agent
        state = _run_safe(run_consistency_agent, state, "Consistency Agent")

        step(75, "Running Ambiguity Agent...")
        from src.agents.ambiguity_agent import run_ambiguity_agent
        state = _run_safe(run_ambiguity_agent, state, "Ambiguity Agent")

        step(90, "Computing Verification Quality Score...")
        state = run_score_engine(state)
        prog.progress(100)
        status_box.empty()

        st.session_state.verification_state = state
        st.session_state.finding_statuses = {f["id"]: "OPEN" for f in state.get("all_findings", [])}
        st.session_state.finding_notes = {}

        _show_result_banner(state)


def _resolve(uploaded, sample_path):
    if uploaded:
        tmp = f"temp_{uploaded.name}"
        with open(tmp, "wb") as f:
            f.write(uploaded.getbuffer())
        return tmp
    if sample_path and os.path.exists(sample_path):
        return sample_path
    return None


def _run_safe(fn, state, name):
    """Run an agent, show its result immediately, never crash the pipeline."""
    try:
        t0 = time.time()
        state = fn(state)
        elapsed = time.time() - t0

        # Show what this agent found
        report_key = name.lower().split()[0] + "_report"
        report = state.get(report_key, {}) or {}
        n = len(report.get("findings", []))
        score = report.get("score", 1.0) * 100
        critical = report.get("critical_count", 0)
        high = report.get("high_count", 0)

        if n == 0:
            st.success(f"✅ **{name}** — No issues found  (score 100%)  [{elapsed:.1f}s]")
        elif critical > 0:
            st.error(f"🔴 **{name}** — {n} finding(s): {critical} CRITICAL, {high} HIGH  (score {score:.0f}%)  [{elapsed:.1f}s]")
        else:
            st.warning(f"🟡 **{name}** — {n} finding(s): {high} HIGH  (score {score:.0f}%)  [{elapsed:.1f}s]")

    except Exception as e:
        st.error(f"❌ **{name}** failed: {e}")
    return state


def _show_parse_summary(state):
    tags = state.get("tag_inventory", [])
    sections = state.get("sections", {})
    sp = st.expander(f"Document parsed — {len(tags)} ISA tags found, {len(sections)} sections detected", expanded=False)
    with sp:
        col_t, col_s = st.columns(2)
        with col_t:
            st.markdown("**ISA Tags found:**")
            chips = " ".join(f'<span class="tag-chip">{t}</span>' for t in sorted(tags))
            st.markdown(chips or "*none detected*", unsafe_allow_html=True)
        with col_s:
            st.markdown("**Sections detected:**")
            for sec in list(sections.keys())[:15]:
                st.markdown(f'<div class="sec-row"><span class="sec-found">✓</span>{sec}</div>',
                            unsafe_allow_html=True)


def _show_result_banner(state):
    vqs = state.get("vqs_scores", {})
    findings = state.get("all_findings", [])
    routing = vqs.get("routing_decision", "")
    composite = vqs.get("composite", 0) * 100
    critical = sum(1 for f in findings if f.get("severity") == "CRITICAL")
    high = sum(1 for f in findings if f.get("severity") == "HIGH")

    if routing == "AUTO_PASS":
        st.success(f"✅  **AUTO PASS** — Score {composite:.1f}%. {len(findings)} minor finding(s). Proceed to Score Dashboard.")
    elif routing == "NEEDS_REVIEW":
        st.warning(f"⚠️  **NEEDS REVIEW** — Score {composite:.1f}%. {len(findings)} finding(s) ({critical} critical, {high} high). Go to Score Dashboard.")
    elif routing == "MANDATORY_REVISION":
        missing = state.get("missing_critical_sections", [])
        detail = f"Missing critical section(s): {', '.join(missing)}" if missing else "Circular interlock detected"
        st.error(f"🚫  **MANDATORY REVISION** — {detail}. Document blocked from PLC code generation.")
    else:
        st.error(f"❌  **FAIL** — Score {composite:.1f}%. {len(findings)} finding(s). Document requires significant revision.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SCORE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.title("Verification Quality Score")

    if not st.session_state.verification_state:
        st.info("No results yet. Go to **Upload & Verify** first.")
        return

    state = st.session_state.verification_state
    vqs = state.get("vqs_scores", {})
    findings = state.get("all_findings", [])
    routing = vqs.get("routing_decision", "PENDING")
    composite = vqs.get("composite", 0)
    routing_info = get_routing_message(routing, composite)

    # ── Routing banner ────────────────────────────────────────────────────────
    color = routing_info["color"]
    msg = f"{routing_info['icon']} **{routing.replace('_', ' ')}** — {routing_info['message']}"
    missing = state.get("missing_critical_sections", [])
    if missing:
        msg += f"  |  Missing critical sections: **{', '.join(missing)}**"
    if color == "green":
        st.success(msg)
    elif color == "orange":
        st.warning(msg)
    else:
        st.error(msg)

    # ── Metric strip ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    _metric_box(m1, f"{composite*100:.1f}%", "Overall VQS Score")
    _metric_box(m2, f"{vqs.get('completeness',1)*100:.0f}%", "Completeness")
    _metric_box(m3, f"{vqs.get('consistency',1)*100:.0f}%", "Consistency")
    _metric_box(m4, f"{vqs.get('ambiguity',1)*100:.0f}%", "Ambiguity")
    critical = sum(1 for f in findings if f.get("severity") == "CRITICAL")
    _metric_box(m5, str(critical), "Critical Issues", red=critical > 0)

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        _gauge(composite * 100, "Overall VQS Score")
    with ch2:
        _radar(vqs)

    st.divider()

    # ── Agent status cards ────────────────────────────────────────────────────
    st.markdown("### Agent Results")
    ag1, ag2, ag3 = st.columns(3)

    comp_report = state.get("completeness_report") or {}
    cons_report = state.get("consistency_report") or {}
    amb_report  = state.get("ambiguity_report") or {}

    _agent_status_card(ag1, "Completeness Agent", comp_report,
                       "ISA-88 / ISA-84 / ISA-18.2", "Missing sections, interlock fields, alarm fields")
    _agent_status_card(ag2, "Consistency Agent", cons_report,
                       "Internal cross-check", "Setpoint mismatches, type contradictions, sequence errors")
    _agent_status_card(ag3, "Ambiguity Agent", amb_report,
                       "Programmability check", "Vague quantities, timing, references, conditions")

    # ── Equipment summary ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Equipment & Document Summary")
    eq1, eq2 = st.columns(2)

    tags = state.get("tag_inventory", [])
    sections = state.get("sections", {})
    setpoints = state.get("setpoint_table", [])
    config_sections = _load_required_sections()

    with eq1:
        st.markdown(f"**ISA Instrument Tags found — {len(tags)} total**")
        if tags:
            chips = " ".join(f'<span class="tag-chip">{t}</span>' for t in sorted(tags))
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No ISA tags detected (format: XX-NNN)")

        st.markdown(f"<br>**Setpoints table — {len(setpoints)} rows**", unsafe_allow_html=True)
        if setpoints:
            for sp in setpoints[:8]:
                st.caption(f"{sp.get('tag','')} | {sp.get('parameter','')} | {sp.get('value','')} {sp.get('units','')}")
            if len(setpoints) > 8:
                st.caption(f"... and {len(setpoints)-8} more")

    with eq2:
        st.markdown("**Section checklist**")
        for req in config_sections:
            found = any(_fuzzy(title, req) for title in sections)
            icon = '<span class="sec-found">✓</span>' if found else '<span class="sec-miss">✗</span>'
            st.markdown(
                f'<div class="sec-row">{icon}{req}</div>',
                unsafe_allow_html=True,
            )

    # ── Findings table ────────────────────────────────────────────────────────
    if findings:
        st.divider()
        st.markdown(f"### All Findings — {len(findings)} total")
        sev_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}
        for f in sorted(findings, key=lambda x: sev_order.get(x.get("severity","INFO"), 3)):
            _compact_row(f)


def _metric_box(col, val, label, red=False):
    color = "#ef4444" if red else "#f1f5f9"
    with col:
        st.markdown(
            f'<div class="metric-box"><div class="val" style="color:{color}">{val}</div>'
            f'<div class="lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )


def _agent_status_card(col, name, report, standard, description):
    with col:
        if not report:
            st.markdown(f"""
<div class="agent-card">
<h4>{name}</h4>
<div class="big" style="color:#475569">—</div>
<div class="sub">Did not run</div>
</div>""", unsafe_allow_html=True)
            return

        score = report.get("score", 1.0) * 100
        n = len(report.get("findings", []))
        critical = report.get("critical_count", 0)
        high = report.get("high_count", 0)

        css = "ok" if score >= 85 else ("warn" if score >= 65 else "bad")
        score_color = "#4ade80" if score >= 85 else ("#fb923c" if score >= 65 else "#f87171")
        st.markdown(f"""
<div class="agent-card {css}">
<h4>{name}</h4>
<div class="big" style="color:{score_color}">{score:.0f}%</div>
<div class="sub">{n} finding(s) &nbsp;·&nbsp; {critical} critical &nbsp;·&nbsp; {high} high</div>
<div class="sub" style="margin-top:6px;color:#475569">{standard}</div>
<div class="sub" style="color:#334155">{description}</div>
</div>""", unsafe_allow_html=True)


def _compact_row(f):
    sev = f.get("severity", "INFO")
    pg = f.get("page_number", 0)
    page_str = f"Page {pg}" if pg else "page ?"
    agent = f.get("agent", "").title()
    desc = f.get("description", "")
    ftype = f.get("finding_type", "")
    st.markdown(
        f'<span class="badge badge-{sev}">{sev}</span> &nbsp;'
        f'<span class="page-badge">{page_str}</span> &nbsp;'
        f'<span style="color:#64748b;font-size:0.78rem">[{agent}]</span> &nbsp;'
        f'<code style="font-size:0.75rem">{ftype}</code> &nbsp; {desc}',
        unsafe_allow_html=True,
    )
    if f.get("quote"):
        st.caption(f"› {f['quote'][:120]}")
    st.markdown('<hr style="border-color:#1e293b;margin:6px 0">', unsafe_allow_html=True)


def _gauge(value, title):
    color = "#22c55e" if value >= 85 else ("#f59e0b" if value >= 65 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                     "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0f172a",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 65],   "color": "#1e293b"},
                {"range": [65, 85],  "color": "#1e293b"},
                {"range": [85, 100], "color": "#1e293b"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": value},
        },
    ))
    fig.update_layout(
        height=260, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font={"color": "#94a3b8"},
        margin=dict(t=50, b=10, l=30, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _radar(vqs):
    cats = ["Completeness", "Consistency", "Ambiguity"]
    vals = [vqs.get(k, 1.0) * 100 for k in ["completeness", "consistency", "ambiguity"]]
    v2 = vals + [vals[0]]
    c2 = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=v2, theta=c2, fill="toself",
        fillcolor="rgba(59,130,246,0.15)",
        line={"color": "#3b82f6", "width": 2},
        marker={"size": 6, "color": "#3b82f6"},
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont={"color": "#64748b", "size": 9},
                            gridcolor="#1e293b", linecolor="#334155"),
            angularaxis=dict(tickfont={"color": "#94a3b8", "size": 11}, gridcolor="#1e293b"),
            bgcolor="#0f172a",
        ),
        paper_bgcolor="#0f172a",
        title={"text": "Score by Dimension", "font": {"color": "#94a3b8", "size": 13}},
        showlegend=False, height=260,
        margin=dict(t=50, b=10, l=40, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — REVIEW FINDINGS
# ═════════════════════════════════════════════════════════════════════════════

_SEV_BORDER = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "WARNING": "#eab308", "INFO": "#3b82f6"}

def page_review():
    st.title("Review Findings")

    if not st.session_state.verification_state:
        st.info("No results yet. Go to **Upload & Verify** first.")
        return

    state = st.session_state.verification_state
    findings = state.get("all_findings", [])

    if not findings:
        st.success("No findings — document passed all checks.")
        _approve_gate(state)
        return

    sev_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}
    sorted_f = sorted(findings, key=lambda f: sev_order.get(f.get("severity", "INFO"), 3))

    # ── Progress ──────────────────────────────────────────────────────────────
    total = len(sorted_f)
    done = sum(1 for f in sorted_f
               if st.session_state.finding_statuses.get(f.get("id"), "OPEN") != "OPEN")
    critical_open = [f for f in sorted_f
                     if f.get("severity") == "CRITICAL"
                     and st.session_state.finding_statuses.get(f.get("id"), "OPEN") == "OPEN"]

    st.progress(done / total if total else 1.0,
                text=f"{done} of {total} findings reviewed  ·  {len(critical_open)} critical still open")

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    f_sev   = fc1.multiselect("Severity", ["CRITICAL","HIGH","WARNING","INFO"],
                               default=["CRITICAL","HIGH","WARNING","INFO"])
    f_agent = fc2.multiselect("Agent", ["completeness","consistency","ambiguity"],
                               default=["completeness","consistency","ambiguity"])
    f_stat  = fc3.multiselect("Status", ["OPEN","ACCEPTED","REJECTED"],
                               default=["OPEN","ACCEPTED","REJECTED"])

    visible = [f for f in sorted_f
               if f.get("severity") in f_sev
               and f.get("agent") in f_agent
               and st.session_state.finding_statuses.get(f.get("id"), "OPEN") in f_stat]

    st.caption(f"Showing {len(visible)} of {total} findings")
    st.divider()

    for f in visible:
        _finding_card(f)

    st.divider()
    _approve_gate(state, critical_open)


def _finding_card(f: dict):
    fid   = f.get("id", "")
    sev   = f.get("severity", "INFO")
    pg    = f.get("page_number", 0)
    cur   = st.session_state.finding_statuses.get(fid, "OPEN")
    border = _SEV_BORDER.get(sev, "#64748b")
    status_icons = {"OPEN": "⬜", "ACCEPTED": "✅", "REJECTED": "❌"}

    header = (
        f'{status_icons.get(cur,"⬜")} '
        f'<span class="badge badge-{sev}">{sev}</span> &nbsp; '
        f'{"📄 Page " + str(pg) if pg else "📄 page ?"} &nbsp;·&nbsp; '
        f'{f.get("description","")[:80]}'
    )
    with st.expander(f"{status_icons.get(cur,'⬜')} [{sev}]  {f.get('description','')[:80]}",
                     expanded=(sev == "CRITICAL" and cur == "OPEN")):

        # Location strip
        loc1, loc2, loc3 = st.columns([1, 1, 2])
        with loc1:
            pg_color = border
            st.markdown(
                f'<div style="background:{pg_color};color:#fff;padding:8px 14px;'
                f'border-radius:8px;font-weight:700;font-size:1rem;display:inline-block">'
                f'📄 {"Page " + str(pg) if pg else "Page unknown"}</div>',
                unsafe_allow_html=True,
            )
        with loc2:
            st.markdown(f"**Agent:** {f.get('agent','').title()}")
            if f.get("section_title"):
                st.markdown(f"**Section:** {f.get('section_title')}")
        with loc3:
            st.markdown(f"**Type:** `{f.get('finding_type','')}`")
            st.markdown(f"**Source:** {f.get('evidence_source','—')}")

        st.markdown("---")

        # What is wrong
        st.markdown("#### 🔎 What is the problem?")
        st.markdown(f.get("description", "—"))

        # Evidence
        if f.get("quote"):
            st.markdown("#### 📋 Exact text from document")
            st.code(f.get("quote"), language=None)

        # Two-column: rule + why
        r1, r2 = st.columns(2)
        with r1:
            if f.get("rule_violated") or f.get("rule_text"):
                st.markdown("#### 📖 Rule violated")
                if f.get("rule_violated"):
                    st.markdown(f"**{f['rule_violated']}**")
                if f.get("rule_text"):
                    st.info(f.get("rule_text"))
        with r2:
            if f.get("why_it_matters"):
                st.markdown("#### ⚠️ Why it matters for PLC coding")
                st.warning(f.get("why_it_matters"))

        # Fix
        st.markdown("#### ✅ Suggested fix")
        st.success(f.get("suggestion", "No suggestion available."))

        # Decision
        st.markdown("---")
        d1, d2 = st.columns([1, 2])
        with d1:
            new_status = st.radio(
                "Engineer decision",
                ["OPEN", "ACCEPTED", "REJECTED"],
                index=["OPEN","ACCEPTED","REJECTED"].index(cur),
                key=f"r_{fid}",
                horizontal=True,
            )
            st.session_state.finding_statuses[fid] = new_status
        with d2:
            note = st.text_input(
                "Note (optional)",
                value=st.session_state.finding_notes.get(fid, ""),
                key=f"n_{fid}",
                placeholder="e.g. Confirmed with P&ID rev 3 / not applicable for test document",
            )
            st.session_state.finding_notes[fid] = note


def _approve_gate(state, critical_open=None):
    if critical_open is None:
        critical_open = []

    routing = state.get("vqs_scores", {}).get("routing_decision", "")
    st.markdown("### Approval Gate")

    if routing == "MANDATORY_REVISION":
        missing = state.get("missing_critical_sections", [])
        detail = f"Missing: {', '.join(missing)}" if missing else "Circular interlock detected"
        st.error(f"🚫 Cannot approve — **MANDATORY REVISION** ({detail}). Fix in source document and re-upload.")
        return

    if critical_open:
        st.error(f"🔴 Cannot approve — **{len(critical_open)} critical finding(s)** still OPEN. "
                 "Review all critical findings above first.")
        return

    st.success("All critical findings reviewed. Document ready for approval.")
    st.markdown(
        "By approving you confirm: findings reviewed, rejected items accepted as risk, "
        "document is suitable for PLC code generation."
    )
    if st.button("✅  Approve — Save Verification Report", type="primary"):
        _save_report(state)
        st.success("Report saved to `output/verification_report.json`. PLC code generation may proceed.")
        st.balloons()


def _save_report(state):
    os.makedirs("output", exist_ok=True)
    report = {
        "document_path": state.get("document_path", ""),
        "vqs_scores": state.get("vqs_scores", {}),
        "tag_inventory": state.get("tag_inventory", []),
        "total_findings": len(state.get("all_findings", [])),
        "findings": [
            {
                **f,
                "engineer_status": st.session_state.finding_statuses.get(f.get("id",""), "OPEN"),
                "engineer_note":   st.session_state.finding_notes.get(f.get("id",""), ""),
            }
            for f in state.get("all_findings", [])
        ],
        "approved": True,
    }
    with open("output/verification_report.json", "w") as fp:
        json.dump(report, fp, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_required_sections() -> list:
    try:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("required_sections", [])
    except Exception:
        return ["General Description", "Operating Modes", "Startup Sequence",
                "Shutdown Sequence", "Emergency Shutdown", "Interlocks", "Alarms", "Setpoints"]


def _fuzzy(a: str, b: str, threshold: float = 0.60) -> bool:
    from difflib import SequenceMatcher
    if b.lower() in a.lower():
        return True
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


# ── Router ────────────────────────────────────────────────────────────────────

if page == "Upload & Verify":
    page_upload()
elif page == "Score Dashboard":
    page_dashboard()
elif page == "Review Findings":
    page_review()
