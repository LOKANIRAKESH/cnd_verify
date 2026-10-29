"""
src/ui/app.py
Streamlit dashboard for CND-Verify — full three-agent pipeline with detailed finding cards.
Each finding card shows: page number, section, rule violated, why it matters, quoted text, fix.
"""

import streamlit as st
import plotly.graph_objects as go
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.document_loader import prepare_document
from src.orchestration.graph import compile_graph
from src.scoring.score_engine import get_routing_message

st.set_page_config(
    page_title="CND-Verify",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("CND-Verify")
st.sidebar.caption("Control Narrative Pre-Verification System")
st.sidebar.markdown("*Detects errors before PLC code generation*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Upload & Verify", "Score Dashboard", "Review Findings"],
    index=0,
)

if st.session_state.get("verification_state"):
    vqs = st.session_state.verification_state.get("vqs_scores", {})
    composite = vqs.get("composite", 0)
    routing = vqs.get("routing_decision", "")
    icon = {"AUTO_PASS": "✅", "NEEDS_REVIEW": "⚠️", "FAIL": "❌", "MANDATORY_REVISION": "🚫"}.get(routing, "")
    st.sidebar.divider()
    st.sidebar.metric("Current Score", f"{composite*100:.0f} / 100")
    st.sidebar.caption(f"{icon} {routing.replace('_', ' ')}")

st.sidebar.divider()
st.sidebar.caption("M.Tech Research Project")
st.sidebar.caption("Mentor: Padmaja Bodanapu, GE Vernova")

# ── Session state ─────────────────────────────────────────────────────────────

defaults = {
    "verification_state": None,
    "finding_statuses": {},
    "finding_notes": {},
    "approved": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — UPLOAD AND VERIFY
# ─────────────────────────────────────────────────────────────────────────────

def page_upload():
    st.title("Upload Control Narrative Document")
    st.markdown(
        "Upload a CND as a **PDF** or **plain text** file. "
        "Three AI agents will check it for **completeness**, **consistency**, and **ambiguity** "
        "before any PLC code is generated."
    )

    col_upload, col_sample = st.columns(2)
    with col_upload:
        st.subheader("Upload your file")
        uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    with col_sample:
        st.subheader("Or use a sample document")
        sample_options = {
            "— Select —": None,
            "Pump Station 01 — Clean (no errors)": "data/synthetic_cnds/pump_station_01_clean.txt",
            "Pump Station 02 — Completeness errors (6 injected)": "data/synthetic_cnds/pump_station_02_completeness_errors.txt",
            "Pump Station 03 — Consistency errors (4 injected)": "data/synthetic_cnds/pump_station_03_consistency_errors.txt",
        }
        selected_sample = st.selectbox("Sample document", list(sample_options.keys()))

    st.divider()

    # What the agents check
    with st.expander("What does each agent check?", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Completeness Agent**")
            st.markdown(
                "- Required sections present (ISA-88)\n"
                "- Interlock fields complete (ISA-84)\n"
                "- Alarm fields complete (ISA-18.2)\n"
                "- All referenced tags defined"
            )
        with c2:
            st.markdown("**Consistency Agent**")
            st.markdown(
                "- Setpoint values match prose vs. table\n"
                "- Same tag described consistently\n"
                "- Sequence steps don't contradict\n"
                "- Units match across document"
            )
        with c3:
            st.markdown("**Ambiguity Agent**")
            st.markdown(
                "- No vague quantities (no number)\n"
                "- No vague timing (no duration)\n"
                "- No unresolved references ('it')\n"
                "- No incomplete conditions"
            )

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_button = st.button("Run Verification", type="primary", use_container_width=True)

    if run_button:
        file_path = _resolve_file_path(uploaded_file, sample_options.get(selected_sample))
        if not file_path:
            st.warning("Please upload a file or select a sample document.")
            return

        # Run pipeline
        progress_bar = st.progress(0, text="Starting verification pipeline...")

        with st.spinner("Step 1/5 — Parsing document and extracting structure..."):
            state = prepare_document(file_path)
        progress_bar.progress(20, text="Step 2/5 — Running Completeness Agent...")

        with st.spinner("Step 2/5 — Running Completeness Agent (checking sections, interlocks, alarms)..."):
            from src.agents.completeness_agent import run_completeness_agent
            state = run_completeness_agent(state)
        progress_bar.progress(40, text="Step 3/5 — Running Consistency Agent...")

        with st.spinner("Step 3/5 — Running Consistency Agent (checking setpoints, equipment types)..."):
            from src.agents.consistency_agent import run_consistency_agent
            state = run_consistency_agent(state)
        progress_bar.progress(60, text="Step 4/5 — Running Ambiguity Agent...")

        with st.spinner("Step 4/5 — Running Ambiguity Agent (detecting vague language)..."):
            from src.agents.ambiguity_agent import run_ambiguity_agent
            state = run_ambiguity_agent(state)
        progress_bar.progress(80, text="Step 5/5 — Computing Verification Quality Score...")

        with st.spinner("Step 5/5 — Computing Verification Quality Score..."):
            from src.scoring.score_engine import run_score_engine
            state = run_score_engine(state)
        progress_bar.progress(100, text="Verification complete.")

        st.session_state.verification_state = state
        st.session_state.finding_statuses = {
            f["id"]: "OPEN" for f in state.get("all_findings", [])
        }
        st.session_state.finding_notes = {}
        st.session_state.approved = False

        findings = state.get("all_findings", [])
        critical = sum(1 for f in findings if f.get("severity") == "CRITICAL")
        routing = state.get("vqs_scores", {}).get("routing_decision", "")

        if routing == "AUTO_PASS":
            st.success(f"Verification complete — **AUTO PASS**. {len(findings)} finding(s), none critical.")
        elif routing == "NEEDS_REVIEW":
            st.warning(f"Verification complete — **{len(findings)} findings** require engineer review ({critical} critical).")
        else:
            st.error(f"Verification complete — **{routing}**. {len(findings)} findings ({critical} critical). Document requires significant revision.")

        st.info("Go to **Score Dashboard** to see the results.")


def _resolve_file_path(uploaded_file, sample_path: str | None) -> str | None:
    if uploaded_file:
        temp_path = f"temp_upload_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    if sample_path and os.path.exists(sample_path):
        return sample_path
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — SCORE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def page_dashboard():
    st.title("Verification Quality Score Dashboard")

    if not st.session_state.verification_state:
        st.info("No verification results yet. Go to **Upload & Verify** first.")
        return

    state = st.session_state.verification_state
    vqs = state.get("vqs_scores", {})
    findings = state.get("all_findings", [])

    composite = vqs.get("composite", 0.0)
    routing = vqs.get("routing_decision", "PENDING")
    routing_info = get_routing_message(routing, composite)

    # ── Score strip ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Overall VQS", f"{composite*100:.1f}%")
    with m2:
        st.metric("Completeness", f"{vqs.get('completeness', 1)*100:.1f}%")
    with m3:
        st.metric("Consistency", f"{vqs.get('consistency', 1)*100:.1f}%")
    with m4:
        st.metric("Ambiguity", f"{vqs.get('ambiguity', 1)*100:.1f}%")
    with m5:
        critical_count = sum(1 for f in findings if f.get("severity") == "CRITICAL")
        st.metric("Critical Issues", critical_count)

    st.divider()

    # ── Routing decision banner ───────────────────────────────────────────────
    color = routing_info["color"]
    msg = f"{routing_info['icon']} **{routing.replace('_', ' ')}** — {routing_info['message']}"
    if color == "green":
        st.success(msg)
    elif color == "orange":
        st.warning(msg)
    else:
        st.error(msg)

    # ── Charts row ────────────────────────────────────────────────────────────
    chart_col, radar_col = st.columns(2)
    with chart_col:
        _gauge_chart(composite * 100, "Overall VQS Score")
    with radar_col:
        _radar_chart(vqs)

    # ── Findings breakdown ────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Findings Overview — {len(findings)} total")

    if not findings:
        st.success("No findings — document appears complete, consistent, and unambiguous.")
        return

    # Group by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}
    sorted_findings = sorted(findings, key=lambda f: severity_order.get(f.get("severity", "INFO"), 3))

    # Summary counts by agent
    agents = ["completeness", "consistency", "ambiguity"]
    agent_counts = {a: sum(1 for f in findings if f.get("agent") == a) for a in agents}
    sev_counts = {s: sum(1 for f in findings if f.get("severity") == s)
                  for s in ["CRITICAL", "HIGH", "WARNING", "INFO"]}

    tab_overview, tab_by_agent = st.tabs(["By Severity", "By Agent"])

    with tab_overview:
        for sev, count in sev_counts.items():
            if count == 0:
                continue
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "WARNING": "🟡", "INFO": "🔵"}[sev]
            with st.expander(f"{icon} {sev} — {count} finding(s)", expanded=(sev == "CRITICAL")):
                sev_findings = [f for f in sorted_findings if f.get("severity") == sev]
                for f in sev_findings:
                    _compact_finding_card(f)

    with tab_by_agent:
        for agent_name in agents:
            agent_findings = [f for f in sorted_findings if f.get("agent") == agent_name]
            if not agent_findings:
                continue
            with st.expander(f"**{agent_name.title()} Agent** — {len(agent_findings)} finding(s)"):
                for f in agent_findings:
                    _compact_finding_card(f)


def _compact_finding_card(f: dict):
    """Compact card for the dashboard overview."""
    sev = f.get("severity", "INFO")
    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "WARNING": "🟡", "INFO": "🔵"}.get(sev, "⚪")
    pg = f.get("page_number", 0)
    page_badge = f"**Page {pg}**" if pg else "*page unknown*"

    st.markdown(
        f"{icon} {page_badge} &nbsp;·&nbsp; `{f.get('finding_type', '')}` &nbsp;·&nbsp; "
        f"{f.get('description', '')}"
    )
    if f.get("quote"):
        st.caption(f"› {f['quote'][:120]}...")
    st.markdown("---")


def _gauge_chart(value: float, title: str):
    color = "green" if value >= 85 else ("orange" if value >= 65 else "red")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 14}},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 65], "color": "#ffcccc"},
                {"range": [65, 85], "color": "#fff3cc"},
                {"range": [85, 100], "color": "#ccffcc"},
            ],
        },
    ))
    fig.update_layout(height=270, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


def _radar_chart(vqs: dict):
    categories = ["Completeness", "Consistency", "Ambiguity"]
    values = [
        vqs.get("completeness", 1.0) * 100,
        vqs.get("consistency", 1.0) * 100,
        vqs.get("ambiguity", 1.0) * 100,
    ]
    v_closed = values + [values[0]]
    c_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=v_closed, theta=c_closed, fill="toself",
        fillcolor="rgba(0, 100, 200, 0.2)",
        line_color="rgba(0, 100, 200, 0.8)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, title="Score by Dimension",
        height=270, margin=dict(t=50, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — REVIEW FINDINGS (detailed cards with full evidence)
# ─────────────────────────────────────────────────────────────────────────────

_SEV_COLOR = {
    "CRITICAL": "#dc2626",   # red
    "HIGH":     "#ea580c",   # orange
    "WARNING":  "#ca8a04",   # amber
    "INFO":     "#2563eb",   # blue
}
_SEV_BG = {
    "CRITICAL": "#fef2f2",
    "HIGH":     "#fff7ed",
    "WARNING":  "#fefce8",
    "INFO":     "#eff6ff",
}


def page_review():
    st.title("Review Findings")
    st.markdown(
        "Each card shows the exact location, the standard violated, "
        "why it matters for PLC coding, and a specific suggested fix. "
        "Mark each finding **Accepted** or **Rejected** before approving the document."
    )

    if not st.session_state.verification_state:
        st.info("No verification results yet. Go to **Upload & Verify** first.")
        return

    state = st.session_state.verification_state
    findings = state.get("all_findings", [])

    if not findings:
        st.success("No findings — document passed all checks. You can proceed with PLC code generation.")
        _show_approve_button(state)
        return

    severity_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}
    sorted_findings = sorted(findings, key=lambda f: severity_order.get(f.get("severity", "INFO"), 3))

    # Progress bar
    total = len(sorted_findings)
    actioned = sum(
        1 for f in sorted_findings
        if st.session_state.finding_statuses.get(f.get("id"), "OPEN") != "OPEN"
    )
    critical_open = [
        f for f in sorted_findings
        if f.get("severity") == "CRITICAL"
        and st.session_state.finding_statuses.get(f.get("id"), "OPEN") == "OPEN"
    ]

    st.progress(actioned / total, text=f"{actioned} of {total} findings reviewed")
    if critical_open:
        st.warning(f"{len(critical_open)} critical finding(s) still unreviewed — required before approval")
    st.divider()

    # Filter controls
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        filter_sev = st.multiselect(
            "Filter by severity", ["CRITICAL", "HIGH", "WARNING", "INFO"],
            default=["CRITICAL", "HIGH", "WARNING", "INFO"]
        )
    with filter_col2:
        filter_agent = st.multiselect(
            "Filter by agent", ["completeness", "consistency", "ambiguity"],
            default=["completeness", "consistency", "ambiguity"]
        )
    with filter_col3:
        filter_status = st.multiselect(
            "Filter by status", ["OPEN", "ACCEPTED", "REJECTED"],
            default=["OPEN", "ACCEPTED", "REJECTED"]
        )

    visible_findings = [
        f for f in sorted_findings
        if f.get("severity") in filter_sev
        and f.get("agent") in filter_agent
        and st.session_state.finding_statuses.get(f.get("id"), "OPEN") in filter_status
    ]

    st.caption(f"Showing {len(visible_findings)} of {total} findings")
    st.divider()

    for f in visible_findings:
        _render_finding_card(f)

    st.divider()
    _show_approve_button(state, critical_open=critical_open)


def _render_finding_card(f: dict):
    fid = f.get("id", "")
    sev = f.get("severity", "INFO")
    current_status = st.session_state.finding_statuses.get(fid, "OPEN")

    sev_color = _SEV_COLOR.get(sev, "#6b7280")
    status_icon = {"OPEN": "⬜", "ACCEPTED": "✅", "REJECTED": "❌"}.get(current_status, "⬜")
    page_num = f.get("page_number", 0)
    page_display = f"Page {page_num}" if page_num else "Page unknown"

    # Card header label
    short_desc = f.get("description", "")
    if len(short_desc) > 90:
        short_desc = short_desc[:87] + "..."

    header = f"{status_icon} [{sev}] {short_desc}"

    with st.expander(header, expanded=(sev == "CRITICAL" and current_status == "OPEN")):

        # ── Location strip ────────────────────────────────────────────────────
        loc_col, agent_col, type_col = st.columns([1, 1, 2])
        with loc_col:
            st.markdown(
                f"<div style='background:{sev_color};color:white;padding:6px 12px;"
                f"border-radius:6px;font-weight:bold;font-size:1.1em;display:inline-block'>"
                f"📄 {page_display}</div>",
                unsafe_allow_html=True,
            )
        with agent_col:
            st.markdown(f"**Agent:** {f.get('agent', '').title()}")
        with type_col:
            st.markdown(f"**Type:** `{f.get('finding_type', '')}`")
            if f.get("section_title"):
                st.markdown(f"**Section:** {f.get('section_title')}")

        st.markdown("")

        # ── WHAT is wrong ─────────────────────────────────────────────────────
        st.markdown("#### What is the problem?")
        st.markdown(f.get("description", "No description available."))

        # ── EVIDENCE — exact quote from document ─────────────────────────────
        if f.get("quote"):
            st.markdown("#### Evidence — exact text from document")
            st.code(f.get("quote"), language=None)

        # ── BASIS — rule violated ─────────────────────────────────────────────
        if f.get("rule_violated") or f.get("rule_text"):
            st.markdown("#### What rule does this violate?")
            if f.get("rule_violated"):
                st.markdown(f"**{f['rule_violated']}**")
            if f.get("rule_text"):
                st.info(f.get("rule_text"))

        # ── WHY IT MATTERS ────────────────────────────────────────────────────
        if f.get("why_it_matters"):
            st.markdown("#### Why does this matter for PLC programming?")
            st.warning(f.get("why_it_matters"))

        # ── HOW TO FIX ────────────────────────────────────────────────────────
        st.markdown("#### Suggested fix")
        st.success(f.get("suggestion", "No suggestion available."))

        if f.get("evidence_source"):
            st.caption(f"Source: {f.get('evidence_source')}")

        # ── Engineer decision ─────────────────────────────────────────────────
        st.markdown("---")
        dec_col1, dec_col2 = st.columns([1, 2])
        with dec_col1:
            st.markdown("**Engineer Decision:**")
            new_status = st.radio(
                f"Decision for {fid}",
                ["OPEN", "ACCEPTED", "REJECTED"],
                index=["OPEN", "ACCEPTED", "REJECTED"].index(current_status),
                key=f"radio_{fid}",
                label_visibility="collapsed",
                horizontal=True,
            )
            st.session_state.finding_statuses[fid] = new_status
        with dec_col2:
            note = st.text_input(
                "Optional note (reason for decision)",
                value=st.session_state.finding_notes.get(fid, ""),
                key=f"note_{fid}",
                placeholder="e.g. 'Confirmed with P&ID revision 3' or 'Not applicable — test CND'",
                label_visibility="visible",
            )
            st.session_state.finding_notes[fid] = note


def _show_approve_button(state: dict, critical_open: list = None):
    if critical_open is None:
        critical_open = []

    vqs = state.get("vqs_scores", {})
    routing = vqs.get("routing_decision", "PENDING")

    st.subheader("Approval Gate")

    if routing == "MANDATORY_REVISION":
        st.error(
            "Cannot approve — document has a **MANDATORY REVISION** flag. "
            "This is caused by: circular interlocks or missing critical sections (Interlocks, Alarms, ESD). "
            "These must be corrected in the source document before re-verification."
        )
        return

    if critical_open:
        st.error(
            f"Cannot approve — **{len(critical_open)} CRITICAL finding(s)** are still marked OPEN. "
            "Review each critical finding above and mark it as ACCEPTED or REJECTED."
        )
        st.markdown("Outstanding critical findings:")
        for f in critical_open:
            pg = f.get("page_number", 0)
            st.markdown(f"- `{f.get('id')}` — Page {pg} — {f.get('description', '')[:80]}")
        return

    st.success("All critical findings reviewed. Document is ready for approval.")
    st.markdown(
        "By clicking **Approve**, you confirm that:\n"
        "- You have reviewed all flagged findings\n"
        "- You accept responsibility for any rejected findings\n"
        "- The document is suitable for PLC code generation"
    )

    if st.button("Approve Document — Save Report", type="primary"):
        _save_approval_report(state)
        st.session_state.approved = True
        st.success(
            "Document approved. Report saved to `output/verification_report.json`. "
            "PLC code generation can now proceed."
        )
        st.balloons()


def _save_approval_report(state: dict):
    os.makedirs("output", exist_ok=True)
    report = {
        "document_path": state.get("document_path", ""),
        "vqs_scores": state.get("vqs_scores", {}),
        "routing_decision": state.get("vqs_scores", {}).get("routing_decision", ""),
        "total_findings": len(state.get("all_findings", [])),
        "findings_with_status": [
            {
                **f,
                "engineer_status": st.session_state.finding_statuses.get(f.get("id", ""), "OPEN"),
                "engineer_note": st.session_state.finding_notes.get(f.get("id", ""), ""),
            }
            for f in state.get("all_findings", [])
        ],
        "approved": True,
    }
    with open("output/verification_report.json", "w") as fp:
        json.dump(report, fp, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

if page == "Upload & Verify":
    page_upload()
elif page == "Score Dashboard":
    page_dashboard()
elif page == "Review Findings":
    page_review()
