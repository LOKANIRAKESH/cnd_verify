# CND-Verify — Setup Guide

## What This System Does

CND-Verify checks a Control Narrative Document for three types of problems before PLC code is written:
1. **Completeness** — missing sections, missing interlock fields, missing alarm fields
2. **Consistency** — setpoint values that contradict each other, tag type conflicts
3. **Ambiguity** — vague language that cannot be converted to PLC code

---

## One-Time Setup (Run These Once)

### Step 1 — Install Ollama

Download from: https://ollama.com/download

After installing, open Command Prompt and run:
```
ollama pull llama3.1
```
This downloads the Llama 3.1 8B model (~4.7 GB). Wait for it to finish.

Verify:
```
ollama list
```
You should see `llama3.1` in the list.

### Step 2 — Install Python Packages

Open Command Prompt in the `cnd-verify` folder and run:
```
pip install -r requirements.txt
```

### Step 3 — Load Domain Knowledge into ChromaDB

Run this once to populate the vector database:
```
python populate_kb.py
```

Expected output:
```
Loaded 45 chunks from ISA-18.2-Alarms
Loaded 38 chunks from ISA-88-Modes
Loaded 52 chunks from ISA-5.1-Tags
Loaded 67 chunks from Control-Glossary
Done. Total chunks in domain knowledge base: 202
```

---

## Running the Dashboard

```
streamlit run src/ui/app.py
```

The dashboard will open in your browser at: http://localhost:8501

---

## Testing with Sample Documents

Three sample documents are in `data/synthetic_cnds/`:

| File | Description | Expected Result |
|---|---|---|
| `pump_station_01_clean.txt` | Clean document, no errors | HIGH score, AUTO PASS |
| `pump_station_02_completeness_errors.txt` | 6 completeness errors | LOW score, FAIL or NEEDS_REVIEW |
| `pump_station_03_consistency_errors.txt` | 4 consistency errors | MEDIUM score, NEEDS_REVIEW |

To test: Go to Upload & Verify page → select a sample document → click Run Verification.

---

## Project Structure

```
cnd-verify/
├── data/
│   ├── synthetic_cnds/         ← Test documents
│   ├── ground_truth/           ← Known error labels for evaluation
│   └── knowledge_base/         ← ISA standard content (text files)
├── src/
│   ├── orchestration/state.py  ← Shared data contract
│   ├── ingestion/document_loader.py  ← PDF/TXT parser
│   ├── agents/completeness_agent.py  ← Completeness checker
│   ├── scoring/score_engine.py       ← VQS scoring
│   └── ui/app.py               ← Streamlit dashboard
├── chroma_db/                  ← Created automatically by populate_kb.py
├── populate_kb.py              ← Run once to load domain knowledge
├── requirements.txt
└── config.yaml                 ← All settings
```

---

## Troubleshooting

**Error: "ollama connection refused"**
- Make sure Ollama is running. Open Ollama from Start menu, or run `ollama serve` in a terminal.

**Error: "collection not found"**
- Run `python populate_kb.py` first.

**Error: "module not found"**
- Make sure you are running commands from the `cnd-verify` directory (not a subdirectory).
- Run `pip install -r requirements.txt` again.

**Streamlit shows blank page**
- Refresh the browser. If still blank, stop Streamlit (Ctrl+C) and run again.

---

## Full Pipeline (Session 2 — Complete)

All three agents and the full orchestration are now in place:

| File | What it does |
|------|-------------|
| `src/agents/completeness_agent.py` | Checks sections, interlock fields, alarm fields, tag definitions |
| `src/agents/consistency_agent.py` | Checks setpoint prose vs. table, equipment type contradictions, sequence order |
| `src/agents/ambiguity_agent.py` | Detects vague language (QUANTIFIER, TEMPORAL, REFERENTIAL, MODAL, CONDITIONAL, PASSIVE) |
| `src/orchestration/graph.py` | LangGraph StateGraph wiring all agents sequentially |
| `src/ui/app.py` | Full dashboard — step progress, detailed finding cards with page number, rule violated, why it matters, fix |

Every finding card now shows:
- **Page number** where the error is located
- **Exact quoted text** from the document
- **Rule violated** (short name + full rule text from ISA standard)
- **Why it matters** for PLC programming
- **Specific suggested fix**
