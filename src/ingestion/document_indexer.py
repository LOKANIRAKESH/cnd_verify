"""
src/ingestion/document_indexer.py

PASS 1 — Global Knowledge Store builder.

Indexes every sentence of the CND into a ChromaDB collection so agents
can query cross-chunk context during verification (Pass 2).

Builds three fast-lookup indexes stored in state:
  tag_index     — {tag: [{chunk, page, text, section}]}
  setpoint_index— {tag: {parameter: value, unit, page}}
  definition_map— {tag: {type, description, page}}

These let any agent ask "what does the full document say about PT-101?"
without scanning the full text every time.
"""

import re
import uuid
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger("cnd_verify")

ISA_TAG_RE = re.compile(r"\b([A-Z]{1,3}-[0-9]{1,4}[A-Z]?)\b")
NUM_VALUE_RE = re.compile(r"([\d]+(?:\.[0-9]+)?)\s*(PSI|BAR|°C|°F|C|F|L/MIN|M3/H|GPM|RPM|KW|V|A|HZ|MM|M|%|BARA|BARG|PSIA|PSIG)?", re.IGNORECASE)

# Equipment type keywords → equipment class
EQUIP_TYPE_PATTERNS = {
    "centrifugal pump":   "PUMP",   "pump":         "PUMP",
    "ball valve":         "VALVE",  "gate valve":    "VALVE",
    "butterfly valve":    "VALVE",  "globe valve":   "VALVE",
    "control valve":      "VALVE",  "valve":         "VALVE",
    "solenoid valve":     "VALVE",
    "pressure transmitter": "PT",   "pressure sensor": "PT",
    "flow transmitter":   "FT",     "flow meter":     "FT",
    "level transmitter":  "LT",     "level sensor":   "LT",
    "temperature transmitter": "TT","thermocouple":  "TT",
    "motor":              "MOTOR",  "compressor":    "COMPRESSOR",
    "heat exchanger":     "HX",     "vessel":        "VESSEL",
    "tank":               "VESSEL",
}


def _split_into_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, keeping [PAGE N] context."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _current_page(text_before: str) -> int:
    """Find the last [PAGE N] marker before a given position."""
    pages = re.findall(r"\[PAGE (\d+)\]", text_before)
    return int(pages[-1]) if pages else 1


def build_document_index(state: dict, config: dict) -> dict:
    """
    Pass 1: Index all CND sentences into a per-document ChromaDB collection.

    Returns enriched state with:
      - doc_collection_name  : ChromaDB collection name
      - tag_index            : {tag: [{page, section, text}]}
      - setpoint_index       : {tag: {param: value, unit, page}}
      - definition_map       : {tag: {type, class, description, page}}
      - chunk_groups         : [{section_titles, text, pages}] for parallel Pass 2
    """
    document_text: str = state.get("document_text", "")
    sections: dict      = state.get("sections", {})
    sections_pages: dict = state.get("sections_pages", {})
    doc_path: str        = state.get("document_path", "unknown")

    # Stable collection name based on doc path
    doc_hash = str(abs(hash(doc_path)))[:12]
    collection_name = f"cnd_doc_{doc_hash}"

    # ── Connect ChromaDB ──────────────────────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=config["chromadb"]["path"])
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=config["embeddings"]["model"]
    )

    # Delete old collection for this doc (fresh index each run)
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=collection_name, embedding_function=embed_fn
    )

    # ── Index sentences ───────────────────────────────────────────────────────
    tag_index: dict[str, list[dict]] = {}
    setpoint_index: dict[str, dict]  = {}
    definition_map: dict[str, dict]  = {}

    batch_ids, batch_docs, batch_metas = [], [], []
    BATCH_SIZE = 100

    for section_title, section_text in sections.items():
        page = sections_pages.get(section_title, 1)
        sentences = _split_into_sentences(section_text)

        for sent in sentences:
            # Find ISA tags in this sentence
            tags_in_sent = ISA_TAG_RE.findall(sent)

            # Build tag_index entries
            for tag in tags_in_sent:
                if tag not in tag_index:
                    tag_index[tag] = []
                tag_index[tag].append({
                    "page":    page,
                    "section": section_title,
                    "text":    sent[:300],
                })

            # Detect equipment type definitions: "XV-201 is a ball valve"
            defn_match = re.search(
                r"\b([A-Z]{1,3}-\d{1,4}[A-Z]?)\b[^.]{0,30}\bis\s+a\s+(.{5,50}?)(?:[,.]|$)",
                sent, re.IGNORECASE
            )
            if defn_match:
                tag_d = defn_match.group(1).upper()
                desc  = defn_match.group(2).strip().lower()
                equip_class = next(
                    (v for k, v in EQUIP_TYPE_PATTERNS.items() if k in desc), "UNKNOWN"
                )
                definition_map[tag_d] = {
                    "type":        desc,
                    "class":       equip_class,
                    "description": sent[:200],
                    "page":        page,
                    "section":     section_title,
                }

            # Detect numerical setpoints: "PT-101 … 150 PSI"
            for tag in tags_in_sent:
                num_m = NUM_VALUE_RE.search(sent[sent.find(tag):])
                if num_m and num_m.group(1):
                    try:
                        val   = float(num_m.group(1))
                        unit  = (num_m.group(2) or "").upper()
                        param = _infer_parameter(sent, tag)
                        if tag not in setpoint_index:
                            setpoint_index[tag] = {}
                        setpoint_index[tag][param] = {
                            "value": val, "unit": unit,
                            "page": page, "text": sent[:200],
                        }
                    except ValueError:
                        pass

            # Queue for ChromaDB batch insert
            batch_ids.append(f"sent_{uuid.uuid4().hex[:10]}")
            batch_docs.append(sent[:500])
            batch_metas.append({
                "section": section_title[:60],
                "page":    page,
                "tags":    ",".join(tags_in_sent[:8]),
            })

            if len(batch_ids) >= BATCH_SIZE:
                _flush_batch(collection, batch_ids, batch_docs, batch_metas)
                batch_ids, batch_docs, batch_metas = [], [], []

    if batch_ids:
        _flush_batch(collection, batch_ids, batch_docs, batch_metas)

    logger.info(
        "Document index built: %d sentences, %d tags, %d setpoints, %d definitions",
        len(batch_ids), len(tag_index), len(setpoint_index), len(definition_map),
    )

    # ── Build section chunk groups for parallel Pass 2 ────────────────────────
    chunk_groups = _build_chunk_groups(sections, sections_pages)

    state["doc_collection_name"] = collection_name
    state["tag_index"]           = tag_index
    state["setpoint_index"]      = setpoint_index
    state["definition_map"]      = definition_map
    state["chunk_groups"]        = chunk_groups
    return state


def query_doc_index(collection_name: str, query: str, config: dict, n: int = 3) -> list[dict]:
    """
    Query the document ChromaDB index.
    Returns list of {text, page, section, score} for the top-n results.
    """
    try:
        chroma_client = chromadb.PersistentClient(path=config["chromadb"]["path"])
        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=config["embeddings"]["model"]
        )
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embed_fn
        )
        results = collection.query(query_texts=[query], n_results=n)
        out = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            out.append({
                "text":    doc,
                "page":    meta.get("page", 0),
                "section": meta.get("section", ""),
                "score":   round(1.0 - dist, 3),
            })
        return out
    except Exception as e:
        logger.warning("doc index query failed: %s", e)
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flush_batch(collection, ids, docs, metas):
    try:
        collection.add(ids=ids, documents=docs, metadatas=metas)
    except Exception as e:
        logger.warning("ChromaDB batch insert warning: %s", e)


def _infer_parameter(sentence: str, tag: str) -> str:
    """Guess the parameter type from surrounding words."""
    s = sentence.lower()
    if any(w in s for w in ("trip", "shutdown", "esd")):                    return "trip"
    if any(w in s for w in ("high high", "hh")):                            return "hi_hi"
    if any(w in s for w in ("exceed", "above", "greater than",
                            "high alarm", "hi alarm", "high")):              return "hi"
    if any(w in s for w in ("low low", "ll")):                              return "lo_lo"
    if any(w in s for w in ("below", "falls below", "drops below",
                            "less than", "low alarm", "lo alarm", "low")):  return "lo"
    if any(w in s for w in ("setpoint", "set point", "sp")):                return "sp"
    return "value"


def _build_chunk_groups(sections: dict, sections_pages: dict) -> list[dict]:
    """
    Group sections into 3-5 balanced chunk groups for parallel processing.
    Sections that reference similar content stay together.
    """
    CHUNK_KEYWORDS = [
        ["GENERAL", "MODE", "OVERVIEW", "PURPOSE", "SCOPE"],
        ["STARTUP", "START-UP", "SHUTDOWN", "SHUT-DOWN", "SEQUENCE", "OPERATION"],
        ["INTERLOCK", "SAFETY", "SIS", "ESD", "EMERGENCY"],
        ["ALARM", "SETPOINT", "INSTRUMENT", "TAG"],
        ["EQUIPMENT", "REVISION", "APPENDIX", "REFERENCE"],
    ]

    groups: list[dict] = [{
        "group_id": i,
        "section_titles": [],
        "combined_text": "",
        "pages": [],
    } for i in range(len(CHUNK_KEYWORDS))]

    for title, text in sections.items():
        title_up = title.upper()
        assigned = False
        for i, keywords in enumerate(CHUNK_KEYWORDS):
            if any(kw in title_up for kw in keywords):
                groups[i]["section_titles"].append(title)
                groups[i]["combined_text"] += f"\n\n{title}\n{text}"
                groups[i]["pages"].append(sections_pages.get(title, 0))
                assigned = True
                break
        if not assigned:
            groups[0]["section_titles"].append(title)
            groups[0]["combined_text"] += f"\n\n{title}\n{text}"
            groups[0]["pages"].append(sections_pages.get(title, 0))

    return [g for g in groups if g["section_titles"]]
