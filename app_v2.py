"""
app_v2.py
=========
NL2SQL Streamlit App — Dual Output Version.

Shows:
1. Neural model SQL prediction + execution result
2. Rule-based CFG SQL prediction + execution result
3. NLP pipeline details
4. Schema retrieval info

Run: streamlit run app_v2.py
"""

import sys
import os
import json

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

import streamlit as st
import pandas as pd
from engine_v2 import NL2SQLEngine
from feedback_logger import save_positive, save_correction, get_feedback_stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NL2SQL | BERT + Seq2Seq",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .sql-box {
        background: #1e1e1e; color: #d4d4d4;
        padding: 14px; border-radius: 8px;
        font-family: 'Fira Code', 'Courier New', monospace;
        font-size: 13px; line-height: 1.6;
        overflow-x: auto; margin: 6px 0;
    }
    .sql-keyword  { color: #569cd6; font-weight: bold; }
    .sql-function { color: #dcdcaa; }
    .sql-string   { color: #ce9178; }
    .sql-number   { color: #b5cea8; }
    .rule-box {
        border: 2px solid #4CAF50;
        border-radius: 8px; padding: 12px;
        background: #f9fff9;
    }
    .neural-box {
        border: 2px solid #2196F3;
        border-radius: 8px; padding: 12px;
        background: #f0f7ff;
    }
    .tag-chip {
        display: inline-block; padding: 2px 8px;
        margin: 2px; border-radius: 12px;
        font-size: 12px; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Engine ────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_engine():
    return NL2SQLEngine()

engine = init_engine()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_sql(sql: str) -> str:
    import re
    keywords = ['SELECT','FROM','WHERE','JOIN','ON','GROUP BY','ORDER BY',
                'LIMIT','AND','OR','AS','IN','NOT','BETWEEN','LIKE',
                'HAVING','DISTINCT','IS','NULL','LEFT','RIGHT','INNER',
                'OUTER','DESC','ASC','UNION','BY']
    functions = ['COUNT','SUM','AVG','MIN','MAX','strftime']
    html = sql.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    for kw in sorted(keywords, key=len, reverse=True):
        import re
        html = re.sub(rf'\b({kw})\b',
                      rf'<span class="sql-keyword">\1</span>',
                      html, flags=re.IGNORECASE)
    for fn in functions:
        html = re.sub(rf'\b({fn})\b',
                      rf'<span class="sql-function">\1</span>',
                      html, flags=re.IGNORECASE)
    html = re.sub(r"'([^']*)'",
                  r"<span class='sql-string'>'\1'</span>", html)
    html = re.sub(r'\b(\d+\.?\d*)\b',
                  r'<span class="sql-number">\1</span>', html)
    return f'<div class="sql-box">{html}</div>'


def show_result_block(result: dict):
    """
    Neural-model-only result display:
      1. Neural model SQL prediction
      2. Execution result
      3. NLP pipeline details
      4. Schema retrieval info
    """

    neural_sql   = result.get("neural_sql",   "unavailable")
    neural_error = result.get("neural_error", None)
    neural_data  = result.get("neural_data",  [])
    neural_conf  = result.get("neural_conf",  0.0)
    intent       = result.get("intent",       "N/A")

    # ── SECTION 1: Neural Model SQL ───────────────────────────────────────
    st.markdown("### 🧠 Neural Model Prediction")
    st.markdown(
        '<div class="neural-box">'
        '<b>🧠 BERT + Seq2Seq Neural Model</b>'
        '</div>',
        unsafe_allow_html=True
    )

    if neural_sql and neural_sql != "unavailable":
        st.markdown(format_sql(neural_sql), unsafe_allow_html=True)
        decoding = result.get("neural_decoding", "beam_search (k=5)")
        st.caption(f"Confidence: {neural_conf:.0%}  |  Encoder: BERT  |  Decoder: {decoding}")

        # corrections badge
        corrections = result.get("neural_corrections", [])
        if corrections:
            st.warning(f"🔧 {len(corrections)} schema correction(s) applied")
            with st.expander("View corrections", expanded=False):
                for c in corrections:
                    st.markdown(f"• {c}")

        # greedy vs beam comparison
        greedy_sql = result.get("greedy_sql", "")
        if greedy_sql and greedy_sql != neural_sql:
            with st.expander("🔍 Greedy vs Beam Search comparison", expanded=False):
                st.markdown("**Greedy output (single best token at each step):**")
                st.markdown(format_sql(greedy_sql), unsafe_allow_html=True)
                st.markdown("**Beam search output (explores 5 paths simultaneously):**")
                st.markdown(format_sql(neural_sql), unsafe_allow_html=True)
                st.caption("Beam search selected the higher probability complete sequence")
    else:
        st.error("❌ Neural model could not generate a valid SQL for this query. Try rephrasing.")

    st.divider()

    # ── SECTION 2: Execution Result ───────────────────────────────────────
    st.markdown("### 📊 Query Result")
    if neural_error and not neural_data:
        st.error(f"SQL Execution Error: {neural_error}")
    elif neural_data:
        df = pd.DataFrame(neural_data)
        st.dataframe(df, use_container_width=True,
                     height=min(400, 40 + len(df) * 35))
        st.caption(f"{len(neural_data)} row(s) returned")
    else:
        st.warning("No results returned.")

    st.divider()

    # ── SECTION 3: NLP Pipeline ───────────────────────────────────────────
    with st.expander("🔬 NLP Pipeline Details (Module 2 & 3)", expanded=False):
        prep = result.get("preprocessing", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tokens:**")
            st.code(" → ".join(prep.get("tokens", [])))
            st.markdown("**Lemmas:**")
            st.code(" → ".join(prep.get("lemmas", [])))
        with c2:
            st.markdown("**POS Tags:**")
            pos_str = ", ".join(
                [f"{t}({tag})" for t, tag in prep.get("pos_tags", [])]
            )
            st.code(pos_str)
            st.markdown("**Bigrams:**")
            st.code(", ".join(prep.get("bigrams", [])[:8]))

        st.markdown(f"**Naive Bayes Intent:** `{intent}`")

    # ── SECTION 4: Schema Retrieval ───────────────────────────────────────
    with st.expander("📡 Schema Retrieval — TF-IDF RAG", expanded=False):
        debug = result.get("debug", {})
        ctx   = debug.get("retrieved_context", {})

        if ctx.get("tables"):
            st.markdown("**Tables retrieved (TF-IDF + Synonym scoring):**")
            for t in ctx["tables"]:
                cols_matched = t.get("matched_columns", [])
                col_str = f" — columns: `{', '.join(cols_matched)}`" if cols_matched else ""
                st.markdown(
                    f"• **{t['table']}** — "
                    f"Combined: `{t['score']:.3f}` | "
                    f"TF-IDF: `{t.get('tfidf_score',0):.3f}` | "
                    f"Synonym: `{t.get('synonym_score',0):.3f}`"
                    f"{col_str}"
                )

        if ctx.get("columns"):
            st.markdown("**Columns linked:**")
            for c in ctx["columns"][:6]:
                st.markdown(f"• `{c['table']}.{c['column']}` — score: `{c['score']}`")

        if ctx.get("filters"):
            st.markdown("**Values detected (value linking):**")
            for f in ctx["filters"]:
                st.markdown(
                    f"• `{f['table']}.{f['column']}` {f['op']} `'{f['value']}'`"
                )

        if ctx.get("join_path"):
            st.markdown("**Join path:**")
            for j in ctx["join_path"]:
                st.markdown(f"• `{j}`")

        steps = debug.get("steps", [])
        if steps:
            st.markdown("**Agent Steps:**")
            for step in steps:
                st.markdown(f"• {step}")

    # ── SECTION 5: Explanation ────────────────────────────────────────────
    if result.get("explanation"):
        st.info(result["explanation"])

    # ── SECTION 6: Feedback ───────────────────────────────────────────────
    if neural_sql and neural_sql != "unavailable":
        st.divider()
        st.markdown("**Was this result correct?**")

        fb_key = result.get("turn", 0)
        col_up, col_down, _ = st.columns([1, 1, 5])

        with col_up:
            if st.button("Correct", key=f"up_{fb_key}",
                         use_container_width=True):
                ok = save_positive(
                    st.session_state.get("last_query", ""),
                    neural_sql
                )
                st.success("Saved as correct example." if ok
                           else "Could not save feedback.")

        with col_down:
            if st.button("Incorrect", key=f"down_{fb_key}",
                         use_container_width=True):
                st.session_state[f"show_correction_{fb_key}"] = True

        if st.session_state.get(f"show_correction_{fb_key}", False):
            st.markdown("**Paste the correct SQL below (optional):**")
            correct_sql = st.text_area(
                "Correct SQL",
                placeholder="SELECT ... FROM ... WHERE ...",
                key=f"correction_sql_{fb_key}",
                label_visibility="collapsed",
                height=100,
            )
            if st.button("Submit correction", key=f"submit_{fb_key}"):
                if correct_sql.strip():
                    ok, err, count = save_correction(
                        st.session_state.get("last_query", ""),
                        correct_sql.strip()
                    )
                    if ok:
                        st.success(
                            f"Verified against database — {count} row(s) returned. "
                            f"Saved as training pair for next fine-tuning cycle."
                        )
                        st.session_state[f"show_correction_{fb_key}"] = False
                    else:
                        st.error(f"SQL verification failed: {err}")
                else:
                    st.info("Feedback noted. No correction provided.")
with st.sidebar:
    st.title("🔍 NL2SQL Engine")
    st.caption("BERT Seq2Seq + TF-IDF Schema Linking + Beam Search")
    st.divider()

    info = engine.get_system_info()
    st.markdown("### 📊 System Info")
    st.markdown(f"**Database:** {info['database']}")

    with st.expander("NLP Components", expanded=False):
        for comp in info["components"]:
            st.markdown(f"• {comp}")

    with st.expander("Supported SQL", expanded=False):
        for sql_type in info["supported_sql"]:
            st.markdown(f"• {sql_type}")

    st.divider()
    st.markdown("### 🗄️ Database Schema")
    schema_tables = ["Artist","Album","Customer","Employee","Genre",
                     "Invoice","InvoiceLine","Track","Playlist","MediaType"]
    for tbl in schema_tables:
        schema = engine.retriever.get_table_schema(tbl)
        if schema:
            with st.expander(f"📋 {tbl}", expanded=False):
                for col in schema["columns"]:
                    st.markdown(f"**{col['name']}** ({col['type']})")

    st.divider()
    if st.button("🔄 Reset Conversation", use_container_width=True):
        engine.reset_conversation()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("M.Tech NLP Course — ECL545")
    st.caption("Hybrid NL2SQL System")

    # feedback stats
    stats = get_feedback_stats()
    if stats["positive"] > 0 or stats["corrections"] > 0:
        st.divider()
        st.markdown("### Feedback Collected")
        st.caption(f"Correct examples: {stats['positive']}")
        st.caption(f"User corrections: {stats['corrections']}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Natural Language to SQL")
st.markdown("Ask anything about the **Chinook Digital Music Store** database.")

# Suggestion chips
st.markdown("**Try these:**")
suggestions = engine.get_suggestions()
cols = st.columns(3)
for i, sugg in enumerate(suggestions[:6]):
    with cols[i % 3]:
        if st.button(sugg, key=f"s_{i}", use_container_width=True):
            st.session_state.pending_query = sugg

st.divider()

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            result = msg.get("result", {})
            if result.get("action") == "result":
                show_result_block(result)
            elif result.get("action") == "clarification":
                st.warning(result.get("message", "Could not understand query."))
            elif result.get("action") == "error":
                st.error(result.get("message", "An error occurred."))

# Chat input
pending    = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask a question about the Chinook music database...")
query      = pending or user_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state["last_query"] = query
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            result = engine.query(query)

        if result.get("action") == "result":
            show_result_block(result)
        elif result.get("action") == "clarification":
            st.warning(result.get("message", "Could not understand query."))
            if result.get("neural_sql"):
                show_result_block({**result, "action": "result",
                                   "neural_data": [],
                                   "neural_error": result.get("neural_error", "")})
        elif result.get("action") == "error":
            st.error(result.get("message", "An error occurred."))

    st.session_state.messages.append({"role": "assistant", "result": result})