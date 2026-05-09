"""
NL2SQL Streamlit Web Application
Agent 05 (Frontend Developer): Creates an interactive web UI for the NL2SQL system.
Agent 03 (UI/UX Lead): Designed from user perspective.

Features:
- Chat-like conversational interface
- SQL syntax highlighting
- Results table display
- Debug/explanation panel
- Preprocessing visualization
- Suggestion chips
"""

import sys
import os
import json

# Fix paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "src"))

import streamlit as st
import pandas as pd
from engine_v2 import NL2SQLEngine


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="NL2SQL | Natural Language to SQL",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* SQL code block styling */
    .sql-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 16px;
        border-radius: 8px;
        font-family: 'Fira Code', 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
        overflow-x: auto;
        margin: 8px 0;
    }
    .sql-keyword { color: #569cd6; font-weight: bold; }
    .sql-function { color: #dcdcaa; }
    .sql-string { color: #ce9178; }
    .sql-number { color: #b5cea8; }
    .sql-table { color: #4ec9b0; }
    
    /* Tag chips */
    .tag-chip {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    .tag-ACTION { background: #e3f2fd; color: #1565c0; }
    .tag-TABLE { background: #e8f5e9; color: #2e7d32; }
    .tag-COLUMN { background: #fff3e0; color: #e65100; }
    .tag-AGG_FUNC { background: #f3e5f5; color: #6a1b9a; }
    .tag-VALUE { background: #fce4ec; color: #c62828; }
    .tag-OP { background: #e0f2f1; color: #00695c; }
    .tag-GROUPBY { background: #ede7f6; color: #4527a0; }
    .tag-ORDERBY { background: #e8eaf6; color: #283593; }
    .tag-LIMIT { background: #fff8e1; color: #f57f17; }
    .tag-O { background: #f5f5f5; color: #757575; }
    .tag-SORT_DIR { background: #e1f5fe; color: #0277bd; }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1a73e8;
    }
    .metric-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
    }
    
    /* Chat message styling */
    .user-msg {
        background: #e3f2fd;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .bot-msg {
        background: #f5f5f5;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Initialize Engine (cached)
# ============================================================
@st.cache_resource
def init_engine():
    return NL2SQLEngine()

engine = init_engine()

# ============================================================
# Session State
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ============================================================
# Helper Functions
# ============================================================
def format_sql(sql: str) -> str:
    """Format SQL with syntax highlighting HTML."""
    import re
    keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'GROUP BY', 'ORDER BY',
                'LIMIT', 'AND', 'OR', 'AS', 'IN', 'NOT', 'BETWEEN', 'LIKE',
                'HAVING', 'DISTINCT', 'IS', 'NULL', 'LEFT', 'RIGHT', 'INNER', 'OUTER']
    functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'strftime']
    
    html = sql
    # Escape HTML
    html = html.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Highlight keywords
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = rf'\b({kw})\b'
        html = re.sub(pattern, rf'<span class="sql-keyword">\1</span>', html, flags=re.IGNORECASE)
    
    # Highlight functions
    for fn in functions:
        pattern = rf'\b({fn})\b'
        html = re.sub(pattern, rf'<span class="sql-function">\1</span>', html, flags=re.IGNORECASE)
    
    # Highlight strings
    html = re.sub(r"'([^']*)'", r"<span class='sql-string'>'\1'</span>", html)
    
    # Highlight numbers
    html = re.sub(r'\b(\d+\.?\d*)\b', r'<span class="sql-number">\1</span>', html)
    
    return f'<div class="sql-box">{html}</div>'

def dual_sql_display(result: dict):
    """Show both neural model SQL and rule-based SQL side by side."""
    neural_sql = result.get("neural_sql", "unavailable")
    rule_sql   = result.get("rule_sql",   "no match")
    source     = result.get("final_source", result.get("debug", {}).get("source", "unknown"))

    st.markdown("**Model Comparison:**")
    col_neural, col_rule = st.columns(2)

    with col_neural:
        st.markdown("🧠 **Neural Model Output** (BERT + Seq2Seq)")
        if neural_sql and neural_sql != "unavailable":
            st.markdown(format_sql(neural_sql), unsafe_allow_html=True)
            conf = result.get("neural_conf", 0)
            st.caption(f"Confidence: {conf:.0%}")
        else:
            st.info("Neural model not available")

    with col_rule:
        st.markdown("📐 **Rule-Based CFG Output**")
        if rule_sql and rule_sql != "no match":
            st.markdown(format_sql(rule_sql), unsafe_allow_html=True)
            rule_name = result.get("rule_name", "")
            st.caption(f"Rule matched: `{rule_name}`")
        else:
            st.info("No rule matched — neural model used")

    # show which one was selected
    if "rule:" in source:
        st.success(f"✅ **Final SQL: Rule-based** (more reliable for known patterns)")
    else:
        st.info(f"🔄 **Final SQL: Neural model** (no rule matched this query)")



def dual_sql_display(result: dict):
    """Show both neural model SQL and rule-based SQL side by side."""
    neural_sql = result.get("neural_sql", "unavailable")
    rule_sql   = result.get("rule_sql",   "no match")
    source     = result.get("final_source", result.get("debug", {}).get("source", "unknown"))

    st.markdown("**Model Comparison:**")
    col_neural, col_rule = st.columns(2)

    with col_neural:
        st.markdown("🧠 **Neural Model Output** (BERT + Seq2Seq)")
        if neural_sql and neural_sql != "unavailable":
            st.markdown(format_sql(neural_sql), unsafe_allow_html=True)
            conf = result.get("neural_conf", 0)
            st.caption(f"Confidence: {conf:.0%}")
        else:
            st.info("Neural model not available")

    with col_rule:
        st.markdown("📐 **Rule-Based CFG Output**")
        if rule_sql and rule_sql != "no match":
            st.markdown(format_sql(rule_sql), unsafe_allow_html=True)
            rule_name = result.get("rule_name", "")
            st.caption(f"Rule matched: `{rule_name}`")
        else:
            st.info("No rule matched — neural model used")

    # show which one was selected
    if "rule:" in source:
        st.success(f"✅ **Final SQL: Rule-based** (more reliable for known patterns)")
    else:
        st.info(f"🔄 **Final SQL: Neural model** (no rule matched this query)")




def render_tags(tagged_sequence):
    """Render slot tags as colored chips."""
    html_parts = []
    for token, tag, meta in tagged_sequence:
        css_class = f"tag-{tag}"
        label = f"{token} [{tag}]"
        html_parts.append(f'<span class="tag-chip {css_class}">{label}</span>')
    return " ".join(html_parts)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("🔍 NL2SQL Engine")
    st.caption("Natural Language to SQL Interface")
    
    st.divider()
    
    # System Info
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
    
    # Database Schema
    st.markdown("### 🗄️ Database Schema")
    schema_tables = ["Artist", "Album", "Customer", "Employee", "Genre",
                 "Invoice", "InvoiceLine", "Track", "Playlist", "MediaType"]
    for tbl in schema_tables:
        schema = engine.retriever.get_table_schema(tbl)
        if schema:
            with st.expander(f"📋 {tbl}", expanded=False):
                for col in schema["columns"]:
                    st.markdown(f"**{col['name']}** ({col['type']})")
                    if col.get("description"):
                        st.caption(col["description"])
    
    st.divider()
    
    if st.button("🔄 Reset Conversation", use_container_width=True):
        engine.reset_conversation()
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()
    
    st.divider()
    st.caption("Built for M.Tech NLP Course | Classical NLP + RAG + Agentic AI")


# ============================================================
# Main Content
# ============================================================
st.markdown("# 🔍 Natural Language to SQL")
st.markdown("Ask questions about our e-commerce database in plain English.")

# Suggestion chips
st.markdown("**Try these queries:**")
suggestions = engine.get_suggestions()
cols = st.columns(2)
for i, sugg in enumerate(suggestions[:6]):
    with cols[i % 2]:
        if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
            st.session_state.pending_query = sugg

st.divider()

# Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            result = msg.get("result", {})
            
            if result.get("action") == "result":
                # SQL Display
                st.markdown("**Generated SQL:**")
                st.markdown(format_sql(result["sql"]), unsafe_allow_html=True)
                
                # Metrics row
                mcols = st.columns(4)
                with mcols[0]:
                    st.metric("Intent", result.get("intent", "N/A"))
                with mcols[1]:
                    st.metric("Confidence", f"{result.get('confidence', 0):.0%}")
                with mcols[2]:
                    st.metric("Rows", result.get("row_count", 0))
                with mcols[3]:
                    is_fu = "Yes" if result.get("is_followup") else "No"
                    st.metric("Follow-up", is_fu)
                
                # Results Table
                if result.get("data"):
                    df = pd.DataFrame(result["data"])
                    st.dataframe(df, use_container_width=True, height=min(400, 40 + len(df) * 35))
                
                # Explanation
                if result.get("explanation"):
                    st.info(result["explanation"])
                
                # Debug Expanders
                with st.expander("🔬 NLP Pipeline Details", expanded=False):
                    prep = result.get("preprocessing", {})
                    if prep:
                        st.markdown("**Tokens:**")
                        st.code(" → ".join(prep.get("tokens", [])))
                        
                        st.markdown("**POS Tags:**")
                        pos_str = ", ".join([f"{t}({tag})" for t, tag in prep.get("pos_tags", [])])
                        st.code(pos_str)
                        
                        st.markdown("**Lemmas:**")
                        st.code(" → ".join(prep.get("lemmas", [])))
                        
                        if prep.get("bigrams"):
                            st.markdown("**Bigrams:**")
                            st.code(", ".join(prep["bigrams"]))
                    
                    # Slot tags
                    debug = result.get("debug", {})
                    if debug.get("tagged_sequence"):
                        st.markdown("**Sequence Tags:**")
                        st.markdown(render_tags(debug["tagged_sequence"]), unsafe_allow_html=True)
                    
                    # Parse tree
                    if debug.get("parse_tree"):
                        st.markdown("**Syntax Tree:**")
                        st.json(debug["parse_tree"])
                
                with st.expander("📡 Schema Retrieval (RAG)", expanded=False):
                    debug = result.get("debug", {})
                    ctx = debug.get("retrieved_context", {})
                    if ctx.get("tables"):
                        for t in ctx["tables"]:
                            st.markdown(f"**{t['table']}** — Score: {t['score']:.3f}")
                    
                    steps = debug.get("steps", [])
                    if steps:
                        st.markdown("**Agent Steps:**")
                        for step in steps:
                            st.markdown(f"• {step}")
            
            elif result.get("action") == "clarification":
                st.warning(result.get("message", "Could you rephrase?"))
            
            elif result.get("action") == "error":
                st.error(result.get("message", "An error occurred."))
                if result.get("sql"):
                    st.code(result["sql"], language="sql")


# ============================================================
# Chat Input
# ============================================================
# Check for pending query from suggestion buttons
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask a question about the e-commerce database...")

query = pending or user_input

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            result = engine.query(query)
        
        st.session_state.query_count += 1
        
        if result.get("action") == "result":
            st.markdown("**Generated SQL:**")
            st.markdown(format_sql(result["sql"]), unsafe_allow_html=True)
            
            mcols = st.columns(4)
            with mcols[0]:
                st.metric("Intent", result.get("intent", "N/A"))
            with mcols[1]:
                st.metric("Confidence", f"{result.get('confidence', 0):.0%}")
            with mcols[2]:
                st.metric("Rows", result.get("row_count", 0))
            with mcols[3]:
                is_fu = "Yes" if result.get("is_followup") else "No"
                st.metric("Follow-up", is_fu)
            
            if result.get("data"):
                df = pd.DataFrame(result["data"])
                st.dataframe(df, use_container_width=True, height=min(400, 40 + len(df) * 35))
            
            if result.get("explanation"):
                st.info(result["explanation"])
            
            with st.expander("🔬 NLP Pipeline Details", expanded=False):
                prep = result.get("preprocessing", {})
                if prep:
                    st.markdown("**Tokens:**")
                    st.code(" → ".join(prep.get("tokens", [])))
                    st.markdown("**POS Tags:**")
                    pos_str = ", ".join([f"{t}({tag})" for t, tag in prep.get("pos_tags", [])])
                    st.code(pos_str)
                    st.markdown("**Lemmas:**")
                    st.code(" → ".join(prep.get("lemmas", [])))
                    if prep.get("bigrams"):
                        st.markdown("**Bigrams:**")
                        st.code(", ".join(prep["bigrams"]))
                
                debug = result.get("debug", {})
                if debug.get("tagged_sequence"):
                    st.markdown("**Sequence Tags:**")
                    st.markdown(render_tags(debug["tagged_sequence"]), unsafe_allow_html=True)
                
                if debug.get("parse_tree"):
                    st.markdown("**Syntax Tree:**")
                    st.json(debug["parse_tree"])
            
            with st.expander("📡 Schema Retrieval (RAG)", expanded=False):
                debug = result.get("debug", {})
                ctx = debug.get("retrieved_context", {})
                if ctx.get("tables"):
                    for t in ctx["tables"]:
                        st.markdown(f"**{t['table']}** — Score: {t['score']:.3f}")
                steps = debug.get("steps", [])
                if steps:
                    st.markdown("**Agent Steps:**")
                    for step in steps:
                        st.markdown(f"• {step}")
        
        elif result.get("action") == "clarification":
            st.warning(result.get("message", "Could you rephrase?"))
        
        elif result.get("action") == "error":
            st.error(result.get("message", "An error occurred."))
            if result.get("sql"):
                st.code(result["sql"], language="sql")
    
    # Store result
    st.session_state.messages.append({"role": "assistant", "result": result})
