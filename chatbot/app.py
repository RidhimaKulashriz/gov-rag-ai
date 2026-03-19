import sys
import os

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st
from rag_pipeline import RAGPipeline

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Gov AI",
    layout="centered",
)

# -------------------------
# CUSTOM CSS (🔥 CLEAN UI)
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.chat-bubble-user {
    background-color: #1f2937;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: white;
}
.chat-bubble-ai {
    background-color: #111827;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid #2d3748;
    color: white;
}
.small-text {
    font-size: 12px;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown("""
<h2 style='text-align: center;'>🇮🇳 Gov AI Assistant</h2>
<p style='text-align: center; color: gray;'>Ask questions from government documents</p>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

rag = load_pipeline()

# -------------------------
# SESSION STATE
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------
# INPUT BOX (BOTTOM STYLE)
# -------------------------
query = st.text_input("Ask something...", placeholder="e.g. What is trauma protocol?")

# -------------------------
# HANDLE QUERY
# -------------------------
if query:
    with st.spinner("Thinking..."):

        answer, sources, verified = rag.generate_answer(query)

        st.session_state.chat.append({
            "query": query,
            "answer": answer,
            "sources": sources,
            "verified": verified
        })

# -------------------------
# DISPLAY CHAT
# -------------------------
for chat in reversed(st.session_state.chat):

    # User message
    st.markdown(
        f"<div class='chat-bubble-user'>🧑 {chat['query']}</div>",
        unsafe_allow_html=True
    )

    # AI response
    st.markdown(
        f"<div class='chat-bubble-ai'>🤖 {chat['answer']}</div>",
        unsafe_allow_html=True
    )

    # Sources (collapsible)
    with st.expander("📚 Sources"):
        for s in chat["sources"]:
            st.markdown(
                f"<div class='small-text'>{s['source']} | Chunk {s['chunk_id']}</div>",
                unsafe_allow_html=True
            )

    # Verified
    with st.expander("✅ Verified Citations"):
        if chat["verified"]:
            for v in chat["verified"]:
                st.markdown(
                    f"<div class='small-text'>{v['source']} | Chunk {v['chunk_id']}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div class='small-text'>No strong matches found</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")