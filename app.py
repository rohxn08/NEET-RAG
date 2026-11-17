"""Streamlit UI for NEET RAG assistant."""
import streamlit as st
from src.rag import ask

st.set_page_config(page_title="NEET.info RAG Assistant", page_icon="🩺", layout="centered")

st.title("🩺 NEET.info Counselling Assistant")
st.caption("Answers are grounded in indexed cut-off and seat-matrix data. The system will refuse if the answer is not in its sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about NEET cut-offs, colleges, categories...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving from index..."):
            response = ask(query)

        if response.refused:
            msg = f"**Not in sources.** {response.refusal_reason or ''}"
            st.warning(msg)
        else:
            st.markdown(response.answer)

            if response.sources:
                with st.expander("📄 Sources", expanded=False):
                    for i, src in enumerate(response.sources, 1):
                        st.markdown(f"**[{i}]** `{src.file}` — page {src.page}")
                        st.markdown(f"> {src.text[:300]}...")

            msg = response.answer

        st.session_state.messages.append({"role": "assistant", "content": msg})
