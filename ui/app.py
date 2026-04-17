import streamlit as st
import requests
import os
import json

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Intelligent Research Assistant", layout="wide", page_icon="🤖")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"user_{os.urandom(4).hex()}"
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- SIDEBAR: KNOWLEDGE VAULT & SESSION TOOLS ---
with st.sidebar:
    st.title("📂 Knowledge Vault")
    st.info(f"**Active Session:** `{st.session_state.session_id}`")

    uploaded_file = st.file_uploader("Upload a Research PDF", type="pdf")
    if uploaded_file:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Analyzing and Vectorizing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    params = {"session_id": st.session_state.session_id}
                    response = requests.post(f"{API_URL}/upload", params=params, files=files)
                    if response.status_code == 201:
                        st.success(f"Indexed: {uploaded_file.name}")
                    else:
                        st.error("Ingestion failed.")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.divider()
    st.subheader("🛠️ Session Controls")

    # Day 29: Hardening - Allow users to clear context or switch sessions
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    if st.button("🔄 New Research Thread", use_container_width=True, help="Generates a new ID to isolate data"):
        st.session_state.session_id = f"user_{os.urandom(4).hex()}"
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

# --- MAIN UI: CHAT ---
st.title("🤖 Agentic Research Assistant")
st.caption("Day 29: Production Ready | Multi-User | Source Inspector")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metrics" in message:
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Faithfulness", f"{message['metrics']['faith']:.2f}")
            m_col2.metric("Relevancy", f"{message['metrics']['rel']:.2f}")

# User input
if prompt := st.chat_input("Ask about your research..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        full_response = ""
        current_metrics = {"faith": 0.0, "rel": 0.0}
        current_sources = []

        payload = {"input": prompt, "session_id": st.session_state.session_id}

        try:
            with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data = json.loads(line_str.replace("data: ", ""))

                            if data["type"] == "thought":
                                status_placeholder.status(f"🧠 {data['content']}", state="running")

                            elif data["type"] == "token":
                                full_response += data["content"]
                                response_placeholder.markdown(full_response + "▌")

                            elif data["type"] == "source_chunks":
                                # Day 28/29: Capture raw context for transparency
                                current_sources = data["content"]

                            elif data["type"] == "eval":
                                current_metrics["faith"] = data["faithfulness"]
                                current_metrics["rel"] = data["relevancy"]

            status_placeholder.empty()
            response_placeholder.markdown(full_response)

            # Display Trust Metrics
            c1, c2 = st.columns(2)
            c1.metric("Faithfulness (Groundedness)", f"{current_metrics['faith']:.2f}")
            c2.metric("Answer Relevancy", f"{current_metrics['rel']:.2f}")

            # Day 29: New Source Inspector UI
            if current_sources:
                with st.expander("📄 Inspect Source Evidence"):
                    for i, chunk in enumerate(current_sources):
                        st.markdown(f"**Source Context {i + 1}**")
                        st.info(chunk)
                st.session_state.last_sources = current_sources

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metrics": current_metrics
            })

        except Exception as e:
            st.error(f"System Error: {e}")