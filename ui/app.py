import streamlit as st
import requests
import os
import json

# --- CONFIGURATION ---
# Use 'http://backend:8000' if running in Docker, otherwise 'http://127.0.0.1:8000'
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Intelligent Research Assistant", layout="wide", page_icon="🤖")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    # This ID acts as the 'Owner Key' for documents in ChromaDB
    st.session_state.session_id = f"user_{os.urandom(4).hex()}"

# --- SIDEBAR: UPLOAD ---
with st.sidebar:
    st.title("📂 Knowledge Vault")
    st.info(f"Session ID: `{st.session_state.session_id}`")  # Visual confirmation of isolation
    uploaded_file = st.file_uploader("Upload a Research PDF", type="pdf")

    if uploaded_file:
        if st.button("Index Document"):
            with st.spinner("Processing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

                    # --- CRITICAL CHANGE FOR DAY 26 ---
                    # We pass the session_id as a query parameter to tag the data
                    params = {"session_id": st.session_state.session_id}
                    response = requests.post(f"{API_URL}/upload", params=params, files=files)

                    if response.status_code == 201:
                        st.success(f"Successfully indexed {uploaded_file.name}!")
                    else:
                        st.error(f"Failed to upload: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# --- MAIN UI: CHAT ---
st.title("🤖 Agentic Research Assistant")
st.caption("Multi-User Isolation | Real-time Streaming | RAGAS Metrics")

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metrics" in message:
            col1, col2 = st.columns(2)
            col1.metric("Faithfulness", f"{message['metrics']['faith']:.2f}")
            col2.metric("Relevancy", f"{message['metrics']['rel']:.2f}")

# User input
if prompt := st.chat_input("Ask a question about your research..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Start Assistant Response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        full_response = ""
        metrics = {"faith": 0.0, "rel": 0.0}

        # Session ID is included in payload for both Memory and Metadata filtering
        payload = {"input": prompt, "session_id": st.session_state.session_id}

        try:
            # Using stream=True for Server-Sent Events (SSE)
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

                            elif data["type"] == "eval":
                                metrics["faith"] = data["faithfulness"]
                                metrics["rel"] = data["relevancy"]

            # Cleanup status and show final clean markdown
            status_placeholder.empty()
            response_placeholder.markdown(full_response)

            # 2. Display Trust Metrics
            col1, col2 = st.columns(2)
            col1.metric("Faithfulness (Groundedness)", f"{metrics['faith']:.2f}")
            col2.metric("Answer Relevancy", f"{metrics['rel']:.2f}")

            # Persist to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metrics": metrics
            })

        except Exception as e:
            st.error(f"The agent encountered an error: {e}")