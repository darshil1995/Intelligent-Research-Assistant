import streamlit as st
import requests
import os

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Intelligent Research Assistant", layout="wide")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"user_{os.urandom(4).hex()}"

# --- SIDEBAR: UPLOAD ---
with st.sidebar:
    st.title("📂 Knowledge Vault")
    uploaded_file = st.file_uploader("Upload a Research PDF", type="pdf")

    if uploaded_file:
        if st.button("Index Document"):
            with st.spinner("Processing PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 201:
                    st.success(f"Successfully indexed {uploaded_file.name}!")
                else:
                    st.error("Failed to upload document.")

# --- MAIN UI: CHAT ---
st.title("🤖 Agentic Research Assistant")
st.caption("Powered by GPT-4o-mini, ChromaDB, and RAGAS Metrics")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about your research..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Call the Chat API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {"input": prompt, "session_id": st.session_state.session_id}
            response = requests.post(f"{API_URL}/chat", json=payload)

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                faith = data["faithfulness"]
                rel = data["relevancy"]

                st.markdown(answer)

                # 2. Display Trust Metrics
                col1, col2 = st.columns(2)
                col1.metric("Faithfulness (Groundedness)", f"{faith:.2f}")
                col2.metric("Answer Relevancy", f"{rel:.2f}")

                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("The agent encountered an error.")