import streamlit as st
from agent_graph import build_twin_graph

st.set_page_config(page_title="Agentic AI Twin", page_icon="🧠")
st.title("🧠 My Agentic AI Twin")

# 1. Get the key
raw_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# 2. Safety Step: Remove accidental spaces
api_key = raw_api_key.strip()

if not api_key:
    st.warning("""Please enter your Gemini API Key in the sidebar to start.""")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Talk to your twin..."):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Agentic workflow running..."):
            try:
                graph = build_twin_graph(api_key)
                result = graph.invoke({"user_query": prompt, "iteration_count": 0})
                response = result["final_response"]
                st.chat_message("assistant").write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
