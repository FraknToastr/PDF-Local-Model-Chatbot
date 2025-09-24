import streamlit as st
import base64

# --- App Config ---
st.set_page_config(page_title="Council Meeting Chatbot", page_icon="📝", layout="wide")

# --- Dark theme styling ---
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .chat-bubble-user {
        background-color: #2b2b2b;
        color: #ffffff;
        padding: 10px;
        border-radius: 12px;
        margin: 5px 0;
        text-align: right;
    }
    .chat-bubble-bot {
        background-color: #004578;
        color: #ffffff;
        padding: 10px;
        border-radius: 12px;
        margin: 5px 0;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Embedded logo (base64) ---
LOGO_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAMgAAAAeCAYAAABpP1GsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA25pVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMTM4IDc5LjE1OTgyNCwgMjAxNi8wOS8xNC0wMTowOTowMSAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo5NTQxOUVFMkI1NTMxMUU3OTFGRUU2NkM4QzUxMDc0RiIgeG1wTU06RG9jdW1lbnRJRD0ieG1wLmRpZDo3MkM2QjU3M0I5RDcxMUU3QTI1Q0UxQTNFNzNDQzY0MiIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDo3MkM2QjU3MkI5RDcxMUU3QTI1Q0UxQTNFNzNDQzY0MiIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ0MgMjAxNyAoV2luZG93cykiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDpmMDAwZjU3NS04Zjk0LTQxNDMtYmQ3OC0xOGYwM2JkZTQwN2QiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6OTU0MTlFRTJCNTUzMTFFNzkxRkVFNjZDOEM1MTA3NEYiLz4gPC9yZGY6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"  # truncated for readability, paste full string here

# Render logo from base64
st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{LOGO_BASE64}" width="150"/>
        <h1 style="color:white; margin-top:10px;">Council Meeting Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Example chat UI ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("Ask a question about the council PDFs:")

if user_input:
    # Save user message
    st.session_state["messages"].append(("user", user_input))
    # Mock bot response for testing
    st.session_state["messages"].append(("bot", f"Response to: {user_input}"))

# --- Display chat history ---
for sender, msg in st.session_state["messages"]:
    if sender == "user":
        st.markdown(f"<div class='chat-bubble-user'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>{msg}</div>", unsafe_allow_html=True)
