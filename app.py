import os
import streamlit as st
from openai import OpenAI
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys and Pinecone setup
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=openai_key)
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(pinecone_index_name)

# Streamlit config
st.set_page_config(page_title="Generative AI Assistant", layout="centered")

# Sidebar info
with st.sidebar:
    st.markdown("""
    <h3 style='margin-bottom: 5px;'>📘 About</h3>
    <b>Generative AI Assistant</b><br>
    This assistant is built specifically to help you with Generative AI topics like transformers, embeddings, RAG, GANs, and more.<br><br>
    Feel free to ask follow-up questions or explore related topics!
    """, unsafe_allow_html=True)

    # Light/Dark Mode Toggle
    theme_toggle = st.radio("🌗 Theme", ["Light", "Dark"])
    st.session_state.theme = theme_toggle

    # File upload for knowledge base
    uploaded_file = st.file_uploader("📄 Upload Knowledge Base", type=["txt", "md"])
    if uploaded_file is not None:
        uploaded_text = uploaded_file.read().decode("utf-8")
        st.session_state.uploaded_kb = uploaded_text
        st.success("Knowledge base uploaded successfully!")

    # Reset chat
    if st.button("🧹 Reset Chat"):
        st.session_state.history = []
        st.rerun()

# Apply dark mode theme if selected
if st.session_state.get("theme") == "Dark":
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #0e1117;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #1c1f26;
        }
        .user-bubble {
            background-color: #1f2937 !important;
            color: white !important;
        }
        .assistant-bubble {
            background-color: #374151 !important;
            color: white !important;
        }
        input[type="text"], .stTextInput > div > div > input {
            background-color: #1f2937 !important;
            color: white !important;
            border: 1px solid #4b5563 !important;
        }
        .stButton button {
            background-color: #3b82f6 !important;
            color: white !important;
            border: none;
        }
        .stButton button:hover {
            background-color: #2563eb !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 48px; margin-bottom: 5px;'>🧠 Generative AI Assistant</h1>
        <p style='color: grey;'>Ask me anything based on Generative AI knowledge base.</p>
    </div>
""", unsafe_allow_html=True)

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Custom bubble style
def message_bubble(role, message):
    if role == "User":
        st.markdown(f"""
        <div class='user-bubble' style='padding:10px;border-radius:10px;margin:5px 0;'>
            🧑 <strong>User:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='assistant-bubble' style='padding:10px;border-radius:10px;margin:5px 0;'>
            🤖 <strong>Assistant:</strong> {message}
        </div>
        """, unsafe_allow_html=True)

# Display chat history
st.markdown("<h3 style='margin-top:30px;'>💬 Chat History</h3>", unsafe_allow_html=True)
for sender, msg in st.session_state.history:
    message_bubble(sender, msg)

# User input
with st.form("chat-form", clear_on_submit=True):
    user_input = st.text_input("Your question here:", key="input")
    submitted = st.form_submit_button("Ask")

# When user submits a question
if submitted and user_input:
    st.session_state.history.append(("User", user_input))

    # Get embedding for the question
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_input
    ).data[0].embedding

    # Query Pinecone for relevant chunks
    query_response = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True
    )

    matches = query_response.get("matches", [])
    context = " ".join([match["metadata"].get("text", "") for match in matches])

    # Build prompt dynamically
    if context.strip():
        prompt = f"""You are a friendly and helpful AI assistant. Use the following context from the knowledge base to answer the question.
Be concise, conversational, and helpful. Only focus on Generative AI topics or their related subtopics like embeddings, transformers, RAG, GANs, etc.

Context: {context}

Question: {user_input}
Answer:"""
    else:
        prompt = f"""You are a friendly AI assistant. The user's question may not directly match the provided knowledge base, but you can still answer if it's closely related to generative AI (like LLMs, embeddings, transformers, GANs, etc).
If it's clearly off-topic, gently let the user know. Be warm and polite.

Question: {user_input}
Answer:"""

    # Generate response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    
    answer = response.choices[0].message.content.strip()

    st.session_state.history.append(("Assistant", answer))
    st.rerun()  # Refresh to show latest messages
