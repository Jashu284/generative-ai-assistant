import os
import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch API keys and environment from OS Environment Variables
pinecone_api_key = "pcsk_69FCXk_L3WhjjBFV8wXxcYjp7KjpYGoMAoC1odyUqqw1rsg39Jfm6fMNEVTtewY6hbr6GP"
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Debugging - you can uncomment this to check if environment variables are loaded
# st.write("Pinecone API Key:", pinecone_api_key)
# st.write("Pinecone Env:", pinecone_env)
# st.write("Pinecone Index:", pinecone_index_name)
# st.write("OpenAI API Key:", openai.api_key)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Create index if it does not exist
if pinecone_index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )

index = pc.Index(pinecone_index_name)

# Streamlit config
st.set_page_config(page_title="Generative AI Assistant", layout="centered")

# Sidebar info
with st.sidebar:
    st.markdown("""
    <h3 style='margin-bottom: 5px;'>📘 About</h3>
    <b>Generative AI Assistant</b><br>
    This assistant helps you with Generative AI topics like transformers, embeddings, RAG, GANs, and more.<br><br>
    Feel free to ask follow-up questions!
    """, unsafe_allow_html=True)

    theme_toggle = st.radio("🌗 Theme", ["Light", "Dark"])
    st.session_state.theme = theme_toggle

    uploaded_file = st.file_uploader("📄 Upload Knowledge Base", type=["txt", "md"])
    if uploaded_file:
        uploaded_text = uploaded_file.read().decode("utf-8")
        st.session_state.uploaded_kb = uploaded_text
        st.success("Knowledge base uploaded successfully!")

    if st.button("🧹 Reset Chat"):
        st.session_state.history = []
        st.rerun()

# Apply dark mode theme if selected
if st.session_state.get("theme") == "Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #0e1117; color: white; }
        .sidebar .sidebar-content { background-color: #1c1f26; }
        .user-bubble { background-color: #1f2937 !important; color: white !important; }
        .assistant-bubble { background-color: #374151 !important; color: white !important; }
        input[type="text"], .stTextInput > div > div > input { background-color: #1f2937 !important; color: white !important; border: 1px solid #4b5563 !important; }
        .stButton button { background-color: #3b82f6 !important; color: white !important; border: none; }
        .stButton button:hover { background-color: #2563eb !important; }
        </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 48px; margin-bottom: 5px;'>🧠 Generative AI Assistant</h1>
        <p style='color: grey;'>Ask anything about Generative AI knowledge base.</p>
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

if submitted and user_input:
    st.session_state.history.append(("User", user_input))

    # Get embedding for the question
    embedding = openai.Embedding.create(
        model="text-embedding-3-small",
        input=user_input
    )["data"][0]["embedding"]

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
Be concise, conversational, and helpful. Focus only on Generative AI topics like embeddings, transformers, RAG, GANs, etc.

Context: {context}

Question: {user_input}
Answer:"""
    else:
        prompt = f"""You are a friendly AI assistant. Even if the user's question doesn't match the knowledge base exactly, answer if it's about Generative AI topics (LLMs, transformers, GANs, RAG, etc). If not, gently say it's off-topic.

Question: {user_input}
Answer:"""

    # Generate response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["choices"][0]["message"]["content"].strip()

    st.session_state.history.append(("Assistant", answer))
    st.rerun()
