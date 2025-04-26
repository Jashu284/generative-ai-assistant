import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Initialize clients
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Function to embed the query and get response
def ask_ai(query):
    # Step 1: Embed the query
    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = embedding_response.data[0].embedding

    # Step 2: Search in Pinecone
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    context = "\n".join([match['metadata']['text'] for match in results['matches']])

    # Step 3: Send to OpenAI Chat Completion
    chat_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers based on the provided documents."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    )
    return chat_response.choices[0].message.content.strip()

# Run the app
if __name__ == "__main__":
    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        response = ask_ai(user_input)
        print(f"\n🧠 Answer: {response}\n")
