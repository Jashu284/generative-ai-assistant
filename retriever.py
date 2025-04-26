import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Read documents from knowledge_base/
def read_documents():
    docs = []
    folder_path = "knowledge_base"
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                docs.append((filename, file.read()))
    return docs

# Embed and upload documents
def embed_and_upsert(docs):
    vectors = []
    for doc_id, text in docs:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        vectors.append({
            "id": doc_id,
            "values": embedding,
            "metadata": {"text": text}
        })
    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} documents to Pinecone!")

# Run the embedding and upload
if __name__ == "__main__":
    documents = read_documents()
    embed_and_upsert(documents)
