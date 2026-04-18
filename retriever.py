import os
import pickle
import numpy as np
import faiss
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))

def get_embedding(text):
    time.sleep(1)
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def retrieve(query, top_k=3):
    index = faiss.read_index("index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    query_embedding = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "score": float(distances[0][i])
        })

    return results

if __name__ == "__main__":
    query = "What is the fine for a data breach?"
    results = retrieve(query)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.2f}) ---")
        print(r["chunk"][:300])