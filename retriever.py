import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("API_KEY"))

def get_embedding(text):
    result = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return result.data[0].embedding

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