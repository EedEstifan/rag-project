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

def build_index(chunks):
    print("Embedding chunks...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        print(f"Embedded chunk {i+1}/{len(chunks)}")

    embeddings_np = np.array(embeddings).astype("float32")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    faiss.write_index(index, "index.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done! {len(chunks)} chunks indexed.")

if __name__ == "__main__":
    from loader import load_and_chunk
    chunks = load_and_chunk("data/gdpr.pdf")
    build_index(chunks)