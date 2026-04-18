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
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def build_index(chunks):
    print("Embedding chunks... this may take a minute")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        print(f"Embedded chunk {i+1}/{len(chunks)}")
        time.sleep(1)#to avoid hitting rate limits
    
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