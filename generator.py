import os
from openai import OpenAI
from dotenv import load_dotenv
from retriever import retrieve

load_dotenv()

client = OpenAI(api_key=os.getenv("API_KEY"))

def answer(query):
    chunks = retrieve(query, top_k=3)
    context = "\n\n".join([r["chunk"] for r in chunks])

    prompt = f"""You are a helpful assistant that answers questions about GDPR.
Use the context below to answer the question as best you can.
The context may contain partial information - use what is available.
Only say "I couldn't find a clear answer" if the context has absolutely nothing relevant.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks
    }

if __name__ == "__main__":
    query = "What is the fine for a data breach?"
    result = answer(query)
    print(f"Answer:\n{result['answer']}")
    print(f"\nBased on {len(result['sources'])} retrieved chunks")