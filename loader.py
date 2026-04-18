from pypdf import PdfReader

def load_and_chunk(pdf_path, chunk_size=500):
    reader = PdfReader(pdf_path)
    
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    
    words = full_text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk("data/gdpr.pdf")
    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:\n{chunks[0][:300]}")