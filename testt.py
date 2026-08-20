from embed import load_model, get_embeddings
from ingest import load_pdf, chunk_text

text = load_pdf("test.pdf")
chunks = chunk_text(text)

model = load_model()
embeddings = get_embeddings(model, chunks)
print(embeddings)