from ingest import load_pdf, chunk_text

text = load_pdf("test.pdf")
print("TEXT LENGTH:", len(text))
print("TEXT PREVIEW:", text[:100])

chunks = chunk_text(text)
print("CHUNKS:", len(chunks))
print(chunks)