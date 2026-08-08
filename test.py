from pypdf import PdfReader 
from sentence_transformers import SentenceTransformer
import chromadb

reader = PdfReader("test.pdf")


text = ""
chunks = []

for page in reader.pages:
    text+= page.extract_text()
    while len(text)>=500:
        chunks.append(text[:500])
        text = text[500:]
if text:
    chunks.append(text)
print(len(chunks))

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

print(len(embeddings))
print(len(embeddings[0]))

client = chromadb.Client()

collection = client.create_collection(name="my_pdf")

collection.add(
    documents = chunks,
    embeddings = embeddings,
    ids = [str(i) for i in range(len(chunks))]
)


question = "What is Battery life"
enc = model.encode([question])
result =collection.query(query_embeddings = enc, n_results = 2)

print(result)
