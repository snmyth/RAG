from pypdf import PdfReader 
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
import os




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


question = input("Enter Your Question")
enc = model.encode([question])
result =collection.query(query_embeddings = enc, n_results = 2)

print(result)

load_dotenv()

client_ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

while True:
    question = input("Enter your question (or 'quit' to exit): ")
    if question.lower() == "quit":
        break

    enc = model.encode([question])
    result = collection.query(query_embeddings=enc, n_results=2)

    context = "\n\n".join(result["documents"][0])

    response = client_ai.chat.completions.create(
        model="google/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": f"""Answer the question using only the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}"""
            }
        ]
    )

    print(response.choices[0].message.content)
    print("---")


