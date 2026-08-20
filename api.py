from fastapi import FastAPI, UploadFile, File

from ingest import load_pdf, chunk_text
from embed import load_model, get_embeddings
from store import get_collection, add_to_collection, query_collection
from generate import get_ai_client, generate_answer

app = FastAPI()

# runs once at startup
model = load_model()
collection = get_collection()
client = get_ai_client()


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        text = load_pdf(temp_path)
        chunks = chunk_text(text)
        embeddings = get_embeddings(model, chunks)
        add_to_collection(collection, chunks, embeddings)

    except Exception as e:
        return {"status": "error", "detail": str(e)}

    return {"status": "uploaded", "chunks": len(chunks)}


@app.post("/chat")
def chat(question: str):
    try:
        query_embedding = model.encode([question])
        result = query_collection(collection, query_embedding)
        context = "\n\n".join(result["documents"][0])
        answer = generate_answer(client, context, question)

    except Exception as e:
        return {"status": "error", "detail": str(e)}

    return {"answer": answer}