from ingest import load_pdf, chunk_text
from embed import load_model, get_embeddings
from store import get_collection, add_to_collection, query_collection
from generate import get_ai_client, generate_answer

# setup — runs once
text = load_pdf("test.pdf")
chunks = chunk_text(text)

model = load_model()
embeddings = get_embeddings(model, chunks)

collection = get_collection()
add_to_collection(collection, chunks, embeddings)

client = get_ai_client()

# interactive loop
while True:
    question = input("Enter your question (or 'quit' to exit): ")
    if question.lower() == "quit":
        break

    query_embedding = model.encode([question])
    result = query_collection(collection, query_embedding)

    context = "\n\n".join(result["documents"][0])

    answer = generate_answer(client, context, question)
    print(answer)
    print("---")