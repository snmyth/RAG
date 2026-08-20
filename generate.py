from dotenv import load_dotenv
from openai import OpenAI
import os


def get_ai_client():
    load_dotenv()
    client_ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    return client_ai


def generate_answer(client, context, question, model="google/gemini-2.5-flash-lite"):
    if not context or not context.strip():
        return "I don't have enough information to answer that."

    try:
        response = client.chat.completions.create(
            model=model,
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
    except Exception as e:
        raise ValueError(f"Error occurred while generating answer: {e}")

    return response.choices[0].message.content