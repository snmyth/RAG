import os
from openai import OpenAI

_client = None


def load_model(model_name: str = "text-embedding-3-small"):
    """
    Returns a lightweight handle: just the model name + a shared OpenAI client.
    Kept as a function (instead of a bare string) so calling code that does
    `model = load_model()` then `get_embeddings(model, chunks)` still works
    unchanged.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return model_name


def get_embeddings(model, chunks):
    if not chunks:
        raise ValueError("No chunks provided to embed.")

    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    try:
        # OpenAI's embeddings endpoint accepts a batch of strings directly
        response = _client.embeddings.create(
            model=model,
            input=chunks,
        )
        embeddings = [item.embedding for item in response.data]
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

    return embeddings