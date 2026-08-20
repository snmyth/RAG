from sentence_transformers import SentenceTransformer


def load_model(model_name = 'all-MiniLM-L6-v2'):
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise ValueError(f"An Error Occured{e}")
    return model


def get_embeddings(model, chunks):
    if not chunks:
        raise ValueError("No chunks provided to embed.")

    try:
        embeddings = model.encode(chunks)
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

    return embeddings