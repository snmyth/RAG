import chromadb

def get_collection(name = "my_pdf"):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name)


    return collection


def add_to_collection(collection,chunks,embeddings):
    try:
        collection.add(
            documents = chunks,
            embeddings = embeddings,
            ids = [str(i) for i in range(len(chunks))]
        )

        
        return "Done"

    except Exception as e:
        raise ValueError(f"Error Occured{e}")



def query_collection(collection, query_embedding, n_results=10):
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
    except Exception as e:
        raise ValueError(f"Error occurred while querying: {e}")

    return results

