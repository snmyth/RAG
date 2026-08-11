from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(path):
    try:
        loader = PyPDFLoader(path)
        pages = loader.load()

    except Exception as e:
        raise ValueError(f"An error occured, {e}")

    if not pages:
        raise ValueError(f"There are no pages in the PDF")

    text = "".join(page.page_content for page in pages)

    if not text.strip():
        raise ValueError("No extractable text in PDF")

    return text


def chunk_text(text, size = 500, overlap = 50):
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size = size, chunk_overlap = overlap)

    return splitter.split_text(text)