import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_active_vector_store: Chroma | None = None

# from transcript give embedding
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : "cpu"}
    )

def build_vector_store(transcript : str)->Chroma:
    global _active_vector_store
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    # transcript splitted
    chunks = splitter.split_text(transcript)

    # wrapping in a langchain doc
    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i, chunk in enumerate(chunks)
    ]

    # creating an instance of embedding model
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    _active_vector_store = vector_store
    return vector_store

# loading vector store
def load_vector_store() ->Chroma:
    global _active_vector_store
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    _active_vector_store = vector_store
    return vector_store


def close_vector_store() -> None:
    global _active_vector_store
    if _active_vector_store is not None:
        _active_vector_store._client.close()
        _active_vector_store = None

# retriving things out of the vector store (top k algo)
def get_retriever(vector_store: Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k" : k}
    )