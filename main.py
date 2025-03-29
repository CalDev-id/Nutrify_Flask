from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import chromadb

app = FastAPI()

# Load model dan database
embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
chroma_client = chromadb.PersistentClient(path="Database")
collection = chroma_client.get_or_create_collection(name="nutrition")

@app.get("/")
def read_root():
    return {"message": "timuca"}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

def retrieve_documents(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_docs = []
    for metadata_list in results["metadatas"]:
        if metadata_list:
            retrieved_docs.extend(metadata_list)

    return retrieved_docs

def sentence_similarity(query: str, retrieved_docs):
    if not retrieved_docs:
        return None

    query_embedding = embedding_model.encode([query])
    doc_embeddings = embedding_model.encode([doc['name'] for doc in retrieved_docs])

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    best_match_idx = np.argmax(similarities)

    return retrieved_docs[best_match_idx]

@app.post("/search/")
def search_food(request: QueryRequest):
    retrieved_docs = retrieve_documents(request.query, top_k=request.top_k)

    if not retrieved_docs:
        return {"results": []}

    # Gunakan sentence similarity untuk memilih hasil terbaik dari retrieved_docs
    best_match = sentence_similarity(request.query, retrieved_docs)

    return {"results": best_match}