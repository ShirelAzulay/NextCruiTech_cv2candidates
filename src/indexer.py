import faiss
import numpy as np

def build_faiss_index(embeddings):
    ids = list(embeddings.keys())
    vectors = np.array([embeddings[i]["embedding"] for i in ids], dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, ids
