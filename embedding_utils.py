# embedding_utils.py
import openai
import os
import faiss
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

def get_embedding(text: str) -> List[float]:
    """Generates an embedding for the given text using OpenAI's text-embedding-ada-002 model."""
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except openai.OpenAIError as e:
        print(f"Error generating embedding: {e}")
        # Consider a retry mechanism or a fallback here
        raise

class VectorStore:
    def __init__(self, dimension: int = 1536):
        # text-embedding-ada-002 has a dimension of 1536
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: List[Dict[str, Any]] = [] # To store original text and metadata
        self.doc_id_to_index_range: Dict[str, tuple] = {} # Maps doc_id to (start_idx, end_idx) in metadata/index

    def add_chunks(self, doc_id: str, chunks: List[str], filename: str = "unknown_file") -> None:
        """
        Adds text chunks and their embeddings to the FAISS index.
        Stores original text and metadata.
        """
        embeddings = []
        start_index = len(self.metadata)

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            embeddings.append(embedding)
            # Store metadata for each chunk
            self.metadata.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk # Storing the original text chunk
            })

        if embeddings:
            np_embeddings = np.array(embeddings).astype('float32')
            self.index.add(np_embeddings)
            end_index = len(self.metadata) - 1
            self.doc_id_to_index_range[doc_id] = (start_index, end_index)
            print(f"Added {len(chunks)} chunks for document {doc_id} to FAISS index.")
        else:
            print(f"No embeddings generated for document {doc_id}.")


    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a similarity search on the FAISS index and returns top k relevant chunks.
        Returns the original text and its metadata.
        """
        query_embedding = get_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')

        # D: distances, I: indices
        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: # -1 indicates no valid result
                continue
            metadata = self.metadata[idx]
            results.append({
                "text": metadata["text"],
                "doc_id": metadata["doc_id"],
                "filename": metadata["filename"],
                "chunk_index": metadata["chunk_index"],
                "distance": float(distances[0][i])
            })
        return results

    def clear_doc(self, doc_id: str):
        """Removes all chunks associated with a specific doc_id from the store."""

        if doc_id not in self.doc_id_to_index_range:
            print(f"Document ID {doc_id} not found.")
            return

        print(f"Removing document {doc_id} from the store. Note: FAISS IndexFlatL2 doesn't support efficient deletion.")

        self.metadata = [m for m in self.metadata if m["doc_id"] != doc_id]
        del self.doc_id_to_index_range[doc_id]


vector_store = VectorStore(dimension=1536)
