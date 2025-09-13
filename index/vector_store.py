import os
import pickle
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np

from config import Config


class VectorStore:
    def __init__(self):
        self.index_path = Config.INDEX_PATH
        self.use_openai = Config.USE_OPENAI

        if self.use_openai:
            import openai
            openai.api_key = Config.OPENAI_API_KEY
            self.client = openai.OpenAI()
            self.model_name = Config.OPENAI_EMBEDDING_MODEL
            self.dimension = 1536  # text-embedding-3-small dimension
        else:
            from sentence_transformers import SentenceTransformer
            self.model_name = Config.LOCAL_EMBEDDING_MODEL
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.metadata = []

        # Load existing index if available
        self.load_index()

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI or local model"""
        if self.use_openai:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            embeddings = np.array([item.embedding for item in response.data])
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store"""
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        embeddings = self._generate_embeddings(texts)

        # Ensure float32 format for FAISS
        embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata with text content
        for chunk in chunks:
            metadata_with_text = chunk['metadata'].copy()
            metadata_with_text['text'] = chunk['text']
            self.metadata.append(metadata_with_text)

        # Save updated index
        self.save_index()

    def search(self, query: str, k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = self._generate_embeddings([query])
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                # Get original text from stored metadata
                chunk_text = self.metadata[idx].get('text',
                                                    f"Chunk {self.metadata[idx].get('chunk_id', idx)} from {self.metadata[idx].get('filename', 'unknown')}")
                results.append((chunk_text, self.metadata[idx], float(score)))

        return results

    def save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.faiss")

        # Save metadata
        with open(f"{self.index_path}_metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)

    def load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        try:
            if os.path.exists(f"{self.index_path}.faiss"):
                self.index = faiss.read_index(f"{self.index_path}.faiss")

            if os.path.exists(f"{self.index_path}_metadata.pkl"):
                with open(f"{self.index_path}_metadata.pkl", 'rb') as f:
                    self.metadata = pickle.load(f)
        except Exception as e:
            print(f"Could not load existing index: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

    def clear_index(self) -> None:
        """Clear the entire index"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_documents': self.index.ntotal,
            'dimension': self.dimension,
            'model': self.model_name
        }
