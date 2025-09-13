from typing import List, Dict, Any

from index.vector_store import VectorStore


class RAGRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query"""
        results = self.vector_store.search(query, k=top_k)

        context_chunks = []
        for text, metadata, score in results:
            context_chunks.append({
                'text': text,
                'metadata': metadata,
                'relevance_score': score,
                'source': metadata.get('filename', 'unknown'),
                'chunk_id': metadata.get('chunk_id', 0)
            })

        return context_chunks

    def format_context_for_llm(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved context for LLM input"""
        if not context_chunks:
            return "No relevant context found."

        formatted_context = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['source']
            chunk_id = chunk['chunk_id']
            text = chunk['text']
            score = chunk['relevance_score']

            formatted_context.append(
                f"[{i}] Source: {source} (Chunk {chunk_id}, Relevance: {score:.3f})\n{text}\n"
            )

        return "\n".join(formatted_context)

    def get_citations(self, context_chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate citations from context chunks"""
        citations = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['source']
            chunk_id = chunk['chunk_id']
            citations.append(f"[{i}] {source} (Chunk {chunk_id})")

        return citations
