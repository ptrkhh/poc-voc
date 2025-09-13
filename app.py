import os
import tempfile
from typing import List

import streamlit as st

from config import Config
from index.vector_store import VectorStore
# Import our modules
from ingestion.document_processor import DocumentProcessor
from llm.local_llm import LocalLLM
from retrieval.rag_retriever import RAGRetriever

# Page config
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Load models with caching"""
    vector_store = VectorStore()
    llm = LocalLLM()
    retriever = RAGRetriever(vector_store)
    processor = DocumentProcessor()

    return vector_store, llm, retriever, processor


def process_uploaded_files(uploaded_files: List, processor: DocumentProcessor, vector_store: VectorStore):
    """Process uploaded files and add to vector store"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_chunks = []

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            doc_data = processor.extract_text_from_file(tmp_path)
            chunks = processor.chunk_text(doc_data['text'], {
                'filename': uploaded_file.name,
                'extraction_method': doc_data['method']
            })
            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_path)

        progress_bar.progress((i + 1) / len(uploaded_files))

    if all_chunks:
        status_text.text("Adding documents to vector store...")
        vector_store.add_documents(all_chunks)
        status_text.text(f"Successfully processed {len(all_chunks)} chunks from {len(uploaded_files)} files!")

    progress_bar.empty()
    return len(all_chunks)


def main():
    st.title(f"📚 {Config.PAGE_TITLE}")
    st.markdown("Upload research papers and query them using natural language")

    try:
        vector_store, llm, retriever, processor = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

    # Sidebar for file upload and management
    with st.sidebar:
        st.header("📁 Document Management")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Research Papers",
            type=['pdf', 'tex', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, LaTeX (.tex), Plain Text (.txt)"
        )

        # Process uploaded files
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    num_chunks = process_uploaded_files(uploaded_files, processor, vector_store)
                    if num_chunks > 0:
                        st.success(f"Added {num_chunks} chunks to the knowledge base!")
                        st.rerun()

        # Index management
        st.header("🗂️ Index Management")

        # Show index stats
        stats = vector_store.get_stats()
        st.metric("Documents in Index", stats['total_documents'])
        st.metric("Embedding Dimension", stats['dimension'])
        st.text(f"Model: {stats['model']}")

        # Clear index button
        if st.button("Clear Index", type="secondary"):
            vector_store.clear_index()
            st.success("Index cleared!")
            st.rerun()

        # Model info
        st.header("🤖 Model Info")
        model_type = "OpenAI" if Config.USE_OPENAI else "Local"
        st.text(f"Mode: {model_type}")

        model_info = llm.get_model_info()
        st.text(f"LLM: {model_info['model_name']}")
        st.text(f"Embeddings: {vector_store.model_name}")
        st.text(f"Device: {model_info['device']}")
        st.text(f"Status: {'✅ Ready' if model_info['available'] else '❌ Error'}")

    # Main panel for querying
    st.header("🔍 Query Interface")

    # Query input
    query = st.text_input(
        "Enter your research question:",
        placeholder="e.g., What are the main findings about machine learning in healthcare?",
        help="Ask questions about the uploaded research papers"
    )

    # Search parameters
    top_k = st.slider("Number of results", 3, 10, Config.MAX_CHUNKS_PER_QUERY)

    if query and st.button("Search", type="primary"):
        if stats['total_documents'] == 0:
            st.warning("No documents in the index. Please upload some research papers first.")
        else:
            with st.spinner("Searching and generating answer..."):
                # Retrieve context
                context_chunks = retriever.retrieve_context(query, top_k=top_k)

                if not context_chunks:
                    st.warning("No relevant documents found for your query.")
                else:
                    # Format context for LLM
                    formatted_context = retriever.format_context_for_llm(context_chunks)

                    # Generate answer
                    result = llm.generate_answer(query, formatted_context)

                    # Display results
                    st.header("📝 Generated Answer")

                    if result['error']:
                        st.error(result['answer'])
                    else:
                        st.write(result['answer'])

                        # Show model used
                        st.caption(f"Generated using: {result['model_used']}")

                    # Display retrieved context
                    st.header("📄 Retrieved Context")

                    for i, chunk in enumerate(context_chunks, 1):
                        with st.expander(f"Source {i}: {chunk['source']} (Relevance: {chunk['relevance_score']:.3f})"):
                            st.write(chunk['text'])
                            st.caption(f"Chunk ID: {chunk['chunk_id']}")

                    # Display citations
                    st.header("📚 Citations")
                    citations = retriever.get_citations(context_chunks)
                    for citation in citations:
                        st.write(citation)


if __name__ == "__main__":
    main()
