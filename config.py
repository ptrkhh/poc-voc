import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # Model Configuration
    USE_OPENAI = os.getenv('USE_OPENAI', 'false').lower() == 'true'
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # OpenAI Models
    OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    OPENAI_LLM_MODEL = os.getenv('OPENAI_LLM_MODEL', 'gpt-3.5-turbo')

    # Local Models
    LOCAL_EMBEDDING_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    LOCAL_LLM_MODEL = os.getenv('LOCAL_LLM_MODEL', 'microsoft/DialoGPT-medium')

    # Document Processing
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
    MAX_CHUNKS_PER_QUERY = int(os.getenv('MAX_CHUNKS_PER_QUERY', '5'))

    # FAISS Index
    INDEX_PATH = os.getenv('INDEX_PATH', 'data/faiss_index')

    # Streamlit Config
    PAGE_TITLE = os.getenv('PAGE_TITLE', 'Internal Research RAG System')
