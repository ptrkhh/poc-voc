# Internal Research RAG System

A complete Retrieval-Augmented Generation (RAG) system for querying internal research papers using Streamlit.

## Features

- **Document Processing**: Supports PDF, LaTeX (.tex), and plain text files
- **OCR Support**: Automatic OCR for scanned PDFs using Tesseract
- **Local Embeddings**: Uses sentence-transformers for text embeddings
- **Vector Search**: FAISS-based similarity search with persistence
- **Local LLM**: Runs completely offline using Hugging Face transformers
- **Interactive UI**: Clean Streamlit interface for document upload and querying

## Installation

1. **Clone and setup:**
```bash
git clone <repository>
cd poc-voc
cp .env.example .env
```

2. **Configure environment:**
Edit `.env` file:
- For local deployment: Set `USE_OPENAI=false`
- For Streamlit Cloud: Set `USE_OPENAI=true` and add your `OPENAI_API_KEY`

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install Tesseract OCR (for scanned PDF support):**

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH or set TESSDATA_PREFIX environment variable

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

3. **Optional: Install PyMuPDF for better PDF processing:**
```bash
pip install PyMuPDF
```

## Usage

1. **Start the application:**
```bash
streamlit run app.py
```

2. **Upload documents:**
   - Use the sidebar to upload PDF, LaTeX, or text files
   - Click "Process Documents" to add them to the knowledge base

3. **Query the system:**
   - Enter natural language questions in the main interface
   - Adjust the number of retrieved results with the slider
   - View generated answers with source citations

## Project Structure

```
poc-voc/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── data/                          # Generated data directory
│   ├── faiss_index.faiss         # FAISS vector index (auto-generated)
│   └── faiss_index_metadata.pkl  # Document metadata (auto-generated)
├── ingestion/
│   ├── __init__.py
│   └── document_processor.py      # Text extraction and chunking
├── index/
│   ├── __init__.py
│   └── vector_store.py           # FAISS index management
├── retrieval/
│   ├── __init__.py
│   └── rag_retriever.py          # Context retrieval and formatting
└── llm/
    ├── __init__.py
    └── local_llm.py              # Local LLM wrapper
```

## Configuration

### Model Selection

The system uses lightweight models by default for broader compatibility:

- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers)
- **LLM**: `microsoft/DialoGPT-medium` (fallback to `distilgpt2`)

To use larger models (requires more RAM/GPU):

1. Edit `llm/local_llm.py` and change the model name:
```python
def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
```

2. Edit `index/vector_store.py` for different embedding models:
```python
def __init__(self, index_path: str = "data/faiss_index", model_name: str = "all-mpnet-base-v2"):
```

### Chunking Parameters

Modify chunk size and overlap in `ingestion/document_processor.py`:
```python
def __init__(self):
    self.chunk_size = 500  # tokens per chunk
    self.overlap = 100     # token overlap between chunks
```

## Troubleshooting

### Common Issues

1. **Tesseract not found:**
   - Ensure Tesseract is installed and in PATH
   - Set environment variable: `TESSDATA_PREFIX=/path/to/tessdata`

2. **CUDA out of memory:**
   - The system automatically falls back to CPU
   - Use smaller models or reduce batch sizes

3. **Model download issues:**
   - Models are downloaded automatically on first run
   - Ensure internet connection for initial setup
   - Models are cached locally after first download

4. **PDF processing errors:**
   - The system tries multiple extraction methods
   - For complex PDFs, manual preprocessing may be needed

### Performance Tips

- **GPU acceleration**: Install PyTorch with CUDA support for faster inference
- **Memory usage**: Close other applications when processing large documents
- **Storage**: The FAISS index grows with the number of documents

## Deployment

### Local Deployment
1. Set `USE_OPENAI=false` in `.env`
2. Run: `streamlit run app.py`
3. Models will be downloaded automatically on first run

### Streamlit Cloud Deployment
1. Set `USE_OPENAI=true` in Streamlit Cloud secrets
2. Add `OPENAI_API_KEY` to secrets
3. Deploy from GitHub repository
4. No local model downloads required

## Security Notes

- **Local mode**: All processing happens locally - no external API calls
- **OpenAI mode**: Uses OpenAI API for embeddings and LLM
- Uploaded documents are processed in temporary files and cleaned up
- API keys should be stored in environment variables or Streamlit secrets

## License

This is a proof-of-concept application for internal use.