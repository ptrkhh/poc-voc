import io
import os
import re
from typing import List, Dict, Any

import PyPDF2
import pytesseract
from PIL import Image
from pdfminer.high_level import extract_text

from config import Config


class DocumentProcessor:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.overlap = Config.CHUNK_OVERLAP

    def extract_text_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract text from various file formats"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif ext == '.tex':
            return self._extract_from_tex(file_path)
        elif ext == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _extract_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF, with OCR fallback for scanned PDFs"""
        try:
            # Try regular PDF text extraction first
            text = extract_text(file_path)
            if len(text.strip()) > 100:  # Sufficient text found
                return {
                    'text': text,
                    'filename': os.path.basename(file_path),
                    'method': 'direct'
                }
        except:
            pass

        # Fallback to OCR for scanned PDFs
        return self._ocr_pdf(file_path)

    def _ocr_pdf(self, file_path: str) -> Dict[str, Any]:
        """OCR processing for scanned PDFs"""
        try:
            import fitz  # PyMuPDF for better PDF to image conversion

            doc = fitz.open(file_path)
            text_parts = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")

                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))

                # OCR with Tesseract
                page_text = pytesseract.image_to_string(image)
                text_parts.append(page_text)

            doc.close()

            return {
                'text': '\n\n'.join(text_parts),
                'filename': os.path.basename(file_path),
                'method': 'ocr'
            }
        except ImportError:
            # PyMuPDF not available, use fallback
            return self._fallback_ocr(file_path)
        except Exception as e:
            # Other errors, use fallback
            return self._fallback_ocr(file_path)

    def _fallback_ocr(self, file_path: str) -> Dict[str, Any]:
        """Simplified OCR fallback"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = [page.extract_text() for page in reader.pages if page.extract_text().strip()]

                return {
                    'text': '\n\n'.join(text_parts) if text_parts else "Could not extract text from PDF",
                    'filename': os.path.basename(file_path),
                    'method': 'fallback' if text_parts else 'failed'
                }
        except Exception as e:
            return {
                'text': f"Error processing PDF: {str(e)}",
                'filename': os.path.basename(file_path),
                'method': 'error'
            }

    def _extract_from_tex(self, file_path: str) -> Dict[str, Any]:
        """Extract text from LaTeX files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Remove LaTeX commands and formatting
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', content)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'%.*', '', text)  # Remove comments
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        return {
            'text': text.strip(),
            'filename': os.path.basename(file_path),
            'method': 'latex'
        }

    def _extract_from_txt(self, file_path: str) -> Dict[str, Any]:
        """Extract text from plain text files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        return {
            'text': text,
            'filename': os.path.basename(file_path),
            'method': 'text'
        }

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': len(chunks),
                'start_word': i,
                'end_word': i + len(chunk_words)
            })

            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })

        return chunks
