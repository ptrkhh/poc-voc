#!/usr/bin/env python3
"""
Test script to verify RAG system setup and dependencies
Run this before starting the main application
"""

import importlib
import sys


def test_imports():
    """Test all required imports"""
    required_packages = [
        'streamlit',
        'faiss',
        'sentence_transformers',
        'transformers',
        'torch',
        'PyPDF2',
        'pdfminer',
        'pytesseract',
        'PIL',
        'numpy'
    ]

    print("Testing package imports...")
    failed_imports = []

    for package in required_packages:
        try:
            if package == 'PIL':
                from PIL import Image
            elif package == 'pdfminer':
                from pdfminer.high_level import extract_text
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)

    return failed_imports


def test_models():
    """Test model loading"""
    print("\nTesting model loading...")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model loaded")
    except Exception as e:
        print(f"❌ Sentence transformer: {e}")
        return False

    try:
        from transformers import pipeline
        pipe = pipeline("text-generation", model="distilgpt2")
        print("✅ Text generation model loaded")
    except Exception as e:
        print(f"❌ Text generation model: {e}")
        return False

    return True


def test_tesseract():
    """Test Tesseract OCR"""
    print("\nTesting Tesseract OCR...")

    try:
        import pytesseract
        from PIL import Image
        import numpy as np

        test_image = Image.fromarray(np.ones((50, 100), dtype=np.uint8) * 255)
        pytesseract.image_to_string(test_image)
        print("✅ Tesseract OCR working")
        return True
    except Exception as e:
        print(f"❌ Tesseract OCR: {e}")
        print("   Install Tesseract and add to PATH for OCR support")
        return False


def main():
    print("RAG System Setup Test")
    print("=" * 50)

    # Test imports
    failed_imports = test_imports()

    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False

    # Test models
    if not test_models():
        print("\n❌ Model loading failed")
        return False

    # Test Tesseract (optional)
    test_tesseract()

    print("\n" + "=" * 50)
    print("✅ Setup test completed!")
    print("Run the application with: streamlit run app.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
