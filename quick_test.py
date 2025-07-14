#!/usr/bin/env python3
"""
Quick Installation Test
======================
Tests if all components are working correctly
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ faiss imported successfully")
    except ImportError as e:
        print(f"❌ faiss import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"❌ sentence-transformers import failed: {e}")
        return False
    
    try:
        import openai
        print("✅ openai imported successfully")
    except ImportError as e:
        print(f"❌ openai import failed: {e}")
        print("   Note: OpenAI is optional if you have other LLM APIs")
    
    return True

def test_file_structure():
    """Test if required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'config.py',
        'data_processor.py',
        'embedding_service.py',
        'similarity_search.py',
        'llm_service.py',
        'incident_classifier.py',
        'main_app.py'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_dataset():
    """Test if dataset file exists"""
    print("\n📊 Testing dataset...")
    
    dataset_file = 'Thiqa_Incidents_Example.xlsx'
    if Path(dataset_file).exists():
        print(f"✅ Dataset file found: {dataset_file}")
        try:
            import pandas as pd
            df = pd.read_excel(dataset_file)
            print(f"✅ Dataset loaded successfully: {len(df)} records")
            print(f"✅ Columns: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print(f"❌ Dataset file missing: {dataset_file}")
        print("   Please place the Excel file in the project root directory")
        return False

def test_embedding_model():
    """Test embedding model download and usage"""
    print("\n🤖 Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Downloading embedding model (this may take a moment)...")
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Test with Arabic and English
        test_texts = [
            "مشكلة في تسجيل الدخول",
            "Login problem with system"
        ]
        
        embeddings = model.encode(test_texts)
        print(f"✅ Embedding model working: {embeddings.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

def test_basic_classification():
    """Test basic classification without full system"""
    print("\n🎯 Testing basic classification setup...")
    
    try:
        # Test data processing
        from data_processor import DataProcessor
        processor = DataProcessor()
        
        # Test text cleaning
        test_text = "مشكلة في تسجيل الدخول   !@#"
        cleaned = processor.clean_text(test_text)
        print(f"✅ Text cleaning: '{test_text}' → '{cleaned}'")
        
        # Test language detection
        language = processor.detect_language(cleaned)
        print(f"✅ Language detection: {language}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic classification test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Quick Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Dataset", test_dataset),
        ("Embedding Model", test_embedding_model),
        ("Basic Classification", test_basic_classification)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready.")
        print("\nNext steps:")
        print("1. Add your API key to the .env file")
        print("2. Run: python main.py info")
        print("3. Run: python main.py build-index --sample-size 1000")
        print("4. Test: python main.py classify \"مشكلة في النظام\"")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
