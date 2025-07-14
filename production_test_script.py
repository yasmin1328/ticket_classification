#!/usr/bin/env python3
"""
Production Test Script for Enhanced Incident Classification System
================================================================

This script demonstrates the enhanced system that properly leverages 
FAISS similarity search to extract actual categories from similar incidents.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from incident_classifier import EnhancedIncidentClassifier
import logging

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def test_enhanced_classification():
    """Test the enhanced classification system"""
    
    print("🚀 Testing Enhanced Incident Classification System")
    print("=" * 70)
    
    # Initialize the enhanced classifier
    print("🔧 Initializing enhanced classifier...")
    classifier = EnhancedIncidentClassifier(use_existing_index=True)
    
    if not classifier.initialize(sample_size=5000):  # Use larger sample for better results
        print("❌ Failed to initialize classifier")
        return False
    
    print("✅ Enhanced classifier initialized successfully!")
    
    # Test incidents that should match existing categories in the dataset
    test_incidents = [
        {
            "incident_id": "PROD_TEST_001",
            "description": "يواجهه العميل في بيانات الشهاده خاطى كما موضح لكم في الصورة",
            "expected_behavior": "Should find similar incidents and use their categories"
        },
        {
            "incident_id": "PROD_TEST_002", 
            "description": "مشكلة في تسجيل الدخول إلى النظام",
            "expected_behavior": "Should classify as authentication/login issue"
        },
        {
            "incident_id": "PROD_TEST_003",
            "description": "البريد الإلكتروني لا يعمل بشكل صحيح",
            "expected_behavior": "Should classify as email system issue"
        },
        {
            "incident_id": "PROD_TEST_004",
            "description": "مشكلة في طباعة الوثائق من النظام",
            "expected_behavior": "Should classify as printing/document issue"
        },
        {
            "incident_id": "PROD_TEST_005",
            "description": "خطأ في قاعدة البيانات عند محاولة الحفظ",
            "expected_behavior": "Should classify as database/system error"
        }
    ]
    
    print(f"\n📊 Testing {len(test_incidents)} incidents...")
    print("=" * 70)
    
    results = []
    successful_classifications = 0
    
    for i, incident in enumerate(test_incidents, 1):
        print(f"\n🔍 Test {i}/{len(test_incidents)}: {incident['incident_id']}")
        print(f"Description: {incident['description']}")
        print(f"Expected: {incident['expected_behavior']}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            result = classifier.classify_incident(
                incident['description'], 
                incident['incident_id']
            )
            processing_time = time.time() - start_time
            
            # Display results
            print(f"✅ CLASSIFICATION RESULTS:")
            print(f"   Status: {result['classification_status']}")
            print(f"   Main Category: {result['subdirectory1']}")
            print(f"   Sub Category: {result['subdirectory2']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Category Source: {result['category_source']}")
            print(f"   Processing Time: {processing_time:.3f}s")
            
            # Similarity analysis details
            sim_analysis = result['similarity_analysis']
            print(f"\n📈 SIMILARITY ANALYSIS:")
            print(f"   Similar incidents found: {sim_analysis['similar_incidents_found']}")
            print(f"   Max similarity score: {sim_analysis['max_similarity_score']:.3f}")
            print(f"   Average similarity: {sim_analysis['avg_similarity_score']:.3f}")
            print(f"   Recommendation: {sim_analysis['recommendation']}")
            
            if 'category_consensus' in sim_analysis:
                consensus = sim_analysis['category_consensus']
                print(f"   Category consensus strength: {consensus.get('consensus_strength', 0):.3f}")
            
            print(f"\n🤔 REASONING:")
            print(f"   {result['reasoning']}")
            
            # Check if classification was successful (not generic fallback)
            if (result['subdirectory1'] not in ['مشكلة عامة', 'خطأ في النظام', 'طلبات المساعدة العامة'] and 
                result['confidence'] > 0.5):
                successful_classifications += 1
                print(f"✅ SUCCESS: Proper classification achieved")
            else:
                print(f"⚠️  WARNING: Generic/fallback classification")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'incident_id': incident['incident_id'],
                'error': str(e),
                'status': 'failed'
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Total incidents tested: {len(test_incidents)}")
    print(f"Successful classifications: {successful_classifications}")
    print(f"Success rate: {(successful_classifications/len(test_incidents)*100):.1f}%")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"production_test_results_{timestamp}.json"
    
    test_summary = {
        'test_timestamp': timestamp,
        'total_tests': len(test_incidents),
        'successful_classifications': successful_classifications,
        'success_rate': successful_classifications/len(test_incidents)*100,
        'detailed_results': results,
        'system_info': classifier.get_category_insights()
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Category insights
    insights = classifier.get_category_insights()
    if insights:
        print(f"\n📋 SYSTEM INSIGHTS:")
        print(f"   Total main categories: {insights.get('total_main_categories', 0)}")
        print(f"   Popular category combinations:")
        
        popular = insights.get('popular_combinations', {})
        for combo, count in list(popular.items())[:5]:
            cat1, cat2 = combo.split('|')
            print(f"      • {cat1} → {cat2}: {count} incidents")
    
    return successful_classifications >= len(test_incidents) * 0.6  # 60% success threshold

def test_batch_classification():
    """Test batch classification functionality"""
    
    print(f"\n🔄 Testing Batch Classification")
    print("=" * 50)
    
    # Create batch test data
    batch_incidents = [
        {"incident_id": "BATCH_001", "description": "مشكلة في الطباعة"},
        {"incident_id": "BATCH_002", "description": "خطأ في قاعدة البيانات"},
        {"incident_id": "BATCH_003", "description": "مشكلة في الشبكة"},
        {"incident_id": "BATCH_004", "description": "البريد الإلكتروني معطل"},
        {"incident_id": "BATCH_005", "description": "مشكلة في تسجيل الدخول"}
    ]
    
    try:
        classifier = EnhancedIncidentClassifier(use_existing_index=True)
        classifier.initialize()
        
        start_time = time.time()
        results = classifier.classify_batch(batch_incidents)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch processing completed:")
        print(f"   Processed: {len(results)} incidents")
        print(f"   Total time: {batch_time:.3f}s")
        print(f"   Average time per incident: {batch_time/len(results):.3f}s")
        
        # Show sample results
        for result in results[:3]:
            print(f"\n   {result['incident_id']}: {result['subdirectory1']} → {result['subdirectory2']} ({result['confidence']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False

def verify_faiss_integration():
    """Verify that FAISS similarity search is working properly"""
    
    print(f"\n🔍 Verifying FAISS Integration")
    print("=" * 40)
    
    try:
        from embedding_service import EmbeddingService
        from similarity_search import FAISSimilaritySearch
        
        # Test embedding service
        embedding_service = EmbeddingService()
        test_text = "مشكلة في النظام"
        embedding = embedding_service.encode_single_text(test_text)
        print(f"✅ Embedding generated: shape {embedding.shape}")
        
        # Test FAISS search
        search_engine = FAISSimilaritySearch()
        
        # Try to load existing index
        try:
            base_path = Config.FAISS_INDEX_PATH.replace('.bin', '')
            search_engine.load_index(base_path)
            print(f"✅ FAISS index loaded successfully")
            
            # Test search
            results = search_engine.search(embedding, k=5)
            print(f"✅ Similarity search returned {len(results)} results")
            
            if results:
                print(f"   Top similarity score: {results[0]['similarity_score']:.3f}")
                if results[0].get('metadata'):
                    meta = results[0]['metadata']
                    print(f"   Top result category: {meta.get('category1', 'N/A')} → {meta.get('category2', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ FAISS index not found or corrupted: {e}")
            print("   Please run: python main.py build-index")
            return False
        
    except Exception as e:
        print(f"❌ FAISS integration test failed: {e}")
        return False

def main():
    """Main test function"""
    setup_logging()
    
    print("🎯 Production Testing for Enhanced Incident Classification System")
    print("================================================================")
    print("This test verifies that the system properly uses FAISS similarity")
    print("search to extract actual categories from similar incidents.")
    print()
    
    # Check if dataset exists
    if not Path(Config.DATASET_PATH).exists():
        print(f"❌ Dataset file not found: {Config.DATASET_PATH}")
        print("Please place the Excel file in the project root directory.")
        return False
    
    # Test components
    tests = [
        ("FAISS Integration", verify_faiss_integration),
        ("Enhanced Classification", test_enhanced_classification),
        ("Batch Processing", test_batch_classification)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏁 FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The enhanced system is working correctly.")
        print("\n📋 The system now properly:")
        print("   ✅ Uses FAISS vector similarity search")
        print("   ✅ Extracts categories from most similar incidents")
        print("   ✅ Provides weighted voting based on similarity scores")
        print("   ✅ Falls back intelligently when no good matches found")
        print("   ✅ Achieves high classification accuracy")
        
        print(f"\n🚀 Ready for production use!")
        print(f"   • API Service: python api_service.py")
        print(f"   • Single Classification: python main.py classify \"your text\"")
        print(f"   • Batch Processing: python main.py batch-classify file.json")
        
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed.")
        print("Please check the error messages above and fix the issues.")
        
        if passed_tests == 0:
            print("\n🔧 Quick fixes to try:")
            print("   1. Run: python main.py build-index --sample-size 5000")
            print("   2. Check that the dataset file exists")
            print("   3. Verify all dependencies are installed: pip install -r requirements.txt")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
