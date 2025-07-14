#!/usr/bin/env python3
"""
System Testing and Validation Script
====================================

This script validates the incident classification system by:
- Testing data loading and preprocessing
- Validating embedding generation
- Testing FAISS index operations
- Validating LLM integration
- Running end-to-end classification tests
- Performance benchmarking
"""

import time
import json
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict

from config import Config, validate_config
from data_processor import DataProcessor
from embedding_service import EmbeddingService
from similarity_search import FAISSimilaritySearch
from llm_service import LLMService
from incident_classifier import IncidentClassifier

class SystemTester:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        
    def run_all_tests(self) -> Dict:
        """Run all system tests"""
        print("🧪 Starting System Tests")
        print("=" * 50)
        
        test_suite = [
            ("Configuration Validation", self.test_configuration),
            ("Data Processing", self.test_data_processing),
            ("Embedding Service", self.test_embedding_service),
            ("Similarity Search", self.test_similarity_search),
            ("LLM Service", self.test_llm_service),
            ("End-to-End Classification", self.test_classification),
            ("Performance Benchmark", self.test_performance),
            ("Error Handling", self.test_error_handling)
        ]
        
        results = {
            'total_tests': len(test_suite),
            'passed': 0,
            'failed': 0,
            'test_details': [],
            'timestamp': time.time()
        }
        
        for test_name, test_func in test_suite:
            print(f"\n🔍 Running: {test_name}")
            
            try:
                start_time = time.time()
                test_result = test_func()
                execution_time = time.time() - start_time
                
                test_detail = {
                    'test_name': test_name,
                    'status': 'PASSED' if test_result['success'] else 'FAILED',
                    'execution_time': execution_time,
                    'details': test_result.get('details', ''),
                    'error': test_result.get('error', '')
                }
                
                results['test_details'].append(test_detail)
                
                if test_result['success']:
                    results['passed'] += 1
                    print(f"   ✅ PASSED ({execution_time:.2f}s)")
                else:
                    results['failed'] += 1
                    print(f"   ❌ FAILED: {test_result.get('error', 'Unknown error')}")
                    
                if test_result.get('details'):
                    print(f"   📝 {test_result['details']}")
                    
            except Exception as e:
                results['failed'] += 1
                print(f"   ❌ FAILED: {str(e)}")
                
                test_detail = {
                    'test_name': test_name,
                    'status': 'ERROR',
                    'execution_time': 0,
                    'details': '',
                    'error': str(e)
                }
                results['test_details'].append(test_detail)
        
        # Print summary
        print(f"\n📊 Test Summary")
        print("=" * 50)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']} ✅")
        print(f"Failed: {results['failed']} ❌")
        print(f"Success Rate: {(results['passed']/results['total_tests']*100):.1f}%")
        
        # Save results
        self.save_test_results(results)
        
        return results
    
    def test_configuration(self) -> Dict:
        """Test configuration validation"""
        try:
            validate_config()
            
            # Check required paths
            checks = {
                'dataset_exists': Path(Config.DATASET_PATH).exists(),
                'output_dir_exists': Path(Config.OUTPUT_DIR).exists(),
                'logs_dir_exists': Path(Config.LOGS_DIR).exists(),
                'has_api_key': bool(Config.OPENAI_API_KEY or Config.ANTHROPIC_API_KEY)
            }
            
            all_passed = all(checks.values())
            
            return {
                'success': all_passed,
                'details': f"Config checks: {checks}",
                'error': '' if all_passed else 'Some configuration checks failed'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_data_processing(self) -> Dict:
        """Test data loading and preprocessing"""
        try:
            processor = DataProcessor()
            
            # Load dataset
            data = processor.load_dataset()
            if len(data) == 0:
                return {'success': False, 'error': 'Dataset is empty'}
            
            # Test preprocessing
            processed = processor.preprocess_data()
            if len(processed) == 0:
                return {'success': False, 'error': 'Preprocessing removed all data'}
            
            # Test sample
            sample = processor.sample_data(100)
            
            # Test category distribution
            distribution = processor.get_categories_distribution()
            
            return {
                'success': True,
                'details': f"Loaded {len(data)} records, processed {len(processed)}, sample {len(sample)}"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_embedding_service(self) -> Dict:
        """Test embedding generation"""
        try:
            service = EmbeddingService()
            
            # Test single text encoding
            test_texts = [
                "مشكلة في تسجيل الدخول",
                "Login problem with system",
                "البرنامج لا يعمل بشكل صحيح"
            ]
            
            for text in test_texts:
                embedding = service.encode_single_text(text)
                if embedding.shape[0] != Config.EMBEDDING_DIMENSION:
                    return {
                        'success': False, 
                        'error': f'Wrong embedding dimension: {embedding.shape[0]}'
                    }
            
            # Test batch encoding
            embeddings = service.encode_texts(test_texts)
            if embeddings.shape != (len(test_texts), Config.EMBEDDING_DIMENSION):
                return {
                    'success': False,
                    'error': f'Wrong batch embedding shape: {embeddings.shape}'
                }
            
            # Test model info
            model_info = service.get_model_info()
            
            return {
                'success': True,
                'details': f"Generated embeddings shape: {embeddings.shape}, Model: {model_info['model_name']}"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_similarity_search(self) -> Dict:
        """Test FAISS similarity search"""
        try:
            search_engine = FAISSimilaritySearch()
            
            # Create test data
            test_embeddings = np.random.random((100, Config.EMBEDDING_DIMENSION)).astype('float32')
            test_metadata = [
                {
                    'id': str(i),
                    'description': f'Test description {i}',
                    'category1': f'Category {i % 5}',
                    'category2': f'Subcategory {i % 10}'
                }
                for i in range(100)
            ]
            
            # Build index
            search_engine.build_index(test_embeddings, test_metadata)
            
            # Test search
            query = np.random.random(Config.EMBEDDING_DIMENSION).astype('float32')
            results = search_engine.search(query, k=5)
            
            if len(results) != 5:
                return {'success': False, 'error': f'Expected 5 results, got {len(results)}'}
            
            # Test threshold search
            threshold_results = search_engine.search_with_threshold(query, threshold=0.5)
            
            # Test save/load
            search_engine.save_index('test_index')
            
            new_search_engine = FAISSimilaritySearch()
            new_search_engine.load_index('test_index')
            
            # Clean up
            Path('test_index.bin').unlink(missing_ok=True)
            Path('test_index_metadata.json').unlink(missing_ok=True)
            
            return {
                'success': True,
                'details': f"Index with {len(test_embeddings)} vectors, search returned {len(results)} results"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_llm_service(self) -> Dict:
        """Test LLM service integration"""
        try:
            service = LLMService()
            
            # Test classification (mock similar examples)
            test_description = "مشكلة في تسجيل الدخول"
            similar_examples = [
                {
                    'similarity_score': 0.8,
                    'metadata': {
                        'description': 'مشكلة في كلمة المرور',
                        'category1': 'الأمان',
                        'category2': 'تسجيل الدخول'
                    }
                }
            ]
            
            result = service.classify_incident(test_description, similar_examples)
            
            # Validate result structure
            required_fields = ['subdirectory1', 'subdirectory2', 'confidence', 'reasoning']
            for field in required_fields:
                if field not in result:
                    return {'success': False, 'error': f'Missing field: {field}'}
            
            # Test summarization
            summary = service.summarize_incident(test_description)
            if not summary or len(summary) == 0:
                return {'success': False, 'error': 'Summarization failed'}
            
            return {
                'success': True,
                'details': f"Classification confidence: {result['confidence']:.2f}, Summary length: {len(summary)}"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_classification(self) -> Dict:
        """Test end-to-end classification"""
        try:
            # Use a small sample for testing
            classifier = IncidentClassifier(use_existing_index=False)
            
            # Initialize with small sample
            if not classifier.initialize(sample_size=100):
                return {'success': False, 'error': 'Failed to initialize classifier'}
            
            # Test single classification
            test_incidents = [
                "مشكلة في تسجيل الدخول إلى النظام",
                "البريد الإلكتروني لا يعمل",
                "Password reset not working"
            ]
            
            results = []
            for description in test_incidents:
                result = classifier.classify_incident(description)
                results.append(result)
                
                # Validate result structure
                if 'incident_id' not in result:
                    return {'success': False, 'error': 'Missing incident_id in result'}
                
                if result['processing_time'] > 10:  # Should be fast
                    return {'success': False, 'error': f'Processing too slow: {result["processing_time"]}s'}
            
            # Test batch classification
            batch_input = [
                {'description': desc, 'incident_id': f'TEST_{i}'}
                for i, desc in enumerate(test_incidents)
            ]
            
            batch_results = classifier.classify_batch(batch_input)
            
            if len(batch_results) != len(batch_input):
                return {'success': False, 'error': 'Batch processing count mismatch'}
            
            return {
                'success': True,
                'details': f"Classified {len(results)} single + {len(batch_results)} batch incidents"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_performance(self) -> Dict:
        """Test system performance"""
        try:
            # Test embedding speed
            service = EmbeddingService()
            test_texts = ["Test text"] * 10
            
            start_time = time.time()
            embeddings = service.encode_texts(test_texts, show_progress=False)
            embedding_time = time.time() - start_time
            
            # Test search speed
            search_engine = FAISSimilaritySearch()
            test_embeddings = np.random.random((1000, Config.EMBEDDING_DIMENSION)).astype('float32')
            test_metadata = [{'id': str(i)} for i in range(1000)]
            
            search_engine.build_index(test_embeddings, test_metadata)
            
            query = np.random.random(Config.EMBEDDING_DIMENSION).astype('float32')
            
            start_time = time.time()
            results = search_engine.search(query, k=5)
            search_time = time.time() - start_time
            
            performance_metrics = {
                'embedding_time_per_text': embedding_time / len(test_texts),
                'search_time': search_time,
                'embeddings_per_second': len(test_texts) / embedding_time,
                'searches_per_second': 1 / search_time
            }
            
            # Check if performance meets requirements
            if performance_metrics['embedding_time_per_text'] > 1.0:  # Should be < 1s per text
                return {'success': False, 'error': 'Embedding too slow'}
            
            if performance_metrics['search_time'] > 0.1:  # Should be < 100ms
                return {'success': False, 'error': 'Search too slow'}
            
            return {
                'success': True,
                'details': f"Embedding: {performance_metrics['embeddings_per_second']:.1f}/s, Search: {search_time*1000:.1f}ms"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_error_handling(self) -> Dict:
        """Test error handling and edge cases"""
        try:
            processor = DataProcessor()
            service = EmbeddingService()
            
            # Test empty input
            cleaned = processor.clean_text("")
            if cleaned != "":
                return {'success': False, 'error': 'Empty text cleaning failed'}
            
            # Test malformed input
            cleaned = processor.clean_text(None)
            if cleaned != "":
                return {'success': False, 'error': 'None input cleaning failed'}
            
            # Test very long input
            long_text = "a" * 10000
            embedding = service.encode_single_text(long_text)
            if embedding.shape[0] != Config.EMBEDDING_DIMENSION:
                return {'success': False, 'error': 'Long text embedding failed'}
            
            # Test mixed language
            mixed_text = "Hello مرحبا world عالم"
            embedding = service.encode_single_text(mixed_text)
            language = processor.detect_language(mixed_text)
            
            return {
                'success': True,
                'details': f"Error handling tests passed, mixed language detected as: {language}"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def save_test_results(self, results: Dict) -> None:
        """Save test results to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{Config.OUTPUT_DIR}/test_results_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 Test results saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save test results: {str(e)}")

def benchmark_system():
    """Run performance benchmarks"""
    print("\n🏁 Running Performance Benchmarks")
    print("=" * 50)
    
    try:
        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100]
        test_descriptions = [
            "مشكلة في النظام",
            "Login issue",
            "البريد الإلكتروني معطل",
            "Network connectivity problem"
        ] * 25  # 100 total
        
        service = EmbeddingService()
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            for i in range(0, len(test_descriptions), batch_size):
                batch = test_descriptions[i:i+batch_size]
                embeddings = service.encode_texts(batch, show_progress=False)
            
            total_time = time.time() - start_time
            throughput = len(test_descriptions) / total_time
            
            print(f"Batch size {batch_size:3d}: {throughput:6.1f} texts/sec ({total_time:.2f}s total)")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {str(e)}")

def main():
    """Main testing function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{Config.LOGS_DIR}/test_system.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create output directory
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    tester = SystemTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Run benchmarks if tests passed
    if results['failed'] == 0:
        benchmark_system()
    else:
        print(f"\n⚠️ Skipping benchmarks due to {results['failed']} failed tests")
    
    # Exit with appropriate code
    exit_code = 0 if results['failed'] == 0 else 1
    print(f"\n🏁 Testing complete. Exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    exit(main())
