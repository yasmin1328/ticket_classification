#!/usr/bin/env python3
"""
Enhanced Incident Classification and Summarization System
=========================================================

This is the main application for the Dubai Health AI-powered incident classification system.
Updated to work with the enhanced incident classifier that properly uses FAISS similarity results.

Usage:
    python main_app.py --help
    python main_app.py classify "مشكلة في تسجيل الدخول"
    python main_app.py batch-classify incidents.json
    python main_app.py build-index --sample-size 5000
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config, validate_config
from data_processor import DataProcessor

# Import the enhanced incident classifier
try:
    from incident_classifier import EnhancedIncidentClassifier as IncidentClassifier
except ImportError:
    # Fallback to the original if enhanced version not available
    from incident_classifier import IncidentClassifier

def setup_logging():
    """Setup comprehensive logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{Config.LOGS_DIR}/main_app.log'),
            logging.StreamHandler()
        ]
    )

def setup_directories():
    """Create necessary directories"""
    directories = [Config.OUTPUT_DIR, Config.LOGS_DIR, 'data']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def classify_single_incident(classifier: IncidentClassifier, description: str, incident_id: str = None) -> None:
    """Classify a single incident and print enhanced results"""
    print(f"\n🔍 Classifying incident...")
    print(f"Description: {description}\n")
    
    try:
        result = classifier.classify_incident(description, incident_id)
        
        # Print formatted results with enhanced information
        print("=" * 80)
        print("ENHANCED CLASSIFICATION RESULTS")
        print("=" * 80)
        
        print(f"📋 Incident ID: {result['incident_id']}")
        print(f"⏱️  Processing Time: {result['processing_time']:.3f} seconds")
        print(f"🎯 Classification Status: {result['classification_status']}")
        print(f"✅ Category Verified: {'Yes' if result['category_verified'] else 'No'}")
        print(f"🏷️  Main Category: {result['subdirectory1']}")
        print(f"🏷️  Sub Category: {result['subdirectory2']}")
        print(f"📊 Confidence Score: {result['confidence']:.3f}")
        print(f"🌐 Language Detected: {result['language_detected']}")
        print(f"🎯 Domain Relevance: {result['domain_relevance']:.3f}")
        print(f"🔧 Category Source: {result['category_source']}")
        
        print(f"\n📄 SUMMARY:")
        print(f"{result['summary']}")
        
        print(f"\n🤔 REASONING:")
        print(f"{result['reasoning']}")
        
        if result.get('category_creation_reasoning'):
            print(f"\n🆕 CATEGORY CREATION REASONING:")
            print(f"{result['category_creation_reasoning']}")
        
        print(f"\n📈 ENHANCED SIMILARITY ANALYSIS:")
        sim_analysis = result['similarity_analysis']
        print(f"   Similar incidents found: {sim_analysis['similar_incidents_found']}")
        print(f"   Max similarity score: {sim_analysis['max_similarity_score']:.3f}")
        print(f"   Average similarity score: {sim_analysis['avg_similarity_score']:.3f}")
        print(f"   Recommendation: {sim_analysis['recommendation']}")
        
        # Enhanced similarity details if available
        if 'category_consensus' in sim_analysis:
            consensus = sim_analysis['category_consensus']
            print(f"   Category consensus strength: {consensus.get('consensus_strength', 0):.3f}")
            if 'supporting_examples' in sim_analysis:
                print(f"   Supporting examples: {sim_analysis['supporting_examples']}")
        
        if result.get('error_message'):
            print(f"\n⚠️  ERROR MESSAGE:")
            print(f"{result['error_message']}")
        
        print("=" * 80)
        
        # Save individual result
        timestamp = result['timestamp'].replace(':', '-').replace('.', '-')
        filename = f"{Config.OUTPUT_DIR}/enhanced_single_classification_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Enhanced results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error during classification: {str(e)}")
        sys.exit(1)

def classify_batch_incidents(classifier: IncidentClassifier, input_file: str) -> None:
    """Classify multiple incidents from a JSON file with enhanced processing"""
    try:
        print(f"\n📁 Loading incidents from: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            incidents = json.load(f)
        
        if not isinstance(incidents, list):
            print("❌ Input file must contain a list of incidents")
            return
        
        print(f"🔄 Processing {len(incidents)} incidents with enhanced system...")
        
        results = classifier.classify_batch(incidents)
        
        # Save results using enhanced save method
        output_file = classifier.save_results(results)
        
        # Print enhanced summary statistics
        print("\n" + "=" * 80)
        print("ENHANCED BATCH CLASSIFICATION SUMMARY")
        print("=" * 80)
        
        status_counts = {}
        confidence_scores = []
        category_sources = {}
        
        for result in results:
            status = result['classification_status']
            status_counts[status] = status_counts.get(status, 0) + 1
            confidence_scores.append(result['confidence'])
            
            source = result.get('category_source', 'unknown')
            category_sources[source] = category_sources.get(source, 0) + 1
        
        print(f"📊 Total incidents processed: {len(results)}")
        print(f"📁 Enhanced results saved to: {output_file}")
        
        print(f"\n📈 STATUS DISTRIBUTION:")
        for status, count in status_counts.items():
            percentage = (count / len(results)) * 100
            print(f"   {status}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔧 CATEGORY SOURCE DISTRIBUTION:")
        for source, count in category_sources.items():
            percentage = (count / len(results)) * 100
            print(f"   {source}: {count} ({percentage:.1f}%)")
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            max_confidence = max(confidence_scores)
            min_confidence = min(confidence_scores)
            high_confidence_count = sum(1 for score in confidence_scores if score >= Config.CONFIDENCE_THRESHOLD)
            
            print(f"\n🎯 CONFIDENCE STATISTICS:")
            print(f"   Average: {avg_confidence:.3f}")
            print(f"   Maximum: {max_confidence:.3f}")
            print(f"   Minimum: {min_confidence:.3f}")
            print(f"   High confidence (≥{Config.CONFIDENCE_THRESHOLD}): {high_confidence_count} ({high_confidence_count/len(results)*100:.1f}%)")
        
        # Show category insights if available
        if hasattr(classifier, 'get_category_insights'):
            try:
                insights = classifier.get_category_insights()
                if insights:
                    print(f"\n📋 SYSTEM INSIGHTS:")
                    print(f"   Total main categories available: {insights.get('total_main_categories', 0)}")
                    
                    popular = insights.get('popular_combinations', {})
                    if popular:
                        print(f"   Top category combinations in dataset:")
                        for combo, count in list(popular.items())[:3]:
                            if '|' in combo:
                                cat1, cat2 = combo.split('|', 1)
                                print(f"      • {cat1} → {cat2}: {count} incidents")
            except Exception as e:
                print(f"   (Could not retrieve category insights: {e})")
        
        print("=" * 80)
        
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during batch classification: {str(e)}")
        sys.exit(1)

def build_index(sample_size: int = None) -> None:
    """Build FAISS index from the dataset with enhanced configuration"""
    print(f"\n🏗️  Building enhanced FAISS index from dataset...")
    
    if sample_size:
        print(f"📊 Using sample size: {sample_size}")
    else:
        print(f"📊 Using full dataset")
    
    try:
        classifier = IncidentClassifier(use_existing_index=False)
        success = classifier.initialize(sample_size=sample_size)
        
        if success:
            print("✅ Enhanced index built successfully!")
            
            # Print enhanced index statistics
            if hasattr(classifier, 'get_system_stats'):
                stats = classifier.get_system_stats()
                index_stats = stats.get('index_stats', {})
                
                print(f"\n📈 ENHANCED INDEX STATISTICS:")
                print(f"   Total vectors: {index_stats.get('total_vectors', 'N/A')}")
                print(f"   Dimension: {index_stats.get('dimension', 'N/A')}")
                print(f"   Index type: {index_stats.get('index_type', 'N/A')}")
                print(f"   Categories count: {index_stats.get('categories_count', 'N/A')}")
                
                if 'language_distribution' in index_stats:
                    print(f"   Language distribution: {index_stats['language_distribution']}")
                
                if 'top_categories' in index_stats:
                    print(f"   Top categories:")
                    for cat, count in index_stats['top_categories'][:5]:
                        print(f"      • {cat}: {count} incidents")
            
            # Show category insights if available
            if hasattr(classifier, 'get_category_insights'):
                try:
                    insights = classifier.get_category_insights()
                    if insights:
                        print(f"\n📋 CATEGORY INSIGHTS:")
                        print(f"   Total main categories: {insights.get('total_main_categories', 0)}")
                        
                        popular = insights.get('popular_combinations', {})
                        if popular:
                            print(f"   Most common category combinations:")
                            for combo, count in list(popular.items())[:5]:
                                if '|' in combo:
                                    cat1, cat2 = combo.split('|', 1)
                                    print(f"      • {cat1} → {cat2}: {count} incidents")
                except Exception as e:
                    print(f"   (Could not retrieve category insights: {e})")
        else:
            print("❌ Failed to build enhanced index")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error building enhanced index: {str(e)}")
        sys.exit(1)

def system_info() -> None:
    """Display enhanced system information and health check"""
    print("\n" + "=" * 80)
    print("ENHANCED SYSTEM INFORMATION")
    print("=" * 80)
    
    try:
        # Enhanced configuration info
        from config import get_config_summary
        config_summary = get_config_summary()
        
        print(f"📊 CONFIGURATION:")
        for key, value in config_summary.items():
            print(f"   {key}: {value}")
        
        # Check if files exist
        print(f"\n📂 FILE STATUS:")
        dataset_exists = os.path.exists(Config.DATASET_PATH)
        index_exists = os.path.exists(Config.FAISS_INDEX_PATH)
        
        print(f"   Dataset file: {'✅ Found' if dataset_exists else '❌ Missing'}")
        print(f"   FAISS index: {'✅ Found' if index_exists else '❌ Missing'}")
        
        if dataset_exists:
            # Quick dataset info
            processor = DataProcessor()
            data = processor.load_dataset()
            processed_data = processor.preprocess_data()
            
            print(f"   Dataset records: {len(data)}")
            print(f"   After preprocessing: {len(processed_data)}")
            
            # Show dataset distribution
            distribution = processor.get_categories_distribution()
            print(f"   Unique main categories: {len(distribution.get('subcategory1_count', {}))}")
            print(f"   Unique sub categories: {len(distribution.get('subcategory2_count', {}))}")
            print(f"   Language distribution: {distribution.get('language_distribution', {})}")
        
        # Test enhanced system if index exists
        if index_exists:
            print(f"\n🧪 TESTING ENHANCED SYSTEM:")
            try:
                classifier = IncidentClassifier(use_existing_index=True)
                if classifier.initialize():
                    print(f"   ✅ Enhanced classifier loads successfully")
                    
                    # Quick test classification
                    test_result = classifier.classify_incident("اختبار النظام المحسن")
                    print(f"   ✅ Test classification works")
                    print(f"   Test result: {test_result['subdirectory1']} → {test_result['subdirectory2']}")
                    print(f"   Test confidence: {test_result['confidence']:.3f}")
                    print(f"   Test source: {test_result['category_source']}")
                else:
                    print(f"   ❌ Enhanced classifier failed to initialize")
            except Exception as e:
                print(f"   ❌ Enhanced system test failed: {e}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error getting enhanced system info: {str(e)}")

def create_sample_batch_file() -> None:
    """Create a sample batch input file for testing enhanced system"""
    sample_incidents = [
        {
            "incident_id": "ENHANCED_SAMPLE_001",
            "description": "مشكلة في تسجيل الدخول إلى النظام"
        },
        {
            "incident_id": "ENHANCED_SAMPLE_002",
            "description": "لا أستطيع الوصول إلى البريد الإلكتروني"
        },
        {
            "incident_id": "ENHANCED_SAMPLE_003",
            "description": "البرنامج يتوقف بشكل مفاجئ عند فتح ملف كبير"
        },
        {
            "incident_id": "ENHANCED_SAMPLE_004",
            "description": "Internet connection is very slow"
        },
        {
            "incident_id": "ENHANCED_SAMPLE_005",
            "description": "Password reset is not working"
        },
        {
            "incident_id": "ENHANCED_SAMPLE_006",
            "description": "يواجهه العميل في بيانات الشهاده خاطى"
        },
        {
            "incident_id": "ENHANCED_SAMPLE_007",
            "description": "مشكلة في طباعة الوثائق من النظام"
        }
    ]
    
    filename = "enhanced_sample_incidents.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sample_incidents, f, ensure_ascii=False, indent=2)
    
    print(f"📝 Enhanced sample batch file created: {filename}")

def main():
    """Main application entry point for enhanced system"""
    parser = argparse.ArgumentParser(
        description="Enhanced Incident Classification and Summarization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  %(prog)s classify "مشكلة في تسجيل الدخول"
  %(prog)s classify "Login issue" --incident-id "INC_001"
  %(prog)s batch-classify enhanced_sample_incidents.json
  %(prog)s build-index --sample-size 5000
  %(prog)s info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Classify single incident
    classify_parser = subparsers.add_parser('classify', help='Classify a single incident with enhanced system')
    classify_parser.add_argument('description', help='Incident description')
    classify_parser.add_argument('--incident-id', help='Optional incident ID')
    
    # Batch classify
    batch_parser = subparsers.add_parser('batch-classify', help='Classify multiple incidents with enhanced processing')
    batch_parser.add_argument('input_file', help='JSON file containing list of incidents')
    
    # Build index
    build_parser = subparsers.add_parser('build-index', help='Build enhanced FAISS index from dataset')
    build_parser.add_argument('--sample-size', type=int, help='Use only a sample of the dataset (recommended: 5000+)')
    
    # System info
    subparsers.add_parser('info', help='Display enhanced system information')
    
    # Create sample
    subparsers.add_parser('create-sample', help='Create sample batch input file for enhanced system')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup directories and logging
    setup_directories()
    setup_logging()
    
    print("🚀 Enhanced Incident Classification System")
    print("=" * 50)
    print("✨ Now with proper FAISS similarity-based category extraction!")
    
    if args.command == 'info':
        system_info()
        return
    
    if args.command == 'create-sample':
        create_sample_batch_file()
        return
    
    if args.command == 'build-index':
        build_index(args.sample_size)
        return
    
    # For classification commands, initialize the enhanced classifier
    try:
        print("🔧 Initializing enhanced classifier...")
        classifier = IncidentClassifier(use_existing_index=True)
        
        if not classifier.initialize():
            print("❌ Failed to initialize enhanced classifier. Try building the index first:")
            print("   python main_app.py build-index --sample-size 5000")
            sys.exit(1)
        
        print("✅ Enhanced classifier initialized successfully!")
        
        if args.command == 'classify':
            classify_single_incident(classifier, args.description, args.incident_id)
        
        elif args.command == 'batch-classify':
            classify_batch_incidents(classifier, args.input_file)
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Make sure you have built the index: python main_app.py build-index")
        print("   2. Check that the dataset file exists")
        print("   3. Verify your .env configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()