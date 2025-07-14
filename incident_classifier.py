import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
from collections import Counter

from config import Config, validate_config
from data_processor import DataProcessor
from embedding_service import EmbeddingService
from similarity_search import FAISSimilaritySearch, SimilarityAnalyzer

# Import the enhanced LLM service
try:
    from llm_service import EnhancedLLMService as LLMService
except ImportError:
    from llm_service import LLMService

class EnhancedIncidentClassifier:
    def __init__(self, use_existing_index: bool = True):
        """Initialize the enhanced incident classifier"""
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        validate_config()
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.embedding_service = EmbeddingService()
        self.similarity_search = FAISSimilaritySearch()
        self.llm_service = LLMService()
        
        self.use_existing_index = use_existing_index
        self.is_initialized = False
        
        # Cache for categories extracted from dataset
        self.category_cache = {}
        
    def initialize(self, sample_size: Optional[int] = None) -> bool:
        """Initialize the classifier with training data"""
        try:
            self.logger.info("Initializing enhanced incident classifier...")
            
            # Try to load existing index first
            if self.use_existing_index:
                try:
                    base_path = Config.FAISS_INDEX_PATH.replace('.bin', '')
                    self.similarity_search.load_index(base_path)
                    self.logger.info("Loaded existing FAISS index")
                    self.is_initialized = True
                    self._build_category_cache()
                    return True
                except Exception as e:
                    self.logger.warning(f"Could not load existing index: {e}")
            
            # Build new index
            self.logger.info("Building new index from dataset...")
            
            # Load and process data
            self.data_processor.load_dataset()
            processed_data = self.data_processor.preprocess_data()
            
            # Use sample if specified
            if sample_size and len(processed_data) > sample_size:
                processed_data = self.data_processor.sample_data(sample_size)
                self.logger.info(f"Using sample of {len(processed_data)} records")
            
            # Prepare data for embedding
            records = self.data_processor.prepare_for_embedding()
            
            # Generate embeddings
            embeddings, metadata = self.embedding_service.process_dataset_embeddings(
                records, save_path='data/embeddings'
            )
            
            # Build FAISS index
            self.similarity_search.build_index(embeddings, metadata)
            
            # Save index
            base_path = Config.FAISS_INDEX_PATH.replace('.bin', '')
            self.similarity_search.save_index(base_path)
            
            # Build category cache
            self._build_category_cache()
            
            self.is_initialized = True
            self.logger.info("Enhanced classifier initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing classifier: {str(e)}")
            return False
    
    def _build_category_cache(self):
        """Build cache of categories from the metadata"""
        if not self.similarity_search.metadata:
            return
        
        self.category_cache = {
            'all_categories': {},
            'category_examples': {},
            'popular_combinations': Counter()
        }
        
        for meta in self.similarity_search.metadata:
            cat1 = meta.get('category1', '').strip()
            cat2 = meta.get('category2', '').strip()
            
            if cat1 and cat2:
                # Track all categories
                if cat1 not in self.category_cache['all_categories']:
                    self.category_cache['all_categories'][cat1] = set()
                self.category_cache['all_categories'][cat1].add(cat2)
                
                # Track popular combinations
                combo = f"{cat1}|{cat2}"
                self.category_cache['popular_combinations'][combo] += 1
                
                # Store examples for each combination
                if combo not in self.category_cache['category_examples']:
                    self.category_cache['category_examples'][combo] = []
                
                if len(self.category_cache['category_examples'][combo]) < 5:
                    self.category_cache['category_examples'][combo].append({
                        'description': meta.get('description', ''),
                        'incident_id': meta.get('incident_id', '')
                    })
        
        # Convert sets to lists for JSON serialization
        for cat1 in self.category_cache['all_categories']:
            self.category_cache['all_categories'][cat1] = list(self.category_cache['all_categories'][cat1])
        
        self.logger.info(f"Built category cache with {len(self.category_cache['all_categories'])} main categories")
    
    def classify_incident(self, 
                         description: str, 
                         incident_id: str = None,
                         language: str = 'ar') -> Dict:
        """Enhanced incident classification with proper FAISS integration"""
        
        start_time = time.time()
        
        if not self.is_initialized:
            raise ValueError("Classifier not initialized. Call initialize() first.")
        
        try:
            # Generate unique incident ID if not provided
            if incident_id is None:
                incident_id = f"INCIDENT_{int(time.time() * 1000)}"
            
            # Clean and preprocess description
            cleaned_description = self.data_processor.clean_text(description)
            detected_language = self.data_processor.detect_language(cleaned_description)
            
            # Generate embedding for the description
            query_embedding = self.embedding_service.encode_single_text(cleaned_description)
            
            # Search for similar incidents with enhanced parameters
            similar_results = self.similarity_search.search_with_threshold(
                query_embedding, 
                threshold=Config.SIMILARITY_THRESHOLD * 0.8,  # Lower threshold for more candidates
                k=Config.TOP_K_SIMILAR * 2  # Get more candidates
            )
            
            # Enhanced similarity analysis
            similarity_analysis = self._enhanced_similarity_analysis(similar_results, cleaned_description)
            
            # Classification using similarity + LLM
            classification_result = self._enhanced_classification(
                cleaned_description, similar_results, detected_language, similarity_analysis
            )
            
            # Generate summary
            summary = self.llm_service.summarize_incident(cleaned_description, detected_language)
            
            # Create final result
            processing_time = time.time() - start_time
            
            result = {
                "incident_id": incident_id,
                "confidence": classification_result.get('confidence', 0.0),
                "category_verified": classification_result.get('confidence', 0.0) >= Config.CONFIDENCE_THRESHOLD,
                "subdirectory1": classification_result.get('subdirectory1', ''),
                "subdirectory2": classification_result.get('subdirectory2', ''),
                "original_description": description,
                "processed_description": cleaned_description,
                "summary": summary,
                "classification_status": self._determine_enhanced_status(similarity_analysis, classification_result),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "reasoning": classification_result.get('reasoning', ''),
                "category_source": classification_result.get('category_source', 'unknown'),
                "category_creation_reasoning": classification_result.get('category_creation_reasoning', ''),
                "domain_relevance": classification_result.get('domain_relevance', 0.0),
                "language_detected": detected_language,
                "similarity_analysis": {
                    "similar_incidents_found": len(similar_results),
                    "max_similarity_score": similarity_analysis.get('max_similarity', 0.0),
                    "avg_similarity_score": similarity_analysis.get('avg_similarity', 0.0),
                    "recommendation": similarity_analysis.get('recommendation', 'unknown'),
                    "category_consensus": similarity_analysis.get('category_consensus', {}),
                    "supporting_examples": classification_result.get('supporting_examples', 0)
                },
                "error_message": classification_result.get('error_message', '')
            }
            
            self.logger.info(f"Enhanced classification for {incident_id}: {result['subdirectory1']} -> {result['subdirectory2']} (confidence: {result['confidence']:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced classification: {str(e)}")
            return self._create_error_result(incident_id, description, str(e), time.time() - start_time)
    
    def _enhanced_similarity_analysis(self, similar_results: List[Dict], query_text: str) -> Dict:
        """Enhanced similarity analysis with category consensus"""
        
        if not similar_results:
            return {
                'total_results': 0,
                'has_similar': False,
                'confidence': 0.0,
                'recommendation': 'create_new_category',
                'category_consensus': {}
            }
        
        # Basic statistics
        similarities = [r['similarity_score'] for r in similar_results]
        analysis = {
            'total_results': len(similar_results),
            'has_similar': len(similar_results) > 0,
            'max_similarity': max(similarities),
            'min_similarity': min(similarities),
            'avg_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities)
        }
        
        # Category consensus analysis
        category_votes = Counter()
        category_details = {}
        total_weighted_score = 0
        
        for result in similar_results:
            if 'metadata' not in result or not result['metadata']:
                continue
                
            meta = result['metadata']
            cat1 = meta.get('category1', '').strip()
            cat2 = meta.get('category2', '').strip()
            
            if cat1 and cat2:
                combo = f"{cat1}|{cat2}"
                weight = result['similarity_score']
                
                category_votes[combo] += weight
                total_weighted_score += weight
                
                if combo not in category_details:
                    category_details[combo] = {
                        'cat1': cat1,
                        'cat2': cat2,
                        'count': 0,
                        'total_similarity': 0,
                        'max_similarity': 0,
                        'examples': []
                    }
                
                category_details[combo]['count'] += 1
                category_details[combo]['total_similarity'] += weight
                category_details[combo]['max_similarity'] = max(
                    category_details[combo]['max_similarity'], weight
                )
                
                if len(category_details[combo]['examples']) < 3:
                    category_details[combo]['examples'].append({
                        'description': meta.get('description', ''),
                        'similarity': weight
                    })
        
        # Find consensus
        if category_votes:
            top_category = category_votes.most_common(1)[0]
            consensus_category = top_category[0]
            consensus_weight = top_category[1]
            consensus_strength = consensus_weight / total_weighted_score if total_weighted_score > 0 else 0
            
            analysis['category_consensus'] = {
                'top_category': consensus_category,
                'consensus_strength': consensus_strength,
                'vote_weight': consensus_weight,
                'details': category_details.get(consensus_category, {}),
                'all_categories': dict(category_votes.most_common())
            }
        
        # Enhanced recommendation logic
        max_sim = analysis['max_similarity']
        consensus_strength = analysis.get('category_consensus', {}).get('consensus_strength', 0)
        
        if max_sim >= 0.85 and consensus_strength >= 0.6:
            analysis['recommendation'] = 'classify_existing_high_confidence'
        elif max_sim >= 0.75 and consensus_strength >= 0.5:
            analysis['recommendation'] = 'classify_existing'
        elif max_sim >= 0.6:
            analysis['recommendation'] = 'review_similar'
        else:
            analysis['recommendation'] = 'create_new_category'
        
        analysis['confidence'] = min(max_sim * consensus_strength * 1.2, 1.0) if consensus_strength > 0 else max_sim * 0.7
        
        return analysis
    
    def _enhanced_classification(self, 
                               description: str, 
                               similar_results: List[Dict], 
                               language: str,
                               similarity_analysis: Dict) -> Dict:
        """Enhanced classification using similarity + LLM"""
        
        recommendation = similarity_analysis.get('recommendation', 'create_new_category')
        
        # Use similarity-based classification as primary method
        classification_result = self.llm_service.classify_incident(description, similar_results, language)
        
        # Add similarity context
        classification_result['similarity_context'] = similarity_analysis.get('category_consensus', {})
        
        # Boost confidence if we have strong consensus
        consensus_strength = similarity_analysis.get('category_consensus', {}).get('consensus_strength', 0)
        if consensus_strength >= 0.7:
            classification_result['confidence'] = min(classification_result.get('confidence', 0) + 0.1, 1.0)
            classification_result['reasoning'] += f" (إجماع قوي من الحوادث المشابهة: {consensus_strength:.2f})"
        
        return classification_result
    
    def _determine_enhanced_status(self, similarity_analysis: Dict, classification_result: Dict) -> str:
        """Determine enhanced classification status"""
        
        confidence = classification_result.get('confidence', 0.0)
        recommendation = similarity_analysis.get('recommendation', 'unknown')
        consensus_strength = similarity_analysis.get('category_consensus', {}).get('consensus_strength', 0)
        
        if classification_result.get('error_message'):
            return 'error'
        elif confidence >= 0.85 and consensus_strength >= 0.7:
            return 'classified_high_confidence'
        elif confidence >= Config.CONFIDENCE_THRESHOLD:
            return 'classified'
        elif recommendation == 'review_similar':
            return 'needs_review'
        elif recommendation.startswith('create_new'):
            return 'new_category'
        else:
            return 'uncertain'
    
    def classify_batch(self, incidents: List[Dict]) -> List[Dict]:
        """Enhanced batch classification with progress tracking"""
        results = []
        
        self.logger.info(f"Starting enhanced batch classification of {len(incidents)} incidents")
        
        # Pre-compute all embeddings for efficiency
        descriptions = [incident.get('description', '') for incident in incidents]
        cleaned_descriptions = [self.data_processor.clean_text(desc) for desc in descriptions]
        
        # Batch embed for efficiency
        try:
            batch_embeddings = self.embedding_service.encode_texts(cleaned_descriptions, show_progress=True)
        except Exception as e:
            self.logger.error(f"Batch embedding failed: {e}")
            # Fallback to individual processing
            return self._classify_batch_individual(incidents)
        
        for i, (incident, cleaned_desc, embedding) in enumerate(zip(incidents, cleaned_descriptions, batch_embeddings)):
            try:
                incident_id = incident.get('incident_id', f"BATCH_{int(time.time() * 1000)}_{i}")
                
                # Use pre-computed embedding
                similar_results = self.similarity_search.search_with_threshold(
                    embedding, 
                    threshold=Config.SIMILARITY_THRESHOLD * 0.8,
                    k=Config.TOP_K_SIMILAR * 2
                )
                
                # Enhanced analysis and classification
                similarity_analysis = self._enhanced_similarity_analysis(similar_results, cleaned_desc)
                classification_result = self._enhanced_classification(
                    cleaned_desc, similar_results, 'ar', similarity_analysis
                )
                
                # Create result
                result = {
                    "incident_id": incident_id,
                    "confidence": classification_result.get('confidence', 0.0),
                    "category_verified": classification_result.get('confidence', 0.0) >= Config.CONFIDENCE_THRESHOLD,
                    "subdirectory1": classification_result.get('subdirectory1', ''),
                    "subdirectory2": classification_result.get('subdirectory2', ''),
                    "original_description": incident.get('description', ''),
                    "processed_description": cleaned_desc,
                    "classification_status": self._determine_enhanced_status(similarity_analysis, classification_result),
                    "reasoning": classification_result.get('reasoning', ''),
                    "category_source": classification_result.get('category_source', 'enhanced'),
                    "similarity_analysis": {
                        "similar_incidents_found": len(similar_results),
                        "max_similarity_score": similarity_analysis.get('max_similarity', 0.0),
                        "avg_similarity_score": similarity_analysis.get('avg_similarity', 0.0),
                        "recommendation": similarity_analysis.get('recommendation', 'unknown')
                    }
                }
                
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Enhanced batch processing: {i + 1}/{len(incidents)} incidents")
                    
            except Exception as e:
                self.logger.error(f"Error processing incident {i}: {str(e)}")
                results.append(self._create_error_result(
                    incident.get('incident_id'), 
                    incident.get('description', ''), 
                    str(e), 
                    0.0
                ))
        
        self.logger.info(f"Enhanced batch classification complete. Processed {len(results)} incidents")
        return results
    
    def _classify_batch_individual(self, incidents: List[Dict]) -> List[Dict]:
        """Fallback individual classification for batch processing"""
        results = []
        for incident in incidents:
            try:
                result = self.classify_incident(
                    incident.get('description', ''),
                    incident.get('incident_id')
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Individual classification failed: {e}")
                results.append(self._create_error_result(
                    incident.get('incident_id'), 
                    incident.get('description', ''), 
                    str(e), 
                    0.0
                ))
        return results
    
    def _create_error_result(self, incident_id: str, description: str, error_msg: str, processing_time: float) -> Dict:
        """Create enhanced error result"""
        return {
            "incident_id": incident_id or f"ERROR_{int(time.time() * 1000)}",
            "confidence": 0.0,
            "category_verified": False,
            "subdirectory1": "خطأ في النظام",
            "subdirectory2": "فشل في التصنيف",
            "original_description": description,
            "processed_description": description,
            "summary": "حدث خطأ أثناء معالجة الحادثة",
            "classification_status": "error",
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "reasoning": f"فشل في تصنيف الحادثة: {error_msg}",
            "category_source": "error",
            "domain_relevance": 0.0,
            "language_detected": "unknown",
            "similarity_analysis": {
                "similar_incidents_found": 0,
                "max_similarity_score": 0.0,
                "avg_similarity_score": 0.0,
                "recommendation": "error"
            },
            "error_message": error_msg
        }
    
    def get_category_insights(self) -> Dict:
        """Get insights about the categories in the system"""
        if not self.category_cache:
            return {}
        
        insights = {
            'total_main_categories': len(self.category_cache['all_categories']),
            'popular_combinations': dict(self.category_cache['popular_combinations'].most_common(10)),
            'category_distribution': self.category_cache['all_categories'],
            'examples_per_category': {
                combo: len(examples) 
                for combo, examples in self.category_cache['category_examples'].items()
            }
        }
        
        return insights
    
    def save_results(self, results: List[Dict], filename: str = None) -> str:
        """Save enhanced classification results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{Config.OUTPUT_DIR}/enhanced_classification_results_{timestamp}.json"
        
        try:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata to results
            metadata = {
                'generation_time': datetime.now().isoformat(),
                'total_incidents': len(results),
                'classifier_version': 'enhanced',
                'category_insights': self.get_category_insights()
            }
            
            output = {
                'metadata': metadata,
                'results': results
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Enhanced results saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

# Alias for backward compatibility
IncidentClassifier = EnhancedIncidentClassifier