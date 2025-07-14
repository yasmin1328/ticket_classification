import faiss
import numpy as np
import json
import pickle
import os
from typing import List, Dict, Tuple, Optional
import logging
from config import Config

class FAISSimilaritySearch:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.metadata = None
        self.dimension = Config.EMBEDDING_DIMENSION
        
    def build_index(self, embeddings: np.ndarray, metadata: List[Dict], 
                   index_type: str = 'IVF') -> None:
        """Build FAISS index from embeddings"""
        try:
            self.logger.info(f"Building FAISS index with {len(embeddings)} vectors")
            
            # Ensure embeddings are float32
            embeddings = embeddings.astype('float32')
            self.dimension = embeddings.shape[1]
            
            if index_type == 'IVF' and len(embeddings) > 1000:
                # Use IVF index for larger datasets
                nlist = min(int(np.sqrt(len(embeddings))), 1000)
                quantizer = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                
                # Train the index
                self.logger.info("Training IVF index...")
                self.index.train(embeddings)
                
            else:
                # Use flat index for smaller datasets
                self.logger.info("Using flat index...")
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
            
            # Add vectors to index
            self.index.add(embeddings)
            self.metadata = metadata
            
            self.logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")
            
        except Exception as e:
            self.logger.error(f"Error building index: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = None) -> List[Dict]:
        """Search for similar vectors"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if k is None:
            k = Config.TOP_K_SIMILAR
        
        try:
            # Ensure query is the right shape and type
            query_embedding = query_embedding.astype('float32')
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    result = {
                        'rank': i + 1,
                        'similarity_score': float(score),
                        'index': int(idx),
                        'metadata': self.metadata[idx] if self.metadata else None
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            raise
    
    def search_with_threshold(self, query_embedding: np.ndarray, 
                            threshold: float = None, k: int = None) -> List[Dict]:
        """Search with similarity threshold filtering"""
        if threshold is None:
            threshold = Config.SIMILARITY_THRESHOLD
        
        results = self.search(query_embedding, k)
        
        # Filter by threshold
        filtered_results = [r for r in results if r['similarity_score'] >= threshold]
        
        self.logger.info(f"Found {len(filtered_results)} results above threshold {threshold}")
        return filtered_results
    
    def get_category_examples(self, category1: str, category2: str = None, 
                            max_examples: int = 10) -> List[Dict]:
        """Get examples from specific categories"""
        if self.metadata is None:
            return []
        
        examples = []
        for meta in self.metadata:
            if meta['category1'] == category1:
                if category2 is None or meta['category2'] == category2:
                    examples.append(meta)
                    if len(examples) >= max_examples:
                        break
        
        return examples
    
    def get_all_categories(self) -> Dict[str, List[str]]:
        """Get all unique categories from the metadata"""
        if self.metadata is None:
            return {}
        
        categories = {}
        for meta in self.metadata:
            cat1 = meta['category1']
            cat2 = meta['category2']
            
            if cat1 not in categories:
                categories[cat1] = []
            
            if cat2 and cat2 not in categories[cat1]:
                categories[cat1].append(cat2)
        
        return categories
    
    def save_index(self, base_path: str = None) -> None:
        """Save FAISS index and metadata to disk"""
        if base_path is None:
            base_path = Config.FAISS_INDEX_PATH.replace('.bin', '')
        
        try:
            # Save FAISS index
            index_path = f"{base_path}.bin"
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = f"{base_path}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Index saved to {index_path}")
            self.logger.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self, base_path: str = None) -> None:
        """Load FAISS index and metadata from disk"""
        if base_path is None:
            base_path = Config.FAISS_INDEX_PATH.replace('.bin', '')
        
        try:
            # Load FAISS index
            index_path = f"{base_path}.bin"
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = f"{base_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                self.logger.warning(f"Metadata file not found: {metadata_path}")
                self.metadata = None
            
            self.logger.info(f"Index loaded: {self.index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            raise
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        if self.index is None:
            return {'status': 'not_built'}
        
        stats = {
            'status': 'ready',
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'index_type': type(self.index).__name__,
            'is_trained': getattr(self.index, 'is_trained', True),
            'has_metadata': self.metadata is not None,
            'metadata_count': len(self.metadata) if self.metadata else 0
        }
        
        if self.metadata:
            # Category statistics
            categories = {}
            languages = {}
            for meta in self.metadata:
                cat1 = meta.get('category1', 'unknown')
                lang = meta.get('language', 'unknown')
                
                categories[cat1] = categories.get(cat1, 0) + 1
                languages[lang] = languages.get(lang, 0) + 1
            
            stats['categories_count'] = len(categories)
            stats['top_categories'] = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]
            stats['language_distribution'] = languages
        
        return stats

class SimilarityAnalyzer:
    """Helper class for analyzing similarity search results"""
    
    @staticmethod
    def analyze_results(results: List[Dict], query_text: str = None) -> Dict:
        """Analyze similarity search results"""
        if not results:
            return {
                'total_results': 0,
                'has_similar': False,
                'confidence': 0.0,
                'recommendation': 'create_new_category'
            }
        
        analysis = {
            'total_results': len(results),
            'has_similar': len(results) > 0,
            'max_similarity': max(r['similarity_score'] for r in results),
            'min_similarity': min(r['similarity_score'] for r in results),
            'avg_similarity': np.mean([r['similarity_score'] for r in results]),
            'confidence': max(r['similarity_score'] for r in results) if results else 0.0
        }
        
        # Category analysis
        categories = {}
        for result in results:
            if result['metadata']:
                cat1 = result['metadata']['category1']
                categories[cat1] = categories.get(cat1, 0) + 1
        
        analysis['category_distribution'] = categories
        
        # Determine recommendation
        max_sim = analysis['max_similarity']
        if max_sim >= Config.CONFIDENCE_THRESHOLD:
            analysis['recommendation'] = 'classify_existing'
            analysis['suggested_category'] = max(categories.items(), key=lambda x: x[1])[0] if categories else None
        elif max_sim >= Config.SIMILARITY_THRESHOLD:
            analysis['recommendation'] = 'review_similar'
        else:
            analysis['recommendation'] = 'create_new_category'
        
        return analysis

# Setup logging
def setup_search_logging():
    """Setup logging for similarity search"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{Config.LOGS_DIR}/similarity_search.log'),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    setup_search_logging()
    
    # Test the similarity search
    search_engine = FAISSimilaritySearch()
    
    # Create dummy data for testing
    dummy_embeddings = np.random.random((100, 384)).astype('float32')
    dummy_metadata = [{'id': str(i), 'category1': f'cat_{i%5}', 'description': f'desc_{i}'} for i in range(100)]
    
    # Build index
    search_engine.build_index(dummy_embeddings, dummy_metadata)
    
    # Test search
    query = np.random.random(384).astype('float32')
    results = search_engine.search(query, k=5)
    
    print(f"Found {len(results)} results")
    print("Index stats:", search_engine.get_index_stats())
