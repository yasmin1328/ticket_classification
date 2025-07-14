import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import logging
import pickle
import os
from tqdm import tqdm
from config import Config

class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = None
        self.embeddings_cache = {}
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Encode a list of texts into embeddings"""
        if self.model is None:
            self.load_model()
        
        try:
            self.logger.info(f"Encoding {len(texts)} texts")
            
            # Filter empty texts
            non_empty_texts = [text if text.strip() else "empty" for text in texts]
            
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding texts: {str(e)}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Encode a single text into embedding"""
        if self.model is None:
            self.load_model()
        
        # Check cache first
        cache_key = hash(text)
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        try:
            if not text.strip():
                text = "empty"
            
            embedding = self.model.encode([text], normalize_embeddings=True)[0]
            
            # Cache the result if cache is not full
            if len(self.embeddings_cache) < Config.CACHE_SIZE:
                self.embeddings_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error encoding single text: {str(e)}")
            raise
    
    def process_dataset_embeddings(self, records: List[Dict], save_path: str = None) -> Tuple[np.ndarray, List[Dict]]:
        """Process embeddings for the entire dataset"""
        self.logger.info(f"Processing embeddings for {len(records)} records")
        
        # Extract descriptions
        descriptions = [record['description'] for record in records]
        
        # Generate embeddings
        embeddings = self.encode_texts(descriptions, show_progress=True)
        
        # Prepare metadata
        metadata = []
        for i, record in enumerate(records):
            meta = {
                'index': i,
                'id': record['id'],
                'incident_id': record['incident_id'],
                'description': record['description'],
                'category1': record['category1'],
                'category2': record['category2'],
                'language': record['language'],
                'has_arabic': record['has_arabic']
            }
            metadata.append(meta)
        
        # Save if path provided
        if save_path:
            self.save_embeddings(embeddings, metadata, save_path)
        
        return embeddings, metadata
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], base_path: str):
        """Save embeddings and metadata to disk"""
        try:
            # Save embeddings as numpy array
            embeddings_path = f"{base_path}_embeddings.npy"
            np.save(embeddings_path, embeddings)
            
            # Save metadata as JSON
            metadata_path = f"{base_path}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved embeddings to {embeddings_path}")
            self.logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, base_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """Load embeddings and metadata from disk"""
        try:
            # Load embeddings
            embeddings_path = f"{base_path}_embeddings.npy"
            embeddings = np.load(embeddings_path)
            
            # Load metadata
            metadata_path = f"{base_path}_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Loaded embeddings shape: {embeddings.shape}")
            self.logger.info(f"Loaded {len(metadata)} metadata records")
            
            return embeddings, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            self.load_model()
        
        info = {
            'model_name': self.model_name,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
        }
        
        return info

# Utility functions
def setup_embedding_logging():
    """Setup logging for embedding service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{Config.LOGS_DIR}/embedding_service.log'),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    setup_embedding_logging()
    
    # Test the embedding service
    service = EmbeddingService()
    
    # Test single text encoding
    test_text = "مشكلة في تسجيل الدخول"
    embedding = service.encode_single_text(test_text)
    print(f"Test embedding shape: {embedding.shape}")
    
    # Test model info
    info = service.get_model_info()
    print("Model info:", info)
