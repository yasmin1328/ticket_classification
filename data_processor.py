import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from langdetect import detect, DetectorFactory
import logging
from config import Config

# Set seed for consistent language detection
DetectorFactory.seed = 0

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data = None
        self.processed_data = None
        
    def load_dataset(self, file_path: str = None) -> pd.DataFrame:
        """Load the Excel dataset"""
        if file_path is None:
            file_path = Config.DATASET_PATH
            
        try:
            self.logger.info(f"Loading dataset from {file_path}")
            self.data = pd.read_excel(file_path)
            self.logger.info(f"Loaded {len(self.data)} records")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text - FIXED Arabic handling"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Arabic and English
        # Keep Arabic characters in their original form - don't apply reshaping here
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', ' ', text)
        
        # Remove extra spaces again after cleaning
        text = re.sub(r'\s+', ' ', text)
        
        # DO NOT apply arabic_reshaper here - it corrupts the text for processing
        # Arabic reshaping should only be done for display purposes, not data processing
        
        return text.strip()
    
    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text for better processing"""
        if not text:
            return text
            
        # Normalize Arabic characters
        # Replace different forms of alef with standard alef
        text = re.sub(r'[آأإا]', 'ا', text)
        
        # Replace different forms of yeh with standard yeh
        text = re.sub(r'[يى]', 'ي', text)
        
        # Replace different forms of teh marbuta
        text = re.sub(r'[ة]', 'ه', text)
        
        # Remove diacritics (optional - might want to keep for some applications)
        text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
        
        return text
    
    def contains_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        return bool(arabic_pattern.search(text))
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            if len(text.strip()) < 3:
                return 'unknown'
            
            if self.contains_arabic(text):
                return 'ar'
            
            lang = detect(text)
            return lang if lang in Config.SUPPORTED_LANGUAGES else 'en'
        except:
            return 'ar' if self.contains_arabic(text) else 'en'
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the loaded data with proper Arabic handling"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_dataset() first.")
        
        self.logger.info("Starting data preprocessing with fixed Arabic handling")
        
        # Create a copy for processing
        processed = self.data.copy()
        
        # Clean column names
        processed.columns = processed.columns.str.strip()
        
        # Handle missing values
        processed = processed.fillna('')
        
        # Clean text fields with proper Arabic handling
        if 'Description' in processed.columns:
            self.logger.info("Processing descriptions...")
            processed['Description'] = processed['Description'].apply(self.clean_text)
            # Apply Arabic normalization for better matching
            processed['Description_normalized'] = processed['Description'].apply(self.normalize_arabic_text)
            processed['description_language'] = processed['Description'].apply(self.detect_language)
            processed['description_length'] = processed['Description'].str.len()
        
        # Clean category fields without corrupting Arabic text
        if 'Subcategory_Thiqah' in processed.columns:
            self.logger.info("Processing Subcategory_Thiqah...")
            processed['Subcategory_Thiqah'] = processed['Subcategory_Thiqah'].apply(self.clean_text)
            # Keep original Arabic form for categories
            
        if 'Subcategory2_Thiqah' in processed.columns:
            self.logger.info("Processing Subcategory2_Thiqah...")
            processed['Subcategory2_Thiqah'] = processed['Subcategory2_Thiqah'].apply(self.clean_text)
        
        # Filter out empty descriptions
        processed = processed[processed['Description'].str.len() > 5]
        
        # Limit description length
        processed['Description'] = processed['Description'].str[:Config.MAX_DESCRIPTION_LENGTH]
        
        # Create unique identifier
        processed['unique_id'] = processed.index.astype(str)
        
        # Add metadata
        processed['has_arabic'] = processed['Description'].apply(self.contains_arabic)
        
        self.processed_data = processed
        self.logger.info(f"Preprocessing complete. {len(processed)} records remain after filtering")
        
        # Log some samples to verify Arabic text is preserved correctly
        arabic_samples = processed[processed['has_arabic']].head(3)
        for idx, row in arabic_samples.iterrows():
            self.logger.info(f"Arabic sample: {row['Description'][:100]}...")
            self.logger.info(f"Category: {row.get('Subcategory_Thiqah', 'N/A')} -> {row.get('Subcategory2_Thiqah', 'N/A')}")
        
        return processed
    
    def get_categories_distribution(self) -> Dict:
        """Get distribution of categories"""
        if self.processed_data is None:
            return {}
        
        distribution = {
            'total_records': len(self.processed_data),
            'subcategory1_count': self.processed_data['Subcategory_Thiqah'].value_counts().to_dict(),
            'subcategory2_count': self.processed_data['Subcategory2_Thiqah'].value_counts().to_dict(),
            'language_distribution': self.processed_data['description_language'].value_counts().to_dict(),
            'arabic_content': self.processed_data['has_arabic'].value_counts().to_dict()
        }
        
        return distribution
    
    def sample_data(self, n: int = 1000) -> pd.DataFrame:
        """Get a sample of the data for testing"""
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        if len(self.processed_data) <= n:
            return self.processed_data
        
        return self.processed_data.sample(n=n, random_state=42)
    
    def prepare_for_embedding(self) -> List[Dict]:
        """Prepare data for embedding generation"""
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        records = []
        for idx, row in self.processed_data.iterrows():
            record = {
                'id': row.get('unique_id', str(idx)),
                'incident_id': row.get('Incident', ''),
                'description': row['Description'],  # Use original cleaned text, not reshaped
                'category1': row.get('Subcategory_Thiqah', ''),
                'category2': row.get('Subcategory2_Thiqah', ''),
                'language': row.get('description_language', 'ar'),
                'has_arabic': row.get('has_arabic', False),
                'description_length': row.get('description_length', 0)
            }
            records.append(record)
        
        return records
    
    def display_arabic_text(self, text: str) -> str:
        """Apply Arabic reshaping only for display purposes"""
        if not self.contains_arabic(text):
            return text
            
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            
            reshaped_text = arabic_reshaper.reshape(text)
            display_text = get_display(reshaped_text)
            return display_text
        except ImportError:
            self.logger.warning("arabic_reshaper or python-bidi not available for display")
            return text
        except Exception as e:
            self.logger.warning(f"Arabic reshaping for display failed: {e}")
            return text
    
    def validate_arabic_categories(self) -> Dict:
        """Validate that Arabic categories are properly preserved"""
        if self.processed_data is None:
            return {}
        
        validation = {
            'total_categories': 0,
            'arabic_categories': 0,
            'corrupted_categories': 0,
            'sample_categories': []
        }
        
        # Check categories
        for col in ['Subcategory_Thiqah', 'Subcategory2_Thiqah']:
            if col in self.processed_data.columns:
                categories = self.processed_data[col].dropna().unique()
                
                for cat in categories[:10]:  # Sample first 10
                    validation['total_categories'] += 1
                    
                    if self.contains_arabic(cat):
                        validation['arabic_categories'] += 1
                        
                        # Check for corrupted text (isolated form characters)
                        if re.search(r'[\uFE80-\uFEFF]', cat):
                            validation['corrupted_categories'] += 1
                            
                        validation['sample_categories'].append({
                            'category': cat,
                            'column': col,
                            'is_arabic': True,
                            'might_be_corrupted': bool(re.search(r'[\uFE80-\uFEFF]', cat))
                        })
        
        return validation

# Utility functions
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{Config.LOGS_DIR}/data_processor.log'),
            logging.StreamHandler()
        ]
    )

def test_arabic_processing():
    """Test Arabic text processing"""
    processor = DataProcessor()
    
    # Test cases
    test_texts = [
        "المدفوعات",
        "إصدار الفاتورة", 
        "تسجيل الدخول",
        "ﺕﺎﻋﻮﻓﺪﻤﻟﺍ",  # Corrupted form
        "مشكلة في النظام"
    ]
    
    print("Testing Arabic text processing:")
    for text in test_texts:
        cleaned = processor.clean_text(text)
        normalized = processor.normalize_arabic_text(cleaned)
        is_arabic = processor.contains_arabic(cleaned)
        
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print(f"Normalized: {normalized}")
        print(f"Is Arabic: {is_arabic}")
        print("-" * 40)

if __name__ == "__main__":
    setup_logging()
    
    # Test Arabic processing
    test_arabic_processing()
    
    # Process actual data if available
    try:
        processor = DataProcessor()
        
        # Load and process data
        data = processor.load_dataset()
        processed = processor.preprocess_data()
        
        # Validate Arabic categories
        validation = processor.validate_arabic_categories()
        print("\nArabic Categories Validation:")
        for key, value in validation.items():
            print(f"{key}: {value}")
        
        # Print statistics
        stats = processor.get_categories_distribution()
        print("\nData Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict) and key.endswith('_count'):
                print(f"{key}: {len(value)} unique values")
                # Show first few categories
                if key in ['subcategory1_count', 'subcategory2_count']:
                    print("  Sample categories:")
                    for cat, count in list(value.items())[:5]:
                        print(f"    {cat}: {count}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error testing with actual data: {e}")