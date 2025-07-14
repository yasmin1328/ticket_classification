import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    
    # Alternative LLM APIs (if OpenAI not available)
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    
    # Embedding Configuration
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    EMBEDDING_DIMENSION = 384
    
    # Alternative embedding settings
    USE_ALTERNATIVE_EMBEDDING = os.getenv('USE_ALTERNATIVE_EMBEDDING', 'false').lower() == 'true'
    ALTERNATIVE_EMBEDDING_METHOD = os.getenv('ALTERNATIVE_EMBEDDING_METHOD', 'tfidf')  # 'tfidf', 'openai', 'hybrid'
    
    # FAISS Configuration
    FAISS_INDEX_PATH = 'data/faiss_index.bin'
    FAISS_METADATA_PATH = 'data/faiss_metadata.json'
    
    # Enhanced Classification Parameters
    # Lowered thresholds for better category extraction from similar incidents
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.4'))  # Lowered from 0.5 for more candidates
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))   # Lowered from 0.7 for production use
    TOP_K_SIMILAR = int(os.getenv('TOP_K_SIMILAR', '10'))  # Increased from 5 for better analysis
    
    # Enhanced similarity search parameters
    SIMILARITY_SEARCH_MULTIPLIER = 2  # Get 2x more candidates for better analysis
    MIN_SIMILARITY_FOR_EXISTING = 0.6  # Minimum similarity to use existing category
    HIGH_CONFIDENCE_THRESHOLD = 0.85   # Threshold for high confidence classification
    CONSENSUS_STRENGTH_THRESHOLD = 0.7  # Minimum consensus strength for category voting
    
    # Processing Configuration
    MAX_DESCRIPTION_LENGTH = int(os.getenv('MAX_DESCRIPTION_LENGTH', '500'))
    PROCESSING_BATCH_SIZE = int(os.getenv('PROCESSING_BATCH_SIZE', '100'))
    
    # Data Paths
    DATASET_PATH = os.getenv('DATASET_PATH', 'Thiqa_Incidents_Example.xlsx')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'results')
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    
    # Language Configuration
    SUPPORTED_LANGUAGES = ['ar', 'en']
    DEFAULT_LANGUAGE = 'ar'
    
    # Performance Settings
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    CACHE_SIZE = int(os.getenv('CACHE_SIZE', '1000'))
    
    # Production Settings
    PRODUCTION_MODE = os.getenv('PRODUCTION_MODE', 'false').lower() == 'true'
    ENABLE_DETAILED_LOGGING = os.getenv('ENABLE_DETAILED_LOGGING', 'true').lower() == 'true'
    
    # Enhanced classification weights for similarity-based voting
    SIMILARITY_WEIGHT = 0.4      # Weight for similarity scores in confidence calculation
    CONSENSUS_WEIGHT = 0.3       # Weight for category consensus in confidence calculation  
    EXAMPLES_WEIGHT = 0.3        # Weight for number of supporting examples
    
    # Fallback classification keywords (Arabic and English)
    CATEGORY_KEYWORDS = {
        ('أنظمة المصادقة والهوية', 'تسجيل الدخول'): [
            'دخول', 'تسجيل', 'مصادقة', 'كلمة المرور', 'باسورد', 'login', 'password', 
            'authentication', 'signin', 'account'
        ],
        ('إدارة الهوية والوصول', 'بيانات الهوية والشهادات'): [
            'شهادة', 'هوية', 'بيانات شخصية', 'معلومات', 'certificate', 'identity',
            'personal data', 'profile', 'شهاده', 'بياناتي'
        ],
        ('البريد الإلكتروني والرسائل', 'مشاكل الاتصال'): [
            'بريد', 'إيميل', 'email', 'outlook', 'رسائل', 'message', 'mail',
            'رسالة', 'inbox', 'ايميل'
        ],
        ('الشبكة والاتصالات', 'انقطاع الخدمة'): [
            'شبكة', 'انترنت', 'اتصال', 'network', 'internet', 'connection',
            'wifi', 'connectivity', 'موقع', 'site'
        ],
        ('التطبيقات والبرمجيات', 'أخطاء التشغيل'): [
            'برنامج', 'تطبيق', 'software', 'application', 'خطأ', 'error',
            'crash', 'freeze', 'app', 'program', 'نظام', 'system'
        ],
        ('الأجهزة والمعدات', 'مشاكل الأجهزة'): [
            'جهاز', 'طابعة', 'hardware', 'printer', 'computer', 'device',
            'equipment', 'طباعة', 'print', 'scanner'
        ],
        ('إدارة البيانات والملفات', 'فقدان البيانات'): [
            'بيانات', 'ملف', 'data', 'file', 'backup', 'save', 'حفظ',
            'folder', 'مجلد', 'document', 'وثيقة'
        ],
        ('خدمات قواعد البيانات', 'أخطاء قاعدة البيانات'): [
            'قاعدة بيانات', 'database', 'sql', 'query', 'connection',
            'timeout', 'data', 'table', 'جدول'
        ]
    }

class ProductionConfig(Config):
    """Production-specific configuration"""
    SIMILARITY_THRESHOLD = 0.3   # More liberal for production
    CONFIDENCE_THRESHOLD = 0.6   # Balanced for production use
    TOP_K_SIMILAR = 15           # More examples for better decisions
    ENABLE_DETAILED_LOGGING = True
    PRODUCTION_MODE = True

class DevelopmentConfig(Config):
    """Development-specific configuration"""  
    SIMILARITY_THRESHOLD = 0.5   # More conservative for testing
    CONFIDENCE_THRESHOLD = 0.7   # Higher threshold for development
    TOP_K_SIMILAR = 5            # Fewer for faster testing
    ENABLE_DETAILED_LOGGING = True
    PRODUCTION_MODE = False

class TestingConfig(Config):
    """Testing-specific configuration"""
    SIMILARITY_THRESHOLD = 0.4
    CONFIDENCE_THRESHOLD = 0.6
    TOP_K_SIMILAR = 8
    PROCESSING_BATCH_SIZE = 50   # Smaller batches for testing
    ENABLE_DETAILED_LOGGING = True

# Select configuration based on environment
def get_config():
    """Get configuration based on environment"""
    env = os.getenv('APP_ENV', 'production').lower()
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'development':
        return DevelopmentConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return Config()

# Use the appropriate configuration
Config = get_config()

# Validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check API keys
    has_openai = bool(Config.OPENAI_API_KEY)
    has_anthropic = bool(Config.ANTHROPIC_API_KEY)
    
    if not has_openai and not has_anthropic:
        # Only warn, system can work without LLM in similarity mode
        print("⚠️  WARNING: No LLM API key provided. System will use similarity-based classification only.")
    
    # Check dataset
    if not os.path.exists(Config.DATASET_PATH):
        errors.append(f"Dataset file not found: {Config.DATASET_PATH}")
    
    # Check thresholds
    if not (0 <= Config.SIMILARITY_THRESHOLD <= 1):
        errors.append(f"SIMILARITY_THRESHOLD must be between 0 and 1, got: {Config.SIMILARITY_THRESHOLD}")
    
    if not (0 <= Config.CONFIDENCE_THRESHOLD <= 1):
        errors.append(f"CONFIDENCE_THRESHOLD must be between 0 and 1, got: {Config.CONFIDENCE_THRESHOLD}")
    
    # Create necessary directories
    for directory in [Config.OUTPUT_DIR, Config.LOGS_DIR, 'data']:
        os.makedirs(directory, exist_ok=True)
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        'environment': os.getenv('APP_ENV', 'production'),
        'similarity_threshold': Config.SIMILARITY_THRESHOLD,
        'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
        'top_k_similar': Config.TOP_K_SIMILAR,
        'has_openai_key': bool(Config.OPENAI_API_KEY),
        'has_anthropic_key': bool(Config.ANTHROPIC_API_KEY),
        'dataset_path': Config.DATASET_PATH,
        'production_mode': getattr(Config, 'PRODUCTION_MODE', False),
        'total_keyword_categories': len(Config.CATEGORY_KEYWORDS)
    }

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        'environment': os.getenv('APP_ENV', 'production'),
        'similarity_threshold': Config.SIMILARITY_THRESHOLD,
        'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
        'top_k_similar': Config.TOP_K_SIMILAR,
        'has_openai_key': bool(Config.OPENAI_API_KEY),
        'has_anthropic_key': bool(Config.ANTHROPIC_API_KEY),
        'dataset_path': Config.DATASET_PATH,
        'production_mode': getattr(Config, 'PRODUCTION_MODE', False),
        'total_keyword_categories': len(Config.CATEGORY_KEYWORDS)
    }

if __name__ == "__main__":
    # Test configuration
    try:
        validate_config()
        summary = get_config_summary()
        
        print("✅ Configuration Valid")
        print("\n📊 Configuration Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Configuration Error: {e}")