#!/usr/bin/env python3
"""
Flask API Service for Incident Classification System
===================================================

This module provides a REST API wrapper for the incident classification system.
It enables integration with external systems via HTTP endpoints.

Endpoints:
- POST /classify - Classify single incident
- POST /classify/batch - Classify multiple incidents
- GET /health - Health check
- GET /stats - System statistics
- GET /categories - Get available categories
"""

from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import threading
import queue
import uuid

from config import Config, validate_config
from incident_classifier import IncidentClassifier
from data_processor import DataProcessor

# Global classifier instance
classifier_instance = None
classifier_lock = threading.Lock()

class APIError(Exception):
    """Custom API error class"""
    def __init__(self, message: str, status_code: int = 400, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)

def create_app():
    """Create Flask application"""
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False  # Support Unicode in JSON responses
    
    # Enable CORS for cross-origin requests
    CORS(app)
    
    # Setup logging
    setup_api_logging()
    
    # Initialize classifier in background
    threading.Thread(target=initialize_classifier_background, daemon=True).start()
    
    return app

def setup_api_logging():
    """Setup logging for API service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{Config.LOGS_DIR}/api_service.log'),
            logging.StreamHandler()
        ]
    )

def initialize_classifier_background():
    """Initialize classifier in background thread"""
    global classifier_instance
    
    try:
        logging.info("Initializing classifier...")
        classifier = IncidentClassifier(use_existing_index=True)
        
        if classifier.initialize():
            with classifier_lock:
                classifier_instance = classifier
            logging.info("Classifier initialized successfully")
        else:
            logging.error("Failed to initialize classifier")
            
    except Exception as e:
        logging.error(f"Error initializing classifier: {e}")

def get_classifier() -> IncidentClassifier:
    """Get classifier instance with thread safety"""
    with classifier_lock:
        if classifier_instance is None:
            raise APIError("Classifier not initialized", 503, "CLASSIFIER_NOT_READY")
        return classifier_instance

def validate_request_data(data: dict, required_fields: List[str]) -> None:
    """Validate request data"""
    if not data:
        raise APIError("No JSON data provided", 400, "NO_DATA")
    
    for field in required_fields:
        if field not in data:
            raise APIError(f"Missing required field: {field}", 400, "MISSING_FIELD")
        
        if not data[field] or not str(data[field]).strip():
            raise APIError(f"Field '{field}' cannot be empty", 400, "EMPTY_FIELD")

def create_error_response(error: APIError) -> tuple:
    """Create standardized error response"""
    response = {
        "success": False,
        "error": {
            "message": error.message,
            "code": error.error_code or "UNKNOWN_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    }
    return jsonify(response), error.status_code

def create_success_response(data: dict, message: str = "Success") -> dict:
    """Create standardized success response"""
    response = {
        "success": True,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    return jsonify(response)

# Create Flask app
app = create_app()

@app.errorhandler(APIError)
def handle_api_error(error):
    """Handle custom API errors"""
    return create_error_response(error)

@app.errorhandler(Exception)
def handle_generic_error(error):
    """Handle unexpected errors"""
    logging.error(f"Unexpected error: {error}")
    api_error = APIError("Internal server error", 500, "INTERNAL_ERROR")
    return create_error_response(api_error)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if classifier is ready
        with classifier_lock:
            classifier_ready = classifier_instance is not None
        
        # Check configuration
        try:
            validate_config()
            config_valid = True
        except:
            config_valid = False
        
        health_data = {
            "status": "healthy" if classifier_ready and config_valid else "degraded",
            "classifier_ready": classifier_ready,
            "config_valid": config_valid,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        return create_success_response(health_data, "Health check completed")
        
    except Exception as e:
        raise APIError(f"Health check failed: {str(e)}", 500, "HEALTH_CHECK_FAILED")

@app.route('/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        classifier = get_classifier()
        stats = classifier.get_system_stats()
        
        return create_success_response(stats, "System statistics retrieved")
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Failed to get system stats: {str(e)}", 500, "STATS_ERROR")

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available categories from the training data"""
    try:
        classifier = get_classifier()
        
        if hasattr(classifier.similarity_search, 'get_all_categories'):
            categories = classifier.similarity_search.get_all_categories()
        else:
            categories = {}
        
        category_data = {
            "categories": categories,
            "total_main_categories": len(categories),
            "total_subcategories": sum(len(subcats) for subcats in categories.values())
        }
        
        return create_success_response(category_data, "Categories retrieved")
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Failed to get categories: {str(e)}", 500, "CATEGORIES_ERROR")

@app.route('/classify', methods=['POST'])
def classify_incident():
    """Classify a single incident"""
    start_time = time.time()
    
    try:
        # Validate request
        data = request.get_json()
        validate_request_data(data, ['description'])
        
        # Extract parameters
        description = data['description']
        incident_id = data.get('incident_id')
        language = data.get('language', 'ar')
        
        # Validate description length
        if len(description) > Config.MAX_DESCRIPTION_LENGTH:
            raise APIError(
                f"Description too long. Maximum {Config.MAX_DESCRIPTION_LENGTH} characters allowed",
                400,
                "DESCRIPTION_TOO_LONG"
            )
        
        # Get classifier and classify
        classifier = get_classifier()
        result = classifier.classify_incident(description, incident_id, language)
        
        # Add API metadata
        result['api_processing_time'] = time.time() - start_time
        result['api_version'] = '1.0.0'
        
        return create_success_response(result, "Incident classified successfully")
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Classification failed: {str(e)}", 500, "CLASSIFICATION_ERROR")

@app.route('/classify/batch', methods=['POST'])
def classify_batch():
    """Classify multiple incidents"""
    start_time = time.time()
    
    try:
        # Validate request
        data = request.get_json()
        
        if not data or 'incidents' not in data:
            raise APIError("Missing 'incidents' field in request", 400, "MISSING_INCIDENTS")
        
        incidents = data['incidents']
        
        if not isinstance(incidents, list):
            raise APIError("'incidents' must be a list", 400, "INVALID_INCIDENTS_FORMAT")
        
        if len(incidents) == 0:
            raise APIError("Incidents list cannot be empty", 400, "EMPTY_INCIDENTS_LIST")
        
        if len(incidents) > 100:  # Limit batch size
            raise APIError("Batch size cannot exceed 100 incidents", 400, "BATCH_TOO_LARGE")
        
        # Validate each incident
        for i, incident in enumerate(incidents):
            if not isinstance(incident, dict):
                raise APIError(f"Incident {i} must be an object", 400, "INVALID_INCIDENT_FORMAT")
            
            if 'description' not in incident or not incident['description']:
                raise APIError(f"Incident {i} missing description", 400, "MISSING_DESCRIPTION")
        
        # Get classifier and process batch
        classifier = get_classifier()
        results = classifier.classify_batch(incidents)
        
        # Calculate statistics
        batch_stats = {
            'total_incidents': len(results),
            'processing_time': time.time() - start_time,
            'average_time_per_incident': (time.time() - start_time) / len(results),
            'status_distribution': {}
        }
        
        # Count statuses
        for result in results:
            status = result.get('classification_status', 'unknown')
            batch_stats['status_distribution'][status] = batch_stats['status_distribution'].get(status, 0) + 1
        
        response_data = {
            'results': results,
            'batch_statistics': batch_stats,
            'api_version': '1.0.0'
        }
        
        return create_success_response(response_data, f"Batch of {len(results)} incidents classified")
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Batch classification failed: {str(e)}", 500, "BATCH_CLASSIFICATION_ERROR")

@app.route('/classify/async', methods=['POST'])
def classify_async():
    """Start asynchronous classification job"""
    try:
        # Validate request
        data = request.get_json()
        validate_request_data(data, ['incidents'])
        
        incidents = data['incidents']
        if len(incidents) > 1000:  # Larger limit for async
            raise APIError("Async batch size cannot exceed 1000 incidents", 400, "ASYNC_BATCH_TOO_LARGE")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # For now, just return job ID (in production, implement proper async processing)
        job_data = {
            'job_id': job_id,
            'status': 'queued',
            'incidents_count': len(incidents),
            'estimated_completion': datetime.now().isoformat()
        }
        
        return create_success_response(job_data, "Async classification job started")
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Failed to start async job: {str(e)}", 500, "ASYNC_JOB_ERROR")

@app.route('/classify/async/<job_id>', methods=['GET'])
def get_async_status(job_id):
    """Get status of asynchronous classification job"""
    # Placeholder for async job status
    # In production, implement proper job tracking
    
    job_status = {
        'job_id': job_id,
        'status': 'completed',  # queued, processing, completed, failed
        'progress': 100,
        'results_available': True,
        'completion_time': datetime.now().isoformat()
    }
    
    return create_success_response(job_status, "Job status retrieved")

@app.route('/debug/info', methods=['GET'])
def debug_info():
    """Debug information endpoint (development only)"""
    if not app.debug:
        raise APIError("Debug endpoint not available in production", 403, "DEBUG_DISABLED")
    
    try:
        debug_data = {
            'classifier_initialized': classifier_instance is not None,
            'config': {
                'similarity_threshold': Config.SIMILARITY_THRESHOLD,
                'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
                'embedding_model': Config.EMBEDDING_MODEL,
                'max_description_length': Config.MAX_DESCRIPTION_LENGTH
            },
            'system_info': {
                'python_version': f"{sys.version}",
                'flask_version': flask.__version__
            }
        }
        
        return create_success_response(debug_data, "Debug info retrieved")
        
    except Exception as e:
        raise APIError(f"Debug info failed: {str(e)}", 500, "DEBUG_ERROR")

def run_development_server(host='127.0.0.1', port=5000, debug=True):
    """Run development server"""
    print(f"🚀 Starting API service on http://{host}:{port}")
    print("📖 API Documentation:")
    print(f"   Health check: GET  http://{host}:{port}/health")
    print(f"   Classify:     POST http://{host}:{port}/classify")
    print(f"   Batch:        POST http://{host}:{port}/classify/batch")
    print(f"   Stats:        GET  http://{host}:{port}/stats")
    print(f"   Categories:   GET  http://{host}:{port}/categories")
    
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == "__main__":
    import argparse
    import sys
    import flask
    
    parser = argparse.ArgumentParser(description="Incident Classification API Service")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    
    args = parser.parse_args()
    
    if args.production:
        print("⚠️  Production mode - use proper WSGI server like Gunicorn")
        print("Example: gunicorn -w 4 -b 0.0.0.0:5000 api_service:app")
    else:
        run_development_server(args.host, args.port, args.debug)
