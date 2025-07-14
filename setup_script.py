#!/usr/bin/env python3
"""
Setup Script for Incident Classification System
===============================================

This script automates the setup process for the incident classification system.
It handles installation, configuration, and initial data preparation.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import time

def print_step(step_num, total_steps, description):
    """Print formatted step information"""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("=" * 60)

def run_command(command, description="", check=True):
    """Run a shell command with error handling"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error {description}: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'results', 'logs', 'models']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python packages...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
        return False
    
    print("✅ All dependencies installed successfully")
    return True

def setup_environment():
    """Setup environment configuration"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        # Copy example env file
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys")
        return False
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("❌ No .env.example file found")
        return False

def download_models():
    """Download and cache embedding models"""
    print("Downloading embedding models...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        print(f"Downloading model: {model_name}")
        
        model = SentenceTransformer(model_name)
        print("✅ Model downloaded successfully")
        
        # Test the model
        test_embedding = model.encode(["Test text"])
        print(f"✅ Model test successful, embedding shape: {test_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False

def check_dataset():
    """Check if dataset file exists"""
    dataset_path = Path('Thiqa_Incidents_Example.xlsx')
    
    if dataset_path.exists():
        print(f"✅ Dataset file found: {dataset_path}")
        
        # Quick validation
        try:
            import pandas as pd
            df = pd.read_excel(dataset_path)
            print(f"✅ Dataset loaded successfully: {len(df)} records")
            
            # Check required columns
            required_columns = ['Incident', 'Description', 'Subcategory_Thiqah', 'Subcategory2_Thiqah']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️  Warning: Missing columns: {missing_columns}")
                return False
            else:
                print("✅ All required columns present")
                return True
                
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please place the Thiqa_Incidents_Example.xlsx file in the project root")
        return False

def build_initial_index(sample_size=1000):
    """Build initial FAISS index for testing"""
    print(f"Building initial FAISS index with sample size: {sample_size}")
    
    try:
        command = f"{sys.executable} main.py build-index --sample-size {sample_size}"
        if run_command(command, "building index"):
            print("✅ Initial index built successfully")
            return True
        else:
            print("❌ Failed to build initial index")
            return False
            
    except Exception as e:
        print(f"❌ Error building index: {e}")
        return False

def run_system_test():
    """Run system validation tests"""
    print("Running system validation tests...")
    
    try:
        command = f"{sys.executable} test_system.py"
        if run_command(command, "running tests"):
            print("✅ System tests passed")
            return True
        else:
            print("❌ Some system tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def test_classification():
    """Test classification with sample incident"""
    print("Testing classification with sample incident...")
    
    try:
        test_description = "مشكلة في تسجيل الدخول إلى النظام"
        command = f'{sys.executable} main.py classify "{test_description}"'
        
        if run_command(command, "testing classification"):
            print("✅ Classification test successful")
            return True
        else:
            print("❌ Classification test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing classification: {e}")
        return False

def create_sample_files():
    """Create sample input files for testing"""
    print("Creating sample files...")
    
    try:
        # Create sample incidents file
        sample_incidents = [
            {
                "incident_id": "SETUP_TEST_001",
                "description": "مشكلة في تسجيل الدخول إلى النظام"
            },
            {
                "incident_id": "SETUP_TEST_002", 
                "description": "البريد الإلكتروني لا يعمل بشكل صحيح"
            },
            {
                "incident_id": "SETUP_TEST_003",
                "description": "Password reset functionality is not working"
            }
        ]
        
        with open('sample_setup_test.json', 'w', encoding='utf-8') as f:
            json.dump(sample_incidents, f, ensure_ascii=False, indent=2)
        
        print("✅ Sample files created")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample files: {e}")
        return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Incident Classification System")
    parser.add_argument('--skip-models', action='store_true', help='Skip model download')
    parser.add_argument('--skip-index', action='store_true', help='Skip index building')
    parser.add_argument('--skip-tests', action='store_true', help='Skip system tests')
    parser.add_argument('--sample-size', type=int, default=1000, help='Sample size for initial index')
    
    args = parser.parse_args()
    
    print("🚀 Incident Classification System Setup")
    print("=" * 60)
    print("This script will set up the complete incident classification system.")
    print("Make sure you have placed the dataset file (Thiqa_Incidents_Example.xlsx) in the project root.")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Setup steps
    steps = [
        ("Check Python Version", check_python_version),
        ("Create Directories", create_directories),
        ("Install Dependencies", install_dependencies),
        ("Setup Environment", setup_environment),
        ("Check Dataset", check_dataset),
        ("Download Models", download_models if not args.skip_models else lambda: True),
        ("Build Initial Index", lambda: build_initial_index(args.sample_size) if not args.skip_index else True),
        ("Create Sample Files", create_sample_files),
        ("Run System Tests", run_system_test if not args.skip_tests else lambda: True),
        ("Test Classification", test_classification if not args.skip_tests else lambda: True)
    ]
    
    total_steps = len(steps)
    failed_steps = []
    
    start_time = time.time()
    
    for i, (step_name, step_func) in enumerate(steps, 1):
        print_step(i, total_steps, step_name)
        
        try:
            if not step_func():
                failed_steps.append(step_name)
                print(f"⚠️  Step '{step_name}' failed but continuing...")
        except Exception as e:
            print(f"❌ Step '{step_name}' failed with error: {e}")
            failed_steps.append(step_name)
    
    # Summary
    setup_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if failed_steps:
        print(f"⚠️  Setup completed with {len(failed_steps)} warnings:")
        for step in failed_steps:
            print(f"   - {step}")
        
        print(f"\n📋 Next steps:")
        if "Setup Environment" in failed_steps:
            print("   1. Edit .env file with your API keys")
        if "Check Dataset" in failed_steps:
            print("   2. Place dataset file in project root")
        if "Build Initial Index" in failed_steps:
            print("   3. Run: python main.py build-index")
        
    else:
        print("✅ Setup completed successfully!")
        print(f"\n🎉 System is ready to use!")
        print(f"\n📋 Quick start commands:")
        print(f"   python main.py info                    # System information")
        print(f'   python main.py classify "sample text"  # Test classification')
        print(f"   python main.py batch-classify sample_setup_test.json  # Batch test")
    
    print(f"\n⏱️  Total setup time: {setup_time:.1f} seconds")
    
    if failed_steps:
        return 1
    else:
        return 0

if __name__ == "__main__":
    exit(main())
