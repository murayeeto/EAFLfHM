#!/usr/bin/env python3
"""
Setup script for the Edge-Aware Federated Learning System
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "results",
        "logs", 
        "data/raw",
        "data/processed",
        "visualizations",
        "checkpoints"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created")

def check_optional_dependencies():
    """Check for optional dependencies"""
    print("🔍 Checking optional dependencies...")
    
    optional_deps = {
        "docker": "Docker (for containerized deployment)",
        "nvidia-smi": "NVIDIA GPU support",
        "git": "Git (for version control)"
    }
    
    for cmd, description in optional_deps.items():
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            print(f"   ✅ {description}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"   ⚠️  {description} - not found (optional)")

def run_basic_tests():
    """Run basic system tests"""
    print("🧪 Running basic tests...")
    try:
        subprocess.check_call([sys.executable, "tests/test_system.py"])
        print("✅ Basic tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Some tests failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration for testing"""
    print("⚙️ Creating sample configuration...")
    
    # This would typically copy or modify the default config
    # For now, we'll just verify the config exists
    if os.path.exists("config/config.py"):
        print("✅ Configuration file found")
    else:
        print("❌ Configuration file not found")

def display_usage_instructions():
    """Display usage instructions"""
    print("\\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    print("\\n📖 Usage Instructions:")
    print("\\n1. Run a simple experiment:")
    print("   python main.py --num-clients 5 --num-rounds 10")
    print("\\n2. Run all predefined experiments:")
    print("   python scripts/run_experiments.py")
    print("\\n3. Visualize results:")
    print("   python scripts/visualize_results.py")
    print("\\n4. Run in different modes:")
    print("   python main.py --mode edge-server --edge-server-id edge_1")
    print("   python main.py --mode client --client-id device_1")
    print("\\n5. Docker deployment:")
    print("   cd docker")
    print("   docker-compose up")
    print("\\n6. Run tests:")
    print("   python tests/test_system.py")
    print("\\n📁 Important directories:")
    print("   - results/     : Experiment results")
    print("   - logs/        : System logs")
    print("   - visualizations/ : Generated plots")
    print("\\n🔗 For more information, see README.md")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Setup Edge-Aware FL System")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    print("🚀 Setting up Edge-Aware Federated Learning System")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not args.no_deps:
        if not install_requirements():
            print("⚠️  Warning: Requirements installation failed")
    
    # Create directories
    create_directories()
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Create sample config
    create_sample_config()
    
    # Run tests
    if not args.skip_tests:
        if not run_basic_tests():
            print("⚠️  Warning: Some tests failed")
    
    # Display usage instructions
    display_usage_instructions()

if __name__ == "__main__":
    main()
