#!/usr/bin/env python3
"""
InsightDock Setup Script
Helps users set up the application quickly
"""

import os
import sys
import shutil

def setup_config_files():
    """Copy template files to actual config files if they don't exist."""
    
    # Copy Token.txt template if Token.txt doesn't exist
    if not os.path.exists("Token.txt") and os.path.exists("Token.txt.template"):
        shutil.copy("Token.txt.template", "Token.txt")
        print("✅ Created Token.txt from template")
        print("⚠️  Please edit Token.txt and add your actual API keys")
    
    # Copy kaggle.json template if kaggle.json doesn't exist
    if not os.path.exists("kaggle.json") and os.path.exists("kaggle.json.template"):
        shutil.copy("kaggle.json.template", "kaggle.json")
        print("✅ Created kaggle.json from template")
        print("⚠️  Please edit kaggle.json and add your Kaggle credentials")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies."""
    try:
        import subprocess
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def main():
    """Main setup function."""
    print("🚀 InsightDock Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Setup config files
    setup_config_files()
    
    # Install dependencies
    if "--install-deps" in sys.argv:
        install_dependencies()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit Token.txt with your API keys")
    print("2. Edit kaggle.json with your Kaggle credentials")
    print("3. Run: streamlit run streamlit_app.py")
    print("\nFor help, see README.md")

if __name__ == "__main__":
    main()
