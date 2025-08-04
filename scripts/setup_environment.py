#!/usr/bin/env python3
"""
Automated Environment Setup Script
Handles complete project setup with validation and error handling
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description, check=True):
    """Run shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_python_version():
    """Verify Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported. Requires Python 3.8+")
        return False

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_name = "venv"
    
    if not os.path.exists(venv_name):
        if not run_command(f"python -m venv {venv_name}", "Creating virtual environment"):
            return False
    
    # Activation command varies by OS
    if platform.system() == "Windows":
        activate_cmd = f"{venv_name}\\Scripts\\activate"
        pip_cmd = f"{venv_name}\\Scripts\\pip"
    else:
        activate_cmd = f"source {venv_name}/bin/activate"
        pip_cmd = f"{venv_name}/bin/pip"
    
    print(f"📝 To activate virtual environment, run: {activate_cmd}")
    return pip_cmd

def main():
    """Main setup routine"""
    print("🚀 Starting Change Point Analysis Project Setup")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Setup virtual environment
    pip_cmd = setup_virtual_environment()
    if not pip_cmd:
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run analysis: python changepoint_detection.py")
    print("4. Start dashboard: cd src/task3/backend && python app.py")

if __name__ == "__main__":
    main()