#!/usr/bin/env python
"""
Setup script for Migraine Predictor System
Run this to set up and start the application.
"""

import os
import sys
import subprocess

def main():
    print("="*50)
    print("  🧠 Migraine Predictor Setup")
    print("="*50)
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Create virtual environment if it doesn't exist
    venv_path = os.path.join(os.path.dirname(__file__), 'venv')
    if not os.path.exists(venv_path):
        print("\n📦 Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✓ Virtual environment created")
    
    # Determine pip path
    if sys.platform == 'win32':
        pip_path = os.path.join(venv_path, 'Scripts', 'pip')
        python_path = os.path.join(venv_path, 'Scripts', 'python')
    else:
        pip_path = os.path.join(venv_path, 'bin', 'pip')
        python_path = os.path.join(venv_path, 'bin', 'python')
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    try:
        subprocess.run([pip_path, 'install', '-r', requirements_path], check=True)
        print("✓ Dependencies installed")
    except subprocess.CalledProcessError:
        print("⚠️  Some dependencies failed to install. Trying core packages...")
        core_packages = [
            'Flask==2.3.3',
            'Flask-SocketIO==5.3.6',
            'Flask-SQLAlchemy==3.1.1',
            'Flask-CORS==4.0.0',
            'psutil==5.9.5',
            'numpy==1.24.3',
            'python-socketio==5.10.0',
            'eventlet==0.33.3'
        ]
        for package in core_packages:
            try:
                subprocess.run([pip_path, 'install', package], check=True, capture_output=True)
            except:
                pass
        print("✓ Core dependencies installed")
    
    # Create necessary directories
    static_dirs = ['static/css', 'static/js']
    for dir_path in static_dirs:
        full_path = os.path.join(os.path.dirname(__file__), dir_path)
        os.makedirs(full_path, exist_ok=True)
    
    print("\n" + "="*50)
    print("  ✅ Setup Complete!")
    print("="*50)
    print()
    print("To start the application:")
    print()
    if sys.platform == 'win32':
        print("  1. Activate virtual environment:")
        print("     venv\\Scripts\\activate")
        print()
        print("  2. Run the application:")
        print("     python run.py")
    else:
        print("  1. Activate virtual environment:")
        print("     source venv/bin/activate")
        print()
        print("  2. Run the application:")
        print("     python run.py")
    print()
    print("  3. Open http://localhost:5000 in your browser")
    print()
    
    # Ask if user wants to start now
    try:
        response = input("Start the application now? (y/n): ").strip().lower()
        if response == 'y':
            print("\n🚀 Starting Migraine Predictor...")
            run_path = os.path.join(os.path.dirname(__file__), 'run.py')
            subprocess.run([python_path, run_path])
    except KeyboardInterrupt:
        print("\n\nSetup complete. Run 'python run.py' to start.")


if __name__ == '__main__':
    main()
