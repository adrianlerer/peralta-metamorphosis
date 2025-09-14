#!/usr/bin/env python3
"""
Quick API Demo Starter for Universal Analysis Framework
Starts the IntegridAI Suite API and shows available endpoints
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_api_health(port=8000, max_attempts=10):
    """Check if API is running and healthy"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def main():
    print("🚀 UNIVERSAL ANALYSIS FRAMEWORK - API DEMO")
    print("=" * 60)
    
    # Check if we're in the right directory
    framework_path = Path("universal_analysis_framework")
    if not framework_path.exists():
        print("❌ Error: Run this script from the project root directory")
        sys.exit(1)
    
    api_path = framework_path / "integration" / "integrid_api.py"
    if not api_path.exists():
        print("❌ Error: API file not found")
        sys.exit(1)
    
    print("📡 Starting IntegridAI Suite API...")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    print("💊 Health check at: http://localhost:8000/health")
    print("")
    print("🔧 Available Endpoints:")
    endpoints = [
        "POST /analyze - Universal analysis with any domain",
        "POST /mathematical/bounds - Calculate mathematical bounds",
        "POST /ensemble/evaluate - Multi-model ensemble evaluation", 
        "POST /hybridization/analyze - Adaptive hybridization",
        "POST /uncertainty/quantify - Uncertainty quantification",
        "GET /domains - List available analysis domains",
        "GET /stats - API usage statistics"
    ]
    
    for endpoint in endpoints:
        print(f"   • {endpoint}")
    
    print("")
    print("🔥 Starting server...")
    print("   (Press Ctrl+C to stop)")
    print("=" * 60)
    
    try:
        # Start the API server
        result = subprocess.run([
            sys.executable, str(api_path)
        ], cwd=str(framework_path.parent))
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())