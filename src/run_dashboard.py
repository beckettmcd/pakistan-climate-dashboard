#!/usr/bin/env python3
"""
Launcher script for the Pakistan Climate Dashboard
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'plotly', 'geopandas', 
        'shapely', 'folium', 'streamlit_folium'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'data/Sample_District_Climate_Data.csv',
        'data/geoBoundaries-PAK-ADM2 (1).geojson'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        print("Please ensure all data files are in the correct location")
        return False
    
    print("✅ All data files are present")
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Pakistan Climate Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🌐 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {str(e)}")
        return False
    
    return True

def main():
    """Main function"""
    print("🌦️ Pakistan Climate Dashboard Launcher")
    print("=" * 40)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return 1
    
    # Check data files
    print("📁 Checking data files...")
    if not check_data_files():
        return 1
    
    # Run dashboard
    if run_dashboard():
        print("✅ Dashboard completed successfully")
        return 0
    else:
        print("❌ Dashboard failed to run")
        return 1

if __name__ == "__main__":
    sys.exit(main())
