# setup.py
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    packages = [
        "tensorflow>=2.12.0",
        "scikit-learn>=1.2.0", 
        "spacy>=3.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "jupyter>=1.0.0",
        "streamlit>=1.22.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    # Download spaCy model
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ Successfully downloaded spaCy model")
    except subprocess.CalledProcessError:
        print("✗ Failed to download spaCy model")

if __name__ == "__main__":
    print("Installing required packages...")
    install_requirements()
    print("\nSetup complete! You can now run the AI Tools Assignment scripts.")