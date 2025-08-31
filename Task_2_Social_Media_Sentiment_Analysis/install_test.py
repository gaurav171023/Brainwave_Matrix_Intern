# install_test.py
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def test_import(package_name, import_name=None):
    """Test if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} import successful")
        return True
    except ImportError:
        print(f"❌ {package_name} import failed")
        return False

# Essential packages to install
packages = [
    "pandas",
    "numpy", 
    "matplotlib",
    "seaborn",
    "nltk",
    "textblob",
    "vaderSentiment",
    "scikit-learn",
    "wordcloud"
]

print("🔧 Installing required packages...")
print("=" * 50)

for package in packages:
    print(f"\nInstalling {package}...")
    install_package(package)

print("\n🔍 Testing imports...")
print("=" * 30)

# Test imports
test_import("pandas", "pandas")
test_import("numpy", "numpy") 
test_import("matplotlib", "matplotlib.pyplot")
test_import("seaborn", "seaborn")
test_import("nltk", "nltk")
test_import("textblob", "textblob")
test_import("vaderSentiment", "vaderSentiment.vaderSentiment")
test_import("scikit-learn", "sklearn")
test_import("wordcloud", "wordcloud")

print("\n📥 Downloading NLTK data...")
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print("✅ NLTK data downloaded successfully")
except Exception as e:
    print(f"❌ NLTK data download failed: {e}")

print("\n🎉 Installation complete! Try running your scripts again.")