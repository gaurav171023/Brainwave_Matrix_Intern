import os
import pandas as pd

print("BRAINWAVE MATRIX SOLUTIONS - TASK 2")
print("Social Media Sentiment Analysis")
print("=" * 40)

# Check if all key files exist
key_files = [
    "data/raw/social_media_sentiment_data.csv",
    "data/processed/analyzed_sentiment_data.csv", 
    "src/data_collection.py",
    "src/preprocessing.py",
    "src/sentiment_analyzer.py",
    "src/visualizations.py"
]

print("PROJECT STATUS CHECK:")
for file in key_files:
    status = "✅ EXISTS" if os.path.exists(file) else "❌ MISSING"
    print(f"  {file}: {status}")

# Load and show summary
if os.path.exists("data/processed/analyzed_sentiment_data.csv"):
    df = pd.read_csv("data/processed/analyzed_sentiment_data.csv")
    print(f"\nDATA SUMMARY:")
    print(f"  Total Posts: {len(df)}")
    print(f"  Sentiment Distribution:")
    sentiment_counts = df['ensemble_sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pct = count/len(df)*100
        print(f"    {sentiment.title()}: {count} ({pct:.1f}%)")

print(f"\nPROJECT READY FOR SUBMISSION!")