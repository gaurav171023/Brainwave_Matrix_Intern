# preprocessing.py
"""
Text Preprocessing Module for Social Media Sentiment Analysis
Brainwave Matrix Solutions - Task 2
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from textblob import TextBlob
import os

# Download required NLTK data if not present
def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    
    for path, name in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading {name}...")
            nltk.download(name, quiet=True)

# Ensure NLTK data is available
ensure_nltk_data()

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom stopwords for social media
        social_stopwords = {'rt', 'via', 'amp', 'http', 'https', 'www', 'com', 'user'}
        self.stop_words.update(social_stopwords)
        
        # Important negations to preserve
        self.negations = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 
            'neither', 'nowhere', 'cannot', "can't", "won't", 
            "shouldn't", "wouldn't", "couldn't", "doesn't", "don't"
        }
        
        print("✅ TextPreprocessor initialized")
    
    def clean_text(self, text):
        """
        Comprehensive text cleaning for social media data
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions but keep the context
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove retweet indicators
        text = re.sub(r'^rt\s+', '', text)
        text = re.sub(r'\brt\b', '', text)
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove emojis and special characters (keep basic punctuation for now)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords while preserving important negations"""
        if not text:
            return ""
        
        words = word_tokenize(text)
        
        # Keep words that are not stopwords OR are important negations
        filtered_words = []
        for word in words:
            if word not in self.stop_words or word in self.negations:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text):
        """Apply lemmatization to reduce words to root form"""
        if not text:
            return ""
        
        words = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words if word.isalpha()]
        return ' '.join(lemmatized)
    
    def remove_short_words(self, text, min_length=2):
        """Remove words shorter than min_length"""
        if not text:
            return ""
        
        words = text.split()
        filtered_words = [word for word in words if len(word) >= min_length]
        return ' '.join(filtered_words)
    
    def preprocess_single_text(self, text):
        """Apply complete preprocessing pipeline to a single text"""
        # Step 1: Basic cleaning
        cleaned = self.clean_text(text)
        
        # Step 2: Remove stopwords
        no_stopwords = self.remove_stopwords(cleaned)
        
        # Step 3: Lemmatization
        lemmatized = self.lemmatize_text(no_stopwords)
        
        # Step 4: Remove short words
        final_text = self.remove_short_words(lemmatized)
        
        return final_text
    
    def preprocess_dataframe(self, df, text_column='text', save_path=None):
        """
        Complete preprocessing pipeline for the dataframe
        """
        print("🔄 Starting text preprocessing...")
        print(f"📊 Input dataset shape: {df.shape}")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Check if text column exists
        if text_column not in processed_df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")
        
        # Step 1: Basic cleaning
        print("Step 1/6: Cleaning text...")
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_text)
        
        # Step 2: Remove empty texts
        initial_count = len(processed_df)
        processed_df = processed_df[processed_df['cleaned_text'].str.len() > 0]
        removed_empty = initial_count - len(processed_df)
        print(f"   Removed {removed_empty} empty texts")
        
        # Step 3: Remove stopwords
        print("Step 2/6: Removing stopwords...")
        processed_df['no_stopwords'] = processed_df['cleaned_text'].apply(self.remove_stopwords)
        
        # Step 4: Lemmatization
        print("Step 3/6: Applying lemmatization...")
        processed_df['lemmatized_text'] = processed_df['no_stopwords'].apply(self.lemmatize_text)
        
        # Step 5: Calculate text statistics
        print("Step 4/6: Calculating text statistics...")
        processed_df['original_length'] = processed_df[text_column].str.len()
        processed_df['cleaned_length'] = processed_df['cleaned_text'].str.len()
        processed_df['final_length'] = processed_df['lemmatized_text'].str.len()
        processed_df['word_count'] = processed_df['lemmatized_text'].str.split().str.len()
        
        # Step 6: Filter out very short texts (less than 2 words)
        print("Step 5/6: Filtering short texts...")
        before_filter = len(processed_df)
        processed_df = processed_df[processed_df['word_count'] >= 2]
        removed_short = before_filter - len(processed_df)
        print(f"   Removed {removed_short} texts with less than 2 words")
        
        # Step 7: Final processing
        print("Step 6/6: Final processing...")
        processed_df = processed_df.reset_index(drop=True)
        
        # Save processed data if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            processed_df.to_csv(save_path, index=False)
            print(f"💾 Processed data saved to: {save_path}")
        
        print(f"\n✅ Preprocessing complete!")
        print(f"📊 Final dataset shape: {processed_df.shape}")
        print(f"📈 Data reduction: {initial_count - len(processed_df)} rows removed ({(initial_count - len(processed_df))/initial_count*100:.1f}%)")
        print(f"📝 Average word count: {processed_df['word_count'].mean():.1f} words")
        print(f"📏 Average text length: {processed_df['final_length'].mean():.1f} characters")
        
        return processed_df
    
    def get_preprocessing_stats(self, original_df, processed_df):
        """Generate comprehensive preprocessing statistics"""
        stats = {
            'original_count': len(original_df),
            'processed_count': len(processed_df),
            'removal_rate': (len(original_df) - len(processed_df)) / len(original_df) * 100,
            'avg_original_length': original_df['text'].astype(str).str.len().mean(),
            'avg_processed_length': processed_df['lemmatized_text'].str.len().mean(),
            'avg_word_count': processed_df['word_count'].mean(),
            'total_unique_words': len(set(' '.join(processed_df['lemmatized_text']).split())),
            'sentiment_distribution': processed_df['sentiment'].value_counts().to_dict() if 'sentiment' in processed_df.columns else {}
        }
        
        return stats
    
    def show_examples(self, df, text_column='text', n_examples=5):
        """Show before/after preprocessing examples"""
        print(f"\n📝 Preprocessing Examples (showing {n_examples} samples):")
        print("=" * 80)
        
        for i in range(min(n_examples, len(df))):
            print(f"\nExample {i+1}:")
            print(f"Original:    '{df.iloc[i][text_column]}'")
            if 'cleaned_text' in df.columns:
                print(f"Cleaned:     '{df.iloc[i]['cleaned_text']}'")
            if 'lemmatized_text' in df.columns:
                print(f"Final:       '{df.iloc[i]['lemmatized_text']}'")
            if 'sentiment' in df.columns:
                print(f"Sentiment:   {df.iloc[i]['sentiment']}")
            print("-" * 80)

# Main execution
if __name__ == "__main__":
    print("🔄 Starting Text Preprocessing for Social Media Sentiment Analysis")
    print("=" * 70)
    
    # Load the raw data
    input_path = 'data/raw/social_media_sentiment_data.csv'
    output_path = 'data/processed/processed_sentiment_data.csv'
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"📂 Loaded data from: {input_path}")
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Preprocess the data
        processed_df = preprocessor.preprocess_dataframe(
            df, 
            text_column='text', 
            save_path=output_path
        )
        
        # Get preprocessing statistics
        stats = preprocessor.get_preprocessing_stats(df, processed_df)
        
        print(f"\n📊 Preprocessing Statistics:")
        print(f"   Original texts: {stats['original_count']}")
        print(f"   Processed texts: {stats['processed_count']}")
        print(f"   Removal rate: {stats['removal_rate']:.1f}%")
        print(f"   Avg original length: {stats['avg_original_length']:.1f} chars")
        print(f"   Avg processed length: {stats['avg_processed_length']:.1f} chars")
        print(f"   Avg word count: {stats['avg_word_count']:.1f} words")
        print(f"   Total unique words: {stats['total_unique_words']}")
        
        # Show examples
        preprocessor.show_examples(processed_df, 'text', n_examples=3)
        
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"🎯 Ready for Phase 3: Sentiment Analysis!")
        
    except FileNotFoundError:
        print(f"❌ File not found: {input_path}")
        print("Please run data_collection.py first to create the dataset.")
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")