# data_collection.py
"""
Data Collection Module for Social Media Sentiment Analysis
Brainwave Matrix Solutions - Task 2
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import json

class DataCollector:
    def __init__(self, data_path="data/raw/"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        print(f"DataCollector initialized. Data path: {self.data_path}")
    
    def create_sample_data(self):
        """
        Create sample social media data for demonstration
        In a real project, you'd connect to Twitter API or use Kaggle datasets
        """
        print("Creating sample social media sentiment data...")
        
        # Sample topics for analysis
        topics = [
            'iPhone 15', 'Tesla Model 3', 'ChatGPT', 'Climate Change', 
            'Olympics 2024', 'Netflix', 'Amazon', 'Google', 'Microsoft', 'Apple'
        ]
        
        # Realistic positive phrases
        positive_phrases = [
            "love this", "amazing product", "highly recommend", "best ever", 
            "fantastic experience", "incredible quality", "totally worth it",
            "blown away", "exceeded expectations", "perfect choice", "outstanding",
            "brilliant", "awesome", "excellent service", "top notch"
        ]
        
        # Realistic negative phrases
        negative_phrases = [
            "worst purchase", "total disappointment", "waste of money", 
            "terrible quality", "regret buying", "complete failure",
            "not worth it", "poor service", "frustrating experience", "avoid this",
            "horrible", "disgusting", "pathetic", "useless", "broken"
        ]
        
        # Realistic neutral phrases
        neutral_phrases = [
            "it's okay", "average product", "nothing special", "decent enough",
            "could be better", "mixed feelings", "standard quality",
            "as expected", "fair price", "reasonable option", "not bad",
            "acceptable", "ordinary", "typical", "moderate"
        ]
        
        # Generate sample data
        sample_data = []
        
        for i in range(1500):  # Generate 1500 sample tweets
            topic = np.random.choice(topics)
            sentiment_label = np.random.choice(['positive', 'negative', 'neutral'], 
                                             p=[0.4, 0.3, 0.3])  # 40% positive, 30% negative, 30% neutral
            
            # Create realistic text based on sentiment
            if sentiment_label == 'positive':
                phrase = np.random.choice(positive_phrases)
                text = f"{phrase} {topic}! {self._add_social_media_elements()}"
            elif sentiment_label == 'negative':
                phrase = np.random.choice(negative_phrases)
                text = f"{topic} is a {phrase} {self._add_social_media_elements()}"
            else:
                phrase = np.random.choice(neutral_phrases)
                text = f"{topic} is {phrase} {self._add_social_media_elements()}"
            
            # Add realistic social media noise
            text = self._add_realistic_noise(text)
            
            # Create complete record
            sample_data.append({
                'id': f"tweet_{i+1}",
                'text': text,
                'sentiment': sentiment_label,
                'topic': topic,
                'timestamp': self._random_timestamp(),
                'user_id': f"user_{np.random.randint(1, 500)}",
                'retweets': np.random.randint(0, 100),
                'likes': np.random.randint(0, 500),
                'followers': np.random.randint(50, 10000)
            })
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Save the dataset
        filename = os.path.join(self.data_path, 'social_media_sentiment_data.csv')
        df.to_csv(filename, index=False)
        
        print(f"✅ Sample data created successfully!")
        print(f"📁 Dataset saved as: {filename}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📈 Sentiment distribution:")
        print(df['sentiment'].value_counts())
        print(f"📝 Topic distribution:")
        print(df['topic'].value_counts())
        
        return df
    
    def _add_social_media_elements(self):
        """Add realistic social media elements"""
        elements = [
            "#trending", "@mention", "😍", "👍", "👎", "😢", "🔥", "💯",
            "RT @user:", "via @user", "#hashtag", "check this out!",
            "thoughts?", "agree?", "what do you think?", "lol", "omg",
            "#amazing", "#fail", "#love", "#hate", ""
        ]
        return np.random.choice(elements)
    
    def _add_realistic_noise(self, text):
        """Add realistic social media text noise"""
        modifications = [
            lambda x: x.replace('you', 'u') if np.random.random() < 0.3 else x,
            lambda x: x.replace('are', 'r') if np.random.random() < 0.2 else x,
            lambda x: x.replace('your', 'ur') if np.random.random() < 0.2 else x,
            lambda x: x.replace('and', '&') if np.random.random() < 0.1 else x,
            lambda x: x.upper() if np.random.random() < 0.05 else x,
            lambda x: x + "!!!" if np.random.random() < 0.15 else x,
            lambda x: x + "..." if np.random.random() < 0.1 else x,
            lambda x: x  # no change (most common)
        ]
        
        # Apply random modification
        modifier = np.random.choice(modifications)
        return modifier(text)
    
    def _random_timestamp(self):
        """Generate random timestamp within last 30 days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = np.random.randint(0, days_between)
        random_seconds = np.random.randint(0, 24*60*60)
        
        random_date = start_date + timedelta(days=random_days, seconds=random_seconds)
        return random_date.strftime('%Y-%m-%d %H:%M:%S')
    
    def load_kaggle_dataset(self, dataset_name):
        """
        Load dataset from Kaggle (you'll need to download manually)
        Popular datasets:
        - twitter_airline: Twitter US Airline Sentiment
        - sentiment140: Large Twitter sentiment dataset
        """
        try:
            if dataset_name == 'airline':
                # Assuming you've downloaded the airline sentiment dataset
                filepath = os.path.join(self.data_path, 'Tweets.csv')
                df = pd.read_csv(filepath)
                
                # Standardize column names
                df = df.rename(columns={
                    'airline_sentiment': 'sentiment',
                    'text': 'text'
                })
                
                print(f"✅ Loaded airline dataset from {filepath}")
                
            elif dataset_name == 'sentiment140':
                # Sentiment140 dataset format
                filepath = os.path.join(self.data_path, 'sentiment140.csv')
                df = pd.read_csv(filepath, 
                               encoding='latin-1',
                               names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
                
                # Convert sentiment labels (0=negative, 4=positive)
                df['sentiment'] = df['sentiment'].map({0: 'negative', 4: 'positive'})
                print(f"✅ Loaded Sentiment140 dataset from {filepath}")
            
            else:
                print(f"❌ Unknown dataset: {dataset_name}")
                return self.create_sample_data()
            
            print(f"📊 Dataset shape: {df.shape}")
            print(f"📈 Sentiment distribution:")
            print(df['sentiment'].value_counts())
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Dataset {dataset_name} not found in {self.data_path}")
            print("📝 Creating sample data instead...")
            return self.create_sample_data()
    
    def get_data_info(self, df):
        """Get comprehensive information about the dataset"""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        if 'sentiment' in df.columns:
            info['sentiment_distribution'] = df['sentiment'].value_counts().to_dict()
        
        if 'text' in df.columns:
            info['text_stats'] = {
                'avg_length': df['text'].str.len().mean(),
                'max_length': df['text'].str.len().max(),
                'min_length': df['text'].str.len().min()
            }
        
        return info

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Data Collection for Social Media Sentiment Analysis")
    print("=" * 60)
    
    # Initialize data collector
    collector = DataCollector()
    
    # Create sample data (you can change this to load real data later)
    df = collector.create_sample_data()
    
    # Get data information
    data_info = collector.get_data_info(df)
    
    print("\n📊 Dataset Information:")
    print(f"   Shape: {data_info['shape']}")
    print(f"   Columns: {data_info['columns']}")
    print(f"   Sentiment distribution: {data_info['sentiment_distribution']}")
    print(f"   Average text length: {data_info['text_stats']['avg_length']:.1f} characters")
    
    # Save data info to JSON
    info_path = os.path.join("results", "data_info.json")
    os.makedirs("results", exist_ok=True)
    
    with open(info_path, 'w') as f:
        json.dump(data_info, f, indent=2, default=str)
    
    print(f"\n✅ Data collection complete!")
    print(f"📁 Data saved to: data/raw/social_media_sentiment_data.csv")
    print(f"📄 Data info saved to: {info_path}")
    print("\n🎯 Ready for Phase 3: Data Preprocessing!")