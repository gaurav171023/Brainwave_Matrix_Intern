# sentiment_analyzer.py
"""
Multi-Method Sentiment Analysis Module
Brainwave Matrix Solutions - Task 2
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import json

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.ml_model = None
        self.vectorizer = None
        print("✅ SentimentAnalyzer initialized")
        
    def analyze_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        Returns: polarity (-1 to 1), subjectivity (0 to 1), sentiment_label
        """
        if not text or pd.isna(text):
            return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}
        
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert polarity to sentiment label
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    def analyze_vader(self, text):
        """
        Analyze sentiment using VADER
        Returns: compound score and sentiment label
        """
        if not text or pd.isna(text):
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'sentiment': 'neutral'}
        
        scores = self.vader_analyzer.polarity_scores(str(text))
        compound = scores['compound']
        
        # Convert compound score to sentiment label
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'sentiment': sentiment
        }
    
    def train_ml_model(self, df, text_column='lemmatized_text', label_column='sentiment'):
        """
        Train a machine learning model for sentiment analysis
        """
        print("🤖 Training ML sentiment model...")
        
        # Prepare data
        X = df[text_column].fillna('')
        y = df[label_column]
        
        print(f"   Training data shape: {X.shape}")
        print(f"   Label distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        print("   Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        print("   Training logistic regression model...")
        self.ml_model = LogisticRegression(random_state=42, max_iter=1000)
        self.ml_model.fit(X_train_vec, y_train)
        
        # Evaluate model
        train_score = self.ml_model.score(X_train_vec, y_train)
        test_score = self.ml_model.score(X_test_vec, y_test)
        
        print(f"   ✅ ML Model Training Accuracy: {train_score:.3f}")
        print(f"   ✅ ML Model Testing Accuracy: {test_score:.3f}")
        
        # Predictions for detailed evaluation
        y_pred = self.ml_model.predict(X_test_vec)
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        self.save_model()
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def analyze_ml(self, text):
        """
        Analyze sentiment using trained ML model
        """
        if self.ml_model is None or self.vectorizer is None:
            return {'sentiment': 'unknown', 'confidence': 0.0}
        
        if not text or pd.isna(text):
            return {'sentiment': 'neutral', 'confidence': 0.33}
        
        # Vectorize text
        text_vec = self.vectorizer.transform([str(text)])
        
        # Predict
        prediction = self.ml_model.predict(text_vec)[0]
        probabilities = self.ml_model.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': dict(zip(self.ml_model.classes_, probabilities))
        }
    
    def analyze_comprehensive(self, df, text_column='lemmatized_text'):
        """
        Apply all sentiment analysis methods to the dataframe
        """
        print("🔍 Running comprehensive sentiment analysis...")
        
        # Make a copy
        result_df = df.copy()
        
        # Apply TextBlob analysis
        print("   Applying TextBlob analysis...")
        textblob_results = result_df[text_column].apply(self.analyze_textblob)
        result_df['textblob_polarity'] = textblob_results.apply(lambda x: x['polarity'])
        result_df['textblob_subjectivity'] = textblob_results.apply(lambda x: x['subjectivity'])
        result_df['textblob_sentiment'] = textblob_results.apply(lambda x: x['sentiment'])
        
        # Apply VADER analysis
        print("   Applying VADER analysis...")
        vader_results = result_df[text_column].apply(self.analyze_vader)
        result_df['vader_compound'] = vader_results.apply(lambda x: x['compound'])
        result_df['vader_positive'] = vader_results.apply(lambda x: x['positive'])
        result_df['vader_negative'] = vader_results.apply(lambda x: x['negative'])
        result_df['vader_neutral'] = vader_results.apply(lambda x: x['neutral'])
        result_df['vader_sentiment'] = vader_results.apply(lambda x: x['sentiment'])
        
        # Apply ML analysis (if model is trained)
        if self.ml_model is not None:
            print("   Applying ML analysis...")
            ml_results = result_df[text_column].apply(self.analyze_ml)
            result_df['ml_sentiment'] = ml_results.apply(lambda x: x['sentiment'])
            result_df['ml_confidence'] = ml_results.apply(lambda x: x['confidence'])
        
        # Create ensemble prediction
        print("   Creating ensemble predictions...")
        result_df['ensemble_sentiment'] = self.create_ensemble_prediction(result_df)
        
        print("✅ Sentiment analysis complete!")
        return result_df
    
    def create_ensemble_prediction(self, df):
        """
        Create ensemble prediction by combining multiple methods
        """
        ensemble_predictions = []
        
        for idx, row in df.iterrows():
            votes = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            # TextBlob vote (weight: 1)
            if 'textblob_sentiment' in row:
                votes[row['textblob_sentiment']] += 1
            
            # VADER vote (weight: 1)
            if 'vader_sentiment' in row:
                votes[row['vader_sentiment']] += 1
            
            # ML vote (weight: 2 if confidence > 0.6)
            if 'ml_sentiment' in row and row['ml_sentiment'] != 'unknown':
                weight = 2 if row.get('ml_confidence', 0) > 0.6 else 1
                votes[row['ml_sentiment']] += weight
            
            # Get majority vote
            ensemble_prediction = max(votes, key=votes.get)
            ensemble_predictions.append(ensemble_prediction)
        
        return ensemble_predictions
    
    def get_sentiment_summary(self, df):
        """
        Generate comprehensive sentiment analysis summary
        """
        summary = {}
        
        # Overall sentiment distribution for each method
        methods = ['textblob_sentiment', 'vader_sentiment', 'ensemble_sentiment']
        if 'ml_sentiment' in df.columns:
            methods.append('ml_sentiment')
        
        for method in methods:
            if method in df.columns:
                summary[f'{method}_distribution'] = df[method].value_counts(normalize=True).round(3).to_dict()
        
        # Sentiment scores statistics
        if 'textblob_polarity' in df.columns:
            summary['textblob_stats'] = {
                'mean_polarity': round(df['textblob_polarity'].mean(), 3),
                'std_polarity': round(df['textblob_polarity'].std(), 3),
                'mean_subjectivity': round(df['textblob_subjectivity'].mean(), 3),
                'min_polarity': round(df['textblob_polarity'].min(), 3),
                'max_polarity': round(df['textblob_polarity'].max(), 3)
            }
        
        if 'vader_compound' in df.columns:
            summary['vader_stats'] = {
                'mean_compound': round(df['vader_compound'].mean(), 3),
                'std_compound': round(df['vader_compound'].std(), 3),
                'min_compound': round(df['vader_compound'].min(), 3),
                'max_compound': round(df['vader_compound'].max(), 3)
            }
        
        # Method agreement analysis
        if all(col in df.columns for col in ['textblob_sentiment', 'vader_sentiment']):
            agreement = (df['textblob_sentiment'] == df['vader_sentiment']).mean()
            summary['textblob_vader_agreement'] = round(agreement, 3)
        
        # ML model performance (if available)
        if 'ml_confidence' in df.columns:
            summary['ml_stats'] = {
                'mean_confidence': round(df['ml_confidence'].mean(), 3),
                'high_confidence_rate': round((df['ml_confidence'] > 0.8).mean(), 3)
            }
        
        return summary
    
    def save_model(self, filepath='models/sentiment_model.pkl'):
        """Save trained ML model and vectorizer"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'ml_model': self.ml_model,
            'vectorizer': self.vectorizer,
            'model_type': 'LogisticRegression',
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='models/sentiment_model.pkl'):
        """Load pre-trained ML model and vectorizer"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.ml_model = model_data['ml_model']
            self.vectorizer = model_data['vectorizer']
            
            print(f"📂 Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"❌ Model file {filepath} not found")
            return False

# Main execution
if __name__ == "__main__":
    print("🤖 Starting Sentiment Analysis for Social Media Data")
    print("=" * 60)
    
    # Load processed data
    input_path = 'data/processed/processed_sentiment_data.csv'
    output_path = 'data/processed/analyzed_sentiment_data.csv'
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"📂 Loaded processed data from: {input_path}")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Initialize analyzer
        analyzer = SentimentAnalyzer()
        
        # Train ML model (using labeled data)
        if 'sentiment' in df.columns:
            print("\n🔬 Training machine learning model...")
            model_performance = analyzer.train_ml_model(df, 'lemmatized_text', 'sentiment')
        
        # Run comprehensive analysis
        print("\n🔍 Running comprehensive sentiment analysis...")
        analyzed_df = analyzer.analyze_comprehensive(df, 'lemmatized_text')
        
        # Save analyzed data
        analyzed_df.to_csv(output_path, index=False)
        print(f"💾 Analyzed data saved to: {output_path}")
        
        # Get summary
        summary = analyzer.get_sentiment_summary(analyzed_df)
        
        print("\n📊 Sentiment Analysis Summary:")
        print("=" * 50)
        
        for method in ['textblob_sentiment', 'vader_sentiment', 'ensemble_sentiment']:
            if f'{method}_distribution' in summary:
                print(f"\n{method.replace('_', ' ').title()} Distribution:")
                for sentiment, percentage in summary[f'{method}_distribution'].items():
                    print(f"   {sentiment}: {percentage:.1%}")
        
        # Method agreement
        if 'textblob_vader_agreement' in summary:
            print(f"\nTextBlob-VADER Agreement: {summary['textblob_vader_agreement']:.1%}")
        
        # Save summary to JSON
        summary_path = 'results/sentiment_analysis_summary.json'
        os.makedirs('results', exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Summary saved to: {summary_path}")
        
        print(f"\n✅ Sentiment analysis completed successfully!")
        print(f"🎯 Ready for Phase 4: Visualization!")
        
    except FileNotFoundError:
        print(f"❌ File not found: {input_path}")
        print("Please run preprocessing.py first to create the processed dataset.")
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        import traceback
        traceback.print_exc()