# config.py
"""
Configuration file for Social Media Sentiment Analysis Project
Brainwave Matrix Solutions - Task 2
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed')
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'sample')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')
PLOTS_PATH = os.path.join(RESULTS_PATH, 'plots')
MODELS_PATH = os.path.join(BASE_DIR, 'models')

# Data sources configuration
DATA_SOURCES = {
    'kaggle_sentiment140': {
        'url': 'https://www.kaggle.com/datasets/kazanova/sentiment140',
        'description': 'Large Twitter sentiment dataset with 1.6M tweets'
    },
    'twitter_airline': {
        'url': 'https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment',
        'description': 'Twitter data about US airline sentiment'
    },
    'amazon_reviews': {
        'url': 'https://www.kaggle.com/datasets/bittlingmayer/amazonreviews',
        'description': 'Amazon product reviews sentiment'
    }
}

# Analysis configuration
SENTIMENT_METHODS = ['textblob', 'vader', 'ml_model']
VISUALIZATION_TYPES = ['distribution', 'timeline', 'wordcloud', 'comparison']

# Model parameters
ML_MODEL_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95
}

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (12, 8)
COLOR_PALETTE = ['#2E8B57', '#DC143C', '#4682B4']  # Green, Red, Blue

# Text preprocessing parameters
PREPROCESSING_CONFIG = {
    'min_text_length': 3,  # Minimum number of words
    'remove_stopwords': True,
    'apply_lemmatization': True,
    'handle_negations': True
}

# Project metadata
PROJECT_INFO = {
    'title': 'Social Media Sentiment Analysis',
    'author': 'Gaurav Kaushik',  # Update this with your actual name
    'organization': 'Brainwave Matrix Solutions',
    'task_number': 2,
    'submission_deadline': '15-20 days',
    'version': '1.0'
}