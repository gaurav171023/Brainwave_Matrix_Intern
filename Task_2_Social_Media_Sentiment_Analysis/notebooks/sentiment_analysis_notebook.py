# This is the content for: notebooks/01_complete_sentiment_analysis.ipynb
# Copy each cell section into separate Jupyter notebook cells

# =============================================================================
# CELL 1: Project Introduction and Setup
# =============================================================================
"""
# Social Media Sentiment Analysis
## Brainwave Matrix Solutions - Task 2

**Project Overview**: Comprehensive sentiment analysis of social media data using multiple NLP techniques

**Intern**: [Your Name]  
**Date**: August 31, 2025  
**Objective**: Analyze public sentiment towards various topics using advanced NLP methods

### Key Goals:
1. Implement multiple sentiment analysis approaches
2. Compare method performance and accuracy  
3. Generate actionable insights from social media data
4. Create professional visualizations and reports
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("All libraries imported successfully!")
print("Ready to begin sentiment analysis...")

# =============================================================================
# CELL 2: Data Loading and Initial Exploration
# =============================================================================

# Load the analyzed dataset
df = pd.read_csv('../data/processed/analyzed_sentiment_data.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

print("DATASET OVERVIEW")
print("=" * 40)
print(f"Total Posts: {len(df):,}")
print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
print(f"Topics Analyzed: {df['topic'].nunique()}")
print(f"Columns Available: {len(df.columns)}")

# Display basic information
print(f"\nDataset Shape: {df.shape}")
print(f"\nColumn Names:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

# Show first few rows
print(f"\nSample Data:")
df.head()

# =============================================================================
# CELL 3: Data Quality and Distribution Analysis
# =============================================================================

print("DATA QUALITY ASSESSMENT")
print("=" * 30)

# Check for missing values
missing_data = df.isnull().sum()
print("Missing Values:")
for col, missing in missing_data.items():
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
if missing_data.sum() == 0:
    print("  No missing values found!")

# Sentiment distribution
print(f"\nSENTIMENT DISTRIBUTION")
print("=" * 25)
sentiment_counts = df['ensemble_sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment.title()}: {count:,} posts ({percentage:.1f}%)")

# Topic distribution  
print(f"\nTOPIC DISTRIBUTION")
print("=" * 20)
topic_counts = df['topic'].value_counts()
for topic, count in topic_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {topic}: {count:,} posts ({percentage:.1f}%)")

# Text length analysis
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"\nTEXT STATISTICS")
print("=" * 18)
print(f"  Average text length: {df['text_length'].mean():.0f} characters")
print(f"  Average word count: {df['word_count'].mean():.0f} words")
print(f"  Shortest text: {df['text_length'].min()} characters")
print(f"  Longest text: {df['text_length'].max()} characters")

# =============================================================================
# CELL 4: Sentiment Method Comparison
# =============================================================================

print("SENTIMENT METHOD COMPARISON")
print("=" * 35)

# Calculate agreement between methods
methods = ['textblob_sentiment', 'vader_sentiment', 'ml_sentiment']
method_names = ['TextBlob', 'VADER', 'ML Model']

print("Method Agreement with Ensemble:")
for method, name in zip(methods, method_names):
    agreement = (df[method] == df['ensemble_sentiment']).mean() * 100
    print(f"  {name}: {agreement:.1f}% agreement")

# Cross-method agreement matrix
agreement_matrix = pd.DataFrame(index=method_names, columns=method_names)

for i, method1 in enumerate(methods):
    for j, method2 in enumerate(methods):
        if i == j:
            agreement_matrix.iloc[i, j] = 100.0
        else:
            agreement = (df[method1] == df[method2]).mean() * 100
            agreement_matrix.iloc[i, j] = agreement

print(f"\nCross-Method Agreement Matrix:")
print(agreement_matrix.round(1))

# Visualize method comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Method agreement heatmap
sns.heatmap(agreement_matrix.astype(float), annot=True, cmap='Blues', 
            ax=axes[0], fmt='.1f', cbar_kws={'label': 'Agreement %'})
axes[0].set_title('Method Agreement Matrix')

# Method distribution comparison
method_dist = pd.DataFrame()
for method, name in zip(methods, method_names):
    counts = df[method].value_counts()
    method_dist[name] = [counts.get('positive', 0), counts.get('negative', 0), counts.get('neutral', 0)]

method_dist.index = ['Positive', 'Negative', 'Neutral']
method_dist.plot(kind='bar', ax=axes[1], width=0.8)
axes[1].set_title('Sentiment Distribution by Method')
axes[1].set_ylabel('Count')
axes[1].legend(title='Method')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# =============================================================================
# CELL 5: Temporal Analysis and Trends
# =============================================================================

print("TEMPORAL SENTIMENT ANALYSIS")
print("=" * 32)

# Daily sentiment aggregation
daily_sentiment = df.groupby(['date', 'ensemble_sentiment']).size().unstack(fill_value=0)
daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100

# Calculate rolling averages
window = 3  # 3-day rolling average
daily_sentiment_smooth = daily_sentiment.rolling(window=window, center=True).mean()

print(f"Date range: {daily_sentiment.index.min()} to {daily_sentiment.index.max()}")
print(f"Days analyzed: {len(daily_sentiment)}")

# Create temporal visualizations
fig, axes = plt.subplots(3, 1, figsize=(15, 15))

# 1. Daily sentiment counts
daily_sentiment.plot(kind='line', ax=axes[0], marker='o', linewidth=2, markersize=4)
axes[0].set_title('Daily Sentiment Counts Over Time')
axes[0].set_ylabel('Number of Posts')
axes[0].legend(title='Sentiment')
axes[0].grid(True, alpha=0.3)

# 2. Percentage distribution over time
daily_sentiment_pct.plot(kind='area', ax=axes[1], alpha=0.7)
axes[1].set_title('Daily Sentiment Percentage Distribution')
axes[1].set_ylabel('Percentage (%)')
axes[1].legend(title='Sentiment')
axes[1].grid(True, alpha=0.3)

# 3. Smoothed trends
daily_sentiment_smooth.plot(kind='line', ax=axes[2], linewidth=3, alpha=0.8)
axes[2].set_title(f'Sentiment Trends ({window}-Day Moving Average)')
axes[2].set_ylabel('Average Posts per Day')
axes[2].legend(title='Sentiment')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find peak sentiment days
print("\nPEAK SENTIMENT DAYS")
print("=" * 20)
for sentiment in daily_sentiment.columns:
    peak_day = daily_sentiment[sentiment].idxmax()
    peak_count = daily_sentiment[sentiment].max()
    print(f"  {sentiment.title()}: {peak_day} ({peak_count} posts)")

# =============================================================================
# CELL 6: Topic-Based Analysis
# =============================================================================

print("TOPIC-BASED SENTIMENT ANALYSIS")
print("=" * 35)

# Topic sentiment breakdown
topic_sentiment = pd.crosstab(df['topic'], df['ensemble_sentiment'])
topic_sentiment_pct = pd.crosstab(df['topic'], df['ensemble_sentiment'], normalize='index') * 100

print("Topic Sentiment Distribution (%):")
print(topic_sentiment_pct.round(1))

# Calculate topic sentiment scores
topic_scores = {}
for topic in df['topic'].unique():
    topic_data = df[df['topic'] == topic]
    pos_pct = (topic_data['ensemble_sentiment'] == 'positive').mean() * 100
    neg_pct = (topic_data['ensemble_sentiment'] == 'negative').mean() * 100
    sentiment_score = pos_pct - neg_pct  # Net sentiment score
    topic_scores[topic] = sentiment_score

# Sort topics by sentiment score
sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

print(f"\nTOPIC SENTIMENT RANKING")
print("=" * 25)
for i, (topic, score) in enumerate(sorted_topics):
    sentiment_label = "Positive" if score > 10 else "Negative" if score < -10 else "Neutral"
    print(f"  {i+1}. {topic}: {score:+.1f}% ({sentiment_label})")

# Visualize topic analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Topic-Based Sentiment Analysis', fontsize=16, fontweight='bold')

# 1. Topic sentiment heatmap
sns.heatmap(topic_sentiment_pct, annot=True, cmap='RdYlGn', ax=axes[0,0], fmt='.1f')
axes[0,0].set_title('Topic Sentiment Heatmap (%)')

# 2. Topic sentiment scores
topics, scores = zip(*sorted_topics)
colors = ['green' if score > 0 else 'red' if score < -10 else 'gray' for score in scores]
axes[0,1].barh(topics, scores, color=colors)
axes[0,1].set_title('Net Sentiment Score by Topic')
axes[0,1].set_xlabel('Net Sentiment Score (Positive % - Negative %)')
axes[0,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)

# 3. Topic post volume
topic_volume = df['topic'].value_counts()
axes[1,0].bar(topic_volume.index, topic_volume.values, color='skyblue')
axes[1,0].set_title('Post Volume by Topic')
axes[1,0].set_ylabel('Number of Posts')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Engagement by topic and sentiment
if 'likes' in df.columns:
    engagement_data = df.groupby(['topic', 'ensemble_sentiment'])['likes'].mean().unstack()
    engagement_data.plot(kind='bar', ax=axes[1,1], width=0.8)
    axes[1,1].set_title('Average Likes by Topic and Sentiment')
    axes[1,1].set_ylabel('Average Likes')
    axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# =============================================================================
# CELL 7: Advanced Text Analysis
# =============================================================================

from collections import Counter
import re

print("ADVANCED TEXT ANALYSIS")
print("=" * 25)

# Function to extract key words from text
def extract_keywords(text_series, top_n=10):
    """Extract most common words from text series"""
    all_words = ' '.join(text_series.dropna()).lower()
    # Remove common stop words and extract meaningful words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_words)
    return Counter(words).most_common(top_n)

# Extract keywords by sentiment
print("TOP KEYWORDS BY SENTIMENT:")
print("=" * 30)

for sentiment in ['positive', 'negative', 'neutral']:
    sentiment_text = df[df['ensemble_sentiment'] == sentiment]['cleaned_text']
    keywords = extract_keywords(sentiment_text, 10)
    
    print(f"\n{sentiment.upper()} SENTIMENT:")
    for i, (word, count) in enumerate(keywords, 1):
        print(f"  {i:2d}. {word} ({count})")

# Text length by sentiment
print(f"\nTEXT LENGTH BY SENTIMENT")
print("=" * 28)
length_stats = df.groupby('ensemble_sentiment')['word_count'].agg(['mean', 'median', 'std'])
print(length_stats.round(1))

# Visualize text characteristics
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Text Characteristics Analysis', fontsize=16, fontweight='bold')

# 1. Word count distribution by sentiment
for sentiment in df['ensemble_sentiment'].unique():
    sentiment_data = df[df['ensemble_sentiment'] == sentiment]['word_count']
    axes[0,0].hist(sentiment_data, alpha=0.6, label=sentiment, bins=20)
axes[0,0].set_title('Word Count Distribution by Sentiment')
axes[0,0].set_xlabel('Word Count')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# 2. Average word count by sentiment
word_count_avg = df.groupby('ensemble_sentiment')['word_count'].mean()
axes[0,1].bar(word_count_avg.index, word_count_avg.values, 
              color=['#2E8B57', '#DC143C', '#4682B4'])
axes[0,1].set_title('Average Word Count by Sentiment')
axes[0,1].set_ylabel('Average Words')

# 3. Text length vs engagement correlation
if 'likes' in df.columns:
    axes[1,0].scatter(df['word_count'], df['likes'], alpha=0.5, c=df['ensemble_sentiment'].map(
        {'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#4682B4'}))
    axes[1,0].set_title('Text Length vs Engagement')
    axes[1,0].set_xlabel('Word Count')
    axes[1,0].set_ylabel('Likes')

# 4. Sentiment confidence distribution
if 'ensemble_confidence' in df.columns:
    df['ensemble_confidence'].hist(bins=30, ax=axes[1,1], color='lightcoral', alpha=0.7)
    axes[1,1].set_title('Prediction Confidence Distribution')
    axes[1,1].set_xlabel('Confidence Score')
    axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# =============================================================================
# CELL 8: Interactive Visualizations
# =============================================================================

print("Creating Interactive Visualizations...")

# 1. Interactive sentiment timeline
daily_data = df.groupby(['date', 'ensemble_sentiment']).size().unstack(fill_value=0).reset_index()

fig = go.Figure()
colors_plotly = {'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#4682B4'}

for sentiment in ['positive', 'negative', 'neutral']:
    if sentiment in daily_data.columns:
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data[sentiment],
            mode='lines+markers',
            name=sentiment.title(),
            line=dict(width=3, color=colors_plotly[sentiment]),
            marker=dict(size=8),
            hovertemplate=f'<b>{sentiment.title()}</b><br>' +
                         'Date: %{x}<br>' +
                         'Posts: %{y}<br>' +
                         '<extra></extra>'
        ))

fig.update_layout(
    title='Interactive Sentiment Timeline',
    xaxis_title='Date',
    yaxis_title='Number of Posts',
    hovermode='x unified',
    template='plotly_white',
    legend=dict(x=0.02, y=0.98)
)

fig.show()

# 2. Interactive topic analysis
topic_sentiment_df = pd.crosstab(df['topic'], df['ensemble_sentiment'])

fig_topic = go.Figure()
for sentiment in ['positive', 'negative', 'neutral']:
    if sentiment in topic_sentiment_df.columns:
        fig_topic.add_trace(go.Bar(
            name=sentiment.title(),
            x=topic_sentiment_df.index,
            y=topic_sentiment_df[sentiment],
            marker_color=colors_plotly[sentiment],
            hovertemplate='<b>%{x}</b><br>' +
                         f'{sentiment.title()}: %{y}<br>' +
                         '<extra></extra>'
        ))

fig_topic.update_layout(
    title='Interactive Topic Sentiment Analysis',
    xaxis_title='Topic',
    yaxis_title='Number of Posts',
    barmode='group',
    template='plotly_white'
)

fig_topic.show()

# Save interactive plots
fig.write_html('../results/plots/interactive_timeline.html')
fig_topic.write_html('../results/plots/interactive_topic_analysis.html')
print("Interactive visualizations saved as HTML files!")

# =============================================================================
# CELL 9: Statistical Analysis and Insights
# =============================================================================

print("STATISTICAL ANALYSIS AND KEY INSIGHTS")
print("=" * 45)

# 1. Confidence analysis
if 'ensemble_confidence' in df.columns:
    print("CONFIDENCE ANALYSIS:")
    print("=" * 20)
    
    confidence_stats = df.groupby('ensemble_sentiment')['ensemble_confidence'].agg(['mean', 'std', 'min', 'max'])
    print(confidence_stats.round(3))
    
    # High confidence predictions
    high_conf_threshold = 0.8
    high_conf = df[df['ensemble_confidence'] > high_conf_threshold]
    print(f"\nHigh Confidence Predictions (>{high_conf_threshold}):")
    print(f"  Total: {len(high_conf)} posts ({len(high_conf)/len(df)*100:.1f}%)")
    
    high_conf_sentiment = high_conf['ensemble_sentiment'].value_counts()
    for sentiment, count in high_conf_sentiment.items():
        print(f"  {sentiment.title()}: {count} posts")

# 2. Engagement analysis
if 'likes' in df.columns and 'retweets' in df.columns:
    print(f"\nENGAGEMENT ANALYSIS:")
    print("=" * 20)
    
    engagement_stats = df.groupby('ensemble_sentiment')[['likes', 'retweets']].agg(['mean', 'median'])
    print("Average Engagement by Sentiment:")
    print(engagement_stats.round(1))
    
    # Most engaging posts
    df['total_engagement'] = df['likes'] + df['retweets']
    most_engaging = df.nlargest(5, 'total_engagement')[['text', 'ensemble_sentiment', 'topic', 'total_engagement']]
    
    print(f"\nMOST ENGAGING POSTS:")
    for idx, row in most_engaging.iterrows():
        print(f"  Topic: {row['topic']} | Sentiment: {row['ensemble_sentiment']} | Engagement: {row['total_engagement']}")
        print(f"  Text: {row['text'][:80]}...")
        print()

# 3. Topic performance insights
print("TOPIC PERFORMANCE INSIGHTS:")
print("=" * 30)

topic_stats = df.groupby('topic').agg({
    'ensemble_sentiment': lambda x: (x == 'positive').mean() * 100,  # Positive %
    'likes': 'mean',
    'retweets': 'mean'
}).round(2)

topic_stats.columns = ['Positive_Percentage', 'Avg_Likes', 'Avg_Retweets']
topic_stats = topic_stats.sort_values('Positive_Percentage', ascending=False)

print("Topic Rankings (by positivity):")
for topic, row in topic_stats.iterrows():
    print(f"  {topic}:")
    print(f"    Positive Sentiment: {row['Positive_Percentage']:.1f}%")
    print(f"    Avg Engagement: {row['Avg_Likes']:.0f} likes, {row['Avg_Retweets']:.0f} retweets")

# =============================================================================
# CELL 10: Final Summary and Export Results
# =============================================================================

print("GENERATING FINAL SUMMARY REPORT")
print("=" * 40)

# Create comprehensive summary
final_summary = {
    'project_info': {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_posts_analyzed': int(len(df)),
        'analysis_period': f"{df['date'].min()} to {df['date'].max()}",
        'topics_covered': list(df['topic'].unique()),
        'methods_used': ['TextBlob', 'VADER', 'Machine Learning', 'Ensemble']
    },
    'key_metrics': {
        'overall_sentiment_distribution': df['ensemble_sentiment'].value_counts(normalize=True).round(3).to_dict(),
        'average_confidence': float(df['ensemble_confidence'].mean()) if 'ensemble_confidence' in df.columns else 'N/A',
        'high_confidence_predictions': int(len(df[df['ensemble_confidence'] > 0.8])) if 'ensemble_confidence' in df.columns else 'N/A'
    },
    'method_performance': {
        'textblob_agreement': float((df['textblob_sentiment'] == df['ensemble_sentiment']).mean() * 100),
        'vader_agreement': float((df['vader_sentiment'] == df['ensemble_sentiment']).mean() * 100),
        'ml_agreement': float((df['ml_sentiment'] == df['ensemble_sentiment']).mean() * 100)
    },
    'topic_insights': {
        'most_positive_topic': topic_stats.index[0],
        'most_negative_topic': topic_stats.index[-1],
        'topic_rankings': topic_stats.to_dict('index')
    }
}

# Save final summary
with open('../results/reports/final_analysis_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2, default=str)

# Create executive summary
print("EXECUTIVE SUMMARY")
print("=" * 20)
print(f"✓ Analyzed {len(df):,} social media posts")
print(f"✓ Covered {df['topic'].nunique()} different topics")
print(f"✓ Used {len(methods)} different sentiment analysis methods")
print(f"✓ Achieved {final_summary['method_performance']['ml_agreement']:.1f}% ML model agreement")

dominant_sentiment = df['ensemble_sentiment'].value_counts().index[0]
dominant_pct = df['ensemble_sentiment'].value_counts(normalize=True).iloc[0] * 100
print(f"✓ Overall sentiment trend: {dominant_sentiment} ({dominant_pct:.1f}%)")

print(f"\nMost positive topic: {final_summary['topic_insights']['most_positive_topic']}")
print(f"Most negative topic: {final_summary['topic_insights']['most_negative_topic']}")

print(f"\nFiles Generated:")
print(f"  • Complete analyzed dataset")
print(f"  • Statistical visualizations") 
print(f"  • Interactive HTML dashboards")
print(f"  • Word clouds and trend analysis")
print(f"  • Comprehensive JSON summary")

print(f"\n🎉 SENTIMENT ANALYSIS PROJECT COMPLETE!")
print("Ready for LinkedIn post and final submission!")