# visualizations.py
"""
Visualization Module for Social Media Sentiment Analysis
Brainwave Matrix Solutions - Task 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import os
from datetime import datetime
import json

class SentimentVisualizer:
    def __init__(self, save_path="results/plots/"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color scheme
        self.colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4',   # Steel Blue
            'background': '#f8f9fa'
        }
        
        print(f"✅ SentimentVisualizer initialized. Plots will be saved to: {save_path}")
    
    def plot_sentiment_distribution(self, df, methods=['textblob_sentiment', 'vader_sentiment', 'ensemble_sentiment']):
        """Create sentiment distribution plots for different methods"""
        
        # Count available methods
        available_methods = [method for method in methods if method in df.columns]
        n_methods = len(available_methods)
        
        if n_methods == 0:
            print("❌ No sentiment columns found")
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 6))
        if n_methods == 1:
            axes = [axes]
        
        for i, method in enumerate(available_methods):
            # Calculate percentages
            counts = df[method].value_counts()
            percentages = df[method].value_counts(normalize=True) * 100
            
            # Create pie chart
            colors = [self.colors[sentiment] for sentiment in counts.index]
            wedges, texts, autotexts = axes[i].pie(
                counts.values, 
                labels=counts.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            # Beautify text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            axes[i].set_title(f'{method.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        
        plt.suptitle('Sentiment Distribution Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(self.save_path, 'sentiment_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Sentiment distribution plot saved to: {filename}")
    
    def plot_sentiment_timeline(self, df):
        """Create timeline of sentiment over time"""
        if 'timestamp' not in df.columns:
            print("❌ No timestamp column found")
            return
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['datetime'].dt.date
        
        # Group by date and sentiment
        daily_sentiment = df.groupby(['date', 'ensemble_sentiment']).size().unstack(fill_value=0)
        
        # Calculate percentages
        daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Absolute counts
        daily_sentiment.plot(kind='area', stacked=True, ax=ax1, 
                           color=[self.colors['negative'], self.colors['neutral'], self.colors['positive']])
        ax1.set_title('Daily Sentiment Counts Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Posts')
        ax1.legend(title='Sentiment')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Percentages
        daily_sentiment_pct.plot(kind='line', ax=ax2, marker='o', linewidth=2,
                                color=[self.colors['negative'], self.colors['neutral'], self.colors['positive']])
        ax2.set_title('Daily Sentiment Percentage Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Percentage (%)')
        ax2.legend(title='Sentiment')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(self.save_path, 'sentiment_timeline.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Sentiment timeline plot saved to: {filename}")
    
    def plot_method_comparison(self, df):
        """Compare different sentiment analysis methods"""
        methods = ['textblob_sentiment', 'vader_sentiment', 'ensemble_sentiment']
        available_methods = [method for method in methods if method in df.columns]
        
        if len(available_methods) < 2:
            print("❌ Need at least 2 sentiment methods for comparison")
            return
        
        # Create comparison matrix
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Plot 1: Side-by-side comparison
        comparison_data = []
        for method in available_methods:
            method_counts = df[method].value_counts()
            for sentiment, count in method_counts.items():
                comparison_data.append({
                    'Method': method.replace('_sentiment', '').title(),
                    'Sentiment': sentiment,
                    'Count': count,
                    'Percentage': count / len(df) * 100
                })
        
        comp_df = pd.DataFrame(comparison_data)
        
        sns.barplot(data=comp_df, x='Method', y='Percentage', hue='Sentiment', ax=axes[0])
        axes[0].set_title('Method Comparison: Sentiment Percentages', fontweight='bold')
        axes[0].set_ylabel('Percentage (%)')
        
        # Plot 2: Agreement heatmap (if we have TextBlob and VADER)
        if 'textblob_sentiment' in df.columns and 'vader_sentiment' in df.columns:
            confusion_matrix = pd.crosstab(df['textblob_sentiment'], df['vader_sentiment'], normalize='index')
            sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues', ax=axes[1])
            axes[1].set_title('TextBlob vs VADER Agreement', fontweight='bold')
            axes[1].set_xlabel('VADER Sentiment')
            axes[1].set_ylabel('TextBlob Sentiment')
        
        # Plot 3: Sentiment scores distribution
        if 'textblob_polarity' in df.columns:
            axes[2].hist(df['textblob_polarity'], bins=30, alpha=0.7, color=self.colors['positive'])
            axes[2].set_title('TextBlob Polarity Score Distribution', fontweight='bold')
            axes[2].set_xlabel('Polarity Score')
            axes[2].set_ylabel('Frequency')
            axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.8)
        
        # Plot 4: VADER compound scores
        if 'vader_compound' in df.columns:
            axes[3].hist(df['vader_compound'], bins=30, alpha=0.7, color=self.colors['neutral'])
            axes[3].set_title('VADER Compound Score Distribution', fontweight='bold')
            axes[3].set_xlabel('Compound Score')
            axes[3].set_ylabel('Frequency')
            axes[3].axvline(x=0, color='red', linestyle='--', alpha=0.8)
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(self.save_path, 'method_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Method comparison plot saved to: {filename}")
    
    def create_wordclouds(self, df):
        """Create word clouds for different sentiments"""
        sentiments = ['positive', 'negative', 'neutral']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, sentiment in enumerate(sentiments):
            # Filter texts by sentiment
            sentiment_texts = df[df['ensemble_sentiment'] == sentiment]['lemmatized_text']
            
            if len(sentiment_texts) > 0:
                # Combine all texts
                text_combined = ' '.join(sentiment_texts.astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, 
                    height=300,
                    background_color='white',
                    colormap='viridis' if sentiment == 'positive' else 'Reds' if sentiment == 'negative' else 'Blues',
                    max_words=100,
                    relative_scaling=0.5
                ).generate(text_combined)
                
                # Plot
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment.title()} Sentiment Word Cloud', 
                                fontsize=14, fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No {sentiment} texts found', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{sentiment.title()} Sentiment Word Cloud')
                axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(self.save_path, 'sentiment_wordclouds.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Word clouds saved to: {filename}")
    
    def plot_topic_sentiment(self, df):
        """Analyze sentiment by topic"""
        if 'topic' not in df.columns:
            print("❌ No topic column found")
            return
        
        # Create topic-sentiment cross-tabulation
        topic_sentiment = pd.crosstab(df['topic'], df['ensemble_sentiment'], normalize='index') * 100
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot 1: Heatmap
        sns.heatmap(topic_sentiment, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=33.33, ax=ax1, cbar_kws={'label': 'Percentage (%)'})
        ax1.set_title('Sentiment Distribution by Topic (%)', fontweight='bold')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Topic')
        
        # Plot 2: Stacked bar chart
        topic_sentiment.plot(kind='bar', stacked=True, ax=ax2, 
                           color=[self.colors['negative'], self.colors['neutral'], self.colors['positive']])
        ax2.set_title('Sentiment Percentage by Topic', fontweight='bold')
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Percentage (%)')
        ax2.legend(title='Sentiment')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(self.save_path, 'topic_sentiment_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Topic sentiment analysis plot saved to: {filename}")
    
    def create_interactive_dashboard(self, df):
        """Create interactive Plotly dashboard"""
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Sentiment Timeline', 
                          'Method Comparison', 'Topic Analysis'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Plot 1: Sentiment Distribution (Pie Chart)
        sentiment_counts = df['ensemble_sentiment'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                name="Sentiment Distribution",
                marker_colors=[self.colors[s] for s in sentiment_counts.index]
            ),
            row=1, col=1
        )
        
        # Plot 2: Timeline (if timestamp available)
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['datetime'].dt.date
            daily_sentiment = df.groupby(['date', 'ensemble_sentiment']).size().reset_index(name='count')
            
            for sentiment in ['positive', 'negative', 'neutral']:
                sentiment_data = daily_sentiment[daily_sentiment['ensemble_sentiment'] == sentiment]
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data['count'],
                        mode='lines+markers',
                        name=sentiment,
                        line=dict(color=self.colors[sentiment])
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Method Comparison
        methods = ['textblob_sentiment', 'vader_sentiment', 'ensemble_sentiment']
        available_methods = [m for m in methods if m in df.columns]
        
        comparison_data = []
        for method in available_methods:
            for sentiment in ['positive', 'negative', 'neutral']:
                count = (df[method] == sentiment).sum()
                comparison_data.append({
                    'Method': method.replace('_sentiment', '').title(),
                    'Sentiment': sentiment,
                    'Count': count
                })
        
        comp_df = pd.DataFrame(comparison_data)
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_data = comp_df[comp_df['Sentiment'] == sentiment]
            fig.add_trace(
                go.Bar(
                    x=sentiment_data['Method'],
                    y=sentiment_data['Count'],
                    name=f'{sentiment}',
                    marker_color=self.colors[sentiment]
                ),
                row=2, col=1
            )
        
        # Plot 4: Topic Analysis
        if 'topic' in df.columns:
            topic_sentiment = df.groupby(['topic', 'ensemble_sentiment']).size().unstack(fill_value=0)
            
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in topic_sentiment.columns:
                    fig.add_trace(
                        go.Bar(
                            x=topic_sentiment.index,
                            y=topic_sentiment[sentiment],
                            name=f'{sentiment} (topic)',
                            marker_color=self.colors[sentiment],
                            showlegend=False
                        ),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title_text="Social Media Sentiment Analysis Dashboard",
            title_x=0.5,
            title_font_size=20,
            height=800,
            showlegend=True
        )
        
        # Save interactive plot
        filename = os.path.join(self.save_path, 'interactive_dashboard.html')
        fig.write_html(filename)
        
        print(f"💾 Interactive dashboard saved to: {filename}")
        return fig
    
    def plot_sentiment_scores_analysis(self, df):
        """Analyze and visualize sentiment scores in detail"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: TextBlob Polarity vs Subjectivity
        if 'textblob_polarity' in df.columns and 'textblob_subjectivity' in df.columns:
            scatter = axes[0,0].scatter(df['textblob_subjectivity'], df['textblob_polarity'], 
                                      c=df['ensemble_sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1}),
                                      cmap='RdYlGn', alpha=0.6)
            axes[0,0].set_xlabel('Subjectivity')
            axes[0,0].set_ylabel('Polarity')
            axes[0,0].set_title('TextBlob: Polarity vs Subjectivity', fontweight='bold')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0,0].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: VADER Score Components
        if all(col in df.columns for col in ['vader_positive', 'vader_negative', 'vader_neutral']):
            vader_components = df[['vader_positive', 'vader_negative', 'vader_neutral']].mean()
            axes[0,1].bar(vader_components.index, vader_components.values, 
                         color=[self.colors['positive'], self.colors['negative'], self.colors['neutral']])
            axes[0,1].set_title('Average VADER Score Components', fontweight='bold')
            axes[0,1].set_ylabel('Average Score')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Score distributions
        if 'textblob_polarity' in df.columns:
            axes[1,0].hist(df['textblob_polarity'], bins=30, alpha=0.7, color=self.colors['positive'])
            axes[1,0].set_title('TextBlob Polarity Distribution', fontweight='bold')
            axes[1,0].set_xlabel('Polarity Score')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.8)
        
        # Plot 4: VADER compound distribution
        if 'vader_compound' in df.columns:
            axes[1,1].hist(df['vader_compound'], bins=30, alpha=0.7, color=self.colors['neutral'])
            axes[1,1].set_title('VADER Compound Score Distribution', fontweight='bold')
            axes[1,1].set_xlabel('Compound Score')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(self.save_path, 'sentiment_scores_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Sentiment scores analysis saved to: {filename}")
    
    def generate_insights_report(self, df):
        """Generate comprehensive insights from the analysis"""
        
        insights = {
            'dataset_overview': {
                'total_posts': len(df),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else 'N/A',
                'unique_topics': df['topic'].nunique() if 'topic' in df.columns else 'N/A',
                'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 'N/A'
            },
            'sentiment_insights': {},
            'method_performance': {},
            'topic_insights': {},
            'key_findings': []
        }
        
        # Overall sentiment insights
        if 'ensemble_sentiment' in df.columns:
            sentiment_dist = df['ensemble_sentiment'].value_counts(normalize=True)
            insights['sentiment_insights'] = {
                'dominant_sentiment': sentiment_dist.index[0],
                'sentiment_distribution': sentiment_dist.to_dict(),
                'sentiment_balance': 'balanced' if sentiment_dist.std() < 0.1 else 'imbalanced'
            }
        
        # Method agreement
        if 'textblob_sentiment' in df.columns and 'vader_sentiment' in df.columns:
            agreement = (df['textblob_sentiment'] == df['vader_sentiment']).mean()
            insights['method_performance']['textblob_vader_agreement'] = round(agreement, 3)
        
        # Topic-based insights
        if 'topic' in df.columns:
            topic_sentiment = df.groupby('topic')['ensemble_sentiment'].apply(
                lambda x: x.value_counts(normalize=True).to_dict()
            ).to_dict()
            insights['topic_insights'] = topic_sentiment
        
        # Generate key findings
        findings = []
        
        # Finding 1: Overall sentiment
        if 'ensemble_sentiment' in df.columns:
            dominant = insights['sentiment_insights']['dominant_sentiment']
            percentage = insights['sentiment_insights']['sentiment_distribution'][dominant] * 100
            findings.append(f"Overall sentiment is predominantly {dominant} ({percentage:.1f}% of posts)")
        
        # Finding 2: Method agreement
        if 'textblob_vader_agreement' in insights['method_performance']:
            agreement = insights['method_performance']['textblob_vader_agreement'] * 100
            findings.append(f"TextBlob and VADER show {agreement:.1f}% agreement in sentiment classification")
        
        # Finding 3: Topic insights
        if 'topic' in df.columns:
            # Find most positive topic
            topic_pos_scores = {}
            for topic, sentiments in insights['topic_insights'].items():
                topic_pos_scores[topic] = sentiments.get('positive', 0)
            
            most_positive_topic = max(topic_pos_scores, key=topic_pos_scores.get)
            findings.append(f"'{most_positive_topic}' receives the most positive sentiment")
        
        # Finding 4: Text characteristics
        if 'word_count' in df.columns:
            avg_words = df['word_count'].mean()
            findings.append(f"Average post length is {avg_words:.1f} words after preprocessing")
        
        insights['key_findings'] = findings
        
        # Save insights
        insights_path = os.path.join('results', 'insights_report.json')
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        print(f"📄 Insights report saved to: {insights_path}")
        
        # Print key findings
        print("\n🔍 Key Insights:")
        print("=" * 40)
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding}")
        
        return insights
    
    def create_all_visualizations(self, df):
        """Create all visualizations in one go"""
        print("🎨 Creating all visualizations...")
        print("=" * 50)
        
        try:
            # 1. Sentiment distribution
            print("1/5: Creating sentiment distribution plots...")
            self.plot_sentiment_distribution(df)
            
            # 2. Timeline analysis
            print("2/5: Creating timeline analysis...")
            self.plot_sentiment_timeline(df)
            
            # 3. Method comparison
            print("3/5: Creating method comparison...")
            self.plot_method_comparison(df)
            
            # 4. Word clouds
            print("4/5: Creating word clouds...")
            self.create_wordclouds(df)
            
            # 5. Topic analysis
            print("5/5: Creating topic analysis...")
            self.plot_topic_sentiment(df)
            
            # 6. Interactive dashboard
            print("Bonus: Creating interactive dashboard...")
            self.create_interactive_dashboard(df)
            
            # 7. Generate insights
            print("📊 Generating insights report...")
            insights = self.generate_insights_report(df)
            
            print(f"\n🎉 All visualizations created successfully!")
            print(f"📁 Check the {self.save_path} folder for all plots")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()

# Main execution
if __name__ == "__main__":
    print("🎨 Starting Visualization Generation for Sentiment Analysis")
    print("=" * 65)
    
    # Load analyzed data
    input_path = 'data/processed/analyzed_sentiment_data.csv'
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"📂 Loaded analyzed data from: {input_path}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📈 Available columns: {list(df.columns)}")
        
        # Initialize visualizer
        visualizer = SentimentVisualizer()
        
        # Create all visualizations
        visualizer.create_all_visualizations(df)
        
        print(f"\n✅ Visualization generation completed!")
        print(f"🎯 Ready for Phase 4: Report Generation!")
        
    except FileNotFoundError:
        print(f"❌ File not found: {input_path}")
        print("Please run sentiment_analyzer.py first to create the analyzed dataset.")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()