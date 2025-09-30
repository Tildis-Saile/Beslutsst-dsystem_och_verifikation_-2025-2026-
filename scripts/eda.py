import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpotifyEDA:
    """
    A comprehensive EDA class for Spotify tracks dataset analysis.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the EDA class with the dataset.
        
        Args:
            data_path (str, optional): Path to the CSV file containing Spotify tracks data.
                                     If None, will automatically find the dataset.
        """
        if data_path is None:
            self.data_path = self._find_dataset()
        else:
            self.data_path = data_path
        self.df = None
        self.load_data()
    
    def _find_dataset(self):
        """Automatically find the dataset file."""
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Possible paths to try
        possible_paths = [
            os.path.join(script_dir, '../data/dataset.csv'),  # From scripts folder
            os.path.join(script_dir, '../../data/dataset.csv'),  # From deeper subfolder
            'data/dataset.csv',  # From project root
            '../data/dataset.csv',  # From scripts folder (relative)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        # If not found, return the default relative path
        return '../data/dataset.csv'
    
    def load_data(self):
        """Load and perform initial data inspection."""
        print(f"Loading Spotify tracks dataset from: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully!")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def basic_info(self):
        """Display basic information about the dataset."""
        print("\n" + "="*60)
        print(" BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"\n Dataset Dimensions: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        print(f"\n Column Names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\n Data Types:")
        print(self.df.dtypes)
        
        print(f"\n Missing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        }).sort_values('Missing Count', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print(f"\n Basic Statistics for Numeric Columns:")
        print(self.df.describe())
    
    def analyze_tracks(self):
        """Analyze track-related features."""
        print("\n" + "="*60)
        print(" TRACK ANALYSIS")
        print("="*60)
        
        # Track name analysis
        print(f"\n Total unique tracks: {self.df['track_name'].nunique():,}")
        print(f" Duplicate tracks: {self.df['track_name'].duplicated().sum():,}")
        
        # Most popular tracks
        print(f"\n Top 10 Most Popular Tracks:")
        top_tracks = self.df.nlargest(10, 'popularity')[['track_name', 'artists', 'popularity', 'track_genre']]
        for i, (_, row) in enumerate(top_tracks.iterrows(), 1):
            print(f"  {i:2d}. {row['track_name']} - {row['artists']} ({row['popularity']})")
        
        # Track duration analysis
        print(f"\n Track Duration Analysis:")
        duration_min = self.df['duration_ms'] / (1000 * 60)  # Convert to minutes
        print(f"  Average duration: {duration_min.mean():.2f} minutes")
        print(f"  Median duration: {duration_min.median():.2f} minutes")
        print(f"  Shortest track: {duration_min.min():.2f} minutes")
        print(f"  Longest track: {duration_min.max():.2f} minutes")
        
        # Explicit content
        explicit_pct = (self.df['explicit'].sum() / len(self.df)) * 100
        print(f"\n Explicit content: {self.df['explicit'].sum():,} tracks ({explicit_pct:.1f}%)")
    
    def analyze_genres(self):
        """Analyze genre distribution and patterns."""
        print("\n" + "="*60)
        print(" GENRE ANALYSIS")
        print("="*60)
        
        # Genre distribution
        genre_counts = self.df['track_genre'].value_counts()
        print(f"\n Total unique genres: {len(genre_counts)}")
        print(f"\n Top 15 Genres by Track Count:")
        for i, (genre, count) in enumerate(genre_counts.head(15).items(), 1):
            pct = (count / len(self.df)) * 100
            print(f"  {i:2d}. {genre:<20} {count:>6,} tracks ({pct:5.1f}%)")
        
        # Genre popularity analysis
        print(f"\n Most Popular Genres (by average popularity):")
        genre_popularity = self.df.groupby('track_genre')['popularity'].agg(['mean', 'count']).round(2)
        genre_popularity = genre_popularity[genre_popularity['count'] >= 100]  # Filter genres with at least 100 tracks
        top_popular_genres = genre_popularity.nlargest(10, 'mean')
        for i, (genre, row) in enumerate(top_popular_genres.iterrows(), 1):
            print(f"  {i:2d}. {genre:<20} Avg: {row['mean']:5.1f} ({row['count']:>4} tracks)")
    
    def analyze_artists(self):
        """Analyze artist-related features."""
        print("\n" + "="*60)
        print(" ARTIST ANALYSIS")
        print("="*60)
        
        # Artist statistics
        print(f"\n Total unique artists: {self.df['artists'].nunique():,}")
        
        # Most prolific artists
        print(f"\n Top 15 Most Prolific Artists (by track count):")
        artist_counts = self.df['artists'].value_counts()
        for i, (artist, count) in enumerate(artist_counts.head(15).items(), 1):
            print(f"  {i:2d}. {artist:<30} {count:>4} tracks")
        
        # Most popular artists
        print(f"\n Most Popular Artists (by average popularity):")
        artist_popularity = self.df.groupby('artists')['popularity'].agg(['mean', 'count']).round(2)
        artist_popularity = artist_popularity[artist_popularity['count'] >= 10]  # Filter artists with at least 10 tracks
        top_popular_artists = artist_popularity.nlargest(10, 'mean')
        for i, (artist, row) in enumerate(top_popular_artists.iterrows(), 1):
            print(f"  {i:2d}. {artist:<30} Avg: {row['mean']:5.1f} ({row['count']:>3} tracks)")
    
    def analyze_audio_features(self):
        """Analyze audio features and their relationships."""
        print("\n" + "="*60)
        print(" AUDIO FEATURES ANALYSIS")
        print("="*60)
        
        # Audio features columns
        audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                         'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']
        
        print(f"\n Audio Features Summary Statistics:")
        audio_stats = self.df[audio_features].describe()
        print(audio_stats.round(3))
        
        # Feature correlations
        print(f"\n Audio Features Correlation Matrix:")
        corr_matrix = self.df[audio_features].corr()
        
        # Find strongest correlations (excluding self-correlation)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"\n Top 10 Strongest Feature Correlations:")
        for i, (feat1, feat2, corr) in enumerate(corr_pairs[:10], 1):
            print(f"  {i:2d}. {feat1:<15} <-> {feat2:<15} {corr:6.3f}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations for the dataset."""
        print("\n" + "="*60)
        print(" CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 10
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Genre distribution (top 15)
        plt.subplot(4, 3, 1)
        genre_counts = self.df['track_genre'].value_counts().head(15)
        genre_counts.plot(kind='bar', color='skyblue', alpha=0.7)
        plt.title('Top 15 Genres by Track Count', fontsize=12, fontweight='bold')
        plt.xlabel('Genre')
        plt.ylabel('Number of Tracks')
        plt.xticks(rotation=45, ha='right')
        
        # 2. Popularity distribution
        plt.subplot(4, 3, 2)
        plt.hist(self.df['popularity'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.title('Track Popularity Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Popularity Score')
        plt.ylabel('Frequency')
        plt.axvline(self.df['popularity'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["popularity"].mean():.1f}')
        plt.legend()
        
        # 3. Track duration distribution
        plt.subplot(4, 3, 3)
        duration_min = self.df['duration_ms'] / (1000 * 60)
        plt.hist(duration_min, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
        plt.title('Track Duration Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Frequency')
        plt.axvline(duration_min.mean(), color='red', linestyle='--', 
                   label=f'Mean: {duration_min.mean():.1f} min')
        plt.legend()
        
        # 4. Audio features heatmap
        plt.subplot(4, 3, 4)
        audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        corr_matrix = self.df[audio_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Audio Features Correlation Heatmap', fontsize=12, fontweight='bold')
        
        # 5. Energy vs Valence scatter plot
        plt.subplot(4, 3, 5)
        plt.scatter(self.df['energy'], self.df['valence'], alpha=0.5, s=1, color='purple')
        plt.xlabel('Energy')
        plt.ylabel('Valence')
        plt.title('Energy vs Valence Scatter Plot', fontsize=12, fontweight='bold')
        
        # 6. Danceability vs Energy scatter plot
        plt.subplot(4, 3, 6)
        plt.scatter(self.df['danceability'], self.df['energy'], alpha=0.5, s=1, color='orange')
        plt.xlabel('Danceability')
        plt.ylabel('Energy')
        plt.title('Danceability vs Energy Scatter Plot', fontsize=12, fontweight='bold')
        
        # 7. Genre popularity box plot
        plt.subplot(4, 3, 7)
        top_genres = self.df['track_genre'].value_counts().head(8).index
        genre_pop_data = [self.df[self.df['track_genre'] == genre]['popularity'].values 
                         for genre in top_genres]
        plt.boxplot(genre_pop_data, labels=top_genres)
        plt.title('Popularity Distribution by Top Genres', fontsize=12, fontweight='bold')
        plt.xlabel('Genre')
        plt.ylabel('Popularity')
        plt.xticks(rotation=45, ha='right')
        
        # 8. Tempo distribution
        plt.subplot(4, 3, 8)
        plt.hist(self.df['tempo'], bins=50, color='lightblue', alpha=0.7, edgecolor='black')
        plt.title('Tempo Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Tempo (BPM)')
        plt.ylabel('Frequency')
        plt.axvline(self.df['tempo'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["tempo"].mean():.1f} BPM')
        plt.legend()
        
        # 9. Key distribution
        plt.subplot(4, 3, 9)
        key_counts = self.df['key'].value_counts().sort_index()
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_counts.index = [key_names[i] for i in key_counts.index]
        key_counts.plot(kind='bar', color='lightgreen', alpha=0.7)
        plt.title('Key Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Musical Key')
        plt.ylabel('Number of Tracks')
        plt.xticks(rotation=45)
        
        # 10. Mode distribution
        plt.subplot(4, 3, 10)
        mode_counts = self.df['mode'].value_counts()
        mode_labels = ['Minor', 'Major']
        mode_counts.index = mode_labels
        mode_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        plt.title('Major vs Minor Mode Distribution', fontsize=12, fontweight='bold')
        plt.ylabel('')
        
        # 11. Acousticness vs Instrumentalness
        plt.subplot(4, 3, 11)
        plt.scatter(self.df['acousticness'], self.df['instrumentalness'], alpha=0.5, s=1, color='teal')
        plt.xlabel('Acousticness')
        plt.ylabel('Instrumentalness')
        plt.title('Acousticness vs Instrumentalness', fontsize=12, fontweight='bold')
        
        # 12. Speechiness vs Liveness
        plt.subplot(4, 3, 12)
        plt.scatter(self.df['speechiness'], self.df['liveness'], alpha=0.5, s=1, color='gold')
        plt.xlabel('Speechiness')
        plt.ylabel('Liveness')
        plt.title('Speechiness vs Liveness', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../resources/spotify_eda_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Visualizations saved as 'spotify_eda_visualizations.png' in the resources folder.")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print(" EDA SUMMARY REPORT")
        print("="*60)
        
        # Data quality summary
        missing_data = self.df.isnull().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        data_completeness = ((total_cells - missing_data) / total_cells) * 100
        
        print(f"\n DATA QUALITY SUMMARY:")
        print(f"  • Total tracks: {len(self.df):,}")
        print(f"  • Total features: {len(self.df.columns)}")
        print(f"  • Data completeness: {data_completeness:.1f}%")
        print(f"  • Missing values: {missing_data:,}")
        
        # Genre insights
        unique_genres = self.df['track_genre'].nunique()
        most_common_genre = self.df['track_genre'].mode().iloc[0]
        most_common_count = self.df['track_genre'].value_counts().iloc[0]
        
        print(f"\n GENRE INSIGHTS:")
        print(f"  • Unique genres: {unique_genres}")
        print(f"  • Most common genre: {most_common_genre} ({most_common_count:,} tracks)")
        print(f"  • Genre diversity: {most_common_count/len(self.df)*100:.1f}% of tracks are {most_common_genre}")
        
        # Audio features insights
        high_energy_tracks = (self.df['energy'] > 0.8).sum()
        high_danceability_tracks = (self.df['danceability'] > 0.8).sum()
        acoustic_tracks = (self.df['acousticness'] > 0.8).sum()
        
        print(f"\n AUDIO FEATURES INSIGHTS:")
        print(f"  • High energy tracks (>0.8): {high_energy_tracks:,} ({high_energy_tracks/len(self.df)*100:.1f}%)")
        print(f"  • High danceability tracks (>0.8): {high_danceability_tracks:,} ({high_danceability_tracks/len(self.df)*100:.1f}%)")
        print(f"  • Acoustic tracks (>0.8): {acoustic_tracks:,} ({acoustic_tracks/len(self.df)*100:.1f}%)")
        
        # Popularity insights
        avg_popularity = self.df['popularity'].mean()
        high_popularity_tracks = (self.df['popularity'] > 70).sum()
        
        print(f"\n POPULARITY INSIGHTS:")
        print(f"  • Average popularity: {avg_popularity:.1f}")
        print(f"  • High popularity tracks (>70): {high_popularity_tracks:,} ({high_popularity_tracks/len(self.df)*100:.1f}%)")
        
        # Duration insights
        avg_duration_min = (self.df['duration_ms'] / (1000 * 60)).mean()
        short_tracks = ((self.df['duration_ms'] / (1000 * 60)) < 2).sum()
        long_tracks = ((self.df['duration_ms'] / (1000 * 60)) > 5).sum()
        
        print(f"\n DURATION INSIGHTS:")
        print(f"  • Average duration: {avg_duration_min:.1f} minutes")
        print(f"  • Short tracks (<2 min): {short_tracks:,} ({short_tracks/len(self.df)*100:.1f}%)")
        print(f"  • Long tracks (>5 min): {long_tracks:,} ({long_tracks/len(self.df)*100:.1f}%)")
        
        print(f"\n EDA analysis completed successfully!")
    
    def run_full_analysis(self):
        """Run the complete EDA analysis."""
        print(" Starting Comprehensive Spotify Dataset EDA Analysis")
        print("="*80)
        
        self.basic_info()
        self.analyze_tracks()
        self.analyze_genres()
        self.analyze_artists()
        self.analyze_audio_features()
        self.create_visualizations()
        self.generate_summary_report()
        
        print("\n EDA Analysis Complete!")
        print(" Check 'spotify_eda_visualizations.png' for visualizations")

def main():
    """Main function to run the EDA analysis."""
    # Initialize EDA analyzer (will auto-find dataset)
    eda = SpotifyEDA()
    
    # Run complete analysis
    eda.run_full_analysis()

if __name__ == "__main__":
    main()
