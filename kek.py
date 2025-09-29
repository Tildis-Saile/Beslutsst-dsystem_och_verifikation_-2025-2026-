import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    """
    A content-based recommender system for Spotify tracks based on genres and artists.
    """
    def __init__(self, spotify_data_path):
        """
        Initializes the recommender by loading the Spotify data.
        
        Args:
            spotify_data_path (str): The file path to the Spotify dataset CSV file.
        """
        self.spotify_df = pd.read_csv(spotify_data_path)  # CHANGED: movies_df -> spotify_df
        self.tfidf_matrix = None
        self.track_indices = None  # CHANGED: movie_indices -> track_indices

    def fit(self):
        """
        Preprocesses the data and computes the TF-IDF matrix for track genres and artists.
        This method must be called before `recommend`.
        """

        # CHANGED: Use 'genres' and 'artists' columns, fill missing values
        self.spotify_df['track_genre'] = self.spotify_df['track_genre'].fillna('')
        self.spotify_df['artists'] = self.spotify_df['artists'].fillna('')

        # Combine genres and artists for richer content representation
        self.spotify_df['content'] = self.spotify_df['track_genre'] + ' ' + self.spotify_df['artists']

        # TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english')

        # TF-IDF-matrix on combined content
        self.tfidf_matrix = tfidf.fit_transform(self.spotify_df['content'])

        # Create index for track names
        self.track_indices = pd.Series(self.spotify_df.index, index=self.spotify_df['track_name']).drop_duplicates()
        # CHANGED: 'title' -> 'track_name'

    def recommend(self, track_name, num_recommendations=5):
        """
        Recommends tracks similar to a given track name.
        
        Args:
            track_name (str): The name of the track to get recommendations for.
            num_recommendations (int): The number of recommendations to return.
            
        Returns:
            list: A list of recommended track names. Returns an empty list
                  if the track name is not found or if the model hasn't been fitted.
        """
        if self.tfidf_matrix is None or self.track_indices is None:
            print("The model has not been fitted yet. Please call the 'fit' method first.")
            return []
            
        if track_name not in self.track_indices:
            print(f"Track '{track_name}' not found in the dataset.")
            return []

        # Get the index of the track
        idx = self.track_indices[track_name]
        
        # If duplicate track names exist, idx will be a Series. Get the first index.
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Calculate cosine similarity
        sim_scores = cosine_similarity(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix)[0]
        
        # Enumerate and sort similarity scores
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # --- MODIFIED LOGIC TO HANDLE DUPLICATES ---
        # Filter out all songs with the same name as the input track
        filtered_scores = [score for score in sim_scores if self.spotify_df['track_name'].iloc[score[0]] != track_name]
        
        # Get the top N recommendations from the filtered list
        top_scores = filtered_scores[:num_recommendations]
        track_indices = [i[0] for i in top_scores]
        # --- END MODIFIED LOGIC ---

        # Get top recommendations excluding the track itself
        #sim_scores = sim_scores[1:num_recommendations + 1]
        #track_indices = [i[0] for i in sim_scores]
        
        # Return the track names
        return self.spotify_df['track_name'].iloc[track_indices].tolist()  # CHANGED: 'title' -> 'track_name'

if __name__ == '__main__':
    # Simple main function to test
    # Not part of the unit test 
    
    # CHANGED: Use Spotify dataset path
    recommender = ContentBasedRecommender(spotify_data_path='data/dataset.csv')
    
    # Fit the model
    recommender.fit()
    
    # Get and print recommendations for a track
    # CHANGED: Use a track name from your dataset
    track_name_to_test = "Back In Black"  # <-- Change to a track in your dataset
    recommendations = recommender.recommend(track_name_to_test, num_recommendations=5)
    
    if recommendations:
        print(f"Recommendations for '{track_name_to_test}':")
        for track in recommendations:
            print(f"- {track}")