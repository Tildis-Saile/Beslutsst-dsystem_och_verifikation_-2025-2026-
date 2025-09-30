import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collaborative_filtering import CollaborativeFiltering  
from basic_recommender import ContentBasedRecommender  

class HybridRecommender:
    """
    A hybrid recommendation system that combines content-based and collaborative filtering approaches.
    """
    def __init__(self, spotify_data_path=None):
        self.content_recommender = ContentBasedRecommender(spotify_data_path)
        self.collaborative_recommender = CollaborativeFiltering(spotify_data_path)
        
        # Fit the content-based recommender
        self.content_recommender.fit()
        
        # Fit collaborative filtering models
        print("Fitting collaborative filtering models...")
        self.collaborative_recommender.fit_svd(n_components=20)
        self.collaborative_recommender.fit_user_based_cf(n_neighbors=10)
        self.collaborative_recommender.fit_item_based_cf(n_neighbors=10)

    def recommend(self, track_name, num_recommendations=5):
        """
        Provides recommendations by combining results from both content-based and collaborative filtering methods.
        
        Args:
            track_name (str): The name of the track to get recommendations for.
            num_recommendations (int): The number of recommendations to return.
            
        Returns:
            list: A list of recommended track names.
        """
        # Get content-based recommendations
        content_recommendations = self.content_recommender.recommend(track_name, num_recommendations)
        
        # Get collaborative filtering recommendations
        # We need to find a user who has listened to this track
        collaborative_recommendations = self._get_collaborative_recommendations(track_name, num_recommendations)

        # Combine recommendations and remove duplicates
        combined_recommendations = list(set(content_recommendations + collaborative_recommendations))

        # Return the top N recommendations
        return combined_recommendations[:num_recommendations]
    
    def _get_collaborative_recommendations(self, track_name, num_recommendations):
        """
        Get collaborative filtering recommendations for a track by finding users who listened to it.
        """
        try:
            # Find the track in the collaborative filtering data
            track_data = self.collaborative_recommender.user_item_df[
                self.collaborative_recommender.user_item_df['track_name'] == track_name
            ]
            
            if track_data.empty:
                print(f"Track '{track_name}' not found in collaborative filtering data.")
                return []
            
            # Get a user who has listened to this track
            user_id = track_data['user_id'].iloc[0]
            
            # Get recommendations using different methods and combine them
            recommendations = []
            
            # Try SVD recommendations
            try:
                svd_recs = self.collaborative_recommender.recommend_svd(user_id, num_recommendations)
                recommendations.extend(svd_recs)
            except Exception as e:
                print(f"SVD recommendation error: {e}")
            
            # Try user-based recommendations
            try:
                user_recs = self.collaborative_recommender.recommend_user_based(user_id, num_recommendations)
                recommendations.extend(user_recs)
            except Exception as e:
                print(f"User-based recommendation error: {e}")
            
            # Try item-based recommendations
            try:
                item_recs = self.collaborative_recommender.recommend_item_based(user_id, num_recommendations)
                recommendations.extend(item_recs)
            except Exception as e:
                print(f"Item-based recommendation error: {e}")
            
            # Remove duplicates and return
            return list(set(recommendations))[:num_recommendations]
            
        except Exception as e:
            print(f"Error getting collaborative recommendations: {e}")
            return []

if __name__ == '__main__':
    # Example usage
    hybrid_recommender = HybridRecommender()
    track_name_to_test = "Back In Black"  # Change to a track in your dataset
    recommendations = hybrid_recommender.recommend(track_name_to_test, num_recommendations=5)

    if recommendations:
        print(f"Hybrid Recommendations for '{track_name_to_test}':")
        for track in recommendations:
            print(f"- {track}")