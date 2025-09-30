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

    # Get a few sample users from the collaborative filtering data to test with
    user_item_df = hybrid_recommender.collaborative_recommender.user_item_df
    
    # Take 3 sample users to demonstrate
    sample_user_ids = user_item_df['user_id'].unique()[:3] 

    # For each sample user, find a track they listened to and get recommendations
    for user_id in sample_user_ids:
        print("-" * 40)
        # Find a track this user has listened to to use as a seed
        user_tracks = user_item_df[user_item_df['user_id'] == user_id]
        
        if not user_tracks.empty:
            # Use the first track found for this user as the input for recommendations
            track_name_to_test = user_tracks['track_name'].iloc[0]
            
            print(f"Testing with a track from user '{user_id}': '{track_name_to_test}'")
            
            recommendations = hybrid_recommender.recommend(track_name_to_test, num_recommendations=5)

            if recommendations:
                print(f"Hybrid Recommendations based on '{track_name_to_test}':")
                for i, track in enumerate(recommendations, 1):
                    print(f"{i}. {track}")
            else:
                print(f"Could not find recommendations for '{track_name_to_test}'.")
        else:
            print(f"User '{user_id}' has no tracks in the dataset.")
    
    print("-" * 40)