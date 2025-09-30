import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
import os
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    """
    A collaborative filtering recommender system for Spotify tracks.
    """
    
    def __init__(self, data_path=None):
        """Initialize the collaborative filtering recommender."""
        if data_path is None:
            data_path = self._find_dataset()
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
            
        self.df = pd.read_csv(data_path)
        self.user_item_matrix = None
        self.svd_model = None
        self.nmf_model = None
        self.user_neighbors = None
        self.item_neighbors = None
        
        # Create user-item matrix from implicit feedback
        self._create_user_item_matrix()
    
    def _find_dataset(self):
        """Automatically find the dataset file."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, '../data/dataset.csv'),
            os.path.join(script_dir, '../../data/dataset.csv'),
            'data/dataset.csv',
            '../data/dataset.csv',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        return '../data/dataset.csv'
    
    def _create_user_item_matrix(self):
        """Create user-item matrix from implicit feedback based on genres."""
        print("Creating user-item matrix from implicit feedback...")
        
        # Create synthetic users based on genre preferences
        genres = self.df['track_genre'].unique()
        user_item_data = []
        
        for genre in genres:
            genre_tracks = self.df[self.df['track_genre'] == genre]
            
            for _, track in genre_tracks.iterrows():
                # Calculate implicit rating based on popularity and duration
                popularity_score = track['popularity'] / 100.0
                duration_score = min(track['duration_ms'] / (5 * 60 * 1000), 1.0)
                implicit_rating = (popularity_score * 0.7 + duration_score * 0.3)
                
                user_item_data.append({
                    'user_id': f"genre_{genre}",
                    'item_id': track['track_id'],
                    'rating': implicit_rating,
                    'track_name': track['track_name']
                })
        
        # Convert to DataFrame and create sparse matrix
        self.user_item_df = pd.DataFrame(user_item_data)
        user_ids = self.user_item_df['user_id'].unique()
        item_ids = self.user_item_df['item_id'].unique()
        
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Create sparse matrix
        rows = [self.user_to_idx[user] for user in self.user_item_df['user_id']]
        cols = [self.item_to_idx[item] for item in self.user_item_df['item_id']]
        ratings = self.user_item_df['rating'].values
        
        self.user_item_matrix = csr_matrix((ratings, (rows, cols)), 
                                         shape=(len(user_ids), len(item_ids)))
        
        print(f"User-item matrix created: {self.user_item_matrix.shape}")
        print(f"Number of users: {len(user_ids)}, Items: {len(item_ids)}")
        print(f"Matrix density: {self.user_item_matrix.nnz / (len(user_ids) * len(item_ids)):.4f}")
    
    def fit_svd(self, n_components=50):
        """Fit SVD model."""
        print(f"Fitting SVD model with {n_components} components...")
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd_factors = self.svd_model.fit_transform(self.user_item_matrix)
        print(f"SVD explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.4f}")
    
    def fit_nmf(self, n_components=50):
        """Fit NMF model."""
        print(f"Fitting NMF model with {n_components} components...")
        self.nmf_model = NMF(n_components=n_components, random_state=42, max_iter=200)
        self.nmf_factors = self.nmf_model.fit_transform(self.user_item_matrix)
        print(f"NMF reconstruction error: {self.nmf_model.reconstruction_err_:.4f}")
    
    def fit_user_based_cf(self, n_neighbors=20):
        """Fit user-based collaborative filtering model."""
        print(f"Fitting user-based CF with {n_neighbors} neighbors...")
        self.user_neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        self.user_neighbors.fit(self.user_item_matrix)
    
    def fit_item_based_cf(self, n_neighbors=20):
        """Fit item-based collaborative filtering model."""
        print(f"Fitting item-based CF with {n_neighbors} neighbors...")
        item_user_matrix = self.user_item_matrix.T
        self.item_neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        self.item_neighbors.fit(item_user_matrix)
    
    def recommend_svd(self, user_id, n_recommendations=5):
        """Get recommendations using SVD."""
        if self.svd_model is None:
            raise ValueError("SVD model not fitted. Call fit_svd() first.")
        
        if user_id not in self.user_to_idx:
            print(f"User '{user_id}' not found in training data.")
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_factors = self.svd_factors[user_idx]
        
        # Calculate predicted ratings for all items
        item_factors = self.svd_model.components_.T
        predicted_ratings = np.dot(item_factors, user_factors)
        
        # Get items not already rated by user
        user_items = self.user_item_matrix[user_idx].toarray().flatten()
        unrated_items = np.where(user_items == 0)[0]
        
        if len(unrated_items) == 0:
            print("No unrated items found for this user.")
            return []
        
        # Get top recommendations
        unrated_predictions = predicted_ratings[unrated_items]
        top_indices = np.argsort(unrated_predictions)[::-1][:n_recommendations]
        top_item_indices = unrated_items[top_indices]
        
        # Convert back to track names
        recommendations = []
        for item_idx in top_item_indices:
            item_id = self.idx_to_item[item_idx]
            track_name = self.user_item_df[self.user_item_df['item_id'] == item_id]['track_name'].iloc[0]
            recommendations.append(track_name)
        
        return recommendations
    
    def recommend_user_based(self, user_id, n_recommendations=5):
        """Get recommendations using user-based collaborative filtering."""
        if self.user_neighbors is None:
            raise ValueError("User-based model not fitted. Call fit_user_based_cf() first.")
        
        if user_id not in self.user_to_idx:
            print(f"User '{user_id}' not found in training data.")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get similar users
        distances, neighbor_indices = self.user_neighbors.kneighbors(
            self.user_item_matrix[user_idx], n_neighbors=6
        )
        
        # Remove self from neighbors
        neighbor_indices = neighbor_indices[0][1:]
        distances = distances[0][1:]
        
        # Calculate weighted scores
        user_items = self.user_item_matrix[user_idx].toarray().flatten()
        weighted_scores = np.zeros(self.user_item_matrix.shape[1])
        
        for i, neighbor_idx in enumerate(neighbor_indices):
            neighbor_items = self.user_item_matrix[neighbor_idx].toarray().flatten()
            similarity = 1 - distances[i]
            weighted_scores += neighbor_items * similarity
        
        # Get unrated items and top recommendations
        unrated_items = np.where(user_items == 0)[0]
        if len(unrated_items) == 0:
            print("No unrated items found for this user.")
            return []
        
        unrated_scores = weighted_scores[unrated_items]
        top_indices = np.argsort(unrated_scores)[::-1][:n_recommendations]
        top_item_indices = unrated_items[top_indices]
        
        # Convert back to track names
        recommendations = []
        for item_idx in top_item_indices:
            item_id = self.idx_to_item[item_idx]
            track_name = self.user_item_df[self.user_item_df['item_id'] == item_id]['track_name'].iloc[0]
            recommendations.append(track_name)
        
        return recommendations
    
    def recommend_item_based(self, user_id, n_recommendations=5):
        """Get recommendations using item-based collaborative filtering."""
        if self.item_neighbors is None:
            raise ValueError("Item-based model not fitted. Call fit_item_based_cf() first.")
        
        if user_id not in self.user_to_idx:
            print(f"User '{user_id}' not found in training data.")
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_items = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Get items the user has rated
        rated_items = np.where(user_items > 0)[0]
        if len(rated_items) == 0:
            print("User has no rated items.")
            return []
        
        # Find similar items to user's rated items
        all_similar_items = set()
        for item_idx in rated_items:
            distances, similar_item_indices = self.item_neighbors.kneighbors(
                self.user_item_matrix[:, item_idx].T, n_neighbors=6
            )
            similar_items = similar_item_indices[0][1:]
            all_similar_items.update(similar_items)
        
        # Remove items user has already rated
        unrated_similar_items = list(all_similar_items - set(rated_items))
        if len(unrated_similar_items) == 0:
            print("No similar unrated items found.")
            return []
        
        # Calculate scores based on similarity to rated items
        scores = np.zeros(len(unrated_similar_items))
        for i, item_idx in enumerate(unrated_similar_items):
            for rated_item in rated_items:
                item_similarity = cosine_similarity(
                    self.user_item_matrix[:, item_idx].T,
                    self.user_item_matrix[:, rated_item].T
                )[0][0]
                scores[i] += item_similarity * user_items[rated_item]
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        top_item_indices = [unrated_similar_items[i] for i in top_indices]
        
        # Convert back to track names
        recommendations = []
        for item_idx in top_item_indices:
            item_id = self.idx_to_item[item_idx]
            track_name = self.user_item_df[self.user_item_df['item_id'] == item_id]['track_name'].iloc[0]
            recommendations.append(track_name)
        
        return recommendations
    
    def evaluate_model(self, max_users=10):
        """Simple evaluation of the collaborative filtering model."""
        print("Evaluating collaborative filtering model...")
        
        evaluation_results = {}
        user_ids = list(self.user_to_idx.keys())[:max_users]
        
        for method_name in ['svd', 'user_based', 'item_based']:
            print(f"Evaluating {method_name}...")
            
            total_recommendations = 0
            successful_recommendations = 0
            
            for user_id in user_ids:
                try:
                    if method_name == 'svd':
                        recommendations = self.recommend_svd(user_id, 3)
                    elif method_name == 'user_based':
                        recommendations = self.recommend_user_based(user_id, 3)
                    elif method_name == 'item_based':
                        recommendations = self.recommend_item_based(user_id, 3)
                    
                    total_recommendations += len(recommendations)
                    successful_recommendations += len(recommendations)
                
                except Exception as e:
                    print(f"Error evaluating {method_name} for user {user_id}: {e}")
                    continue
            
            success_rate = successful_recommendations / total_recommendations if total_recommendations > 0 else 0
            evaluation_results[method_name] = {
                'success_rate': success_rate,
                'total_recommendations': total_recommendations,
                'successful_recommendations': successful_recommendations
            }
        
        return evaluation_results

def main():
    """Demo function to test the collaborative filtering recommender."""
    print("Collaborative Filtering Recommender Demo")
    print("=" * 50)
    
    try:
        # Initialize recommender
        print("Initializing collaborative filtering recommender...")
        recommender = CollaborativeFiltering()
        
        # Fit all models
        print("\nFitting models...")
        recommender.fit_svd(n_components=20)
        recommender.fit_nmf(n_components=20)
        recommender.fit_user_based_cf(n_neighbors=10)
        recommender.fit_item_based_cf(n_neighbors=10)
        
        # Get some user IDs for testing
        user_ids = list(recommender.user_to_idx.keys())[:5]
        
        print(f"\nTesting recommendations for {len(user_ids)} users...")
        
        for user_id in user_ids:
            print(f"\n--- Recommendations for {user_id} ---")
            
            # SVD recommendations
            try:
                svd_recs = recommender.recommend_svd(user_id, 3)
                print(f"SVD: {svd_recs}")
            except Exception as e:
                print(f"SVD error: {e}")
            
            # User-based recommendations
            try:
                user_recs = recommender.recommend_user_based(user_id, 3)
                print(f"User-based: {user_recs}")
            except Exception as e:
                print(f"User-based error: {e}")
            
            # Item-based recommendations
            try:
                item_recs = recommender.recommend_item_based(user_id, 3)
                print(f"Item-based: {item_recs}")
            except Exception as e:
                print(f"Item-based error: {e}")
        
        # Evaluate models
        print(f"\nEvaluating models...")
        evaluation_results = recommender.evaluate_model()
        
        print(f"\nEvaluation Results:")
        for method, results in evaluation_results.items():
            print(f"{method}: {results}")
        
        print(f"\nCollaborative filtering demo completed successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()