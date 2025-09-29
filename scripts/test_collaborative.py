"""
Test script for Collaborative Filtering Recommender
==================================================

This script demonstrates the collaborative filtering functionality
and compares different approaches.
"""

from collaborative_filtering import CollaborativeFilteringRecommender
import time

def test_collaborative_filtering():
    """Test the collaborative filtering recommender."""
    print("Testing Collaborative Filtering Recommender")
    print("=" * 50)
    
    try:
        # Initialize recommender
        print("1. Initializing recommender...")
        start_time = time.time()
        recommender = CollaborativeFilteringRecommender()
        init_time = time.time() - start_time
        print(f"   Initialization completed in {init_time:.2f} seconds")
        
        # Fit SVD model
        print("\n2. Fitting SVD model...")
        start_time = time.time()
        recommender.fit_svd(n_components=15)
        svd_time = time.time() - start_time
        print(f"   SVD fitting completed in {svd_time:.2f} seconds")
        
        # Fit user-based CF
        print("\n3. Fitting user-based collaborative filtering...")
        start_time = time.time()
        recommender.fit_user_based_cf(n_neighbors=5)
        user_cf_time = time.time() - start_time
        print(f"   User-based CF completed in {user_cf_time:.2f} seconds")
        
        # Fit item-based CF
        print("\n4. Fitting item-based collaborative filtering...")
        start_time = time.time()
        recommender.fit_item_based_cf(n_neighbors=5)
        item_cf_time = time.time() - start_time
        print(f"   Item-based CF completed in {item_cf_time:.2f} seconds")
        
        # Get user preferences
        print("\n5. Analyzing user preferences...")
        user_genres = recommender.get_user_genre_preferences(max_users=10)  # Limit for performance
        print(f"   Found {len(user_genres)} users with preferences")
        
        # Test recommendations
        print("\n6. Testing recommendations...")
        test_users = list(user_genres.keys())[:3]  # Test first 3 users
        
        for i, user_id in enumerate(test_users, 1):
            print(f"\n   User {i}: {user_id}")
            print(f"   Preferred genres: {user_genres[user_id]}")
            
            # SVD recommendations
            try:
                svd_recs = recommender.recommend_svd(user_id, 3)
                print(f"   SVD recommendations: {svd_recs}")
            except Exception as e:
                print(f"   SVD error: {e}")
            
            # User-based recommendations
            try:
                user_recs = recommender.recommend_user_based(user_id, 3)
                print(f"   User-based recommendations: {user_recs}")
            except Exception as e:
                print(f"   User-based error: {e}")
            
            # Item-based recommendations
            try:
                item_recs = recommender.recommend_item_based(user_id, 3)
                print(f"   Item-based recommendations: {item_recs}")
            except Exception as e:
                print(f"   Item-based error: {e}")
        
        # Performance summary
        print(f"\n7. Performance Summary:")
        print(f"   Total initialization time: {init_time:.2f}s")
        print(f"   SVD fitting time: {svd_time:.2f}s")
        print(f"   User-based CF time: {user_cf_time:.2f}s")
        print(f"   Item-based CF time: {item_cf_time:.2f}s")
        print(f"   Total time: {init_time + svd_time + user_cf_time + item_cf_time:.2f}s")
        
        print(f"\n✅ Collaborative filtering test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def compare_methods():
    """Compare different collaborative filtering methods."""
    print("\n" + "="*60)
    print("COMPARING COLLABORATIVE FILTERING METHODS")
    print("="*60)
    
    try:
        recommender = CollaborativeFilteringRecommender()
        
        # Fit all models
        recommender.fit_svd(n_components=10)
        recommender.fit_user_based_cf(n_neighbors=5)
        recommender.fit_item_based_cf(n_neighbors=5)
        
        # Get a test user
        user_ids = list(recommender.user_to_idx.keys())
        test_user = user_ids[0] if user_ids else None
        
        if not test_user:
            print("No users found for comparison.")
            return
        
        print(f"Comparing methods for user: {test_user}")
        print("-" * 40)
        
        # Test each method
        methods = {
            'SVD': lambda: recommender.recommend_svd(test_user, 5),
            'User-based CF': lambda: recommender.recommend_user_based(test_user, 5),
            'Item-based CF': lambda: recommender.recommend_item_based(test_user, 5)
        }
        
        for method_name, method_func in methods.items():
            try:
                start_time = time.time()
                recommendations = method_func()
                end_time = time.time()
                
                print(f"{method_name}:")
                print(f"  Recommendations: {recommendations}")
                print(f"  Time: {end_time - start_time:.4f}s")
                print()
                
            except Exception as e:
                print(f"{method_name}: Error - {e}")
                print()
        
    except Exception as e:
        print(f"Error during comparison: {e}")

def main():
    """Main function to run all tests."""
    test_collaborative_filtering()
    compare_methods()

if __name__ == "__main__":
    main()
