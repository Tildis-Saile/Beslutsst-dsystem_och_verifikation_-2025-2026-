"""
Quick test script for Collaborative Filtering Recommender
========================================================

This is a simplified version that runs faster and avoids potential bottlenecks.
"""

from collaborative_filtering import CollaborativeFilteringRecommender
import time
import numpy as np

def quick_test():
    """Quick test of collaborative filtering functionality."""
    print("Quick Collaborative Filtering Test")
    print("=" * 40)
    
    try:
        # Initialize recommender
        print("1. Initializing recommender...")
        start_time = time.time()
        recommender = CollaborativeFilteringRecommender()
        init_time = time.time() - start_time
        print(f"   ✓ Initialized in {init_time:.2f}s")
        print(f"   Matrix size: {recommender.user_item_matrix.shape}")
        
        # Fit SVD model only (fastest)
        print("\n2. Fitting SVD model...")
        start_time = time.time()
        recommender.fit_svd(n_components=10)  # Smaller for speed
        svd_time = time.time() - start_time
        print(f"   ✓ SVD fitted in {svd_time:.2f}s")
        
        # Get a few test users
        print("\n3. Testing recommendations...")
        user_ids = list(recommender.user_to_idx.keys())[:3]  # Just 3 users
        print(f"   Testing with {len(user_ids)} users")
        
        for i, user_id in enumerate(user_ids, 1):
            print(f"\n   User {i}: {user_id}")
            
            # Test SVD recommendations
            try:
                start_time = time.time()
                recommendations = recommender.recommend_svd(user_id, 3)
                rec_time = time.time() - start_time
                print(f"   SVD recommendations: {recommendations}")
                print(f"   Time: {rec_time:.3f}s")
            except Exception as e:
                print(f"   SVD error: {e}")
        
        print(f"\n✅ Quick test completed successfully!")
        print(f"Total time: {init_time + svd_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_matrix_properties():
    """Test the user-item matrix properties."""
    print("\n" + "="*40)
    print("MATRIX PROPERTIES TEST")
    print("="*40)
    
    try:
        recommender = CollaborativeFilteringRecommender()
        matrix = recommender.user_item_matrix
        
        print(f"Matrix shape: {matrix.shape}")
        print(f"Number of users: {matrix.shape[0]}")
        print(f"Number of items: {matrix.shape[1]}")
        print(f"Non-zero elements: {matrix.nnz}")
        print(f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
        
        # Check sparsity
        sparsity = 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        print(f"Sparsity: {sparsity:.4f}")
        
        # Sample some data
        print(f"\nSample user-item interactions:")
        for i in range(min(3, matrix.shape[0])):
            user_items = matrix[i].toarray().flatten()
            non_zero = np.where(user_items > 0)[0]
            print(f"  User {i}: {len(non_zero)} items, max rating: {user_items.max():.3f}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run quick tests."""
    quick_test()
    test_matrix_properties()

if __name__ == "__main__":
    main()
