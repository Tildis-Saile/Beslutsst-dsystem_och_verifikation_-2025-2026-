from eda import SpotifyEDA

def main():
    """Run the EDA analysis with progress indicators."""
    print("Spotify Dataset EDA Analysis")
    print("=" * 50)
    
    try:
        # Initialize the EDA analyzer (will auto-find dataset)
        print("Initializing EDA analyzer...")
        eda = SpotifyEDA()  # Will automatically find the dataset
        
        # Run the complete analysis
        print("Running comprehensive analysis...")
        eda.run_full_analysis()
        
        print("\nAnalysis completed successfully!")
        print("Check the generated visualizations and console output for insights.")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Make sure the dataset file exists in the correct location.")
        print("Expected locations:")
        print("  - data/dataset.csv (from project root)")
        print("  - ../data/dataset.csv (from scripts folder)")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check your data and try again.")

if __name__ == "__main__":
    main()
