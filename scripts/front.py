import customtkinter as ctk
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from basic_recommender import ContentBasedRecommender
from collaborative_filtering import CollaborativeFiltering
from hybrid_recommender import HybridRecommender
import os
from dotenv import load_dotenv

# --- USER CONFIGURATION ---
# IMPORTANT: You must set these environment variables with your Spotify API credentials.
# 1. Go to the Spotify Developer Dashboard: https://developer.spotify.com/dashboard
# 2. Create an app to get your Client ID and Client Secret.
# 3. In your app settings, add a Redirect URI: http://localhost:8888/callback
# 4. Set the environment variables on your system. For example, in Windows PowerShell:
# $env:SPOTIPY_CLIENT_ID = 'YOUR_CLIENT_ID'
# $env:SPOTIPY_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
# $env:SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# Load environment variables from a .env file
load_dotenv()

class SpotifyRecommenderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.recommenders = {}
        self.current_recommender = None
        self.spotify_client = None
        self.search_results = {} # To store track URIs

        # --- Window Setup ---
        self.title("Advanced Spotify Recommender")
        self.geometry("600x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        # --- Spotify Authentication ---
        self.auth_button = ctk.CTkButton(self, text="Connect to Spotify", command=self.authenticate_spotify)
        self.auth_button.pack(pady=10)
        
        self.status_label = ctk.CTkLabel(self, text="Please connect to Spotify.")
        self.status_label.pack(pady=5)

        # --- Initialize Recommenders ---
        self.init_recommenders()

        # --- Recommender Selection ---
        self.recommender_frame = ctk.CTkFrame(self)
        self.recommender_label = ctk.CTkLabel(self.recommender_frame, text="Choose Recommendation System:")
        self.recommender_label.pack(pady=5)
        
        self.recommender_var = ctk.StringVar(value="Content-Based")
        self.recommender_menu = ctk.CTkOptionMenu(
            self.recommender_frame, 
            values=["Content-Based", "Collaborative Filtering", "Hybrid"],
            command=self.change_recommender
        )
        self.recommender_menu.pack(pady=5)

        # --- UI Widgets ---
        self.search_frame = ctk.CTkFrame(self)
        
        self.search_label = ctk.CTkLabel(self.search_frame, text="Enter an Artist/Band Name:")
        self.search_label.pack(pady=5)

        self.search_entry = ctk.CTkEntry(self.search_frame, width=300, placeholder_text="e.g., AC/DC, The Beatles, Queen")
        self.search_entry.pack(pady=5)

        self.search_button = ctk.CTkButton(self.search_frame, text="Search Artist", command=self.search_spotify_artist)
        self.search_button.pack(pady=10)

        self.results_listbox = ctk.CTkTextbox(self, width=550, height=150)
        self.results_listbox.insert("1.0", "Choose a recommendation system above, then enter an artist name and click Search Artist.")
        
        # Recommendations dropdown
        self.recommendations_frame = ctk.CTkFrame(self)
        self.recommendations_label = ctk.CTkLabel(self.recommendations_frame, text="Top 5 Recommendations:")
        self.recommendations_label.pack(pady=5)
        
        self.recommendations_dropdown = ctk.CTkOptionMenu(
            self.recommendations_frame,
            values=["Select a song first"],
            state="disabled"
        )
        self.recommendations_dropdown.pack(pady=5)
        
        self.play_button = ctk.CTkButton(self.recommendations_frame, text="Play Selected Recommendation", command=self.play_selected_recommendation, state="disabled")
        self.play_button.pack(pady=10)

    def init_recommenders(self):
        """Initialize all recommendation systems."""
        try:
            self.status_label.configure(text="Loading recommendation systems...")
            self.update_idletasks()
            
            # Initialize Content-Based Recommender
            print("Loading Content-Based Recommender...")
            self.recommenders["Content-Based"] = ContentBasedRecommender()
            self.recommenders["Content-Based"].fit()
            
            # Initialize Collaborative Filtering Recommender
            print("Loading Collaborative Filtering Recommender...")
            self.recommenders["Collaborative Filtering"] = CollaborativeFiltering()
            self.recommenders["Collaborative Filtering"].fit_svd(n_components=20)
            self.recommenders["Collaborative Filtering"].fit_user_based_cf(n_neighbors=10)
            self.recommenders["Collaborative Filtering"].fit_item_based_cf(n_neighbors=10)
            
            # Initialize Hybrid Recommender
            print("Loading Hybrid Recommender...")
            self.recommenders["Hybrid"] = HybridRecommender()
            
            # Set default recommender
            self.current_recommender = self.recommenders["Content-Based"]
            self.status_label.configure(text="All recommendation systems loaded! Please connect to Spotify.")
            
        except Exception as e:
            self.status_label.configure(text=f"Error loading recommenders: {e}", text_color="red")
            print(f"Error initializing recommenders: {e}")

    def change_recommender(self, choice):
        """Change the current recommendation system."""
        if choice in self.recommenders:
            self.current_recommender = self.recommenders[choice]
            self.status_label.configure(text=f"Switched to {choice} recommendation system", text_color="green")
            print(f"Switched to {choice} recommendation system")
            
            # Clear previous search results when switching systems
            if hasattr(self, 'search_results'):
                self.search_results = {}
            if hasattr(self, 'results_listbox'):
                self.results_listbox.delete("1.0", "end")
                self.results_listbox.insert("1.0", f"Now using {choice} recommendation system.\nEnter an artist name and click Search Artist.")
            
            # Reset recommendations dropdown
            if hasattr(self, 'recommendations_dropdown'):
                self.recommendations_dropdown.configure(values=["Select a song first"], state="disabled")
            if hasattr(self, 'play_button'):
                self.play_button.configure(state="disabled")
        else:
            self.status_label.configure(text=f"Error: {choice} not found", text_color="red")

    def authenticate_spotify(self):
        try:
            # Scope needed to view and control playback
            scope = "user-modify-playback-state user-read-playback-state"
            
            self.spotify_client = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
            
            # Check if authentication was successful
            user = self.spotify_client.current_user()
            self.status_label.configure(text=f"Connected as {user['display_name']}", text_color="green")
            self.auth_button.pack_forget() # Hide auth button
            
            # Show the main UI
            self.recommender_frame.pack(pady=10)
            self.search_frame.pack(pady=10)
            self.results_listbox.pack(pady=10, padx=10, fill="both", expand=True)
            self.recommendations_frame.pack(pady=10)

        except Exception as e:
            self.status_label.configure(text=f"Authentication failed. Check credentials/setup.", text_color="red")
            print(f"Error during authentication: {e}")

    def search_spotify_artist(self):
        artist_query = self.search_entry.get()
        if not artist_query or not self.spotify_client:
            return

        self.results_listbox.delete("1.0", "end")
        self.search_results = {}
        self.recommendations_dropdown.configure(values=["Select a song first"], state="disabled")
        self.play_button.configure(state="disabled")

        # Search for artist first
        artist_results = self.spotify_client.search(q=f"artist:{artist_query}", type='artist', limit=1)
        if not artist_results['artists']['items']:
            self.results_listbox.insert("1.0", f"No artist found for '{artist_query}' on Spotify.")
            return

        artist = artist_results['artists']['items'][0]
        artist_name = artist['name']
        
        # Get top tracks for this artist
        tracks_results = self.spotify_client.artist_top_tracks(artist['id'])
        tracks = tracks_results['tracks']

        if not tracks:
            self.results_listbox.insert("1.0", f"No tracks found for {artist_name} on Spotify.")
            return

        # Show current recommendation system
        current_system = self.recommender_menu.get()
        self.results_listbox.insert("1.0", f"Using {current_system} recommendation system\n")
        self.results_listbox.insert("end", f"Top tracks by {artist_name} - Click a song to get recommendations:\n\n")
        
        for track in tracks:
            display_name = f"{track['name']}"
            self.search_results[display_name] = track['uri']
            
            # Insert text and add a tag for binding
            tag_name = f"track_{track['id']}"
            self.results_listbox.insert("end", f"{display_name}\n", (tag_name,))
            self.results_listbox.tag_config(tag_name, foreground="#1DB954") # Spotify green
            self.results_listbox.tag_bind(tag_name, "<Button-1>", lambda e, dn=display_name: self.select_track(dn))

    def select_track(self, display_name):
        self.selected_track_name = display_name.split(' - ')[0] # Get just the track name
        self.status_label.configure(text=f"Selected: {self.selected_track_name} - Getting recommendations...")
        self.update_idletasks()
        
        # Get recommendations from the current recommender
        try:
            if self.recommender_menu.get() == "Collaborative Filtering":
                # For collaborative filtering, we need to find a user who has this track
                track_data = self.current_recommender.user_item_df[
                    self.current_recommender.user_item_df['track_name'] == self.selected_track_name
                ]
                if track_data.empty:
                    self.status_label.configure(text=f"Track '{self.selected_track_name}' not found in collaborative filtering data.")
                    return
                user_id = track_data['user_id'].iloc[0]
                local_recs = self.current_recommender.recommend_svd(user_id, 5)
            else:
                # For content-based and hybrid, use the standard recommend method
                local_recs = self.current_recommender.recommend(self.selected_track_name, num_recommendations=5)
        except Exception as e:
            self.status_label.configure(text=f"Error getting recommendations: {e}", text_color="red")
            print(f"Error getting recommendations: {e}")
            return

        if not local_recs:
            self.status_label.configure(text=f"Could not find recommendations for {self.selected_track_name}.")
            return

        # Store recommendations for playing
        self.current_recommendations = local_recs
        
        # Update dropdown with recommendations
        self.recommendations_dropdown.configure(values=local_recs, state="normal")
        self.recommendations_dropdown.set(local_recs[0])  # Set first recommendation as default
        self.play_button.configure(state="normal")
        
        self.status_label.configure(text=f"Selected: {self.selected_track_name} - {len(local_recs)} recommendations ready!", text_color="green")

    def play_selected_recommendation(self):
        """Play the selected recommendation from the dropdown."""
        if not hasattr(self, 'current_recommendations') or not self.spotify_client:
            return

        selected_recommendation = self.recommendations_dropdown.get()
        if not selected_recommendation or selected_recommendation == "Select a song first":
            return

        self.status_label.configure(text=f"Searching for '{selected_recommendation}' on Spotify...")
        self.update_idletasks()

        # Search for the recommended track on Spotify
        try:
            result = self.spotify_client.search(q=selected_recommendation, type='track', limit=1)
            if result['tracks']['items']:
                track_uri = result['tracks']['items'][0]['uri']
                
                # Play the track
                self.spotify_client.start_playback(uris=[track_uri])
                self.status_label.configure(text=f"Playing: {selected_recommendation}", text_color="green")
            else:
                self.status_label.configure(text=f"Could not find '{selected_recommendation}' on Spotify.", text_color="red")
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 404:
                self.status_label.configure(text="Error: No active Spotify device found.", text_color="red")
            else:
                self.status_label.configure(text=f"Error: {e.msg}", text_color="red")



if __name__ == '__main__':
    print("Starting Advanced Spotify Recommender...")
    print("This app supports Content-Based, Collaborative Filtering, and Hybrid recommendation systems.")
    
    app = SpotifyRecommenderApp()
    app.mainloop()