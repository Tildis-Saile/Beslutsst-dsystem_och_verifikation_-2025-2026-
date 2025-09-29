import customtkinter as ctk
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from kek import ContentBasedRecommender
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
    def __init__(self, recommender):
        super().__init__()

        self.recommender = recommender
        self.spotify_client = None
        self.search_results = {} # To store track URIs

        # --- Window Setup ---
        self.title("Spotify Recommender")
        self.geometry("500x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        # --- Spotify Authentication ---
        self.auth_button = ctk.CTkButton(self, text="Connect to Spotify", command=self.authenticate_spotify)
        self.auth_button.pack(pady=10)
        
        self.status_label = ctk.CTkLabel(self, text="Please connect to Spotify.")
        self.status_label.pack(pady=5)

        # --- UI Widgets ---
        self.search_frame = ctk.CTkFrame(self)
        
        self.search_label = ctk.CTkLabel(self.search_frame, text="Enter a Song Name:")
        self.search_label.pack(pady=5)

        self.search_entry = ctk.CTkEntry(self.search_frame, width=300, placeholder_text="e.g., Back In Black")
        self.search_entry.pack(pady=5)

        self.search_button = ctk.CTkButton(self.search_frame, text="Search", command=self.search_spotify_tracks)
        self.search_button.pack(pady=10)

        self.results_listbox = ctk.CTkTextbox(self, width=450, height=150)
        
        self.recommend_button = ctk.CTkButton(self, text="Get Recommendations & Play", command=self.get_and_play_recommendations, state="disabled")

    def authenticate_spotify(self):
        try:
            # Scope needed to view and control playback
            scope = "user-modify-playback-state user-read-playback-state"
            
            self.spotify_client = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
            
            # Check if authentication was successful
            user = self.spotify_client.current_user()
            self.status_label.configure(text=f"Connected as {user['display_name']}", text_color="green")
            self.auth_button.pack_forget() # Hide auth button
            
            # Show the main search UI
            self.search_frame.pack(pady=10)
            self.results_listbox.pack(pady=10, padx=10, fill="both", expand=True)
            self.recommend_button.pack(pady=10)

        except Exception as e:
            self.status_label.configure(text=f"Authentication failed. Check credentials/setup.", text_color="red")
            print(f"Error during authentication: {e}")

    def search_spotify_tracks(self):
        query = self.search_entry.get()
        if not query or not self.spotify_client:
            return

        self.results_listbox.delete("1.0", "end")
        self.search_results = {}
        self.recommend_button.configure(state="disabled")

        results = self.spotify_client.search(q=query, type='track', limit=10)
        tracks = results['tracks']['items']

        if not tracks:
            self.results_listbox.insert("1.0", "No tracks found on Spotify.")
            return

        self.results_listbox.insert("1.0", "Click a song to select it:\n\n")
        for track in tracks:
            display_name = f"{track['name']} - {track['artists'][0]['name']}"
            self.search_results[display_name] = track['uri']
            
            # Insert text and add a tag for binding
            tag_name = f"track_{track['id']}"
            self.results_listbox.insert("end", f"{display_name}\n", (tag_name,))
            self.results_listbox.tag_config(tag_name, foreground="#1DB954") # Spotify green
            self.results_listbox.tag_bind(tag_name, "<Button-1>", lambda e, dn=display_name: self.select_track(dn))

    def select_track(self, display_name):
        self.selected_track_name = display_name.split(' - ')[0] # Get just the track name
        self.status_label.configure(text=f"Selected: {self.selected_track_name}")
        self.recommend_button.configure(state="normal")

    def get_and_play_recommendations(self):
        if not hasattr(self, 'selected_track_name') or not self.spotify_client:
            return

        self.status_label.configure(text=f"Getting recommendations for {self.selected_track_name}...")
        self.update_idletasks() # Force GUI update

        # Get recommendations from your local model
        local_recs = self.recommender.recommend(self.selected_track_name, num_recommendations=5)

        if not local_recs:
            self.status_label.configure(text=f"Could not find recommendations for {self.selected_track_name}.")
            return

        # Find the recommended tracks on Spotify to get their URIs
        track_uris_to_play = []
        for track_name in local_recs:
            result = self.spotify_client.search(q=track_name, type='track', limit=1)
            if result['tracks']['items']:
                track_uris_to_play.append(result['tracks']['items'][0]['uri'])
        
        if not track_uris_to_play:
            self.status_label.configure(text="Found recommendations, but couldn't find them on Spotify.")
            return

        # Play the tracks on the user's active device
        try:
            self.spotify_client.start_playback(uris=track_uris_to_play)
            self.status_label.configure(text=f"Playing recommendations on Spotify!", text_color="green")
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 404:
                self.status_label.configure(text="Error: No active Spotify device found.", text_color="red")
            else:
                self.status_label.configure(text=f"Error: {e.msg}", text_color="red")


if __name__ == '__main__':
    print("Loading and fitting the recommender model...")
    # This might take a moment depending on your dataset size
    recommender = ContentBasedRecommender(spotify_data_path='data/dataset.csv')
    recommender.fit()
    print("Model ready.")
    
    app = SpotifyRecommenderApp(recommender)
    app.mainloop()