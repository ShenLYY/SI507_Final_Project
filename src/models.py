# src/models.py
import math

class Track:
    """
    Represents a musical track with normalized audio features.
    """
    def __init__(self, track_id: str, name: str, artist: str, genre: str, features: dict):
        self.track_id = track_id
        self.name = name
        self.artist = artist
        self.genre = genre
        self.features = features  # Normalized dictionary of audio features

    def __repr__(self):
        return f"{self.name} — {self.artist}"

    def calculate_distance(self, other_track: 'Track') -> float:
        """
        Calculates the Euclidean distance between this track and another 
        based on their core audio features.
        
        Args:
            other_track (Track): The track to compare against.
            
        Returns:
            float: The Euclidean distance. A lower distance means higher similarity.
        """
        feature_keys = ["energy", "valence", "danceability", "acousticness", "tempo"]
        
        # Extract features into ordered lists
        vector_a = [self.features[k] for k in feature_keys]
        vector_b = [other_track.features[k] for k in feature_keys]
        
        # Calculate and return standard Euclidean distance
        return math.dist(vector_a, vector_b)