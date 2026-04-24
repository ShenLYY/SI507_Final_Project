# src/loader.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models import Track


class SpotifyDataLoader:
    """
    Responsible for ingesting, cleaning, and normalizing Spotify track data 
    from a CSV file into a list of Track objects.
    """
    
    def __init__(self, filepath: str):
        """
        Initializes the data loader.
        
        Args:
            filepath (str): The relative or absolute path to the dataset CSV.
        """
        self.filepath = filepath
        self.feature_keys = ["energy", "valence", "danceability", "acousticness", "tempo"]

    def load(self) -> list[Track]:
        """
        Orchestrates the data processing pipeline.
        
        Returns:
            list[Track]: A list of cleaned, normalized Track objects.
        """
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            print(f"Error: Could not find the dataset at '{self.filepath}'.")
            return []

        df = self._clean_data(df)
        df = self._normalize_features(df)
        return self._build_tracks(df)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows with missing essential data to prevent downstream errors.
        """
        # Drop rows missing audio features OR missing name/artist metadata
        return df.dropna(subset=self.feature_keys + ["track_name", "artists"])

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scales audio features to a standardized [0, 1] range using Min-Max scaling.
        """
        scaler = MinMaxScaler()
        # Using .copy() prevents Pandas SettingWithCopyWarnings during transformation
        df_normalized = df.copy() 
        df_normalized[self.feature_keys] = scaler.fit_transform(df_normalized[self.feature_keys])
        return df_normalized

    def _build_tracks(self, df: pd.DataFrame) -> list[Track]:
        """
        Converts the processed DataFrame rows into instantiated Track objects.
        """
        tracks = []
        for _, row in df.iterrows():
            features = {k: row[k] for k in self.feature_keys}
            track = Track(
                track_id=row["track_id"],
                name=row["track_name"],
                artist=row["artists"],
                genre=row["track_genre"],
                features=features
            )
            tracks.append(track)

        return tracks