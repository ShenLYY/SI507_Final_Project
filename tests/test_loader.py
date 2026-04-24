# test_loader.py
import sys
sys.path.insert(0, "src")

import pytest
import pandas as pd
from loader import SpotifyDataLoader
from models import Track

# --- Fixtures ---

@pytest.fixture
def mock_csv_path(tmp_path):
    """
    Creates a temporary CSV file with specific edge cases:
    - Row 1: Perfectly valid
    - Row 2: Missing an audio feature ('danceability')
    - Row 3: Missing essential metadata ('track_name')
    """
    data = {
        # Add letters to force Pandas to read these as strings
        "track_id": ["id_1", "id_2", "id_3"], 
        "track_name": ["Song A", "Song B", None], 
        "artists": ["Artist A", "Artist B", "Artist C"],
        "track_genre": ["pop", "pop", "rock"],
        "energy": [0.1, 0.9, 0.5],
        "valence": [0.2, 0.8, 0.5],
        "danceability": [0.3, None, 0.5],  
        "acousticness": [0.4, 0.6, 0.5],
        "tempo": [100.0, 150.0, 120.0]
    }
    df = pd.DataFrame(data)
    
    # tmp_path is a pytest tool that provides a temporary directory unique to the test invocation
    file_path = tmp_path / "mock_dataset.csv"
    df.to_csv(file_path, index=False)
    
    return str(file_path)

# --- Tests ---

def test_missing_values_dropped(mock_csv_path):
        loader = SpotifyDataLoader(mock_csv_path)
        tracks = loader.load()
        
        assert len(tracks) == 1
        
        # Update this line to match "id_1"
        assert tracks[0].track_id == "id_1" 
        assert tracks[0].name == "Song A"

def test_min_max_scaler_bounds(tmp_path):
    # Create a fresh mock dataset with wild, unscaled numbers
    data = {
        "track_id": ["1", "2", "3"],
        "track_name": ["A", "B", "C"],
        "artists": ["X", "Y", "Z"],
        "track_genre": ["pop", "pop", "pop"],
        "energy": [10, 50, 100], 
        "valence": [0, 5, 10],   
        "danceability": [-5, 0, 5],
        "acousticness": [100, 200, 300],
        "tempo": [60, 120, 180]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "scale_test.csv"
    df.to_csv(file_path, index=False)
    
    loader = SpotifyDataLoader(str(file_path))
    tracks = loader.load()
    
    # Verify that every feature on every track is strictly between 0.0 and 1.0
    for track in tracks:
        for feature, value in track.features.items():
            assert 0.0 <= value <= 1.0, f"{feature} value {value} is out of bounds!"

def test_file_not_found():
    loader = SpotifyDataLoader("data/this_file_does_not_exist.csv")
    tracks = loader.load()
    
    # Based on our refactored load() method, it should catch the error and return an empty list
    assert isinstance(tracks, list)
    assert len(tracks) == 0