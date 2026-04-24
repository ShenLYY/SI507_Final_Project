# run this temporarily from project root: python src/check.py
import sys
sys.path.insert(0, "src")

from loader import SpotifyDataLoader
from graph import MusicGraph, path_to_feature_table

# Load a small sample
loader = SpotifyDataLoader("data/dataset.csv")
tracks = loader.load()
sample = [t for t in tracks if t.genre == "pop"][:200]

# Build graph
mg = MusicGraph()
mg.add_tracks(sample)
mg.build_edges(sample)

# Pick two track IDs that exist
id_a = sample[0].track_id
id_b = sample[5].track_id

path = mg.shortest_path_with_data(id_a, id_b)
if path:
    df = path_to_feature_table(path)
    print(df[["step", "track", "energy", "valence", "energy_delta", "valence_delta"]])
else:
    print("No path found — try different track indices")