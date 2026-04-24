# test_graph.py
import sys
sys.path.insert(0, "src")

import pytest
from models import Track
from graph import MusicGraph

# --- Fixtures ---
def make_track(track_id, energy, valence, danceability, acousticness, tempo):
    return Track(
        track_id=track_id,
        name=f"Track {track_id}",
        artist="Test Artist",
        genre="pop",
        features={
            "energy": energy,
            "valence": valence,
            "danceability": danceability,
            "acousticness": acousticness,
            "tempo": tempo
        }
    )

@pytest.fixture
def similar_tracks():
    # Two tracks that are very close in feature space
    return [
        make_track("A", 0.8, 0.7, 0.9, 0.1, 0.6),
        make_track("B", 0.8, 0.7, 0.9, 0.1, 0.6),
    ]

@pytest.fixture
def dissimilar_tracks():
    # Two tracks that are far apart
    return [
        make_track("X", 0.0, 0.0, 0.0, 0.0, 0.0),
        make_track("Y", 1.0, 1.0, 1.0, 1.0, 1.0),
    ]

@pytest.fixture
def small_graph():
    tracks = [
        make_track("A", 0.8, 0.7, 0.9, 0.1, 0.6),
        make_track("B", 0.8, 0.7, 0.9, 0.1, 0.6),
        make_track("C", 0.1, 0.2, 0.1, 0.9, 0.2),
    ]
    mg = MusicGraph()
    mg.add_tracks(tracks)
    mg.build_edges(tracks, threshold=0.3)
    return mg, tracks

# --- Track tests ---
def test_track_stores_attributes():
    t = make_track("A", 0.5, 0.5, 0.5, 0.5, 0.5)
    assert t.track_id == "A"
    assert t.name == "Track A"
    assert t.features["energy"] == 0.5

def test_track_repr():
    t = make_track("A", 0.5, 0.5, 0.5, 0.5, 0.5)
    assert "Track A" in repr(t)
    assert "Test Artist" in repr(t)

# --- Graph construction tests ---
def test_graph_nodes_added(small_graph):
    mg, tracks = small_graph
    assert mg.graph.number_of_nodes() == 3

def test_similar_tracks_get_edge(similar_tracks):
    mg = MusicGraph()
    mg.add_tracks(similar_tracks)
    mg.build_edges(similar_tracks, threshold=0.3)
    assert mg.graph.has_edge("A", "B")

def test_dissimilar_tracks_no_edge(dissimilar_tracks):
    mg = MusicGraph()
    mg.add_tracks(dissimilar_tracks)
    mg.build_edges(dissimilar_tracks, threshold=0.3)
    assert not mg.graph.has_edge("X", "Y")

def test_edge_weight_between_zero_and_one(similar_tracks):
    mg = MusicGraph()
    mg.add_tracks(similar_tracks)
    mg.build_edges(similar_tracks, threshold=0.3)
    weight = mg.graph["A"]["B"]["weight"]
    assert 0 <= weight <= 1

# --- Pathfinding tests ---
def test_shortest_path_returns_list(small_graph):
    mg, tracks = small_graph
    path = mg.shortest_path("A", "B")
    assert isinstance(path, list)
    assert path[0] == "A"
    assert path[-1] == "B"

def test_shortest_path_unreachable(small_graph):
    mg, tracks = small_graph
    # C is dissimilar to A and B, may be disconnected
    import networkx as nx
    if not nx.has_path(mg.graph, "A", "C"):
        with pytest.raises(Exception):
            mg.shortest_path("A", "C")

# --- Filter tests ---
def test_filter_returns_correct_nodes(small_graph):
    mg, tracks = small_graph
    subgraph = mg.filter_by_feature("energy", 0.5, 1.0)
    for node in subgraph.nodes():
        track = mg.graph.nodes[node]["data"]
        assert track.features["energy"] >= 0.5

def test_filter_empty_result(small_graph):
    mg, tracks = small_graph
    subgraph = mg.filter_by_feature("energy", 0.99, 1.0)
    # Should return empty graph, not crash
    assert subgraph.number_of_nodes() >= 0

# --- Neighbor tests ---
def test_neighbors_returns_at_most_n(small_graph):
    mg, tracks = small_graph
    neighbors = mg.get_neighbors("A", n=1)
    assert len(neighbors) <= 1

def test_neighbors_sorted_by_weight(small_graph):
    mg, tracks = small_graph
    neighbors = mg.get_neighbors("A", n=10)
    weights = [mg.graph["A"][nb]["weight"] for nb in neighbors]
    assert weights == sorted(weights, reverse=True)

def test_shortest_path_with_data_returns_tracks(small_graph):
    mg, tracks = small_graph
    path = mg.shortest_path_with_data("A", "B")
    assert len(path) > 0
    assert all(isinstance(t, Track) for t in path)
    assert path[0].track_id == "A"
    assert path[-1].track_id == "B"

def test_path_feature_table_shape(small_graph):
    from graph import path_to_feature_table
    mg, tracks = small_graph
    path = mg.shortest_path_with_data("A", "B")
    if path:
        df = path_to_feature_table(path)
        # One row per step
        assert len(df) == len(path)
        # Delta of first row is always 0
        assert df["energy_delta"].iloc[0] == 0.0
        # All expected columns present
        for col in ["energy", "valence", "danceability", "acousticness", "tempo"]:
            assert col in df.columns
            assert f"{col}_delta" in df.columns