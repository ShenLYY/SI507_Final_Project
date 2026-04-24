# src/graph.py
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from models import Track
import networkx.algorithms.community as nx_comm  # Add this line
import math

class MusicGraph:
    """
    Represents a weighted similarity graph of music tracks.
    Nodes are Track objects; edges connect tracks whose normalized
    audio feature vectors fall within a defined Euclidean distance threshold.
    """
    def __init__(self):
        self.graph = nx.Graph()

    def add_tracks(self, tracks: list[Track]):
        for track in tracks:
            self.graph.add_node(track.track_id, data=track)

    def build_edges(self, tracks: list[Track], threshold=0.3):
        """
        Builds graph edges by comparing all pairs of tracks. 
        If the distance between two tracks is below the threshold, they are connected.
        """
        # We use a nested loop to compare every track to every other track exactly once
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                
                track_a = tracks[i]
                track_b = tracks[j]
                
                # Delegate the math to the Track object!
                dist = track_a.calculate_distance(track_b)
                
                if dist < threshold:
                    # Convert distance to a similarity score for the edge weight
                    similarity = 1 - dist
                    self.graph.add_edge(
                        track_a.track_id,
                        track_b.track_id,
                        weight=similarity
                    )

    def get_neighbors(self, track_id, n=20):
        neighbors = self.graph[track_id]
        return sorted(
            neighbors,
            key=lambda x: neighbors[x]["weight"],
            reverse=True
        )[:n]

    def shortest_path(self, start_id, end_id):
        return nx.shortest_path(self.graph, start_id, end_id, weight="weight")

    def filter_by_feature(self, feature, min_val, max_val):
        matching = [
            n for n, d in self.graph.nodes(data=True)
            if min_val <= d["data"].features[feature] <= max_val
        ]
        return self.graph.subgraph(matching)

    def shortest_path_with_data(self, start_id, end_id) -> list[Track]:
        """Returns ordered list of Track objects along shortest path."""
        try:
            path_ids = nx.shortest_path(self.graph, start_id, end_id, weight="weight")
            return [self.graph.nodes[node_id]["data"] for node_id in path_ids]
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []
    def get_communities(self):
        """Detects sub-genres/communities using the Louvain method."""
        # This groups nodes that are highly connected to each other
        return nx_comm.louvain_communities(self.graph, weight='weight')
    
    def get_node_colors(self, feature="energy") -> dict:
        """
        Returns a dict mapping track_id to a hex color.
        Handles both continuous audio features and discrete communities.
        """
        colors = {}
        
        # --- NEW: Handle Community Coloring ---
        if feature == "community":
            communities = self.get_communities()
            # A distinct palette for up to 20 sub-genres
            palette = [
                "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
                "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
                "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"
            ]
            for i, comm in enumerate(communities):
                color = palette[i % len(palette)]
                for node_id in comm:
                    colors[node_id] = color
            return colors

        # --- EXISTING: Handle Audio Features (Energy, Valence, etc.) ---
        for node_id, data in self.graph.nodes(data=True):
            value = data["data"].features.get(feature, 0.5)
            # Interpolate between blue (low) and red (high)
            r = int(value * 255)
            b = int((1 - value) * 255)
            colors[node_id] = f"#{r:02x}20{b:02x}"
            
        return colors
    def get_graph_insights(self):
        """
        Calculates centrality and degree to find the most representative 
        and most isolated tracks in the current graph.
        """
        if self.graph.number_of_nodes() == 0:
            return None, None

        # 1. Betweenness Centrality: The track that bridges the most paths
        centrality = nx.betweenness_centrality(self.graph, weight="weight")
        most_central_id = max(centrality, key=centrality.get)
        most_central_track = self.graph.nodes[most_central_id]["data"]

        # 2. Lowest Degree: The track with the fewest edges (most isolated/unique)
        degrees = dict(self.graph.degree())
        most_unique_id = min(degrees, key=degrees.get)
        most_unique_track = self.graph.nodes[most_unique_id]["data"]

        return most_central_track, most_unique_track


import pandas as pd

def path_to_feature_table(path: list[Track]) -> pd.DataFrame:
    """
    Converts a list of Track objects into a DataFrame showing
    audio features at each step and the delta from the previous step.
    """
    feature_keys = ["energy", "valence", "danceability", "acousticness", "tempo"]
    rows = []

    for i, track in enumerate(path):
        row = {
            "step": i + 1,
            "track": str(track),
            "energy": round(track.features["energy"], 3),
            "valence": round(track.features["valence"], 3),
            "danceability": round(track.features["danceability"], 3),
            "acousticness": round(track.features["acousticness"], 3),
            "tempo": round(track.features["tempo"], 3),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add delta columns showing change from previous step
    for key in feature_keys:
        df[f"{key}_delta"] = df[key].diff().round(3).fillna(0)

    return df