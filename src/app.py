import sys
sys.path.insert(0, "src")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components

from loader import SpotifyDataLoader
from graph import MusicGraph, path_to_feature_table
from models import Track

import wikipedia


st.set_page_config(page_title="Music Graph Explorer", layout="wide")
st.title("🎵 Music Graph Explorer")

# --- Render functions ---

def render_graph(mg: MusicGraph, tracks: list, feature="energy"):
    import hashlib
    import os
    import networkx as nx  # Added import for the layout calculation

    # Generate a unique cache key based on graph state and feature
    cache_key = hashlib.md5(
        f"{len(mg.graph.nodes)}{len(mg.graph.edges)}{feature}".encode()
    ).hexdigest()
    html_path = f"graph_{cache_key}.html"

    # Only rebuild the HTML file if it doesn't already exist
    if not os.path.exists(html_path):
        net = Network(height="500px", width="100%", bgcolor="#1a1a2e", font_color="white")
        colors = mg.get_node_colors(feature)
        
        # 1. Calculate positions in Python before building the Pyvis network
        # seed=42 ensures the layout looks the exact same every time for this specific graph
        pos = nx.spring_layout(mg.graph, weight="weight", seed=42)

        for node_id, data in mg.graph.nodes(data=True):
            track = data["data"]
            
            # --- NEW: Safely handle the hover text ---
            if feature == "community":
                hover_text = "Sub-genre Cluster"
            else:
                # Only try to round/display if it's a real audio feature
                hover_text = f"{feature}: {round(track.features.get(feature, 0), 2)}"
            # -----------------------------------------
            
            net.add_node(
                node_id,
                label=track.name[:20],
                title=f"{track.name}\n{track.artist}\n{hover_text}", # Updated this line
                color=colors.get(node_id, "#ffffff"), # Added .get() for safety
                size=15,
                x=pos[node_id][0] * 1000, 
                y=pos[node_id][1] * 1000,
                physics=False 
            )
            
        for u, v, data in mg.graph.edges(data=True):
            net.add_edge(u, v, value=data["weight"])
            
        # 3. Turn off the global physics engine completely to prevent browser lag
        net.toggle_physics(False)
        
        net.save_graph(html_path)

    with open(html_path) as f:
        components.html(f.read(), height=500)

def render_feature_chart(track: Track, key: str = None):
    feature_keys = ["energy", "valence", "danceability", "acousticness", "tempo"]
    values = [round(track.features[k], 3) for k in feature_keys]
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=feature_keys + [feature_keys[0]],
        fill="toself",
        fillcolor="rgba(29, 185, 84, 0.2)",
        line=dict(color="rgba(29, 185, 84, 0.8)")
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=f"{track.name} — {track.artist}",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def safe_match(text, query):
    return isinstance(text, str) and query.lower() in text.lower()

# --- Load data ---

@st.cache_data
def load_data():
    loader = SpotifyDataLoader("data/dataset.csv")
    return loader.load()

tracks = load_data()

# --- Sidebar ---
st.sidebar.header("Settings")
genre = st.sidebar.selectbox("Genre", sorted(set(t.genre for t in tracks)))

# ADD "community" to this list!
color_feature = st.sidebar.selectbox(
    "Color nodes by",
    ["community", "energy", "valence", "danceability", "acousticness", "tempo"]
)
threshold = st.sidebar.slider("Similarity threshold", 0.1, 0.5, 0.3)

# --- Build graph ---

@st.cache_data
def load_genre_tracks(genre):
    df = pd.read_csv("data/dataset.csv")
    feature_keys = ["energy", "valence", "danceability", "acousticness", "tempo"]
    df = df.dropna(subset=feature_keys)
    genre_df = df[df["track_genre"] == genre]

    # Sort by popularity if column exists, otherwise take a random sample
    if "popularity" in df.columns:
        genre_df = genre_df.sort_values("popularity", ascending=False)

    sample_df = genre_df.head(150)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    sample_df = sample_df.copy()
    sample_df[feature_keys] = scaler.fit_transform(sample_df[feature_keys])

    from models import Track
    return [
        Track(
            track_id=row["track_id"],
            name=row["track_name"],
            artist=row["artists"],
            genre=row["track_genre"],
            features={k: row[k] for k in feature_keys}
        )
        for _, row in sample_df.iterrows()
    ]

@st.cache_data
def build_genre_graph(genre, threshold):
    """Only rebuilds edges when genre or threshold actually changes."""
    sample = load_genre_tracks(genre)
    mg = MusicGraph()
    mg.add_tracks(sample)
    mg.build_edges(sample, threshold=threshold)
    return mg, sample

mg, sample = build_genre_graph(genre, threshold)
st.sidebar.success(f"{mg.graph.number_of_nodes()} nodes, {mg.graph.number_of_edges()} edges")


@st.cache_data(show_spinner=False)
def fetch_artist_bio(artist_name: str) -> str:
    """
    Fetches a short biography from Wikipedia.
    Fails gracefully if the artist is ambiguous or not found.
    """
    try:
        # Search specifically for the artist by adding "musician" or "band" 
        # to help Wikipedia find the right page
        query = f"{artist_name} musician"
        
        # Fetch the top page result
        search_results = wikipedia.search(query)
        if not search_results:
            return "No biography found."
            
        # Get a 2-sentence summary of the first result
        bio = wikipedia.summary(search_results[0], sentences=2, auto_suggest=False)
        return bio
        
    except wikipedia.exceptions.DisambiguationError:
        return f"Multiple Wikipedia entries exist for '{artist_name}'. Could not fetch a specific biography."
    except wikipedia.exceptions.PageError:
        return "No biography found on Wikipedia."
    except Exception:
        # Catch-all for network issues
        return "Could not load biography at this time."

# --- Tabs ---

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Search & Explore",
    "🛤️ Pathfinding",
    "🎚️ Cluster Filter",
    "💾 Playlist Export"
])

# Mode 1: Search & Explore
with tab1:
    st.subheader("Search & Explore")
    with st.expander("💡 Graph Insights (Algorithmic Analysis)", expanded=False):
        central_track, unique_track = mg.get_graph_insights()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔗 Most Central Track**")
            if central_track:
                st.write(f"**{central_track.name}** — {central_track.artist}")
                st.caption("This track acts as a bridge between different sonic clusters.")
                
        with col_b:
            st.markdown("**🏝️ Most Unique Track (Outlier)**")
            if unique_track:
                st.write(f"**{unique_track.name}** — {unique_track.artist}")
                st.caption("This track has the fewest similarities to others in this neighborhood.")
    # ------------------------------------------------------------
    query = st.text_input("Enter a track or artist name")

    matches = [t for t in sample if
               safe_match(t.name, query) or
               safe_match(t.artist, query)] if query else []

    if matches:
        selected = st.selectbox("Select track", matches, format_func=str)
        col1, col2 = st.columns([2, 1])
        with col1:
            render_graph(mg, sample, feature=color_feature)
        with col2:
            render_feature_chart(selected, key=f"explore_{selected.track_id}")
            # --- NEW: Display the Wikipedia Bio ---
            st.markdown("---")
            st.markdown(f"**About {selected.artist}**")
            
            # This calls the cached API function
            bio_text = fetch_artist_bio(selected.artist)
            
            # Display it as a nice, subtle caption card
            st.info(bio_text)
            # --------------------------------------
    elif query:
        all_tracks = load_data()
        cross_genre = [t for t in all_tracks if
                       safe_match(t.name, query) or
                       safe_match(t.artist, query)][:5]
        if cross_genre:
            st.info(f"Not found in current genre. Found {len(cross_genre)} match(es) across all genres:")
            for t in cross_genre:
                st.write(f"**{t.name}** — {t.artist} ({t.genre})")
            st.caption("Switch the genre in the sidebar to explore these tracks.")
        else:
            st.warning("No results found anywhere in the dataset.")
    else:
        render_graph(mg, sample, feature=color_feature)

# Mode 2: Pathfinding

with tab2:
    st.subheader("Find the sonic path between two tracks")
    col1, col2 = st.columns(2)
    with col1:
        start_query = st.text_input("Start track")
        start_matches = [t for t in sample if
                         safe_match(t.name, start_query)] if start_query else []
        start_track = st.selectbox("Select start", start_matches, format_func=str) if start_matches else None
    with col2:
        end_query = st.text_input("End track")
        end_matches = [t for t in sample if
                       safe_match(t.name, end_query)] if end_query else []
        end_track = st.selectbox("Select end", end_matches, format_func=str) if end_matches else None

    if start_track and end_track:
        if st.button("Find Path"):
            path = mg.shortest_path_with_data(start_track.track_id, end_track.track_id)
            if path:
                df = path_to_feature_table(path)
                st.line_chart(df.set_index("step")[["energy", "valence", "danceability"]])
                st.dataframe(df[["step", "track", "energy", "valence", "danceability"]])
                st.subheader("Feature profiles along the path")
                cols = st.columns(len(path))
                for i, (col, track) in enumerate(zip(cols, path)):
                    with col:
                        render_feature_chart(track, key=f"path_step_{i}_{track.track_id}")
            else:
                st.warning("No path found between these tracks — they may be in disconnected parts of the graph.")

# Mode 3: Cluster Filter
with tab3:
    st.subheader("Filter by audio features")
    energy_range = st.slider("Energy", 0.0, 1.0, (0.4, 1.0))
    valence_range = st.slider("Valence", 0.0, 1.0, (0.0, 1.0))
    filtered = mg.filter_by_feature("energy", *energy_range)
    filtered = filtered.subgraph([
        n for n in filtered.nodes()
        if valence_range[0] <= mg.graph.nodes[n]["data"].features["valence"] <= valence_range[1]
    ])
    st.write(f"Showing {filtered.number_of_nodes()} tracks in this range")
    render_graph(mg, sample, feature=color_feature)

# Mode 4: Playlist Export
with tab4:
    st.subheader("Export a playlist")
    selected_tracks = st.multiselect("Select tracks to export", sample, format_func=str)
    if selected_tracks:
        export_df = pd.DataFrame([{
            "track": str(t),
            "genre": t.genre,
            "energy": round(t.features["energy"], 3),
            "valence": round(t.features["valence"], 3),
            "danceability": round(t.features["danceability"], 3),
        } for t in selected_tracks])
        st.dataframe(export_df)
        st.download_button(
            "Download as CSV",
            export_df.to_csv(index=False),
            file_name="my_playlist.csv",
            mime="text/csv"
        )