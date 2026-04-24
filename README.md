
# 🎵 Music Graph Explorer

A graph-based music exploration and recommendation system built with Python and Streamlit. The app models musical similarity as a network — where nodes are tracks and edges connect songs with similar audio profiles — revealing connections and sonic journeys that a flat list or table cannot show.

---

## What It Does

Music Graph Explorer lets you explore 114,000+ Spotify tracks through four interactive modes and analytical features:

- **Search & Explore** — Search for a track or artist and visualize its neighborhood in the similarity graph. Nodes can be colored by audio feature intensity or by **algorithmically detected sub-genre communities**. Selecting a track displays a radar chart of its audio fingerprint and automatically fetches a live biography of the artist via the **Wikipedia API**.
- **Graph Insights** — An algorithmic analysis panel that calculates the "Kevin Bacon" of the current genre (the track with the highest betweenness centrality bridging different sonic clusters) and the most unique outlier track.
- **Pathfinding** — Find the shortest "sonic path" between two tracks, with a step-by-step line chart showing how energy, valence, and danceability shift across each hop.
- **Cluster Filter** — Use sliders to isolate regions of the graph by energy and valence range, revealing clusters of sonically similar music.
- **Playlist Export** — Select any set of tracks from the current graph and download them as a CSV playlist.

---

## Project Structure

```text
SI507_Final_Project/
├── data/
│   └── dataset.csv           # Spotify Tracks Dataset (from Kaggle)
├── src/
│   ├── models.py             # Track class with distance calculation logic
│   ├── loader.py             # SpotifyDataLoader class pipeline
│   ├── graph.py              # MusicGraph class + advanced network algorithms
│   └── app.py                # Streamlit frontend & UI logic
├── tests/
│   ├── test_graph.py         # Pytest suite for graph logic and pathfinding
│   └── test_loader.py        # Pytest suite for data ingestion edge cases
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SI507_Final_Project
```

### 2. Download the dataset

Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle and place the CSV file at:

```text
data/dataset.csv
```

### 3. Create and activate a virtual environment

Requires **Python 3.13+**.

```bash
python3 -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run src/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Dependencies

```text
pandas
networkx
streamlit
pyvis
scikit-learn
plotly
pytest
wikipedia
```

---

## How the Graph Works

**Data & Architecture:** Each track is represented as a node storing its metadata and a normalized vector of five audio features (energy, valence, danceability, acousticness, and tempo). The `Track` object is responsible for calculating its Euclidean distance to other tracks. If this distance falls below a similarity threshold (default: 0.3), an edge is drawn. 

**Performance & Layout:** To ensure a lightning-fast user experience without browser lag, the physical node layout coordinates are calculated completely offline using NetworkX's Spring Layout. Pyvis physics simulations are disabled, allowing for instant rendering of the structural graph.

**Advanced Algorithms:** The graph goes beyond simple plotting by utilizing the Louvain community detection method to group nodes into hidden sub-genres, and Betweenness Centrality to identify the structural importance of tracks within the network. 

---

## Running the Tests

```bash
pytest tests/ -v
```

The test suite provides high-coverage validation of the system, including:
- **Data Ingestion (`test_loader.py`):** Ensuring graceful handling of missing files, dropping corrupted CSV rows, and verifying Min-Max scaler boundaries.
- **Graph Logic (`test_graph.py`):** Validating track construction, edge weight accuracy, pathfinding behaviors (including handling disconnected nodes), filter correctness, and neighbor sorting logic. 

---

## Data Sources

- **Audio Features:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — 114,000+ tracks across 114 genres with full Spotify audio feature metadata.
- **Artist Biographies:** Live integrations with the Python `wikipedia` library to fetch contextual artist information.
