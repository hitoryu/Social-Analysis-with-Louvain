# src/data_collection.py
import networkx as nx
import pandas as pd
import urllib.request
import gzip
import os
from typing import List, Tuple

class DataCollector:
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.edges = []
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        os.makedirs(f"{self.data_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.data_dir}/processed", exist_ok=True)
    
    def method_1_synthetic_data(self) -> List[Tuple]:
        """Create synthetic Facebook-like data"""
        print("Creating synthetic social network data...")
        
        # Create a graph with clear community structure
        G = nx.planted_partition_graph(4, 25, 0.7, 0.05, seed=42)
        self.edges = list(G.edges())
        
        print(f"Created synthetic network with {len(self.edges)} edges")
        return self.edges
    
    def method_2_snap_facebook(self) -> List[Tuple]:
        """Download and load real Facebook data from SNAP"""
        print("📥 Downloading real Facebook data from Stanford SNAP...")
        
        try:
            # Download the dataset
            url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
            local_path = f"{self.data_dir}/raw/facebook_combined.txt.gz"
            
            # Download if not exists
            if not os.path.exists(local_path):
                print("Downloading dataset (this may take a moment)...")
                urllib.request.urlretrieve(url, local_path)
                print("✅ Download completed!")
            
            # Extract and load edges
            self.edges = []
            with gzip.open(local_path, 'rt') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        nodes = line.strip().split()
                        if len(nodes) == 2:
                            self.edges.append((int(nodes[0]), int(nodes[1])))
            
            print(f"✅ Loaded SNAP Facebook dataset: {len(self.edges)} edges")
            return self.edges
            
        except Exception as e:
            print(f"❌ Error loading SNAP data: {e}")
            print("🔄 Falling back to synthetic data...")
            return self.method_1_synthetic_data()
    
    def method_3_snap_twitter(self) -> List[Tuple]:
        """Load Twitter social network from SNAP"""
        print("📥 Loading Twitter social network...")
        
        try:
            url = "https://snap.stanford.edu/data/twitter_combined.txt.gz"
            local_path = f"{self.data_dir}/raw/twitter_combined.txt.gz"
            
            if not os.path.exists(local_path):
                print("Downloading Twitter dataset...")
                urllib.request.urlretrieve(url, local_path)
            
            self.edges = []
            with gzip.open(local_path, 'rt') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        nodes = line.strip().split()
                        if len(nodes) == 2:
                            self.edges.append((int(nodes[0]), int(nodes[1])))
            
            print(f"✅ Loaded SNAP Twitter dataset: {len(self.edges)} edges")
            return self.edges
            
        except Exception as e:
            print(f"❌ Error loading Twitter data: {e}")
            return self.method_1_synthetic_data()
    
    def get_dataset_info(self, edges: List[Tuple]) -> dict:
        """Get information about the dataset"""
        G = nx.Graph()
        G.add_edges_from(edges)
        
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected_components': nx.number_connected_components(G),
            'avg_clustering': nx.average_clustering(G)
        }
    
    def save_data(self, filename: str):
        """Save data to file"""
        filepath = f"{self.data_dir}/raw/{filename}"
        with open(filepath, 'w') as f:
            for edge in self.edges:
                f.write(f"{edge[0]},{edge[1]}\n")
        print(f"Data saved to: {filepath}")