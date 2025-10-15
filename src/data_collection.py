# src/data_collection.py
import networkx as nx
import pandas as pd
import urllib.request
import gzip
import os
from typing import List, Tuple

class DataCollector:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.edges = []
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        os.makedirs(f"{self.data_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.data_dir}/processed", exist_ok=True)
        os.makedirs(f"{self.data_dir}/kaggle", exist_ok=True)
    
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
        print("Downloading real Facebook data from Stanford SNAP...")
        
        try:
            # Download the dataset
            url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
            local_path = f"{self.data_dir}/raw/facebook_combined.txt.gz"
            
            # Download if not exists
            if not os.path.exists(local_path):
                print("Downloading dataset (this may take a moment)...")
                urllib.request.urlretrieve(url, local_path)
                print("Download completed!")
            
            # Extract and load edges
            self.edges = []
            with gzip.open(local_path, 'rt') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        nodes = line.strip().split()
                        if len(nodes) == 2:
                            self.edges.append((int(nodes[0]), int(nodes[1])))
            
            print(f"Loaded SNAP Facebook dataset: {len(self.edges)} edges")
            return self.edges
            
        except Exception as e:
            print(f"Error loading SNAP data: {e}")
            print("Falling back to synthetic data...")
            return self.method_1_synthetic_data()
    
    def method_4_kaggle_facebook(self, dataset_path: str = None) -> List[Tuple]:
        """Load Facebook data from Kaggle dataset"""
        print("Loading Facebook data from Kaggle...")
        
        try:
            # If no specific path provided, try common Kaggle dataset structures
            if dataset_path is None:
                # Try to find Kaggle dataset in common locations
                possible_paths = [
                    f"{self.data_dir}/kaggle/facebook_edges.csv",
                    f"{self.data_dir}/kaggle/edges.csv",
                    f"{self.data_dir}/kaggle/facebook.csv",
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        dataset_path = path
                        break
            
            if dataset_path is None:
                raise FileNotFoundError("No Kaggle dataset found. Please download manually.")
            
            print(f"Loading from: {dataset_path}")
            
            # Handle different file formats
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
                # Common column names in Kaggle datasets
                if 'source' in df.columns and 'target' in df.columns:
                    self.edges = list(zip(df['source'], df['target']))
                elif 'node1' in df.columns and 'node2' in df.columns:
                    self.edges = list(zip(df['node1'], df['node2']))
                elif 'user1' in df.columns and 'user2' in df.columns:
                    self.edges = list(zip(df['user1'], df['user2']))
                else:
                    # Assume first two columns are edges
                    self.edges = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
                    
            elif dataset_path.endswith('.txt') or dataset_path.endswith('.edges'):
                # Edge list format
                self.edges = []
                with open(dataset_path, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            nodes = line.strip().split()
                            if len(nodes) >= 2:
                                self.edges.append((nodes[0], nodes[1]))
            
            print(f"Loaded Kaggle Facebook dataset: {len(self.edges)} edges")
            return self.edges
            
        except Exception as e:
            print(f"Error loading Kaggle data: {e}")
            print("Falling back to SNAP data...")
            return self.method_2_snap_facebook()
    
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