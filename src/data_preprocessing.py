# src/data_preprocessing.py
import networkx as nx
import pandas as pd
from typing import List, Tuple, Dict, Any

class DataPreprocessor:
    def __init__(self, edges: List[Tuple]):
        self.edges = edges
        self.G = nx.Graph()
    
    def build_graph(self) -> nx.Graph:
        """Build graph from edge list"""
        self.G.add_edges_from(self.edges)
        print(f"Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G
    
    def clean_graph(self, min_degree: int = 1) -> nx.Graph:
        """Clean the graph"""
        # Remove low-degree nodes
        nodes_to_remove = [node for node, degree in dict(self.G.degree()).items() 
                          if degree < min_degree]
        self.G.remove_nodes_from(nodes_to_remove)
        
        # Keep only largest connected component
        if not nx.is_connected(self.G):
            largest_cc = max(nx.connected_components(self.G), key=len)
            self.G = self.G.subgraph(largest_cc).copy()
        
        print(f"Cleaned graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G
    
    def analyze_basic_stats(self) -> Dict[str, Any]:
        """Analyze basic network statistics"""
        stats = {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'average_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
            'clustering_coefficient': nx.average_clustering(self.G),
        }
        
        print("NETWORK STATISTICS")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return stats