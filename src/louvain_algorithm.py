# src/louvain_algorithm.py
import community as community_louvain
import networkx as nx
import numpy as np
from typing import Dict, Tuple  # ADD THIS IMPORT

class LouvainCommunityDetection:
    def __init__(self, graph: nx.Graph):
        self.G = graph
        self.partition = None
        self.modularity_history = []
        self.execution_time = 0
    
    def detect_communities(self, resolution: float = 1.0, random_state: int = None) -> Tuple[Dict, float]:
        """Detect communities using Louvain algorithm"""
        import time
        start_time = time.time()
        
        # Use python-louvain implementation
        self.partition = community_louvain.best_partition(
            self.G, 
            resolution=resolution, 
            random_state=random_state
        )
        
        self.execution_time = time.time() - start_time
        
        # Calculate modularity
        final_modularity = community_louvain.modularity(self.partition, self.G)
        self.modularity_history.append(final_modularity)
        
        print(f"Execution time: {self.execution_time:.2f} seconds")
        print(f"Final Modularity: {final_modularity:.4f}")
        
        return self.partition, final_modularity
    
    def analyze_partition(self) -> Dict:
        """Analyze the partition results"""
        if self.partition is None:
            raise ValueError("Run Louvain algorithm first")
            
        community_sizes = {}
        for node, comm_id in self.partition.items():
            if comm_id not in community_sizes:
                community_sizes[comm_id] = 0
            community_sizes[comm_id] += 1
        
        print(f"\n=== PARTITION ANALYSIS ===")
        print(f"Number of communities: {len(community_sizes)}")
        
        sizes = list(community_sizes.values())
        print(f"Community sizes - Min: {min(sizes)}, Max: {max(sizes)}, Avg: {np.mean(sizes):.2f}")
        
        return community_sizes