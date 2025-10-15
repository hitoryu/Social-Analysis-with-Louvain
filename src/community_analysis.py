# src/community_analysis.py
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class CommunityAnalyzer:
    def __init__(self, graph: nx.Graph, partition: Dict):
        self.G = graph
        self.partition = partition
        self.community_data = None
        
    def extract_community_metrics(self) -> Dict:
        """Extract metrics for each community"""
        community_metrics = defaultdict(lambda: {
            'nodes': [],
            'internal_edges': 0,
            'external_edges': 0,
            'density': 0,
            'clustering_coeff': 0
        })
        
        # Classify nodes by community
        for node, comm_id in self.partition.items():
            community_metrics[comm_id]['nodes'].append(node)
        
        # Calculate metrics for each community
        for comm_id, metrics in community_metrics.items():
            nodes = metrics['nodes']
            subgraph = self.G.subgraph(nodes)
            
            # Calculate internal and external edges
            internal_edges = subgraph.number_of_edges()
            total_edges_from_comm = sum(self.G.degree(node) for node in nodes)
            external_edges = total_edges_from_comm - 2 * internal_edges
            
            metrics['internal_edges'] = internal_edges
            metrics['external_edges'] = external_edges
            metrics['size'] = len(nodes)
            metrics['density'] = nx.density(subgraph)
            metrics['clustering_coeff'] = nx.average_clustering(subgraph)
            metrics['internal_external_ratio'] = internal_edges / (external_edges + 1e-8)
        
        self.community_data = community_metrics
        
        print("COMMUNITY METRICS")
        for comm_id, metrics in community_metrics.items():
            print(f"Community {comm_id}: {metrics['size']} nodes, Density: {metrics['density']:.3f}")
        
        return community_metrics
    
    def identify_community_roles(self) -> Dict:
        """Identify roles of communities"""
        roles = {}
        
        for comm_id, metrics in self.community_data.items():
            ratio = metrics['internal_external_ratio']
            density = metrics['density']
            
            if ratio > 2.0 and density > 0.5:
                role = "Tight-knit Community"
            elif ratio > 1.0:
                role = "Balanced Community"
            elif density > 0.3:
                role = "Loosely Connected"
            else:
                role = "Sparse Community"
                
            roles[comm_id] = role
        
        return roles
    
    def find_influential_nodes(self, top_k: int = 5) -> Dict:
        """Find most influential nodes"""
        centrality_measures = {
            'degree': nx.degree_centrality(self.G),
            'betweenness': nx.betweenness_centrality(self.G),
            'closeness': nx.closeness_centrality(self.G),
        }
        
        influential_nodes = {}
        
        for measure_name, centrality in centrality_measures.items():
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
            influential_nodes[measure_name] = [
                (node, score, self.partition[node]) 
                for node, score in sorted_nodes
            ]
        
        print("TOP INFLUENTIAL NODES")
        for measure, nodes in influential_nodes.items():
            print(f"{measure.capitalize()} Centrality:")
            for node, score, community in nodes:
                print(f"  Node {node}: {score:.3f} (Community {community})")
        
        return influential_nodes