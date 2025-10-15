# src/report_generator.py
import networkx as nx
import numpy as np
from datetime import datetime
import os
from typing import Dict

class ReportGenerator:
    def __init__(self, graph: nx.Graph, partition: Dict, metrics: Dict, execution_time: float):
        self.G = graph
        self.partition = partition
        self.metrics = metrics
        self.execution_time = execution_time
        
        # Ensure results directory exists
        os.makedirs('results/reports', exist_ok=True)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive report"""
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'network_stats': self._get_network_statistics(),
            'community_stats': self._get_community_statistics(),
            'algorithm_performance': self._get_algorithm_performance(),
            'key_findings': self._get_key_findings()
        }
        
        self._save_report(report)
        return report
    
    def _get_network_statistics(self) -> Dict:
        """Get network statistics"""
        return {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'network_density': nx.density(self.G),
            'average_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
            'clustering_coefficient': nx.average_clustering(self.G),
        }
    
    def _get_community_statistics(self) -> Dict:
        """Get community statistics"""
        community_sizes = [len([n for n in self.G.nodes() if self.partition[n] == comm_id]) 
                          for comm_id in set(self.partition.values())]
        
        return {
            'total_communities': len(set(self.partition.values())),
            'largest_community': max(community_sizes),
            'smallest_community': min(community_sizes),
            'average_community_size': np.mean(community_sizes),
            'modularity_score': self.metrics.get('modularity', 0)
        }
    
    def _get_algorithm_performance(self) -> Dict:
        """Get algorithm performance"""
        return {
            'execution_time_seconds': self.execution_time,
            'nodes_processed_per_second': self.G.number_of_nodes() / self.execution_time,
        }
    
    def _get_key_findings(self) -> Dict:
        """Get key findings"""
        modularity = self.metrics.get('modularity', 0)
        return {
            'community_structure': "Clear" if modularity > 0.3 else "Unclear",
            'network_type': self._classify_network_type(),
            'recommendations': self._generate_recommendations()
        }
    
    def _classify_network_type(self) -> str:
        """Classify network type"""
        density = nx.density(self.G)
        if density > 0.1:
            return "Dense Network"
        elif density > 0.01:
            return "Medium Network"
        else:
            return "Sparse Network"
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations"""
        modularity = self.metrics.get('modularity', 0)
        if modularity > 0.7:
            return "Excellent community structure. Focus on detailed community analysis."
        elif modularity > 0.4:
            return "Good community structure. Analyze influential nodes."
        else:
            return "Weak community structure. Consider different parameters or data."
    
    def _save_report(self, report: Dict):
        """Save report to file"""
        with open('results/reports/comprehensive_report.txt', 'w', encoding='utf-8') as f:
            f.write("FACEBOOK COMMUNITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for section, content in report.items():
                f.write(f"\n--- {section.upper()} ---\n")
                if isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{content}\n")
        
        print("Report saved to: results/reports/comprehensive_report.txt")