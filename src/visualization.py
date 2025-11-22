# visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import time
import os

class ResultVisualizer:
    def __init__(self, graph: nx.Graph, partition: dict):
        self.G = graph
        self.partition = partition
        plt.style.use('default')
        
        # Create absolute paths for saving
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.images_dir = os.path.join(self.results_dir, 'images')
        
        # Ensure directories exist
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"Saving images to: {self.images_dir}")
        
    def diagnose_network_issues(self):
        """Check why edges might not be visible"""
        print("NETWORK DIAGNOSTICS:")
        print(f"   Total nodes: {self.G.number_of_nodes()}")
        print(f"   Total edges: {self.G.number_of_edges()}")
        print(f"   Network density: {nx.density(self.G):.6f}")
        print(f"   Average degree: {sum(dict(self.G.degree()).values()) / self.G.number_of_nodes():.2f}")
        
        # Check if edges exist
        if self.G.number_of_edges() == 0:
            print("CRITICAL: No edges in the graph!")
            return False
        
        # Check graph connectivity
        if nx.is_connected(self.G):
            print("   Graph is fully connected")
        else:
            components = list(nx.connected_components(self.G))
            print(f"   Graph has {len(components)} connected components")
            print(f"   Largest component: {len(max(components, key=len))} nodes")
        
        # Sample some edges to verify they exist
        sample_edges = list(self.G.edges())[:5]
        print(f"   Sample edges: {sample_edges}")
        
        return True

    def plot_network_guaranteed(self, figsize=(14, 10)):
        """Network plot that guarantees visible edges"""
        print("Creating network with guaranteed visible edges...")
        
        # Run diagnostics first
        self.diagnose_network_issues()
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42, k=1, iterations=50)
        
        unique_communities = list(set(self.partition.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        
        print(f"Drawing {self.G.number_of_edges()} edges with high visibility...")
        
        # 1. FIRST draw edges with high visibility
        nx.draw_networkx_edges(
            self.G, pos,
            alpha=0.8,
            edge_color='#2E86AB',
            width=1.5,
            style='-'
        )
        
        # 2. THEN draw nodes (slightly transparent so edges show through)
        for i, comm_id in enumerate(unique_communities):
            nodes = [node for node in self.G.nodes() if self.partition[node] == comm_id]
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=nodes,
                node_color=[colors[i]],
                node_size=150,
                alpha=0.9,
                edgecolors='black',
                linewidths=0.8,
                label=f'Community {comm_id}'
            )
        
        # 3. Add labels for small networks
        if len(self.G.nodes()) <= 100:
            nx.draw_networkx_labels(self.G, pos, font_size=8, font_color='darkred')
        
        plt.title('Network Communities - Clear Edge Visualization', fontsize=16, fontweight='bold')
        
        # Legend for all communities
        if len(unique_communities) <= 18:
            plt.legend(
                bbox_to_anchor=(1.05, 1), 
                loc='upper left',
                fontsize=8,
                ncol=1
            )
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save with high quality
        save_path = os.path.join(self.images_dir, 'network_clear_edges.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network with clear edges saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(2)
        print("You should now see clear lines connecting the circles!")
        
        return plt.gcf()

    def plot_network_for_large_graph(self, figsize=(15, 12)):
        """Optimized for large networks like SNAP Facebook"""
        print("Creating optimized visualization for large network...")
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42, k=0.3, iterations=30)
        
        unique_communities = list(set(self.partition.values()))
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(unique_communities))))
        
        print(f"Network has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        
        # For very large networks, use thinner but visible edges
        edge_alpha = 0.3 if self.G.number_of_edges() > 10000 else 0.5
        edge_width = 0.5 if self.G.number_of_edges() > 10000 else 1.0
        
        # Draw edges first
        nx.draw_networkx_edges(
            self.G, pos, 
            alpha=edge_alpha, 
            edge_color='#FF6B6B',
            width=edge_width
        )
        
        # Draw nodes
        node_size = 20 if self.G.number_of_nodes() > 1000 else 50
        for i, comm_id in enumerate(unique_communities):
            if i >= 10:
                break
            nodes = [node for node in self.G.nodes() if self.partition[node] == comm_id]
            nx.draw_networkx_nodes(
                self.G, pos, nodelist=nodes, 
                node_color=[colors[i]], 
                node_size=node_size, 
                alpha=0.7, 
                label=f'Comm {comm_id}'
            )
        
        plt.title(f'Large Network: {len(self.G.nodes())} users, {len(self.G.edges())} connections', 
                  fontsize=12, fontweight='bold')
        
        # Legend for large graph
        if len(unique_communities) <= 10:
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1))
        
        plt.axis('off')
        plt.tight_layout()
        
        save_path = os.path.join(self.images_dir, 'network_large_optimized.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Large network visualization saved to: {save_path}")
        plt.show(block=False)
        plt.pause(2)
        
        return plt.gcf()

    def plot_network_all_communities(self, figsize=(20, 12)):
        """Plot network with ALL communities visible - UPDATED FOR 18 COMMUNITIES"""
        print("Creating network visualization with ALL communities...")
        
        # Run diagnostics first
        self.diagnose_network_issues()
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42, k=0.3, iterations=30)
        
        unique_communities = sorted(list(set(self.partition.values())))
        print(f"Total communities to display: {len(unique_communities)}")
        
        # Use a colormap with enough colors for all communities
        if len(unique_communities) <= 10:
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        else:
            # Use tab20 for up to 20 communities
            colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_communities))))
        
        print(f"Drawing {self.G.number_of_edges()} edges...")
        
        # 1. Draw edges first
        edge_alpha = 0.3 if self.G.number_of_edges() > 10000 else 0.5
        edge_width = 0.5 if self.G.number_of_edges() > 10000 else 1.0
        
        nx.draw_networkx_edges(
            self.G, pos,
            alpha=edge_alpha,
            edge_color='#2E86AB',
            width=edge_width
        )
        
        # 2. Draw nodes for ALL communities
        node_size = 20 if self.G.number_of_nodes() > 1000 else 50
        
        for i, comm_id in enumerate(unique_communities):
            nodes = [node for node in self.G.nodes() if self.partition[node] == comm_id]
            
            # Cycle colors if more than 20 communities
            if len(unique_communities) > 20:
                color = colors[i % 20]
            else:
                color = colors[i]
            
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=nodes,
                node_color=[color],
                node_size=node_size,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
                label=f'Community {comm_id}'
            )
        
        # 3. UPDATED LEGEND FOR 18 COMMUNITIES
        plt.title(f'Network Communities - All {len(unique_communities)} Communities', 
                  fontsize=14, fontweight='bold')
        
        # Hiển thị legend cho tất cả 18 communities
        if len(unique_communities) <= 18:
            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                fontsize=7,
                ncol=1,
                framealpha=0.9,
                markerscale=0.8,
                handletextpad=0.4,
                title='Communities',
                title_fontsize=9
            )
        elif len(unique_communities) <= 24:
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=6,
                ncol=2,
                framealpha=0.9,
                markerscale=0.7,
                handletextpad=0.3
            )
        else:
            # For too many communities, don't show legend but print info
            print(f"Too many communities ({len(unique_communities)}) for legend. Community IDs: {unique_communities}")
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save with high quality
        save_path = os.path.join(self.images_dir, 'network_all_communities.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network with ALL communities saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(2)
        
        return plt.gcf()

    def plot_network_18_communities_optimized(self, figsize=(22, 12)):
        """Special version optimized exactly for 18 communities"""
        print("Creating optimized visualization for 18 communities...")
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42, k=0.4, iterations=50)
        
        unique_communities = sorted(list(set(self.partition.values())))
        
        if len(unique_communities) != 18:
            print(f"Warning: Expected 18 communities, but found {len(unique_communities)}")
        
        # Use tab20 colormap (exactly 20 colors)
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Draw edges
        nx.draw_networkx_edges(
            self.G, pos,
            alpha=0.4,
            edge_color='gray',
            width=0.8
        )
        
        # Draw nodes for each community
        for i, comm_id in enumerate(unique_communities):
            nodes = [node for node in self.G.nodes() if self.partition[node] == comm_id]
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=nodes,
                node_color=[colors[i]],
                node_size=80,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                label=f'Community {comm_id} ({len(nodes)} nodes)'
            )
        
        # PERFECT LEGEND FOR 18 COMMUNITIES
        plt.legend(
            bbox_to_anchor=(1.01, 1),
            loc='upper left',
            fontsize=7,
            ncol=1,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            markerscale=0.8,
            handlelength=1.0,
            handletextpad=0.5
        )
        
        plt.title(f'Network with {len(unique_communities)} Communities', fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        save_path = os.path.join(self.images_dir, 'network_18_communities_optimized.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimized 18 communities network saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(2)
        return plt.gcf()

    def plot_community_size_distribution(self, community_sizes: list):
        """Plot community size distribution with accurate scaling"""
        print("Generating accurate community size distribution...")
        
        # Create figure with proper layout
        plt.figure(figsize=(14, 6))
        
        # Subplot 1: Histogram with proper bins
        plt.subplot(1, 2, 1)
        
        # Calculate optimal bins based on data range
        max_size = max(community_sizes)
        min_size = min(community_sizes)
        
        # Use more appropriate bin ranges
        if max_size > 1000:
            bins = [0, 50, 100, 200, 300, 500, 1000, max_size + 100]
        else:
            bins = [0, 20, 50, 100, 200, 300, 500, max_size + 50]
        
        # Create histogram with density=False to show actual counts
        counts, bin_edges, patches = plt.hist(community_sizes, bins=bins, 
                                             alpha=0.7, color='skyblue', 
                                             edgecolor='black', 
                                             density=False)  # Show actual counts, not density
        
        plt.xlabel('Community Size')
        plt.ylabel('Number of Communities')
        plt.title('Community Size Distribution (Actual Counts)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (count, patch) in enumerate(zip(counts, patches)):
            if count > 0:
                plt.text(patch.get_x() + patch.get_width()/2, count + 0.1,
                        f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 2: Rank-size plot with proper log scale
        plt.subplot(1, 2, 2)
        
        sizes_sorted = sorted(community_sizes, reverse=True)
        ranks = range(1, len(sizes_sorted) + 1)
        
        plt.plot(ranks, sizes_sorted, 'o-', linewidth=2, color='green', markersize=6)
        plt.xlabel('Community Rank')
        plt.ylabel('Community Size')
        plt.title('Rank-Size Distribution')
        plt.grid(True, alpha=0.3)
        
        # Use log scale only if there's large variation
        if max_size / min_size > 100:  # Only use log if variation > 100x
            plt.yscale('log')
            plt.ylabel('Community Size (log scale)')
        
        # Add some data point labels
        for i, (rank, size) in enumerate(zip(ranks, sizes_sorted)):
            if i < 5 or i % 5 == 0 or i == len(ranks) - 1:  # Label first 5, every 5th, and last
                plt.annotate(f'{size}', (rank, size), 
                            textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        
        # Print debug information
        print(f"Community sizes: {sorted(community_sizes, reverse=True)}")
        print(f"Total communities: {len(community_sizes)}")
        print(f"Size range: {min_size} - {max_size}")
        print(f"Bin edges: {bins}")
        
        save_path = os.path.join(self.images_dir, 'community_size_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Community size distribution saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(1)
        return plt.gcf()

    def plot_community_size_distribution_detailed(self, community_sizes: list):
        """Plot detailed community size distribution with all communities"""
        print("Generating detailed community size distribution...")
        
        plt.figure(figsize=(14, 6))
        
        # Subplot 1: Histogram with all communities
        plt.subplot(1, 2, 1)
        
        # Create bins that cover the full range
        max_size = max(community_sizes)
        min_size = min(community_sizes)
        
        # Create appropriate bins based on data range
        if max_size > 1000:
            bins = np.linspace(0, max_size + 100, 15)
        else:
            bins = np.linspace(0, max_size + 50, 12)
        
        counts, bin_edges, patches = plt.hist(community_sizes, bins=bins, 
                                             alpha=0.7, color='skyblue', 
                                             edgecolor='black')
        
        plt.xlabel('Community Size')
        plt.ylabel('Number of Communities')
        plt.title(f'Community Size Distribution\n(Total: {len(community_sizes)} communities)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (count, patch) in enumerate(zip(counts, patches)):
            if count > 0:
                plt.text(patch.get_x() + patch.get_width()/2, count + 0.1,
                        f'{int(count)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Subplot 2: Rank-size plot with all communities labeled
        plt.subplot(1, 2, 2)
        
        sizes_sorted = sorted(community_sizes, reverse=True)
        ranks = range(1, len(sizes_sorted) + 1)
        
        plt.plot(ranks, sizes_sorted, 'o-', linewidth=2, color='green', markersize=4)
        plt.xlabel('Community Rank')
        plt.ylabel('Community Size')
        plt.title('Rank-Size Distribution (All Communities)')
        plt.grid(True, alpha=0.3)
        
        # Label important points
        label_indices = [0]  # First point
        if len(ranks) > 1:
            label_indices.append(len(ranks)//4)
        if len(ranks) > 2:
            label_indices.append(len(ranks)//2)
        if len(ranks) > 3:
            label_indices.append(len(ranks)-1)  # Last point
        
        for idx in label_indices:
            if idx < len(ranks):
                plt.annotate(f'Rank {ranks[idx]}\nSize: {sizes_sorted[idx]}', 
                            (ranks[idx], sizes_sorted[idx]),
                            textcoords="offset points", 
                            xytext=(10, 10), 
                            ha='left', 
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        
        # Print detailed information
        print(f"=== COMMUNITY SIZE ANALYSIS ===")
        print(f"Total communities: {len(community_sizes)}")
        print(f"Size range: {min_size} - {max_size}")
        print(f"Average size: {np.mean(community_sizes):.1f}")
        print(f"Median size: {np.median(community_sizes):.1f}")
        print(f"Top 5 largest communities: {sorted(community_sizes, reverse=True)[:5]}")
        
        save_path = os.path.join(self.images_dir, 'community_size_detailed.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed community size distribution saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    def plot_centrality_analysis(self, influential_nodes: dict):
        """Plot centrality analysis with improved formatting"""
        print("Generating centrality analysis...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        measures = list(influential_nodes.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, measure in enumerate(measures):
            nodes_data = influential_nodes[measure][:6]  # Top 6 nodes
            
            if not nodes_data:
                continue
                
            nodes = [f"Node {x[0]}" for x in nodes_data]
            scores = [x[1] for x in nodes_data]
            communities = [x[2] for x in nodes_data]
            
            # Create horizontal bar chart for better readability
            bars = axes[idx].barh(nodes, scores, color=colors[idx], alpha=0.7, 
                                 edgecolor='black', height=0.6)
            axes[idx].set_title(f'{measure.capitalize()} Centrality', fontweight='bold', fontsize=12)
            axes[idx].set_xlabel('Centrality Score')
            
            # Add values on bars
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                axes[idx].text(width + 0.001, bar.get_y() + bar.get_height()/2,
                             f'{score:.4f}', ha='left', va='center', 
                             fontsize=9, fontweight='bold')
            
            # Add community info
            for i, (node, comm) in enumerate(zip(nodes, communities)):
                axes[idx].text(-0.05, i, f'Comm {comm}', ha='right', va='center', 
                             fontsize=8, color='gray', transform=axes[idx].transData)
            
            # Set consistent x-limits for comparison
            max_score = max(scores) if scores else 0
            axes[idx].set_xlim(0, max_score * 1.15)
            axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.images_dir, 'centrality_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Centrality analysis saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(1)
        return plt.gcf()

    def create_all_visualizations(self, community_sizes: list, influential_nodes: dict):
        """Create all visualizations with proper timing"""
        print("=" * 60)
        print("STARTING VISUALIZATION PROCESS")
        print("=" * 60)
        
        # Choose the right network visualization based on size
        if self.G.number_of_nodes() > 1000:
            print("Large network detected, using optimized visualization...")
            network_fig = self.plot_network_for_large_graph()
        else:
            print("Small/medium network, using standard visualization...")
            network_fig = self.plot_network_guaranteed()
        
        # Wait for user to see it
        print("Waiting 3 seconds for network graph...")
        plt.pause(3)
        plt.close(network_fig)
        
        # Create community size distribution
        community_fig = self.plot_community_size_distribution(community_sizes)
        print("Waiting 3 seconds for community graph...")
        plt.pause(3)
        plt.close(community_fig)
        
        # Create centrality analysis
        centrality_fig = self.plot_centrality_analysis(influential_nodes)
        print("Waiting 3 seconds for centrality graph...")
        plt.pause(3)
        plt.close(centrality_fig)
        
        print("ALL VISUALIZATIONS COMPLETED!")
        print("Check the 'results/images/' folder for saved images")

    def create_comprehensive_visualizations(self, community_sizes: list, influential_nodes: dict):
        """Create comprehensive visualizations showing ALL communities"""
        print("=" * 60)
        print("CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 60)
        
        # 1. Network with all communities (using updated method)
        print("1. Creating network with ALL communities...")
        network_fig = self.plot_network_all_communities()
        plt.pause(3)
        plt.close(network_fig)
        
        # 2. Special optimized version for 18 communities
        unique_communities = list(set(self.partition.values()))
        if len(unique_communities) == 18:
            print("2. Creating optimized visualization for 18 communities...")
            network_18_fig = self.plot_network_18_communities_optimized()
            plt.pause(3)
            plt.close(network_18_fig)
        
        # 3. Detailed community size distribution
        print("3. Creating detailed community size distribution...")
        self.plot_community_size_distribution_detailed(community_sizes)
        plt.pause(3)
        plt.close()
        
        # 4. Centrality analysis
        print("4. Creating centrality analysis...")
        self.plot_centrality_analysis(influential_nodes)
        plt.pause(3)
        plt.close()
        
        print("COMPREHENSIVE VISUALIZATIONS COMPLETED!")
        print(f"All visualizations saved to: {self.images_dir}")

    def create_community_table(self):
        """Tạo bảng liệt kê chi tiết các community"""
        
        # Tính toán thống kê cho từng community
        community_stats = {}
        
        for node, comm_id in self.partition.items():
            if comm_id not in community_stats:
                community_stats[comm_id] = {
                    'size': 0,
                    'nodes': [],
                    'degree_sum': 0
                }
            
            community_stats[comm_id]['size'] += 1
            community_stats[comm_id]['nodes'].append(node)
            community_stats[comm_id]['degree_sum'] += self.G.degree(node)
        
        # Tạo bảng liệt kê
        print("\n" + "="*60)
        print("BẢNG LIỆT KÊ CÁC COMMUNITY")
        print("="*60)
        print(f"{'Community ID':<15} {'Size':<8} {'Avg Degree':<12} {'Nodes'}")
        print("-"*60)
        
        for comm_id, stats in sorted(community_stats.items()):
            avg_degree = stats['degree_sum'] / stats['size'] if stats['size'] > 0 else 0
            sample_nodes = stats['nodes'][:5]  # Hiển thị 5 node đầu tiên
            nodes_str = str(sample_nodes) + ("..." if len(stats['nodes']) > 5 else "")
            
            print(f"{comm_id:<15} {stats['size']:<8} {avg_degree:<12.2f} {nodes_str}")
        
        # Tổng kết
        print("-"*60)
        print(f"Tổng số communities: {len(community_stats)}")
        print(f"Tổng số nodes: {self.G.number_of_nodes()}")
        
        return community_stats

    def export_community_table_to_file(self, filename="community_table.txt"):
        """Xuất bảng liệt kê community ra file"""
        
        community_stats = self.create_community_table()
        
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("BẢNG LIỆT KÊ CÁC COMMUNITY\n")
            f.write("="*50 + "\n")
            f.write(f"{'Community ID':<15} {'Size':<8} {'Avg Degree':<12} {'Nodes (first 5)'}\n")
            f.write("-"*50 + "\n")
            
            for comm_id, stats in sorted(community_stats.items()):
                avg_degree = stats['degree_sum'] / stats['size'] if stats['size'] > 0 else 0
                sample_nodes = stats['nodes'][:5]
                nodes_str = str(sample_nodes) + ("..." if len(stats['nodes']) > 5 else "")
                
                f.write(f"{comm_id:<15} {stats['size']:<8} {avg_degree:<12.2f} {nodes_str}\n")
            
            f.write("-"*50 + "\n")
        
        print(f"Community table exported to: {filepath}")
        return filepath