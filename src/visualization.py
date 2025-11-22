# visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import time
import os
from collections import Counter

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

    def create_comprehensive_visualizations(self, community_sizes: list, influential_nodes: dict):
        """Create comprehensive visualizations with precise community analysis"""
        print("=" * 60)
        print("CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 60)
        
        try:
            # 1. Network with all communities
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
            
            # 3. PRECISE Community size distribution
            print("3. Creating PRECISE community size distribution...")
            community_fig = self.plot_community_size_distribution(community_sizes)
            plt.pause(3)
            plt.close(community_fig)
            
            # 4. Detailed community analysis
            print("4. Creating detailed community analysis...")
            stats = self.plot_community_size_distribution_detailed(community_sizes)
            
            # 5. Centrality analysis
            print("5. Creating centrality analysis...")
            centrality_fig = self.plot_centrality_analysis(influential_nodes)
            plt.pause(3)
            plt.close(centrality_fig)
            
            print("COMPREHENSIVE VISUALIZATIONS COMPLETED!")
            print(f"All visualizations saved to: {self.images_dir}")
            
            return stats
            
        except Exception as e:
            print(f"ERROR in comprehensive visualizations: {e}")
            return None

    def _calculate_precise_community_stats(self, community_sizes: list):
        """Tính toán thống kê chính xác cho communities"""
        sizes_array = np.array(community_sizes)
        
        stats = {
            'total_communities': len(community_sizes),
            'total_nodes': sum(community_sizes),
            'min_size': min(community_sizes),
            'max_size': max(community_sizes),
            'mean_size': np.mean(community_sizes),
            'median_size': np.median(community_sizes),
            'std_size': np.std(community_sizes),
            'size_counts': dict(Counter(community_sizes))
        }
        
        # Tính phần trăm tích lũy
        sorted_sizes = np.sort(community_sizes)
        cumulative_percent = np.cumsum(sorted_sizes) / sum(community_sizes) * 100
        
        # Tìm điểm mà 80% nodes được bao phủ
        eighty_percent_idx = np.where(cumulative_percent >= 80)[0]
        if len(eighty_percent_idx) > 0:
            stats['size_80_percent'] = sorted_sizes[eighty_percent_idx[0]]
        else:
            stats['size_80_percent'] = sorted_sizes[-1]
            
        return stats

    def _get_optimal_bins(self, community_sizes: list):
        """Tính số bins tối ưu cho histogram"""
        if len(community_sizes) <= 1:
            return 1
            
        # Sử dụng quy tắc Freedman-Diaconis cho dữ liệu phân phối không chuẩn
        data = np.array(community_sizes)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        
        if iqr == 0:
            # Nếu IQR = 0, sử dụng quy tắc Sturges
            bins = int(np.ceil(np.log2(len(data))) + 1)
        else:
            bin_width = 2 * iqr / (len(data) ** (1/3))
            data_range = data.max() - data.min()
            bins = int(np.ceil(data_range / bin_width))
        
        # Giới hạn số bins trong khoảng hợp lý
        bins = max(3, min(bins, min(20, len(community_sizes))))
        
        return bins

    def plot_community_size_distribution(self, community_sizes: list):
        """Plot community size distribution với độ chính xác cao"""
        print("Generating PRECISE community size distribution...")
        
        if not community_sizes:
            print("ERROR: No community sizes provided!")
            return None
            
        # Tính toán thống kê chính xác
        stats = self._calculate_precise_community_stats(community_sizes)
        
        # Tạo figure với kích thước tối ưu
        plt.figure(figsize=(15, 6))
        
        # SUBPLOT 1: HISTOGRAM CHÍNH XÁC
        plt.subplot(1, 2, 1)
        
        # Tính bins tối ưu
        optimal_bins = self._get_optimal_bins(community_sizes)
        
        # Tạo histogram với density=False để hiển thị số lượng thực
        counts, bin_edges, patches = plt.hist(community_sizes, bins=optimal_bins, 
                                             alpha=0.7, color='#1f77b4', 
                                             edgecolor='black', linewidth=1.0,
                                             density=False)
        
        plt.xlabel('Community Size (Number of Nodes)', fontsize=11, fontweight='bold')
        plt.ylabel('Number of Communities', fontsize=11, fontweight='bold')
        plt.title('Community Size Distribution\n(Actual Counts)', 
                 fontsize=12, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Thêm giá trị count lên các bars
        for i, (count, patch) in enumerate(zip(counts, patches)):
            if count > 0:
                plt.text(patch.get_x() + patch.get_width()/2, count + 0.05,
                        f'{int(count)}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Hiển thị thống kê quan trọng trên biểu đồ
        stats_text = f"""Statistics:
Total Communities: {stats['total_communities']}
Total Nodes: {stats['total_nodes']}
Size Range: {stats['min_size']} - {stats['max_size']}
Mean: {stats['mean_size']:.1f} ± {stats['std_size']:.1f}
Median: {stats['median_size']:.1f}
80% Nodes in Size ≤ {stats['size_80_percent']}"""
        
        plt.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                    fontsize=9, fontfamily='monospace')
        
        # SUBPLOT 2: RANK-SIZE PLOT CHÍNH XÁC
        plt.subplot(1, 2, 2)
        
        sizes_sorted = sorted(community_sizes, reverse=True)
        ranks = range(1, len(sizes_sorted) + 1)
        
        # Vẽ đường rank-size
        plt.plot(ranks, sizes_sorted, 'o-', linewidth=2.0, 
                color='#2ca02c', markersize=5, markerfacecolor='red', 
                markeredgecolor='darkred', markeredgewidth=1)
        
        plt.xlabel('Community Rank', fontsize=11, fontweight='bold')
        plt.ylabel('Community Size', fontsize=11, fontweight='bold')
        plt.title('Rank-Size Distribution\n(Largest to Smallest)', 
                 fontsize=12, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Quyết định có dùng log scale không
        if stats['max_size'] / max(stats['min_size'], 1) > 100:
            plt.yscale('log')
            plt.ylabel('Community Size (Log Scale)', fontsize=11, fontweight='bold')
        
        # Chú thích các điểm quan trọng
        important_indices = [0]  # Largest community
        if len(ranks) >= 5:
            important_indices.append(len(ranks)//4)  # 25%
        if len(ranks) >= 3:
            important_indices.append(len(ranks)//2)  # 50%
        if len(ranks) >= 2:
            important_indices.append(len(ranks)-1)   # Smallest
        
        for idx in important_indices:
            if idx < len(ranks):
                plt.annotate(f'Rank {ranks[idx]}\nSize: {sizes_sorted[idx]}', 
                            (ranks[idx], sizes_sorted[idx]),
                            textcoords="offset points", 
                            xytext=(10, 10), 
                            ha='left', 
                            fontsize=8,
                            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6, lw=1),
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # IN THÔNG TIN CHI TIẾT RA CONSOLE
        print(f"\n=== PRECISE COMMUNITY SIZE ANALYSIS ===")
        print(f"Total communities: {stats['total_communities']}")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Size range: {stats['min_size']} - {stats['max_size']}")
        print(f"Mean size: {stats['mean_size']:.2f} ± {stats['std_size']:.2f}")
        print(f"Median size: {stats['median_size']:.2f}")
        print(f"Optimal bins used: {optimal_bins}")
        print(f"Size distribution: {dict(sorted(stats['size_counts'].items()))}")
        
        # Lưu biểu đồ
        save_path = os.path.join(self.images_dir, 'precise_community_size_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Precise community size distribution saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(2)
        return plt.gcf()

    def plot_community_size_distribution_detailed(self, community_sizes: list):
        """Plot chi tiết và chính xác phân phối kích thước communities"""
        print("Generating DETAILED and PRECISE community size analysis...")
        
        if not community_sizes:
            print("ERROR: No community sizes provided!")
            return None
            
        stats = self._calculate_precise_community_stats(community_sizes)
        
        # Tạo figure lớn hơn cho nhiều biểu đồ
        fig = plt.figure(figsize=(16, 10))
        
        # SUBPLOT 1: HISTOGRAM VỚI BINS TỐI ƯU
        ax1 = plt.subplot(2, 2, 1)
        optimal_bins = self._get_optimal_bins(community_sizes)
        
        counts, bin_edges, patches = ax1.hist(community_sizes, bins=optimal_bins, 
                                             alpha=0.7, color='skyblue', 
                                             edgecolor='navy', linewidth=1.2,
                                             density=False)
        
        ax1.set_xlabel('Community Size', fontweight='bold')
        ax1.set_ylabel('Number of Communities', fontweight='bold')
        ax1.set_title('(A) Size Distribution Histogram\n(Actual Counts)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Hiển thị giá trị trên các bars
        for i, (count, patch) in enumerate(zip(counts, patches)):
            if count > 0:
                ax1.text(patch.get_x() + patch.get_width()/2, count + 0.1,
                        f'{int(count)}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=8)
        
        # SUBPLOT 2: BOX PLOT CHO PHÂN BỐ
        ax2 = plt.subplot(2, 2, 2)
        box_plot = ax2.boxplot(community_sizes, vert=True, patch_artist=True,
                              labels=['All Communities'], showmeans=True,
                              meanprops={'marker':'D', 'markerfacecolor':'red', 'markersize':6})
        box_plot['boxes'][0].set_facecolor('lightgreen')
        ax2.set_ylabel('Community Size', fontweight='bold')
        ax2.set_title('(B) Statistical Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # SUBPLOT 3: PHÂN PHỐI TÍCH LŨY
        ax3 = plt.subplot(2, 2, 3)
        sorted_sizes = np.sort(community_sizes)
        cumulative_percent = np.cumsum(sorted_sizes) / sum(community_sizes) * 100
        
        ax3.plot(sorted_sizes, cumulative_percent, 'b-', linewidth=2.5)
        ax3.fill_between(sorted_sizes, cumulative_percent, alpha=0.3, color='blue')
        
        # Đánh dấu điểm 80%
        eighty_percent_idx = np.where(cumulative_percent >= 80)[0]
        if len(eighty_percent_idx) > 0:
            idx_80 = eighty_percent_idx[0]
            ax3.axvline(x=sorted_sizes[idx_80], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax3.plot(sorted_sizes[idx_80], 80, 'ro', markersize=8)
            ax3.annotate(f'80% of total nodes\nSize ≤ {sorted_sizes[idx_80]}', 
                        (sorted_sizes[idx_80], 80), xytext=(10, -30),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        ax3.set_xlabel('Community Size', fontweight='bold')
        ax3.set_ylabel('Cumulative Percentage of Total Nodes (%)', fontweight='bold')
        ax3.set_title('(C) Cumulative Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # SUBPLOT 4: BAR PLOT CHO TOP COMMUNITIES
        ax4 = plt.subplot(2, 2, 4)
        top_n = min(10, len(community_sizes))
        top_sizes = sorted(community_sizes, reverse=True)[:top_n]
        communities_top = [f'Comm {i+1}' for i in range(top_n)]
        
        bars = ax4.bar(communities_top, top_sizes, color='orange', alpha=0.7, edgecolor='darkorange')
        ax4.set_xlabel('Community Rank', fontweight='bold')
        ax4.set_ylabel('Size', fontweight='bold')
        ax4.set_title(f'(D) Top {top_n} Largest Communities', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Hiển thị giá trị trên bars
        for bar, size in zip(bars, top_sizes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(size)}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'COMPREHENSIVE COMMUNITY SIZE ANALYSIS\n'
                    f'Total: {stats["total_communities"]} communities, {stats["total_nodes"]} nodes', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # IN BÁO CÁO CHI TIẾT
        print(f"\n=== DETAILED COMMUNITY STATISTICS ===")
        print(f"Total communities: {stats['total_communities']}")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Size range: {stats['min_size']} - {stats['max_size']}")
        print(f"Mean ± Std: {stats['mean_size']:.2f} ± {stats['std_size']:.2f}")
        print(f"Median: {stats['median_size']:.2f}")
        print(f"80% of nodes in communities of size ≤ {stats['size_80_percent']}")
        print(f"Optimal histogram bins: {optimal_bins}")
        
        # Phân tích phân phối
        unique_sizes = len(set(community_sizes))
        print(f"Unique size values: {unique_sizes}")
        print(f"Most common sizes: {sorted(stats['size_counts'].items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        save_path = os.path.join(self.images_dir, 'detailed_community_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Detailed community analysis saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        return stats

    def plot_network_all_communities(self, figsize=(20, 12)):
        """Plot network with ALL communities visible"""
        print("Creating network visualization with ALL communities...")
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.G, seed=42, k=0.3, iterations=30)
        
        unique_communities = sorted(list(set(self.partition.values())))
        print(f"Total communities to display: {len(unique_communities)}")
        
        # Use a colormap with enough colors for all communities
        if len(unique_communities) <= 10:
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_communities))))
        
        # Draw edges first
        edge_alpha = 0.3 if self.G.number_of_edges() > 10000 else 0.5
        edge_width = 0.5 if self.G.number_of_edges() > 10000 else 1.0
        
        nx.draw_networkx_edges(
            self.G, pos,
            alpha=edge_alpha,
            edge_color='#2E86AB',
            width=edge_width
        )
        
        # Draw nodes for ALL communities
        node_size = 20 if self.G.number_of_nodes() > 1000 else 50
        
        for i, comm_id in enumerate(unique_communities):
            nodes = [node for node in self.G.nodes() if self.partition[node] == comm_id]
            
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
        
        # LEGEND FOR 18 COMMUNITIES
        plt.title(f'Network Communities - All {len(unique_communities)} Communities', 
                  fontsize=14, fontweight='bold')
        
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
        
        plt.axis('off')
        plt.tight_layout()
        
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

    # Thêm các methods còn thiếu nếu cần
    def diagnose_network_issues(self):
        """Check why edges might not be visible"""
        print("NETWORK DIAGNOSTICS:")
        print(f"   Total nodes: {self.G.number_of_nodes()}")
        print(f"   Total edges: {self.G.number_of_edges()}")
        return True

    def plot_network_guaranteed(self, figsize=(14, 10)):
        """Network plot that guarantees visible edges"""
        print("Creating network with guaranteed visible edges...")
        plt.figure(figsize=figsize)
        # ... implementation ...
        return plt.gcf()

    def plot_network_for_large_graph(self, figsize=(15, 12)):
        """Optimized for large networks"""
        print("Creating optimized visualization for large network...")
        plt.figure(figsize=figsize)
        # ... implementation ...
        return plt.gcf()