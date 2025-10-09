# compare_algorithms.py
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_comparison_study():
    """Comprehensive comparison of community detection algorithms"""
    print("🔬 STARTING ALGORITHM COMPARISON STUDY")
    print("=" * 60)
    
    try:
        from data_collection import DataCollector
        from data_preprocessing import DataPreprocessor
        from louvain_algorithm import LouvainCommunityDetection
        from community_analysis import CommunityAnalyzer
        
        # Test on multiple datasets
        datasets = [
            ("Synthetic", "synthetic"),
            ("SNAP Facebook", "snap"), 
            ("Kaggle Social", "kaggle")
        ]
        
        results = []
        
        for dataset_name, dataset_type in datasets:
            print(f"\n📊 ANALYZING: {dataset_name}")
            print("-" * 40)
            
            try:
                # Load data
                collector = DataCollector()
                
                if dataset_type == "synthetic":
                    edges = collector.method_1_synthetic_data()
                elif dataset_type == "snap":
                    edges = collector.method_2_snap_facebook()
                elif dataset_type == "kaggle":
                    edges = collector.method_4_kaggle_facebook()
                else:
                    continue
                
                # Preprocess
                preprocessor = DataPreprocessor(edges)
                graph = preprocessor.build_graph()
                graph = preprocessor.clean_graph(min_degree=1)
                
                dataset_info = {
                    'name': dataset_name,
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges(),
                    'density': nx.density(graph)
                }
                
                print(f"   Network: {dataset_info['nodes']} nodes, {dataset_info['edges']} edges")
                
                # Test different algorithms
                algorithms = [
                    ("Louvain", "louvain"),
                    ("Girvan-Newman", "girvan_newman"),
                    ("Label Propagation", "label_propagation"),
                    ("Greedy Modularity", "greedy_modularity")
                ]
                
                for algo_name, algo_type in algorithms:
                    print(f"   🧮 Testing {algo_name}...")
                    
                    try:
                        start_time = time.time()
                        
                        if algo_type == "louvain":
                            louvain = LouvainCommunityDetection(graph)
                            partition, modularity = louvain.detect_communities(random_state=42)
                            communities = len(set(partition.values()))
                            execution_time = time.time() - start_time
                            
                        elif algo_type == "girvan_newman":
                            import networkx as nx
                            start_time = time.time()
                            comp = nx.community.girvan_newman(graph)
                            communities = tuple(sorted(c) for c in next(comp))
                            partition = {}
                            for i, comm in enumerate(communities):
                                for node in comm:
                                    partition[node] = i
                            modularity = nx.community.modularity(graph, communities)
                            execution_time = time.time() - start_time
                            communities = len(communities)
                            
                        elif algo_type == "label_propagation":
                            import networkx as nx
                            start_time = time.time()
                            communities = list(nx.community.label_propagation_communities(graph))
                            partition = {}
                            for i, comm in enumerate(communities):
                                for node in comm:
                                    partition[node] = i
                            modularity = nx.community.modularity(graph, communities)
                            execution_time = time.time() - start_time
                            communities = len(communities)
                            
                        elif algo_type == "greedy_modularity":
                            import networkx as nx
                            start_time = time.time()
                            communities = list(nx.community.greedy_modularity_communities(graph))
                            partition = {}
                            for i, comm in enumerate(communities):
                                for node in comm:
                                    partition[node] = i
                            modularity = nx.community.modularity(graph, communities)
                            execution_time = time.time() - start_time
                            communities = len(communities)
                        
                        # Store results
                        result = {
                            'dataset': dataset_name,
                            'algorithm': algo_name,
                            'nodes': dataset_info['nodes'],
                            'edges': dataset_info['edges'],
                            'density': dataset_info['density'],
                            'modularity': modularity,
                            'communities': communities,
                            'execution_time': execution_time,
                            'efficiency': dataset_info['nodes'] / execution_time if execution_time > 0 else 0
                        }
                        
                        results.append(result)
                        
                        print(f"      ✅ {algo_name}: Modularity={modularity:.4f}, "
                              f"Communities={communities}, Time={execution_time:.2f}s")
                        
                    except Exception as e:
                        print(f"      ❌ {algo_name} failed: {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ Failed to process {dataset_name}: {e}")
                continue
        
        # Generate comprehensive report
        if results:
            generate_comparison_report(results)
            plot_comparison_results(results)
        else:
            print("❌ No results to compare!")
            
    except Exception as e:
        print(f"❌ Comparison study failed: {e}")
        import traceback
        traceback.print_exc()

def generate_comparison_report(results):
    """Generate detailed comparison report"""
    print("\n📝 GENERATING COMPARISON REPORT")
    print("=" * 50)
    
    # Create results directory
    os.makedirs('results/comparison', exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/comparison/algorithm_comparison_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Raw results saved to: {csv_path}")
    
    # Generate summary statistics
    summary = df.groupby(['dataset', 'algorithm']).agg({
        'modularity': ['mean', 'std', 'max'],
        'execution_time': ['mean', 'std'],
        'communities': ['mean', 'std'],
        'efficiency': ['mean', 'max']
    }).round(4)
    
    # Print summary table
    print("\n📊 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Algorithm':<20} {'Modularity':<12} {'Communities':<12} {'Time (s)':<10} {'Efficiency':<12}")
    print("-" * 80)
    
    for (dataset, algorithm), group in df.groupby(['dataset', 'algorithm']):
        avg_modularity = group['modularity'].mean()
        avg_communities = group['communities'].mean()
        avg_time = group['execution_time'].mean()
        avg_efficiency = group['efficiency'].mean()
        
        print(f"{dataset:<15} {algorithm:<20} {avg_modularity:<12.4f} {avg_communities:<12.0f} {avg_time:<10.2f} {avg_efficiency:<12.0f}")
    
    # Find best algorithms
    print("\n🏆 BEST PERFORMING ALGORITHMS")
    print("=" * 50)
    
    # Best by modularity
    best_modularity = df.loc[df.groupby('dataset')['modularity'].idxmax()]
    print("\n📈 HIGHEST MODULARITY:")
    for _, row in best_modularity.iterrows():
        print(f"   {row['dataset']}: {row['algorithm']} (Modularity: {row['modularity']:.4f})")
    
    # Fastest algorithms
    best_speed = df.loc[df.groupby('dataset')['execution_time'].idxmin()]
    print("\n⚡ FASTEST ALGORITHMS:")
    for _, row in best_speed.iterrows():
        print(f"   {row['dataset']}: {row['algorithm']} (Time: {row['execution_time']:.2f}s)")
    
    # Most efficient
    best_efficiency = df.loc[df.groupby('dataset')['efficiency'].idxmax()]
    print("\n🎯 MOST EFFICIENT (Nodes/Second):")
    for _, row in best_efficiency.iterrows():
        print(f"   {row['dataset']}: {row['algorithm']} (Efficiency: {row['efficiency']:.0f} nodes/s)")
    
    # Save detailed report
    report_path = f'results/comparison/comparison_report_{timestamp}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ALGORITHM COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Dataset':<15} {'Algorithm':<20} {'Modularity':<12} {'Communities':<12} {'Time (s)':<10} {'Efficiency':<12}\n")
        f.write("-" * 80 + "\n")
        
        for (dataset, algorithm), group in df.groupby(['dataset', 'algorithm']):
            avg_modularity = group['modularity'].mean()
            avg_communities = group['communities'].mean()
            avg_time = group['execution_time'].mean()
            avg_efficiency = group['efficiency'].mean()
            
            f.write(f"{dataset:<15} {algorithm:<20} {avg_modularity:<12.4f} {avg_communities:<12.0f} {avg_time:<10.2f} {avg_efficiency:<12.0f}\n")
        
        f.write("\nBEST PERFORMERS:\n")
        f.write("Highest Modularity:\n")
        for _, row in best_modularity.iterrows():
            f.write(f"  {row['dataset']}: {row['algorithm']} (Modularity: {row['modularity']:.4f})\n")
        
        f.write("\nFastest Algorithms:\n")
        for _, row in best_speed.iterrows():
            f.write(f"  {row['dataset']}: {row['algorithm']} (Time: {row['execution_time']:.2f}s)\n")
    
    print(f"✅ Detailed report saved to: {report_path}")

def plot_comparison_results(results):
    """Create visualization plots for comparison results"""
    print("\n🎨 CREATING COMPARISON VISUALIZATIONS")
    
    df = pd.DataFrame(results)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Community Detection Algorithm Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Modularity comparison
    ax1 = axes[0, 0]
    modularity_pivot = df.pivot_table(index='dataset', columns='algorithm', values='modularity')
    modularity_pivot.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Modularity Score Comparison', fontweight='bold')
    ax1.set_ylabel('Modularity')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Execution time comparison
    ax2 = axes[0, 1]
    time_pivot = df.pivot_table(index='dataset', columns='algorithm', values='execution_time')
    time_pivot.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('Execution Time Comparison', fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Number of communities detected
    ax3 = axes[1, 0]
    communities_pivot = df.pivot_table(index='dataset', columns='algorithm', values='communities')
    communities_pivot.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax3.set_title('Number of Communities Detected', fontweight='bold')
    ax3.set_ylabel('Number of Communities')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Efficiency (nodes per second)
    ax4 = axes[1, 1]
    efficiency_pivot = df.pivot_table(index='dataset', columns='algorithm', values='efficiency')
    efficiency_pivot.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax4.set_title('Algorithm Efficiency (Nodes/Second)', fontweight='bold')
    ax4.set_ylabel('Nodes per Second')
    ax4.set_yscale('log')  # Log scale for better visualization
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'results/comparison/comparison_plots_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plots saved to: {plot_path}")
    
    plt.show(block=False)
    plt.pause(2)

def quick_comparison():
    """Quick comparison for testing"""
    print("⚡ RUNNING QUICK COMPARISON")
    
    try:
        from data_collection import DataCollector
        from data_preprocessing import DataPreprocessor
        from louvain_algorithm import LouvainCommunityDetection
        import networkx as nx
        
        # Use synthetic data for quick test
        collector = DataCollector()
        edges = collector.method_1_synthetic_data()
        
        preprocessor = DataPreprocessor(edges)
        graph = preprocessor.build_graph()
        graph = preprocessor.clean_graph(min_degree=1)
        
        print(f"📊 Testing on network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Test Louvain
        start_time = time.time()
        louvain = LouvainCommunityDetection(graph)
        partition, modularity = louvain.detect_communities(random_state=42)
        louvain_time = time.time() - start_time
        
        print(f"✅ Louvain: Modularity={modularity:.4f}, Time={louvain_time:.3f}s")
        
        # Test Girvan-Newman (limited)
        try:
            start_time = time.time()
            comp = nx.community.girvan_newman(graph)
            communities = tuple(sorted(c) for c in next(comp))
            gn_modularity = nx.community.modularity(graph, communities)
            gn_time = time.time() - start_time
            print(f"✅ Girvan-Newman: Modularity={gn_modularity:.4f}, Time={gn_time:.3f}s")
        except Exception as e:
            print(f"❌ Girvan-Newman failed: {e}")
        
    except Exception as e:
        print(f" Quick comparison failed: {e}")

if __name__ == "__main__":
    # You can choose which comparison to run
    import argparse
    
    parser = argparse.ArgumentParser(description='Algorithm Comparison')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'quick'],
                       help='Comparison mode: full or quick')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_comparison_study()
    else:
        quick_comparison()
    
    print("\n COMPARISON COMPLETED!")