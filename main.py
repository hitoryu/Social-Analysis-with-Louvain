# main.py
import sys
import os
import argparse

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='Facebook Community Analysis')
    parser.add_argument('--dataset', type=str, default='snap', 
                       choices=['synthetic', 'snap', 'twitter'],
                       help='Dataset to use: synthetic, snap, or twitter')
    
    args = parser.parse_args()
    
    print("🚀 STARTING FACEBOOK COMMUNITY ANALYSIS PROJECT")
    print("=" * 50)
    
    try:
        from data_collection import DataCollector
        from data_preprocessing import DataPreprocessor
        from louvain_algorithm import LouvainCommunityDetection
        from community_analysis import CommunityAnalyzer
        from visualization import ResultVisualizer
        from report_generator import ReportGenerator
        
        # Step 1: Data Collection
        print(f"\n📥 STEP 1: Loading {args.dataset} dataset...")
        collector = DataCollector()
        
        if args.dataset == 'synthetic':
            edges = collector.method_1_synthetic_data()
        elif args.dataset == 'snap':
            edges = collector.method_2_snap_facebook()
        elif args.dataset == 'twitter':
            edges = collector.method_3_snap_twitter()
        
        # Show dataset info
        dataset_info = collector.get_dataset_info(edges)
        print(f"📊 Dataset Info: {dataset_info['nodes']} nodes, {dataset_info['edges']} edges")
        
        # Save the data
        collector.save_data(f'{args.dataset}_edges.csv')
        
        # Step 2: Data Preprocessing
        print("\n🔧 STEP 2: Preprocessing data...")
        preprocessor = DataPreprocessor(edges)
        graph = preprocessor.build_graph()
        graph = preprocessor.clean_graph(min_degree=1)
        basic_stats = preprocessor.analyze_basic_stats()
        
        # Step 3: Community Detection
        print("\n🎯 STEP 3: Detecting communities with Louvain...")
        louvain = LouvainCommunityDetection(graph)
        partition, modularity = louvain.detect_communities(random_state=42)
        community_sizes = louvain.analyze_partition()
        
        # Step 4: Community Analysis
        print("\n📊 STEP 4: Analyzing communities...")
        analyzer = CommunityAnalyzer(graph, partition)
        community_metrics = analyzer.extract_community_metrics()
        community_roles = analyzer.identify_community_roles()
        influential_nodes = analyzer.find_influential_nodes(top_k=6)
        
        # Step 5: Visualization
        print("\n🎨 STEP 5: Creating visualizations...")
        
        # Ensure results directory exists
        os.makedirs('../results/images', exist_ok=True)
        os.makedirs('../results/reports', exist_ok=True)
        os.makedirs('../results/models', exist_ok=True)
        
        visualizer = ResultVisualizer(graph, partition)
        
        # Use the new method that handles all visualizations properly
        visualizer.create_all_visualizations(
            list(community_sizes.values()), 
            influential_nodes
        )
        
        # Step 6: Reporting
        print("\n📝 STEP 6: Generating report...")
        metrics = {'modularity': modularity}
        report_gen = ReportGenerator(graph, partition, metrics, louvain.execution_time)
        report = report_gen.generate_comprehensive_report()
        
        print("\n" + "=" * 50)
        print("✅ PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📈 Final Modularity: {modularity:.4f}")
        print(f"👥 Communities Detected: {len(set(partition.values()))}")
        print(f"📊 Network Size: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"📁 Results saved in 'results/' folder")
        print(f"🖼️  Visualizations saved in 'results/images/'")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()