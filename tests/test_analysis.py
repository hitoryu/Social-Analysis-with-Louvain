# File: tests/test_analysis.py

import unittest
import sys
import os
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from louvain_algorithm import LouvainCommunityDetection

class TestCommunityDetection(unittest.TestCase):
    
    def setUp(self):
        # Tạo dữ liệu test
        self.test_edges = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)]
        self.preprocessor = DataPreprocessor(self.test_edges)
        self.graph = self.preprocessor.build_graph()
    
    def test_graph_construction(self):
        self.assertEqual(self.graph.number_of_nodes(), 6)
        self.assertEqual(self.graph.number_of_edges(), 7)
    
    def test_louvain_algorithm(self):
        louvain = LouvainCommunityDetection(self.graph)
        partition, modularity = louvain.detect_communities()
        
        self.assertIsInstance(partition, dict)
        self.assertGreater(modularity, 0)
        self.assertGreater(len(set(partition.values())), 0)

if __name__ == '__main__':
    unittest.main()