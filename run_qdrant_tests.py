#!/usr/bin/env python3
"""
Script to run Qdrant vector tests directly without pytest
"""
import sys
import os

# Add the project root to the path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Run the test file directly
    import unittest
    from tests.services.test_qdrant_vector import TestQdrantVector
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQdrantVector)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)