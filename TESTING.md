# Testing Qdrant Vector Service

## Running the Qdrant Vector Tests

The tests for the Qdrant vector service can be run in multiple ways:

### Method 1: Direct Python Execution (Recommended)
```bash
python3 tests/services/test_qdrant_vector.py
```

### Method 2: Using the Test Runner Script
```bash
python3 run_qdrant_tests.py
```

### Method 3: Using pytest (if dependencies are properly installed)
```bash
PYTHONPATH=. python3 -m pytest tests/services/test_qdrant_vector.py -v
```

## Test Results
All 6 tests should pass:
- test_add_documents
- test_delete
- test_get_all_ids
- test_get_documents_by_ids
- test_get_filtered_ids
- test_similarity_search_with_score_by_vector

## Troubleshooting

If you encounter issues with pytest, it's likely due to missing dependencies. The direct Python execution method is the most reliable way to run these tests.

## Dependencies
The tests require:
- Python 3.x
- The project's requirements from `requirements.txt`
- The test requirements from `test_requirements.txt`

For development, you may want to create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r test_requirements.txt