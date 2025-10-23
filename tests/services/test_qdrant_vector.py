import unittest
from unittest.mock import patch, MagicMock
from app.services.vector_store.qdrant_vector import QdrantVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class TestQdrantVector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock embeddings object
        self.mock_embeddings = MagicMock(spec=Embeddings)
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # Create a QdrantVector instance with mock parameters
        with patch('app.services.vector_store.qdrant_vector.Qdrant.__init__', return_value=None):
            self.qdrant_vector = QdrantVector(
                url="http://localhost:6333",
                api_key=None,
                collection_name="test_collection",
                embeddings=self.mock_embeddings
            )
            
    @patch('app.services.vector_store.qdrant_vector.Qdrant.add_documents')
    def test_add_documents(self, mock_add_documents):
        """Test add_documents method."""
        # Create test documents
        docs = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"})
        ]
        ids = ["id1", "id2"]
        
        # Mock the parent class method
        mock_add_documents.return_value = ["id1", "id2"]
        
        # Call the method
        result = self.qdrant_vector.add_documents(docs, ids)
        
        # Assertions
        self.assertEqual(result, ["id1", "id2"])
        mock_add_documents.assert_called_once()
        
    @patch('app.services.vector_store.qdrant_vector.Qdrant.similarity_search_with_score')
    def test_similarity_search_with_score_by_vector(self, mock_similarity_search):
        """Test similarity_search_with_score_by_vector method."""
        # Mock the parent class method
        mock_doc = Document(page_content="Test document", metadata={"source": "test"})
        mock_similarity_search.return_value = [(mock_doc, 0.8)]
        
        # Call the method
        result = self.qdrant_vector.similarity_search_with_score_by_vector(
            embedding=[0.1, 0.2, 0.3, 0.4],
            k=4
        )
        
        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 0.8)  # score
        mock_similarity_search.assert_called_once()
        
    @patch('app.services.vector_store.qdrant_vector.QdrantClient')
    def test_get_all_ids(self, mock_qdrant_client):
        """Test get_all_ids method."""
        # Mock the Qdrant client response
        mock_client_instance = MagicMock()
        mock_client_instance.scroll.return_value = (
            [MagicMock(payload={"file_id": "test_id_1"}), MagicMock(payload={"file_id": "test_id_2"})],
            None  # No next page
        )
        mock_qdrant_client.return_value = mock_client_instance
        
        # Call the method
        result = self.qdrant_vector.get_all_ids()
        
        # Assertions
        self.assertEqual(len(result), 2)
        self.assertIn("test_id_1", result)
        self.assertIn("test_id_2", result)
        
    @patch('app.services.vector_store.qdrant_vector.QdrantClient')
    def test_get_filtered_ids(self, mock_qdrant_client):
        """Test get_filtered_ids method."""
        # Mock the Qdrant client response
        mock_client_instance = MagicMock()
        mock_client_instance.scroll.return_value = (
            [MagicMock(payload={"file_id": "filtered_id_1"})],
            None  # No next page
        )
        mock_qdrant_client.return_value = mock_client_instance
        
        # Call the method
        result = self.qdrant_vector.get_filtered_ids(["filtered_id_1", "filtered_id_2"])
        
        # Assertions
        self.assertEqual(len(result), 1)
        self.assertIn("filtered_id_1", result)
        
    @patch('app.services.vector_store.qdrant_vector.QdrantClient')
    def test_get_documents_by_ids(self, mock_qdrant_client):
        """Test get_documents_by_ids method."""
        # Mock the Qdrant client response
        mock_client_instance = MagicMock()
        mock_point = MagicMock()
        mock_point.payload = {"page_content": "Test content", "source": "test", "file_id": "doc_id_1"}
        mock_client_instance.scroll.return_value = (
            [mock_point],
            None  # No next page
        )
        mock_qdrant_client.return_value = mock_client_instance
        
        # Call the method
        result = self.qdrant_vector.get_documents_by_ids(["doc_id_1"])
        
        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "Test content")
        self.assertEqual(result[0].metadata["source"], "test")
        
    @patch('app.services.vector_store.qdrant_vector.QdrantClient')
    def test_delete(self, mock_qdrant_client):
        """Test delete method."""
        # Mock the Qdrant client
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        
        # Call the method
        self.qdrant_vector.delete(["delete_id_1", "delete_id_2"])
        
        # Assertions
        mock_client_instance.delete.assert_called_once()


if __name__ == '__main__':
    unittest.main()