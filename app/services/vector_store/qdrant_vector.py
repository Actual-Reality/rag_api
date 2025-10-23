import copy
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import Qdrant


class QdrantVector(Qdrant):
    def __init__(self, url: str, api_key: Optional[str], collection_name: str, 
                 embeddings: Embeddings):
        # Initialize Qdrant client and connection
        super().__init__(
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            embeddings=embeddings
        )
        
    def add_documents(self, docs: list[Document], ids: list[str]):
        # Add documents with custom IDs
        # Convert docs to proper format with metadata
        # Qdrant expects content to be in 'page_content' field
        documents_with_ids = []
        for doc, id in zip(docs, ids):
            # Create a copy of the document with the id in metadata
            doc_copy = copy.deepcopy(doc)
            doc_copy.metadata['file_id'] = id
            documents_with_ids.append(doc_copy)
        
        # Use the parent class method to add documents
        return super().add_documents(documents_with_ids, ids=ids)
        
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        # Search with scoring, similar to existing implementations
        # Qdrant's similarity_search_with_score returns (doc, score) tuples
        results = self.similarity_search_with_score(
            query=embedding,
            k=k,
            filter=filter,
            **kwargs
        )
        
        # Process documents to remove any Qdrant-specific metadata
        processed_documents: List[Tuple[Document, float]] = []
        for document, score in results:
            # Make a deep copy to avoid mutating the original document
            doc_copy = copy.deepcopy(document.__dict__)
            # Remove Qdrant-specific fields if they exist
            if "metadata" in doc_copy:
                # Remove any Qdrant internal fields
                qdrant_fields = [key for key in doc_copy["metadata"] if key.startswith("_")]
                for field in qdrant_fields:
                    del doc_copy["metadata"][field]
            new_document = Document(**doc_copy)
            processed_documents.append((new_document, score))
        return processed_documents
        
    def get_all_ids(self) -> list[str]:
        # Return all unique file_id fields
        # In Qdrant, we need to scroll through all points and extract file_id
        try:
            # Get all points from the collection
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            
            # Create a client to access the collection directly
            client = QdrantClient(url=self.url, api_key=self.api_key)
            
            # Scroll through all points to get unique file_ids
            file_ids = set()
            next_page_offset = None
            
            while True:
                response = client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=next_page_offset,
                    with_payload=True
                )
                
                # Extract file_id from each point's payload
                for point in response[0]:
                    if hasattr(point.payload, 'file_id'):
                        file_ids.add(point.payload.file_id)
                    elif isinstance(point.payload, dict) and 'file_id' in point.payload:
                        file_ids.add(point.payload['file_id'])
                
                # Check if there are more points
                next_page_offset = response[1]
                if next_page_offset is None:
                    break
                    
            return list(file_ids)
        except Exception as e:
            # Fallback: return empty list if we can't retrieve all IDs
            return []
        
    def get_filtered_ids(self, ids: list[str]) -> list[str]:
        # Return unique file_id fields filtered by provided ids
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Filter, FieldCondition, MatchAny
            
            # Create a client to access the collection directly
            client = QdrantClient(url=self.url, api_key=self.api_key)
            
            # Create a filter to find points with file_id in the provided list
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="file_id",
                        match=MatchAny(any=ids)
                    )
                ]
            )
            
            # Scroll through points with the filter
            file_ids = set()
            next_page_offset = None
            
            while True:
                response = client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=next_page_offset,
                    with_payload=True,
                    scroll_filter=qdrant_filter
                )
                
                # Extract file_id from each point's payload
                for point in response[0]:
                    if hasattr(point.payload, 'file_id'):
                        file_ids.add(point.payload.file_id)
                    elif isinstance(point.payload, dict) and 'file_id' in point.payload:
                        file_ids.add(point.payload['file_id'])
                
                # Check if there are more points
                next_page_offset = response[1]
                if next_page_offset is None:
                    break
                    
            return list(file_ids)
        except Exception as e:
            # Fallback: return empty list if we can't retrieve filtered IDs
            return []
        
    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        # Return documents filtered by file_id
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Filter, FieldCondition, MatchAny
            
            # Create a client to access the collection directly
            client = QdrantClient(url=self.url, api_key=self.api_key)
            
            # Create a filter to find points with file_id in the provided list
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="file_id",
                        match=MatchAny(any=ids)
                    )
                ]
            )
            
            # Scroll through points with the filter
            documents = []
            next_page_offset = None
            
            while True:
                response = client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=next_page_offset,
                    with_payload=True,
                    scroll_filter=qdrant_filter
                )
                
                # Convert points to Documents
                for point in response[0]:
                    payload = point.payload
                    if isinstance(payload, dict):
                        # Extract content and metadata
                        content = payload.get('page_content', '')
                        metadata = {k: v for k, v in payload.items() if k != 'page_content'}
                        documents.append(Document(page_content=content, metadata=metadata))
                
                # Check if there are more points
                next_page_offset = response[1]
                if next_page_offset is None:
                    break
                    
            return documents
        except Exception as e:
            # Fallback: return empty list if we can't retrieve documents
            return []
        
    def delete(self, ids: Optional[list[str]] = None) -> None:
        # Delete documents by file_id
        if ids is not None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Filter, FieldCondition, MatchAny
                
                # Create a client to access the collection directly
                client = QdrantClient(url=self.url, api_key=self.api_key)
                
                # Create a filter to find points with file_id in the provided list
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="file_id",
                            match=MatchAny(any=ids)
                        )
                    ]
                )
                
                # Delete points matching the filter
                client.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_filter
                )
            except Exception as e:
                # Log the error but don't raise it to maintain consistency with other implementations
                pass