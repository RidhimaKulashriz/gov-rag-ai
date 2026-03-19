from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import uuid

class QdrantVectorStore:
    """
    Vector store using Qdrant for efficient similarity search
    """
    
    def __init__(self, collection_name: str = "gov_documents", host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = 384  # For all-MiniLM-L6-v2
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create collection with appropriate configuration"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.document_type",
                field_type=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.circular_number",
                field_type=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.date",
                field_type=models.PayloadSchemaType.TEXT
            )
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Add documents with their embeddings to vector store
        """
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'content': doc['content'],
                        'metadata': doc['metadata']
                    }
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_embedding: List[float], k: int = 10, 
               filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents with optional filtering
        """
        # Build filter if provided
        search_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value)
                    )
                )
            search_filter = models.Filter(must=must_conditions)
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=search_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'content': result.payload['content'],
                'metadata': result.payload['metadata'],
                'score': result.score
            })
        
        return formatted_results
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(self.collection_name)