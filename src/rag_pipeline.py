import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from .enhanced_ingestion import EnhancedDocumentIngestion
from .enhanced_chunking import SmartDocumentChunker
from .vector_store import QdrantVectorStore
from .supersession_detector import SupersessionDetector
from .comparison_generator import DocumentComparator

class GovernmentRAGPipeline:
    """
    Complete RAG pipeline for government documents with all required features
    """
    
    def __init__(self, collection_name: str = "gov_documents"):
        # Initialize components
        self.ingestion = EnhancedDocumentIngestion()
        self.chunker = SmartDocumentChunker()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = QdrantVectorStore(collection_name)
        self.supersession_detector = SupersessionDetector()
        self.comparator = DocumentComparator()
        
        # Storage for processed documents
        self.processed_documents = []
        self.supersessions = []
    
    def process_documents(self, file_paths: List[str]):
        """
        Process multiple documents through the pipeline
        """
        all_chunks = []
        
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            
            # Ingest document
            document = self.ingestion.process_document(file_path)
            
            # Chunk document
            chunks = self.chunker.chunk_document(document)
            all_chunks.extend(chunks)
            
            print(f"  Created {len(chunks)} chunks")
        
        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        texts = [chunk['content'] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Store in vector database
        print("Storing in vector database...")
        self.vector_store.add_documents(all_chunks, embeddings)
        
        # Store processed documents
        self.processed_documents = all_chunks
        
        # Detect supersessions
        print("Detecting supersessions...")
        self.supersessions = self.supersession_detector.detect_supersessions(
            [{'content': chunk['content'], 'metadata': chunk['metadata']} 
             for chunk in all_chunks]
        )
        
        print(f"Found {len(self.supersessions)} supersession relationships")
        
        return {
            'num_chunks': len(all_chunks),
            'num_supersessions': len(self.supersessions),
            'documents_processed': len(file_paths)
        }
    
    def query(self, question: str, k: int = 15, 
              filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Query the RAG system with filtering options
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(question).tolist()
        
        # Retrieve relevant chunks
        results = self.vector_store.search(
            query_embedding, 
            k=k,
            filter_conditions=filters
        )
        
        # Group results by source for comparison
        grouped_results = self._group_by_source(results)
        
        # Generate comparison if multiple sources
        comparison = None
        if len(grouped_results) > 1:
            comparison = self.comparator.compare_documents(
                [{'content': r[0]['content'], 'metadata': r[0]['metadata']} 
                 for r in grouped_results.values()],
                question
            )
        
        # Check for supersession warnings
        supersession_warnings = self._get_supersession_warnings(results)
        
        return {
            'question': question,
            'results': results,
            'grouped_results': grouped_results,
            'comparison': comparison,
            'supersession_warnings': supersession_warnings,
            'num_sources': len(grouped_results)
        }
    
    def _group_by_source(self, results: List[Dict]) -> Dict[str, List]:
        """Group results by source document"""
        grouped = {}
        for result in results:
            source = result['metadata']['source_file']
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(result)
        return grouped
    
    def _get_supersession_warnings(self, results: List[Dict]) -> List[Dict]:
        """Get supersession warnings for retrieved documents"""
        warnings = []
        
        for result in results:
            source_file = result['metadata']['source_file']
            
            # Check if this document is superseded
            for supersession in self.supersessions:
                if supersession['superseded_document']['file'] == source_file:
                    warnings.append({
                        'type': 'superseded',
                        'message': f"This document is superseded by {supersession['superseding_document']['file']}",
                        'superseding': supersession['superseding_document'],
                        'confidence': supersession['confidence']
                    })
                
                if supersession['superseding_document']['file'] == source_file:
                    warnings.append({
                        'type': 'superseding',
                        'message': f"This document supersedes {supersession['superseded_document']['file']}",
                        'superseded': supersession['superseded_document'],
                        'confidence': supersession['confidence']
                    })
        
        return warnings