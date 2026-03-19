from typing import List, Dict, Any
import re

class SmartDocumentChunker:
    """
    Intelligent chunking that preserves document structure and metadata
    """
    
    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document while preserving structure and metadata
        """
        content = document['content']
        metadata = document['metadata']
        
        # Try to chunk by sections first
        sections = self._split_by_sections(content)
        
        if len(sections) > 1:
            chunks = self._chunk_sections(sections, metadata)
        else:
            chunks = self._chunk_by_tokens(content, metadata)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk['metadata']['chunk_index'] = i
            chunk['metadata']['total_chunks'] = len(chunks)
            chunk['metadata']['chunk_type'] = self._identify_chunk_type(chunk['content'])
        
        return chunks
    
    def _split_by_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split document into logical sections"""
        section_patterns = [
            r'(?:^|\n)(?:[A-Z][A-Z\s]+)(?:\n|$)',
            r'(?:^|\n)(?:\d+\.\s+[A-Z][^\n]+)',
            r'(?:^|\n)(?:Chapter\s+\d+[^\n]*)',
            r'(?:^|\n)(?:Section\s+\d+[^\n]*)'
        ]
        
        sections = []
        current_pos = 0
        content_lines = content.split('\n')
        
        for i, line in enumerate(content_lines):
            for pattern in section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    if current_pos < i:
                        section_content = '\n'.join(content_lines[current_pos:i])
                        if section_content.strip():
                            sections.append({
                                'header': content_lines[current_pos] if current_pos > 0 else 'Introduction',
                                'content': section_content
                            })
                    current_pos = i
                    break
        
        # Add last section
        if current_pos < len(content_lines):
            section_content = '\n'.join(content_lines[current_pos:])
            if section_content.strip():
                sections.append({
                    'header': content_lines[current_pos],
                    'content': section_content
                })
        
        return sections
    
    def _chunk_sections(self, sections: List[Dict[str, Any]], metadata: Dict) -> List[Dict[str, Any]]:
        """Chunk by sections"""
        chunks = []
        for section in sections:
            chunks.append({
                'content': f"# {section['header']}\n\n{section['content']}",
                'metadata': {
                    **metadata,
                    'section_header': section['header']
                }
            })
        return chunks
    
    def _chunk_by_tokens(self, content: str, metadata: Dict) -> List[Dict[str, Any]]:
        """Fallback: chunk by tokens"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            chunks.append({
                'content': chunk_content,
                'metadata': {**metadata}
            })
        
        return chunks
    
    def _identify_chunk_type(self, content: str) -> str:
        """Identify the type of content in chunk"""
        content_lower = content.lower()
        
        if re.search(r'\b(?:circular|notification)\b', content_lower):
            return 'legal_document'
        elif re.search(r'\b(?:section|rule|regulation)\b', content_lower):
            return 'legal_provision'
        elif re.search(r'\b(?:dated|date|issued)\b', content_lower):
            return 'metadata'
        else:
            return 'general'