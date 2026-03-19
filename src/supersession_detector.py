from typing import List, Dict, Any, Optional
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SupersessionDetector:
    """
    Detect when newer documents override older ones on the same topic
    """
    
    def __init__(self):
        self.supersession_keywords = [
            'supersedes', 'supersession', 'overrides', 'replaces',
            'withdrawn', 'cancelled', 'revoked', 'amended',
            'in partial modification', 'in exercise of the powers'
        ]
    
    def detect_supersessions(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect supersession relationships between documents
        """
        supersessions = []
        
        # Group documents by topic
        topic_groups = self._group_by_topic(documents)
        
        for topic, docs in topic_groups.items():
            # Sort by date
            sorted_docs = sorted(docs, key=lambda x: self._extract_date_obj(x))
            
            # Check for supersession relationships
            for i, doc in enumerate(sorted_docs):
                doc_date = self._extract_date_obj(doc)
                doc_content = doc['content'].lower()
                
                # Look for explicit supersession statements
                for keyword in self.supersession_keywords:
                    if keyword in doc_content:
                        # Find which document is being superseded
                        superseded = self._find_superseded_document(
                            doc, sorted_docs[:i]
                        )
                        if superseded:
                            supersessions.append({
                                'superseding_document': {
                                    'file': doc['metadata']['source_file'],
                                    'circular_no': doc['metadata'].get('circular_number'),
                                    'date': doc['metadata'].get('date'),
                                    'subject': doc['metadata'].get('subject')
                                },
                                'superseded_document': {
                                    'file': superseded['metadata']['source_file'],
                                    'circular_no': superseded['metadata'].get('circular_number'),
                                    'date': superseded['metadata'].get('date'),
                                    'subject': superseded['metadata'].get('subject')
                                },
                                'topic': topic,
                                'confidence': self._calculate_confidence(doc, superseded)
                            })
        
        return supersessions
    
    def _group_by_topic(self, documents: List[Dict[str, Any]]) -> Dict[str, List]:
        """Group documents by topic using TF-IDF and clustering"""
        if len(documents) < 2:
            return {'general': documents}
        
        # Extract subjects
        subjects = []
        for doc in documents:
            subject = doc['metadata'].get('subject', '')
            if not subject:
                # Use first 100 chars as subject
                subject = doc['content'][:100]
            subjects.append(subject)
        
        # Vectorize subjects
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        vectors = vectorizer.fit_transform(subjects)
        
        # Simple clustering based on similarity
        groups = {}
        for i, doc in enumerate(documents):
            if len(groups) == 0:
                groups[f'topic_{len(groups)}'] = [doc]
            else:
                assigned = False
                for topic, group_docs in groups.items():
                    # Check similarity with first doc in group
                    group_vector = vectors[documents.index(group_docs[0])]
                    similarity = cosine_similarity(vectors[i], group_vector)[0][0]
                    
                    if similarity > 0.3:  # Threshold for same topic
                        groups[topic].append(doc)
                        assigned = True
                        break
                
                if not assigned:
                    groups[f'topic_{len(groups)}'] = [doc]
        
        return groups
    
    def _extract_date_obj(self, document: Dict[str, Any]) -> datetime:
        """Extract datetime object from document metadata"""
        date_str = document['metadata'].get('date')
        if date_str:
            try:
                # Try different date formats
                for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%B %d, %Y', '%d %B %Y']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
            except:
                pass
        return datetime.min  # Default for documents without date
    
    def _find_superseded_document(self, current_doc: Dict, 
                                   previous_docs: List[Dict]) -> Optional[Dict]:
        """Find which previous document is being superseded"""
        current_content = current_doc['content'].lower()
        
        for prev_doc in previous_docs:
            # Check if previous document is mentioned
            prev_circular = prev_doc['metadata'].get('circular_number', '')
            prev_date = prev_doc['metadata'].get('date', '')
            
            if prev_circular and prev_circular.lower() in current_content:
                return prev_doc
            
            if prev_date and prev_date in current_content:
                return prev_doc
            
            # Check for references to previous subject
            prev_subject = prev_doc['metadata'].get('subject', '')[:50]
            if prev_subject and prev_subject.lower() in current_content:
                return prev_doc
        
        return None
    
    def _calculate_confidence(self, superseding: Dict, superseded: Dict) -> float:
        """Calculate confidence score for supersession detection"""
        confidence = 0.5  # Base confidence
        
        # Boost if circular numbers match
        if (superseding['metadata'].get('circular_number') and 
            superseded['metadata'].get('circular_number')):
            confidence += 0.2
        
        # Boost if subjects are similar
        subj1 = superseding['metadata'].get('subject', '')
        subj2 = superseded['metadata'].get('subject', '')
        if subj1 and subj2:
            # Simple word overlap
            words1 = set(subj1.lower().split())
            words2 = set(subj2.lower().split())
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            confidence += overlap * 0.3
        
        return min(confidence, 1.0)