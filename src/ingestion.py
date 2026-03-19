import os
import fitz  # PyMuPDF for PDF processing
import pytesseract
from pdf2image import convert_from_path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime
import hashlib

class EnhancedDocumentIngestion:
    """
    Enhanced ingestion pipeline for government PDF documents
    Handles both scanned and digitally-born PDFs
    """
    
    def __init__(self, data_path: str = "data/pdfs"):
        self.data_path = data_path
        self.document_types = {
            'circular': r'circular|circular no|no\.?\s*\d+',
            'notification': r'notification|notify|gazette',
            'order': r'order|ordinance|directive',
            'press_release': r'press release|press note|media',
            'guidelines': r'guideline|instruction|manual'
        }
        
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single PDF document with OCR for scanned documents
        """
        # Check if scanned or digital
        is_scanned = self._check_if_scanned(file_path)
        
        if is_scanned:
            text = self._extract_text_ocr(file_path)
        else:
            text = self._extract_text_digital(file_path)
        
        # Extract metadata
        metadata = self._extract_metadata(file_path, text)
        
        # Classify document type
        doc_type = self._classify_document(text)
        
        return {
            'content': text,
            'metadata': {
                **metadata,
                'document_type': doc_type,
                'is_scanned': is_scanned,
                'source_file': os.path.basename(file_path),
                'processed_date': datetime.now().isoformat()
            }
        }
    
    def _check_if_scanned(self, file_path: str) -> bool:
        """Check if PDF is scanned (image-based)"""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text = page.get_text()
                if len(text.strip()) > 100:  # Has significant text
                    return False
            return True
        except:
            return True
    
    def _extract_text_digital(self, file_path: str) -> str:
        """Extract text from digital PDF"""
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text
    
    def _extract_text_ocr(self, file_path: str) -> str:
        """Extract text from scanned PDF using OCR"""
        text = ""
        images = convert_from_path(file_path)
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    
    def _extract_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """Extract structured legal metadata from document"""
        metadata = {
            'circular_number': self._extract_circular_number(content),
            'notification_number': self._extract_notification_number(content),
            'date': self._extract_date(content),
            'issuing_authority': self._extract_authority(content),
            'subject': self._extract_subject(content),
            'file_hash': self._generate_file_hash(file_path)
        }
        return metadata
    
    def _extract_circular_number(self, text: str) -> Optional[str]:
        """Extract circular/notification number"""
        patterns = [
            r'circular\s*(?:no\.?)?\s*([A-Z0-9\-/]+)',
            r'no\.?\s*([A-Z0-9\-/]+)',
            r'f\.?\s*no\.?\s*([A-Z0-9\-/]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_notification_number(self, text: str) -> Optional[str]:
        """Extract notification number"""
        patterns = [
            r'notification\s*(?:no\.?)?\s*([A-Z0-9\-/]+)',
            r'gazette\s*(?:no\.?)?\s*([A-Z0-9\-/]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from document"""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def _extract_authority(self, text: str) -> Optional[str]:
        """Extract issuing authority"""
        authorities = [
            'ministry', 'department', 'board', 'commission', 
            'authority', 'secretariat', 'directorate'
        ]
        for authority in authorities:
            pattern = rf'({authority}\s+of\s+[A-Za-z\s]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_subject(self, text: str) -> Optional[str]:
        """Extract document subject"""
        subject_patterns = [
            r'subject:\s*(.+?)(?:\n|$)',
            r're:\s*(.+?)(?:\n|$)',
            r'regarding\s*(.+?)(?:\n|$)'
        ]
        for pattern in subject_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _classify_document(self, text: str) -> str:
        """Classify document type based on content"""
        text_lower = text.lower()
        for doc_type, pattern in self.document_types.items():
            if re.search(pattern, text_lower):
                return doc_type
        return 'other'
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate unique hash for file"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()