from typing import List, Dict, Any, Tuple
import difflib
from transformers import pipeline

class DocumentComparator:
    """
    Compare multiple documents and generate highlighted differences
    """
    
    def __init__(self):
        # Initialize summarization pipeline
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def compare_documents(self, documents: List[Dict[str, Any]], 
                          query: str) -> Dict[str, Any]:
        """
        Compare multiple documents on the same topic
        """
        if len(documents) < 2:
            return {
                'differences': [],
                'summary': "Not enough documents for comparison",
                'highlights': {}
            }
        
        # Extract key points from each document
        doc_summaries = []
        for doc in documents:
            summary = self._summarize_document(doc['content'], query)
            doc_summaries.append({
                'source': doc['metadata']['source_file'],
                'circular_no': doc['metadata'].get('circular_number', 'Unknown'),
                'date': doc['metadata'].get('date', 'Unknown'),
                'summary': summary
            })
        
        # Find differences between documents
        differences = self._find_differences(doc_summaries)
        
        # Generate markdown with highlights
        markdown_output = self._generate_markdown(doc_summaries, differences)
        
        return {
            'differences': differences,
            'summaries': doc_summaries,
            'markdown': markdown_output,
            'num_documents': len(documents)
        }
    
    def _summarize_document(self, content: str, query: str, max_length: int = 150) -> str:
        """Summarize document content relevant to query"""
        # Truncate content if too long
        if len(content) > 1024:
            content = content[:1024]
        
        try:
            summary = self.summarizer(content, max_length=max_length, 
                                     min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except:
            # Fallback: return first few sentences
            sentences = content.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    def _find_differences(self, summaries: List[Dict]) -> List[Dict]:
        """Find differences between document summaries"""
        differences = []
        
        for i in range(len(summaries)):
            for j in range(i+1, len(summaries)):
                diff = self._compare_two_documents(
                    summaries[i], summaries[j]
                )
                if diff:
                    differences.append(diff)
        
        return differences
    
    def _compare_two_documents(self, doc1: Dict, doc2: Dict) -> Dict:
        """Compare two documents and highlight differences"""
        # Use difflib to find differences
        d = difflib.Differ()
        diff = list(d.compare(doc1['summary'].split(), doc2['summary'].split()))
        
        additions = [word[2:] for word in diff if word.startswith('+ ')]
        deletions = [word[2:] for word in diff if word.startswith('- ')]
        
        if additions or deletions:
            return {
                'document_1': {
                    'source': doc1['source'],
                    'circular_no': doc1['circular_no'],
                    'unique_points': deletions  # Points in doc1 but not in doc2
                },
                'document_2': {
                    'source': doc2['source'],
                    'circular_no': doc2['circular_no'],
                    'unique_points': additions  # Points in doc2 but not in doc1
                },
                'similarity_score': 1 - (len(additions) + len(deletions)) / 
                                    max(len(doc1['summary'].split()), 
                                        len(doc2['summary'].split()))
            }
        return None
    
    def _generate_markdown(self, summaries: List[Dict], 
                          differences: List[Dict]) -> str:
        """Generate markdown output with highlighted differences"""
        markdown = "# Document Comparison Report\n\n"
        
        # Document summaries
        markdown += "## 📄 Document Summaries\n\n"
        for summary in summaries:
            markdown += f"### {summary['source']}\n"
            markdown += f"- **Circular No:** {summary['circular_no']}\n"
            markdown += f"- **Date:** {summary['date']}\n"
            markdown += f"- **Summary:** {summary['summary']}\n\n"
        
        # Differences
        if differences:
            markdown += "## 🔍 Key Differences\n\n"
            for diff in differences:
                markdown += f"### Between {diff['document_1']['source']} and {diff['document_2']['source']}\n\n"
                
                if diff['document_1']['unique_points']:
                    markdown += f"**Unique to {diff['document_1']['source']}:**\n"
                    for point in diff['document_1']['unique_points'][:5]:
                        markdown += f"- ~~{point}~~\n"
                    markdown += "\n"
                
                if diff['document_2']['unique_points']:
                    markdown += f"**Unique to {diff['document_2']['source']}:**\n"
                    for point in diff['document_2']['unique_points'][:5]:
                        markdown += f"- **{point}**\n"
                    markdown += "\n"
                
                markdown += f"**Similarity Score:** {diff['similarity_score']:.2%}\n\n"
        else:
            markdown += "## ✅ No Significant Differences Found\n\n"
        
        return markdown