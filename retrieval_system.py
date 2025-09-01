import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import pickle
from pathlib import Path
from dataclasses import dataclass

# For embeddings - we'll use a simple approach first
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class OrdinanceDocument:
    """Simple document model for retrieval system"""
    id: str
    title: str
    content: str
    doc_type: str
    section: str
    file_path: str



class RetrievalSystem:
    def __init__(self):
        self.documents_df = pd.DataFrame()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.document_vectors = None
        self.documents = []
    
    def load_documents(self, documents: List[OrdinanceDocument]):
        """Load processed documents"""
        self.documents = documents
        
        # Create DataFrame
        doc_data = []
        for doc in documents:
            doc_data.append({
                'id': doc.id,
                'title': doc.title,
                'content': doc.content,
                'doc_type': doc.doc_type,
                'section': doc.section,
                'file_path': doc.file_path,
                'combined_text': f"{doc.title} {doc.content}"
            })
        
        self.documents_df = pd.DataFrame(doc_data)
        
        # Create document vectors
        texts = [doc['combined_text'] for doc in doc_data]
        self.document_vectors = self.vectorizer.fit_transform(texts)
        
        print(f"Loaded {len(documents)} documents into retrieval system")
    
    def retrieve(self, query: str, k: int = 5, doc_type_filter: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant documents"""
        if self.document_vectors is None:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # Filter by document type if specified
        df_filtered = self.documents_df
        similarity_scores = similarities
        
        if doc_type_filter:
            mask = df_filtered['doc_type'] == doc_type_filter
            df_filtered = df_filtered[mask]
            similarity_scores = similarities[mask.values]
        
        # Get top-k results
        if len(df_filtered) == 0:
            return []
        
        top_indices = np.argsort(similarity_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if doc_type_filter:
                actual_idx = df_filtered.iloc[idx].name
                score = similarity_scores[idx]
            else:
                actual_idx = idx
                score = similarities[idx]
            
            doc_row = self.documents_df.iloc[actual_idx]
            
            results.append({
                'id': doc_row['id'],
                'title': doc_row['title'],
                'content': doc_row['content'],
                'doc_type': doc_row['doc_type'],
                'similarity': float(score),
                'section': doc_row['section'],
                'file_path': doc_row['file_path']
            })
        
        return results
    
    def exact_lookup(self, identifier: str) -> Optional[Dict]:
        """Lookup by exact identifier"""
        # Search in titles and content
        matches = self.documents_df[
            (self.documents_df['title'].str.contains(identifier, case=False, na=False)) |
            (self.documents_df['content'].str.contains(identifier, case=False, na=False))
        ]
        
        if not matches.empty:
            best_match = matches.iloc[0]
            return {
                'id': best_match['id'],
                'title': best_match['title'],
                'content': best_match['content'],
                'doc_type': best_match['doc_type'],
                'similarity': 1.0,
                'section': best_match['section'],
                'file_path': best_match['file_path']
            }
        
        return None
    
    def save_state(self, filepath: str):
        """Save retrieval system state"""
        state = {
            'documents_df': self.documents_df,
            'document_vectors': self.document_vectors,
            'vectorizer': self.vectorizer,
            'documents': self.documents
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load retrieval system state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.documents_df = state['documents_df']
        self.document_vectors = state['document_vectors']
        self.vectorizer = state['vectorizer']
        self.documents = state['documents']

# Test retrieval system
if __name__ == "__main__":
    from document_processor import DocumentProcessor, OrdinanceDocument
    
    # Create sample documents
    sample_docs = [
        OrdinanceDocument(
            id="test_001",
            title="Building Permit Requirements",
            content="All construction requires building permits. Applications must include site plans and structural drawings. Contact Building Department at (555) 123-4567.",
            doc_type="building",
            section="Section 1",
            file_path="test.txt"
        ),
        OrdinanceDocument(
            id="test_002", 
            title="Noise Ordinance Regulations",
            content="Construction noise limited to 7 AM - 6 PM weekdays. Maximum 65 dB at property line. Violations result in fines up to $2000.",
            doc_type="noise",
            section="Section 2", 
            file_path="test.txt"
        )
    ]
    
    # Test retrieval
    retrieval = RetrievalSystem()
    retrieval.load_documents(sample_docs)
    
    # Test queries
    test_queries = [
        "What permits do I need for construction?",
        "What are the noise restrictions?",
        "Building Department contact"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieval.retrieve(query, k=2)
        for result in results:
            print(f"  - {result['title']} (similarity: {result['similarity']:.2f})")
