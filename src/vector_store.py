import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Manages FAISS vector storage for document embeddings and similarity search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize vector store with sentence transformer model."""
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        self.documents = []
        self.chunk_metadata = []
        
        logger.info(f"Initialized vector store with model: {model_name}")
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store by creating embeddings."""
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        all_chunks = []
        
        for doc in documents:
            # Create chunks from document content
            chunks = self._create_chunks(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No chunks created from documents")
            return
        
        # Extract text for embedding
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        
        # Create embeddings
        logger.info(f"Creating embeddings for {len(chunk_texts)} text chunks")
        embeddings = self.encoder.encode(chunk_texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.documents.extend(documents)
        self.chunk_metadata.extend(all_chunks)
        
        logger.info(f"Successfully added {len(all_chunks)} chunks to vector store")
        logger.info(f"Total documents in store: {len(self.documents)}")
    
    def _create_chunks(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create text chunks from a document for embedding."""
        chunks = []
        
        # Main document text
        if document.get('full_text'):
            text_chunks = self._split_text(document['full_text'], max_chunk_size=1000)
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'text': chunk_text,
                    'source': document['filename'],
                    'chunk_type': 'full_text',
                    'chunk_id': f"{document['filename']}_text_{i}",
                    'document_type': document.get('document_type', 'unknown')
                })
        
        # Individual transactions
        if document.get('transactions'):
            for j, transaction in enumerate(document['transactions']):
                transaction_text = self._format_transaction_text(transaction)
                chunks.append({
                    'text': transaction_text,
                    'source': document['filename'],
                    'chunk_type': 'transaction',
                    'chunk_id': f"{document['filename']}_txn_{j}",
                    'document_type': document.get('document_type', 'unknown'),
                    'transaction_data': transaction
                })
        
        return chunks
    
    def _split_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks with optional overlap."""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at a sentence or word boundary
            chunk = text[start:end]
            
            # Look for sentence boundary
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ')
            
            if last_period > max_chunk_size * 0.8:
                end = start + last_period + 1
            elif last_newline > max_chunk_size * 0.8:
                end = start + last_newline + 1
            elif last_space > max_chunk_size * 0.8:
                end = start + last_space + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def _format_transaction_text(self, transaction: Dict[str, Any]) -> str:
        """Format transaction data into searchable text."""
        date = transaction.get('date', 'Unknown date')
        description = transaction.get('description', 'No description')
        amount = transaction.get('amount', 0)
        txn_type = transaction.get('type', 'unknown')
        
        return f"Transaction on {date}: {description} - Amount: ${abs(amount):.2f} ({txn_type})"
    
    def search(self, query: str, k: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents/chunks using semantic similarity."""
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Create query embedding
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype(np.float32), min(k * 2, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            chunk = self.chunk_metadata[idx].copy()
            chunk['similarity_score'] = float(score)
            
            # Apply filter if specified
            if filter_type and chunk.get('document_type') != filter_type:
                continue
            
            results.append(chunk)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_transaction_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all transactions in the vector store."""
        transactions = []
        
        for chunk in self.chunk_metadata:
            if chunk.get('chunk_type') == 'transaction' and 'transaction_data' in chunk:
                transactions.append(chunk['transaction_data'])
        
        if not transactions:
            return {'total_transactions': 0}
        
        # Calculate basic statistics
        amounts = [t['amount'] for t in transactions]
        debits = [abs(a) for a in amounts if a < 0]
        credits = [a for a in amounts if a >= 0]
        
        summary = {
            'total_transactions': len(transactions),
            'total_debits': len(debits),
            'total_credits': len(credits),
            'total_debit_amount': sum(debits) if debits else 0,
            'total_credit_amount': sum(credits) if credits else 0,
            'net_amount': sum(amounts),
            'avg_transaction': np.mean(amounts) if amounts else 0
        }
        
        return summary
    
    def save(self, filepath: str) -> None:
        """Save vector store to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'chunk_metadata': self.chunk_metadata,
                    'model_name': self.model_name,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"Vector store saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load(self, filepath: str) -> bool:
        """Load vector store from disk."""
        try:
            # Load FAISS index
            if Path(f"{filepath}.faiss").exists():
                self.index = faiss.read_index(f"{filepath}.faiss")
            else:
                logger.error(f"FAISS index file not found: {filepath}.faiss")
                return False
            
            # Load metadata
            if Path(f"{filepath}.pkl").exists():
                with open(f"{filepath}.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.chunk_metadata = data['chunk_metadata']
                    
                    # Verify model compatibility
                    if data['model_name'] != self.model_name:
                        logger.warning(f"Model mismatch: stored={data['model_name']}, current={self.model_name}")
            else:
                logger.error(f"Metadata file not found: {filepath}.pkl")
                return False
            
            logger.info(f"Vector store loaded from {filepath}")
            logger.info(f"Loaded {len(self.documents)} documents, {len(self.chunk_metadata)} chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Clear all data from vector store."""
        self.index.reset()
        self.documents.clear()
        self.chunk_metadata.clear()
        logger.info("Vector store cleared")