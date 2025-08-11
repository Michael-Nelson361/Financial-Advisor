import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import re
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF document processing and text extraction for financial documents."""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
        
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process all PDF files in a folder and extract text content."""
        folder_path = Path(folder_path)
        documents = []
        
        if not folder_path.exists():
            raise ValueError(f"Folder path does not exist: {folder_path}")
        
        pdf_files = []
        for ext in self.supported_extensions:
            pdf_files.extend(folder_path.glob(f"*{ext}"))
            pdf_files.extend(folder_path.glob(f"**/*{ext}"))  # Include subdirectories
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            try:
                doc_data = self.process_pdf(str(pdf_path))
                if doc_data:
                    documents.append(doc_data)
                    logger.info(f"Successfully processed: {pdf_path.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
                continue
        
        return documents
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and metadata from a single PDF file."""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            pages_content = []
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                pages_content.append({
                    'page_number': page_num + 1,
                    'content': page_text
                })
                full_text += page_text + "\n"
            
            doc.close()
            
            # Extract basic document info
            document_info = {
                'filename': Path(pdf_path).name,
                'filepath': pdf_path,
                'full_text': full_text,
                'pages': pages_content,
                'page_count': len(pages_content),
                'processed_at': datetime.now().isoformat(),
                'document_type': self._identify_document_type(full_text),
                'transactions': self._extract_transactions(full_text)
            }
            
            return document_info
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None
    
    def _identify_document_type(self, text: str) -> str:
        """Identify the type of financial document based on content."""
        text_lower = text.lower()
        
        # Common patterns for different document types
        if any(keyword in text_lower for keyword in ['bank statement', 'checking account', 'savings account']):
            return 'bank_statement'
        elif any(keyword in text_lower for keyword in ['credit card', 'statement of account']):
            return 'credit_card_statement'
        elif any(keyword in text_lower for keyword in ['invoice', 'bill', 'amount due']):
            return 'bill_invoice'
        elif any(keyword in text_lower for keyword in ['investment', 'portfolio', 'stocks', 'bonds']):
            return 'investment_statement'
        elif any(keyword in text_lower for keyword in ['receipt', 'purchase']):
            return 'receipt'
        else:
            return 'unknown'
    
    def _extract_transactions(self, text: str) -> List[Dict[str, Any]]:
        """Extract transaction information from document text."""
        transactions = []
        
        # Common transaction patterns - improved to avoid duplicates
        patterns = [
            # Date, Description, Amount patterns (most specific first)
            r'(\d{1,2}/\d{1,2}/\d{2,4})\s+([A-Za-z][A-Za-z\s\.]+?[A-Za-z])\s+([-$]?\d+\.\d{2})',
            r'(\d{1,2}-\d{1,2}-\d{2,4})\s+([A-Za-z][A-Za-z\s\.]+?[A-Za-z])\s+([-$]?\d+\.\d{2})',
            # More flexible patterns for different formats
            r'(\d{1,2}/\d{1,2}/\d{2,4})\s+([A-Z][A-Z\s]+)\s+([-+$]?\d+\.\d{2})',
        ]
        
        # Track processed transactions to avoid duplicates
        processed_transactions = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                date_str, description, amount_str = match
                
                try:
                    # Clean up description and amount
                    description = description.strip()
                    amount_str = amount_str.replace('$', '').replace(',', '').replace('+', '')
                    amount = float(amount_str)
                    
                    # Parse date
                    date_obj = None
                    for date_format in ['%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y']:
                        try:
                            date_obj = datetime.strptime(date_str, date_format)
                            break
                        except ValueError:
                            continue
                    
                    if date_obj is None:
                        continue  # Skip if date parsing fails
                    
                    # Create unique key to avoid duplicates
                    transaction_key = (date_obj.date(), description, amount)
                    if transaction_key in processed_transactions:
                        continue
                    
                    processed_transactions.add(transaction_key)
                    
                    transaction = {
                        'date': date_obj.isoformat(),
                        'description': description,
                        'amount': amount,
                        'type': 'debit' if amount < 0 else 'credit'
                    }
                    
                    transactions.append(transaction)
                    
                except (ValueError, TypeError):
                    continue  # Skip invalid transactions
        
        # Sort by date (duplicates already removed)
        transactions.sort(key=lambda x: x['date'], reverse=True)
        
        return transactions[:100]  # Limit to most recent 100 transactions
    
    def get_document_summary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of processed documents."""
        if not documents:
            return {'total_documents': 0}
        
        total_transactions = sum(len(doc.get('transactions', [])) for doc in documents)
        document_types = {}
        
        for doc in documents:
            doc_type = doc.get('document_type', 'unknown')
            document_types[doc_type] = document_types.get(doc_type, 0) + 1
        
        # Calculate date range
        all_transactions = []
        for doc in documents:
            all_transactions.extend(doc.get('transactions', []))
        
        date_range = None
        if all_transactions:
            dates = [t['date'] for t in all_transactions]
            date_range = {
                'earliest': min(dates),
                'latest': max(dates)
            }
        
        return {
            'total_documents': len(documents),
            'total_transactions': total_transactions,
            'document_types': document_types,
            'date_range': date_range,
            'processing_complete': True
        }