import streamlit as st
import os
import sys
from pathlib import Path
import time
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_manager import LLMManager
from financial_analyzer import FinancialAnalyzer
from chat_interface import ChatInterface

# Page configuration
st.set_page_config(
    page_title="Financial Advisor LLM",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .status-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if 'financial_analyzer' not in st.session_state:
        st.session_state.financial_analyzer = FinancialAnalyzer()
    
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = None
    
    if 'chat_interface' not in st.session_state:
        st.session_state.chat_interface = None
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None

# Initialize LLM Manager
def init_llm_manager():
    """Initialize the LLM manager with error handling."""
    try:
        if st.session_state.llm_manager is None:
            with st.spinner("Initializing AI model... This may take a moment."):
                st.session_state.llm_manager = LLMManager()
                
                # Test if model is responsive
                if not st.session_state.llm_manager.is_model_available():
                    st.error("❌ LLM model is not responding. Please ensure Ollama is running and the llama3.2:3b model is installed.")
                    st.info("To install the model, run: `ollama pull llama3.2:3b`")
                    return False
                
                st.session_state.chat_interface = ChatInterface(
                    st.session_state.llm_manager,
                    st.session_state.vector_store,
                    st.session_state.financial_analyzer
                )
                
                st.success("✅ AI model initialized successfully!")
                return True
    except Exception as e:
        st.error(f"❌ Error initializing AI model: {str(e)}")
        st.info("Please ensure Ollama is installed and running: https://ollama.ai")
        return False
    
    return True

# Document processing functions
def process_documents(folder_path):
    """Process documents from the selected folder."""
    try:
        with st.spinner("Processing PDF documents..."):
            # Process documents
            documents = st.session_state.document_processor.process_folder(folder_path)
            
            if not documents:
                st.warning("No valid PDF documents found in the selected folder.")
                return False
            
            # Add to vector store
            st.session_state.vector_store.add_documents(documents)
            
            # Update session state
            st.session_state.documents_loaded = True
            st.session_state.processing_status = f"Successfully processed {len(documents)} documents"
            
            return True
            
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        st.session_state.processing_status = f"Error: {str(e)}"
        return False

# UI Components
def render_sidebar():
    """Render the sidebar with document upload and status."""
    st.sidebar.markdown("## 📁 Document Management")
    
    # Folder selection
    folder_path = st.sidebar.text_input(
        "Enter folder path containing PDF financial documents:",
        placeholder="/path/to/your/financial/documents",
        help="Enter the full path to a folder containing your financial PDF files"
    )
    
    if st.sidebar.button("📂 Browse Folder", disabled=True):
        st.sidebar.info("File browser not available. Please enter folder path manually.")
    
    if st.sidebar.button("🔄 Process Documents") and folder_path:
        if os.path.exists(folder_path):
            if process_documents(folder_path):
                st.sidebar.success("Documents processed successfully!")
                st.rerun()
        else:
            st.sidebar.error("Folder path does not exist.")
    
    # Status information
    st.sidebar.markdown("## 📊 Status")
    
    if st.session_state.documents_loaded:
        summary = st.session_state.vector_store.get_transaction_summary()
        
        st.sidebar.markdown(f"""
        **Documents:** {len(st.session_state.vector_store.documents)}  
        **Transactions:** {summary.get('total_transactions', 0)}  
        **Status:** ✅ Ready for analysis
        """)
    else:
        st.sidebar.markdown("**Status:** 📄 No documents loaded")
    
    # Model status
    st.sidebar.markdown("## 🤖 AI Model")
    if st.session_state.llm_manager is not None:
        st.sidebar.markdown("**Status:** ✅ Model ready")
    else:
        st.sidebar.markdown("**Status:** ⏳ Initializing...")
    
    # Clear data button
    if st.sidebar.button("🗑️ Clear All Data"):
        st.session_state.vector_store.clear()
        st.session_state.documents_loaded = False
        st.session_state.chat_history = []
        if st.session_state.chat_interface:
            st.session_state.chat_interface.clear_session()
        st.sidebar.success("All data cleared!")
        st.rerun()

def render_main_interface():
    """Render the main chat interface."""
    st.markdown('<h1 class="main-header">💰 Financial Advisor LLM</h1>', unsafe_allow_html=True)
    
    if not st.session_state.documents_loaded:
        st.markdown("""
        <div class="status-box status-info">
            <h3>👋 Welcome to Your Personal Financial Advisor!</h3>
            <p>To get started:</p>
            <ol>
                <li>Enter the path to a folder containing your financial PDF documents in the sidebar</li>
                <li>Click "Process Documents" to analyze your financial data</li>
                <li>Start chatting with your AI financial advisor!</li>
            </ol>
            <p><strong>Privacy Note:</strong> All processing happens locally on your computer. Your financial data never leaves your device.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize LLM if not already done
    if st.session_state.llm_manager is None:
        if not init_llm_manager():
            return
    
    # Chat interface
    st.markdown("## 💬 Chat with Your Financial Advisor")
    
    # Display chat history
    for message in st.session_state.chat_history:
        render_chat_message(message)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask your financial advisor:",
                placeholder="e.g., What are my biggest spending categories?",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.form_submit_button("Send 💬")
        
        if submitted and user_input:
            process_chat_message(user_input)
    
    # Suggested questions
    if not st.session_state.chat_history:
        render_suggested_questions()

def render_chat_message(message):
    """Render a single chat message."""
    if message['type'] == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
    
    else:  # assistant message
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Financial Advisor:</strong> {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show additional information if available
        if 'metadata' in message and message['metadata']:
            render_message_metadata(message['metadata'])

def render_message_metadata(metadata):
    """Render additional information from chat response."""
    if metadata.get('analysis_insights'):
        with st.expander("📊 Analysis Insights"):
            for insight in metadata['analysis_insights']:
                st.write(f"• {insight}")
    
    if metadata.get('matching_transactions'):
        with st.expander(f"💳 Found {len(metadata['matching_transactions'])} Matching Transactions"):
            for txn in metadata['matching_transactions'][:5]:  # Show top 5
                st.write(f"**{txn['date']}**: {txn['description']} - ${abs(txn['amount']):.2f}")
    
    if metadata.get('financial_summary'):
        summary = metadata['financial_summary']
        if summary.get('total_transactions', 0) > 0:
            with st.expander("💰 Financial Summary"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", summary.get('total_transactions', 0))
                with col2:
                    st.metric("Total Spending", f"${summary.get('total_debit_amount', 0):.2f}")
                with col3:
                    st.metric("Net Amount", f"${summary.get('net_amount', 0):.2f}")

def render_suggested_questions():
    """Render suggested questions for users."""
    if st.session_state.chat_interface:
        suggestions = st.session_state.chat_interface.get_suggested_questions()
        
        st.markdown("### 💡 Suggested Questions:")
        
        cols = st.columns(2)
        for i, question in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(question, key=f"suggestion_{i}"):
                    process_chat_message(question)

def process_chat_message(user_input):
    """Process user chat message and generate response."""
    # Add user message to history
    st.session_state.chat_history.append({
        'type': 'user',
        'content': user_input,
        'timestamp': time.time()
    })
    
    # Generate response
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.chat_interface.process_message(user_input)
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'type': 'assistant',
                'content': response['response'],
                'metadata': response,
                'timestamp': time.time()
            })
            
        except Exception as e:
            st.session_state.chat_history.append({
                'type': 'assistant',
                'content': f"I apologize, but I encountered an error: {str(e)}",
                'timestamp': time.time()
            })
    
    # Rerun to show new messages
    st.rerun()

# Main application
def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_interface()

if __name__ == "__main__":
    main()