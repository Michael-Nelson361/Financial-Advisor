# Financial Advisor LLM Project - Claude Tracking

## Project Overview
A lightweight LLM-powered financial advisor that analyzes local PDF financial documents to provide spending insights and budget recommendations. Designed for personal use with complete privacy (local processing only).

## Technology Stack
- **Frontend:** Streamlit (interactive UI)
- **LLM:** Ollama with Llama 3.2 (3B model)
- **Document Processing:** PyMuPDF, LangChain
- **Vector Store:** FAISS (local embeddings)
- **Backend:** Python with RAG pipeline

## Project Structure
```
Financial-Advisor/
├── src/
│   ├── document_processor.py    # PDF parsing and text extraction
│   ├── llm_manager.py          # Ollama integration and LLM handling
│   ├── vector_store.py         # FAISS vector database management
│   ├── financial_analyzer.py   # Core financial analysis logic
│   └── chat_interface.py       # Chat conversation management
├── ui/
│   └── streamlit_app.py        # Main Streamlit application
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # Project documentation
```

## Key Features
1. **Document Upload:** Folder selection for PDF financial statements
2. **Local Processing:** Complete privacy with local LLM processing
3. **Chat Interface:** Interactive conversation with financial advisor
4. **Spending Analysis:** Automatic identification of spending patterns
5. **Budget Recommendations:** AI-generated budget and financial guidance

## Development Commands
- **Run Application:** `streamlit run ui/streamlit_app.py`
- **Install Dependencies:** `pip install -r requirements.txt`
- **Setup Ollama:** `ollama pull llama3.2:3b`

## Git Workflow
- Frequent commits with descriptive messages
- Feature branch development for major components
- Regular pushes to track progress

## Next Steps
1. Set up project structure and dependencies
2. Implement PDF document processing
3. Integrate Ollama LLM
4. Create Streamlit UI
5. Build financial analysis pipeline
6. Test and refine system