import ollama
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    """Manages Ollama LLM interactions for financial analysis and advice."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """Initialize LLM manager with specified Ollama model."""
        self.model_name = model_name
        self.client = ollama.Client()
        self.conversation_history = []
        
        # Verify model availability
        self._ensure_model_available()
        
        logger.info(f"LLM Manager initialized with model: {model_name}")
    
    def _ensure_model_available(self) -> None:
        """Check if the specified model is available, pull if necessary."""
        try:
            # List available models
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model_name not in available_models:
                logger.info(f"Model {self.model_name} not found. Attempting to pull...")
                self.client.pull(self.model_name)
                logger.info(f"Successfully pulled model: {self.model_name}")
            else:
                logger.info(f"Model {self.model_name} is available")
                
        except Exception as e:
            logger.error(f"Error checking/pulling model: {str(e)}")
            raise RuntimeError(f"Could not ensure model availability: {str(e)}")
    
    def generate_response(self, prompt: str, context_data: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response using the LLM with optional context."""
        try:
            # Build the full prompt with context
            full_prompt = self._build_prompt_with_context(prompt, context_data)
            
            # Generate response using Ollama
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': self._get_system_prompt()
                    },
                    {
                        'role': 'user',
                        'content': full_prompt
                    }
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 1000
                }
            )
            
            assistant_response = response['message']['content']
            
            # Update conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user': prompt,
                'assistant': assistant_response,
                'context_items': len(context_data) if context_data else 0
            })
            
            # Keep only last 10 exchanges to prevent memory issues
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the financial advisor."""
        return """You are a knowledgeable and helpful personal financial advisor. Your role is to:

1. Analyze financial documents and transactions to identify spending patterns
2. Provide practical budgeting advice and recommendations
3. Help users understand their financial situation
4. Suggest ways to save money and improve financial health
5. Explain financial concepts in simple, accessible terms

Guidelines:
- Always be helpful, accurate, and supportive
- Base your advice on the provided financial data when available
- Provide specific, actionable recommendations
- Explain your reasoning clearly
- Be encouraging and positive while being realistic
- Never provide advice that could be harmful or illegal
- Remind users that this is general guidance, not professional financial advice

Remember: You are working with the user's personal financial data that they have provided locally. This data never leaves their computer, ensuring complete privacy."""
    
    def _build_prompt_with_context(self, user_prompt: str, context_data: Optional[List[Dict[str, Any]]]) -> str:
        """Build a comprehensive prompt including relevant context from documents."""
        if not context_data:
            return user_prompt
        
        context_parts = []
        
        # Add relevant document excerpts
        for i, item in enumerate(context_data[:5]):  # Limit to top 5 most relevant
            if item.get('chunk_type') == 'transaction':
                context_parts.append(f"Transaction {i+1}: {item['text']}")
            else:
                # Truncate long text chunks
                text = item['text'][:300] + "..." if len(item['text']) > 300 else item['text']
                context_parts.append(f"Document excerpt {i+1}: {text}")
        
        context_section = "\n".join(context_parts)
        
        full_prompt = f"""Based on the following financial information from the user's documents:

{context_section}

User Question: {user_prompt}

Please provide a helpful response based on the financial data above. If the data doesn't contain enough information to answer the question, let the user know and provide general guidance instead."""
        
        return full_prompt
    
    def analyze_spending_patterns(self, transactions: List[Dict[str, Any]]) -> str:
        """Analyze spending patterns from transaction data."""
        if not transactions:
            return "No transaction data available for analysis."
        
        # Prepare transaction summary for analysis
        prompt = f"""Please analyze the following financial transactions and provide insights about spending patterns:

Number of transactions: {len(transactions)}

Sample transactions:
"""
        
        # Add sample transactions (up to 20)
        for i, txn in enumerate(transactions[:20]):
            date = txn.get('date', 'Unknown')
            desc = txn.get('description', 'No description')[:50]
            amount = txn.get('amount', 0)
            txn_type = txn.get('type', 'unknown')
            
            prompt += f"{i+1}. {date}: {desc} - ${abs(amount):.2f} ({txn_type})\n"
        
        if len(transactions) > 20:
            prompt += f"\n... and {len(transactions) - 20} more transactions."
        
        prompt += """

Please provide:
1. Key spending categories identified
2. Spending patterns or trends
3. Areas where money could potentially be saved
4. Overall financial behavior observations
5. Recommendations for better financial management"""
        
        return self.generate_response(prompt)
    
    def create_budget_recommendations(self, financial_summary: Dict[str, Any]) -> str:
        """Create budget recommendations based on financial data."""
        prompt = f"""Based on the following financial summary, please create budget recommendations:

Financial Summary:
- Total transactions: {financial_summary.get('total_transactions', 0)}
- Total debits (spending): ${financial_summary.get('total_debit_amount', 0):.2f}
- Total credits (income): ${financial_summary.get('total_credit_amount', 0):.2f}
- Net amount: ${financial_summary.get('net_amount', 0):.2f}
- Average transaction: ${financial_summary.get('avg_transaction', 0):.2f}

Please provide:
1. A realistic monthly budget framework
2. Specific spending limits for major categories
3. Savings goals and strategies
4. Tips for improving financial health
5. Warning signs to watch for

Make the recommendations practical and achievable."""
        
        return self.generate_response(prompt)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.conversation_history
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def is_model_available(self) -> bool:
        """Check if the LLM model is available and responsive."""
        try:
            # Simple test query
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Hello, are you working?'
                    }
                ],
                options={'max_tokens': 50}
            )
            return True
        except Exception as e:
            logger.error(f"Model availability check failed: {str(e)}")
            return False