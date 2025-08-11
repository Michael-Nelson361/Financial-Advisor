from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatInterface:
    """Manages chat conversations and coordinates between components."""
    
    def __init__(self, llm_manager, vector_store, financial_analyzer):
        """Initialize chat interface with required components."""
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.financial_analyzer = financial_analyzer
        self.session_data = {}
        
    def process_message(self, user_message: str, search_context: bool = True) -> Dict[str, Any]:
        """Process user message and generate appropriate response."""
        try:
            # Determine query type
            query_type = self._classify_query(user_message)
            
            # Get relevant context if requested
            context_data = []
            if search_context and query_type != 'general':
                context_data = self.vector_store.search(user_message, k=5)
            
            # Handle different types of queries
            if query_type == 'spending_analysis':
                return self._handle_spending_analysis(user_message, context_data)
            elif query_type == 'budget_help':
                return self._handle_budget_help(user_message, context_data)
            elif query_type == 'transaction_search':
                return self._handle_transaction_search(user_message, context_data)
            elif query_type == 'financial_summary':
                return self._handle_financial_summary(user_message, context_data)
            else:
                return self._handle_general_query(user_message, context_data)
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error while processing your request. Please try again.",
                'query_type': 'error',
                'context_used': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def _classify_query(self, message: str) -> str:
        """Classify the type of user query."""
        message_lower = message.lower()
        
        # Spending analysis keywords
        if any(keyword in message_lower for keyword in [
            'spending', 'spend', 'expenses', 'money on', 'cost', 'purchase', 'bought'
        ]):
            return 'spending_analysis'
        
        # Budget-related keywords
        if any(keyword in message_lower for keyword in [
            'budget', 'save', 'saving', 'cut costs', 'reduce', 'afford', 'plan'
        ]):
            return 'budget_help'
        
        # Transaction search keywords
        if any(keyword in message_lower for keyword in [
            'transaction', 'payment', 'charge', 'find', 'search', 'when did', 'show me'
        ]):
            return 'transaction_search'
        
        # Financial summary keywords
        if any(keyword in message_lower for keyword in [
            'summary', 'overview', 'total', 'how much', 'income', 'balance', 'net'
        ]):
            return 'financial_summary'
        
        return 'general'
    
    def _handle_spending_analysis(self, message: str, context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle spending analysis queries."""
        # Get all transactions for comprehensive analysis
        all_transactions = []
        for chunk in self.vector_store.chunk_metadata:
            if chunk.get('chunk_type') == 'transaction' and 'transaction_data' in chunk:
                all_transactions.append(chunk['transaction_data'])
        
        # Generate LLM response with context
        llm_response = self.llm_manager.generate_response(message, context_data)
        
        # Add analytical insights if we have enough data
        analysis_insights = []
        if len(all_transactions) > 10:
            try:
                analysis = self.financial_analyzer.analyze_transactions(all_transactions)
                
                # Extract key insights
                if 'spending_by_category' in analysis and 'by_amount' in analysis['spending_by_category']:
                    top_categories = list(analysis['spending_by_category']['by_amount'].items())[:3]
                    category_list = [f"{cat.replace('_', ' ').title()} (${amt:.2f})" for cat, amt in top_categories]
                    analysis_insights.append(f"Your top spending categories are: {', '.join(category_list)}")
                
                if 'overview' in analysis:
                    total_spent = analysis['overview'].get('total_spent', 0)
                    analysis_insights.append(f"Total spending analyzed: ${total_spent:.2f}")
                
            except Exception as e:
                logger.error(f"Error in spending analysis: {str(e)}")
        
        return {
            'response': llm_response,
            'query_type': 'spending_analysis',
            'context_used': len(context_data) > 0,
            'analysis_insights': analysis_insights,
            'relevant_transactions': len([c for c in context_data if c.get('chunk_type') == 'transaction']),
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_budget_help(self, message: str, context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle budget-related queries."""
        # Get financial summary for budget recommendations
        transaction_summary = self.vector_store.get_transaction_summary()
        
        # Generate budget-focused response
        llm_response = self.llm_manager.generate_response(message, context_data)
        
        # Add budget recommendations if we have enough data
        budget_tips = []
        if transaction_summary.get('total_transactions', 0) > 0:
            try:
                # Generate specific budget recommendations
                budget_response = self.llm_manager.create_budget_recommendations(transaction_summary)
                budget_tips.append("Based on your transaction data:")
                budget_tips.append(budget_response)
                
            except Exception as e:
                logger.error(f"Error generating budget recommendations: {str(e)}")
        
        return {
            'response': llm_response,
            'query_type': 'budget_help',
            'context_used': len(context_data) > 0,
            'budget_recommendations': budget_tips,
            'financial_summary': transaction_summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_transaction_search(self, message: str, context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle transaction search queries."""
        # Filter context to transaction data
        transaction_results = [item for item in context_data if item.get('chunk_type') == 'transaction']
        
        llm_response = self.llm_manager.generate_response(message, transaction_results)
        
        # Format transaction results for display
        formatted_transactions = []
        for item in transaction_results[:10]:  # Limit to 10 results
            if 'transaction_data' in item:
                txn = item['transaction_data']
                formatted_transactions.append({
                    'date': txn.get('date', 'Unknown'),
                    'description': txn.get('description', 'No description'),
                    'amount': txn.get('amount', 0),
                    'type': txn.get('type', 'unknown')
                })
        
        return {
            'response': llm_response,
            'query_type': 'transaction_search',
            'context_used': len(context_data) > 0,
            'matching_transactions': formatted_transactions,
            'total_matches': len(transaction_results),
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_financial_summary(self, message: str, context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle financial summary queries."""
        # Get comprehensive financial summary
        transaction_summary = self.vector_store.get_transaction_summary()
        
        # Get all transactions for detailed analysis
        all_transactions = []
        for chunk in self.vector_store.chunk_metadata:
            if chunk.get('chunk_type') == 'transaction' and 'transaction_data' in chunk:
                all_transactions.append(chunk['transaction_data'])
        
        # Generate analytical summary
        detailed_analysis = {}
        if len(all_transactions) > 5:
            try:
                detailed_analysis = self.financial_analyzer.analyze_transactions(all_transactions)
            except Exception as e:
                logger.error(f"Error in detailed analysis: {str(e)}")
        
        # Generate LLM response with full context
        llm_response = self.llm_manager.generate_response(message, context_data)
        
        return {
            'response': llm_response,
            'query_type': 'financial_summary',
            'context_used': len(context_data) > 0,
            'transaction_summary': transaction_summary,
            'detailed_analysis': detailed_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_general_query(self, message: str, context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle general financial queries."""
        llm_response = self.llm_manager.generate_response(message, context_data)
        
        return {
            'response': llm_response,
            'query_type': 'general',
            'context_used': len(context_data) > 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions based on available data."""
        suggestions = [
            "What are my biggest spending categories?",
            "Help me create a monthly budget",
            "Where can I cut costs to save money?",
            "Show me my recent transactions",
            "What's my financial summary?",
            "Analyze my spending patterns",
            "What recurring payments do I have?",
            "How much did I spend on dining last month?"
        ]
        
        # Customize suggestions based on available data
        transaction_summary = self.vector_store.get_transaction_summary()
        
        if transaction_summary.get('total_transactions', 0) > 50:
            suggestions.append("What are my spending trends over time?")
        
        if transaction_summary.get('total_debits', 0) > 20:
            suggestions.append("Which merchants do I spend the most money with?")
        
        return suggestions[:6]  # Return top 6 suggestions
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation session."""
        history = self.llm_manager.get_conversation_history()
        
        return {
            'total_messages': len(history),
            'session_start': history[0]['timestamp'] if history else None,
            'last_activity': history[-1]['timestamp'] if history else None,
            'query_types_used': list(set(entry.get('query_type', 'unknown') for entry in history)),
            'documents_loaded': len(self.vector_store.documents),
            'transactions_available': self.vector_store.get_transaction_summary().get('total_transactions', 0)
        }
    
    def clear_session(self) -> None:
        """Clear current chat session."""
        self.llm_manager.clear_conversation()
        self.session_data.clear()
        logger.info("Chat session cleared")