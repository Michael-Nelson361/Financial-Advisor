import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """Analyzes financial data and provides insights for budgeting and planning."""
    
    def __init__(self):
        """Initialize the financial analyzer."""
        self.spending_categories = {
            'food_dining': ['restaurant', 'pizza', 'mcdonald', 'starbucks', 'coffee', 'dining', 'food', 'grocery', 'supermarket'],
            'transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'parking', 'car', 'auto', 'transport'],
            'shopping': ['amazon', 'walmart', 'target', 'store', 'retail', 'shopping', 'purchase'],
            'entertainment': ['movie', 'netflix', 'spotify', 'entertainment', 'game', 'music', 'streaming'],
            'utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'cable', 'utility'],
            'healthcare': ['pharmacy', 'doctor', 'hospital', 'medical', 'health', 'insurance', 'dental'],
            'education': ['school', 'university', 'education', 'tuition', 'books', 'course'],
            'financial': ['bank', 'fee', 'interest', 'loan', 'credit', 'payment', 'transfer'],
            'personal_care': ['hair', 'salon', 'beauty', 'cosmetic', 'personal', 'care'],
            'travel': ['hotel', 'flight', 'airline', 'travel', 'vacation', 'booking']
        }
    
    def analyze_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive analysis of transaction data."""
        if not transactions:
            return {'error': 'No transactions to analyze'}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(transactions)
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Basic statistics
        analysis = {
            'overview': self._get_overview_stats(df),
            'spending_by_category': self._categorize_spending(df),
            'monthly_trends': self._analyze_monthly_trends(df),
            'top_merchants': self._get_top_merchants(df),
            'recurring_transactions': self._identify_recurring_transactions(df),
            'financial_health': self._assess_financial_health(df),
            'recommendations': self._generate_recommendations(df)
        }
        
        return analysis
    
    def _get_overview_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic overview statistics."""
        debits = df[df['amount'] < 0]['amount']
        credits = df[df['amount'] >= 0]['amount']
        
        return {
            'total_transactions': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'total_spent': abs(debits.sum()) if not debits.empty else 0,
            'total_income': credits.sum() if not credits.empty else 0,
            'net_amount': df['amount'].sum(),
            'avg_transaction_amount': df['amount'].mean(),
            'largest_expense': abs(debits.min()) if not debits.empty else 0,
            'largest_income': credits.max() if not credits.empty else 0,
            'spending_transactions': len(debits),
            'income_transactions': len(credits)
        }
    
    def _categorize_spending(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Categorize transactions by spending type."""
        spending_df = df[df['amount'] < 0].copy()
        
        if spending_df.empty:
            return {}
        
        category_totals = defaultdict(float)
        category_counts = defaultdict(int)
        uncategorized = []
        
        for _, row in spending_df.iterrows():
            description = str(row['description']).lower()
            amount = abs(row['amount'])
            categorized = False
            
            for category, keywords in self.spending_categories.items():
                if any(keyword in description for keyword in keywords):
                    category_totals[category] += amount
                    category_counts[category] += 1
                    categorized = True
                    break
            
            if not categorized:
                uncategorized.append({
                    'description': row['description'],
                    'amount': amount,
                    'date': row['date'].strftime('%Y-%m-%d')
                })
        
        # Sort by total amount
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'by_amount': {cat: round(amt, 2) for cat, amt in sorted_categories},
            'by_count': dict(category_counts),
            'uncategorized_count': len(uncategorized),
            'uncategorized_amount': sum(item['amount'] for item in uncategorized),
            'top_uncategorized': uncategorized[:10]  # Top 10 uncategorized transactions
        }
    
    def _analyze_monthly_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spending trends by month."""
        df['month'] = df['date'].dt.to_period('M')
        
        monthly_spending = df[df['amount'] < 0].groupby('month')['amount'].sum().abs()
        monthly_income = df[df['amount'] >= 0].groupby('month')['amount'].sum()
        
        trends = {}
        for period in monthly_spending.index:
            month_str = str(period)
            spending = monthly_spending.get(period, 0)
            income = monthly_income.get(period, 0)
            
            trends[month_str] = {
                'spending': round(spending, 2),
                'income': round(income, 2),
                'net': round(income - spending, 2)
            }
        
        return {
            'monthly_data': trends,
            'avg_monthly_spending': round(monthly_spending.mean(), 2) if not monthly_spending.empty else 0,
            'avg_monthly_income': round(monthly_income.mean(), 2) if not monthly_income.empty else 0,
            'spending_trend': self._calculate_trend(monthly_spending)
        }
    
    def _get_top_merchants(self, df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top merchants by spending amount."""
        spending_df = df[df['amount'] < 0].copy()
        
        if spending_df.empty:
            return []
        
        merchant_spending = spending_df.groupby('description')['amount'].agg(['sum', 'count']).abs()
        merchant_spending = merchant_spending.sort_values('sum', ascending=False)
        
        top_merchants = []
        for merchant, data in merchant_spending.head(limit).iterrows():
            top_merchants.append({
                'merchant': merchant,
                'total_spent': round(data['sum'], 2),
                'transaction_count': int(data['count']),
                'avg_per_transaction': round(data['sum'] / data['count'], 2)
            })
        
        return top_merchants
    
    def _identify_recurring_transactions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify potentially recurring transactions."""
        # Group by description and amount to find recurring patterns
        recurring = []
        
        for description in df['description'].unique():
            desc_transactions = df[df['description'] == description]
            
            if len(desc_transactions) >= 3:  # At least 3 occurrences
                amounts = desc_transactions['amount'].unique()
                
                # Check if amounts are consistent (same or very similar)
                if len(amounts) <= 2:  # Same amount or at most 2 different amounts
                    dates = sorted(desc_transactions['date'].tolist())
                    
                    # Calculate intervals between transactions
                    intervals = []
                    for i in range(1, len(dates)):
                        interval = (dates[i] - dates[i-1]).days
                        intervals.append(interval)
                    
                    # Check for regular intervals (monthly, weekly, etc.)
                    if intervals and (25 <= np.mean(intervals) <= 35 or  # Monthly
                                      6 <= np.mean(intervals) <= 8):     # Weekly
                        
                        recurring.append({
                            'description': description,
                            'frequency': len(desc_transactions),
                            'avg_amount': round(desc_transactions['amount'].mean(), 2),
                            'avg_interval_days': round(np.mean(intervals), 1),
                            'last_date': max(dates).strftime('%Y-%m-%d'),
                            'pattern_type': 'monthly' if np.mean(intervals) > 20 else 'weekly'
                        })
        
        # Sort by frequency and amount
        recurring.sort(key=lambda x: (x['frequency'], abs(x['avg_amount'])), reverse=True)
        
        return recurring[:10]  # Return top 10 recurring transactions
    
    def _assess_financial_health(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall financial health based on transaction patterns."""
        total_income = df[df['amount'] >= 0]['amount'].sum()
        total_spending = abs(df[df['amount'] < 0]['amount'].sum())
        
        savings_rate = ((total_income - total_spending) / total_income * 100) if total_income > 0 else 0
        
        # Calculate spending volatility
        monthly_spending = df[df['amount'] < 0].groupby(df['date'].dt.to_period('M'))['amount'].sum().abs()
        spending_volatility = monthly_spending.std() if len(monthly_spending) > 1 else 0
        
        health_score = 0
        health_factors = []
        
        # Savings rate assessment
        if savings_rate >= 20:
            health_score += 30
            health_factors.append("Excellent savings rate")
        elif savings_rate >= 10:
            health_score += 20
            health_factors.append("Good savings rate")
        elif savings_rate >= 0:
            health_score += 10
            health_factors.append("Positive cash flow")
        else:
            health_factors.append("Negative cash flow - spending exceeds income")
        
        # Spending consistency
        if spending_volatility < total_spending * 0.2:
            health_score += 25
            health_factors.append("Consistent spending patterns")
        elif spending_volatility < total_spending * 0.4:
            health_score += 15
            health_factors.append("Moderately consistent spending")
        else:
            health_factors.append("High spending volatility")
        
        # Transaction frequency (regular financial activity)
        days_covered = (df['date'].max() - df['date'].min()).days
        if days_covered > 0:
            transactions_per_day = len(df) / days_covered
            if transactions_per_day > 0.5:  # More than 1 transaction every 2 days
                health_score += 20
                health_factors.append("Active financial management")
        
        # Categorization completeness
        spending_df = df[df['amount'] < 0]
        if not spending_df.empty:
            categorized = 0
            for _, row in spending_df.iterrows():
                description = str(row['description']).lower()
                if any(any(keyword in description for keyword in keywords) 
                      for keywords in self.spending_categories.values()):
                    categorized += 1
            
            categorization_rate = categorized / len(spending_df)
            if categorization_rate > 0.8:
                health_score += 15
                health_factors.append("Well-categorized expenses")
        
        # Income diversity (multiple income sources)
        income_sources = len(df[df['amount'] > 0]['description'].unique())
        if income_sources > 1:
            health_score += 10
            health_factors.append("Multiple income sources")
        
        return {
            'health_score': min(health_score, 100),
            'savings_rate': round(savings_rate, 1),
            'spending_volatility': round(spending_volatility, 2),
            'health_factors': health_factors,
            'assessment': self._get_health_assessment(min(health_score, 100))
        }
    
    def _get_health_assessment(self, score: float) -> str:
        """Get text assessment based on health score."""
        if score >= 80:
            return "Excellent financial health"
        elif score >= 60:
            return "Good financial health"
        elif score >= 40:
            return "Fair financial health - room for improvement"
        elif score >= 20:
            return "Poor financial health - needs attention"
        else:
            return "Critical financial situation - immediate action needed"
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction for a time series."""
        if len(series) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(series))
        y = series.values
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            
            if abs(slope) < series.mean() * 0.01:  # Less than 1% change
                return "stable"
            elif slope > 0:
                return "increasing"
            else:
                return "decreasing"
        except:
            return "stable"
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable financial recommendations."""
        recommendations = []
        
        total_income = df[df['amount'] >= 0]['amount'].sum()
        total_spending = abs(df[df['amount'] < 0]['amount'].sum())
        
        # Income vs spending
        if total_spending > total_income:
            recommendations.append("🚨 Your spending exceeds your income. Consider reducing expenses or increasing income.")
        
        savings_rate = ((total_income - total_spending) / total_income * 100) if total_income > 0 else 0
        
        if savings_rate < 10:
            recommendations.append("💰 Aim to save at least 10-20% of your income for emergency fund and future goals.")
        
        # Category-specific recommendations
        categories = self._categorize_spending(df)
        if categories and 'by_amount' in categories:
            top_category = max(categories['by_amount'].items(), key=lambda x: x[1])
            if top_category[1] > total_spending * 0.3:
                recommendations.append(f"📊 {top_category[0].replace('_', ' ').title()} is your largest expense category (${top_category[1]:.2f}). Look for ways to optimize this spending.")
        
        # Uncategorized transactions
        if categories and categories.get('uncategorized_count', 0) > len(df) * 0.2:
            recommendations.append("🏷️ Many transactions are uncategorized. Better categorization will help with budgeting.")
        
        # Recurring transactions
        monthly_spending = df[df['amount'] < 0].groupby(df['date'].dt.to_period('M'))['amount'].sum().abs()
        if len(monthly_spending) > 1 and monthly_spending.std() > monthly_spending.mean() * 0.3:
            recommendations.append("📈 Your spending varies significantly month-to-month. Consider creating a monthly budget for more consistency.")
        
        # Large transactions
        large_transactions = df[abs(df['amount']) > abs(df['amount']).quantile(0.95)]
        if not large_transactions.empty:
            recommendations.append(f"💸 Review your largest transactions - they significantly impact your budget.")
        
        return recommendations[:8]  # Return top 8 recommendations