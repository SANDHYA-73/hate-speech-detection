"""
Fallback retrieval mechanism for when vector search fails.
"""
import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FallbackRetriever:
    """Simple keyword-based retriever as fallback when vector search fails."""
    
    def __init__(self, policy_docs_dir: str):
        """
        Initialize the FallbackRetriever.
        
        Args:
            policy_docs_dir: Directory containing policy documents
        """
        self.policy_docs_dir = policy_docs_dir
        self.policies = self._load_policies()
        
    def _load_policies(self) -> List[Dict[str, Any]]:
        """Load all policy documents from the directory."""
        policies = []
        
        try:
            if not os.path.exists(self.policy_docs_dir):
                logger.error(f"Policy directory not found: {self.policy_docs_dir}")
                return policies
            
            for filename in os.listdir(self.policy_docs_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.policy_docs_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        policies.append({
                            'content': content,
                            'source': filename,
                        })
                    except Exception as e:
                        logger.error(f"Error reading policy file {filename}: {e}")
            
            logger.info(f"Loaded {len(policies)} policy documents for fallback retrieval")
            return policies
            
        except Exception as e:
            logger.error(f"Error loading policy documents: {e}")
            return []
    
    def retrieve(self, query: str, classification: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant policies based on simple keyword matching.
        
        Args:
            query: The user query text
            classification: The classification label
            
        Returns:
            List of relevant policy documents
        """
        if not self.policies:
            logger.warning("No policies available for fallback retrieval")
            return []
        
        # Convert query and classification to lowercase for matching
        query_lower = query.lower()
        classification_lower = classification.lower()
        
        # Define keywords for each classification
        keywords = {
            'hate': ['hate', 'racial', 'religion', 'gender', 'ethnicity', 'discriminat', 'violen'],
            'toxic': ['toxic', 'harass', 'bully', 'harmful', 'abuse', 'threat'],
            'offensive': ['offensive', 'inappropri', 'hurt', 'slur', 'insult'],
            'ambiguous': ['ambiguous', 'unclear', 'context', 'intent', 'review'],
            'neutral': ['neutral', 'acceptable', 'allow']
        }
        
        # Get keywords for this classification
        classification_keywords = keywords.get(classification_lower, [])
        
        # Add additional keywords from the query
        additional_keywords = [word for word in query_lower.split() 
                              if len(word) > 3 and word not in ['this', 'that', 'with', 'from', 'what', 'when', 'where', 'which']]
        
        all_keywords = classification_keywords + additional_keywords
        
        # Score policies based on keyword matches
        scored_policies = []
        
        for policy in self.policies:
            content_lower = policy['content'].lower()
            score = 0
            
            for keyword in all_keywords:
                if keyword in content_lower:
                    score += content_lower.count(keyword)
            
            if score > 0:
                scored_policies.append({
                    'content': policy['content'],
                    'source': policy['source'],
                    'similarity': min(0.99, score / 10)  # Cap at 0.99, scale appropriately
                })
        
        # Sort by score and return top results
        scored_policies.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = scored_policies[:3]  # Return top 3
        
        # If no results with scores, return the first 2 policies as fallback
        if not top_results and self.policies:
            logger.info("No keyword matches found, returning default policies")
            return [
                {
                    'content': self.policies[0]['content'],
                    'source': self.policies[0]['source'],
                    'similarity': 0.65
                },
                {
                    'content': self.policies[1]['content'] if len(self.policies) > 1 else self.policies[0]['content'],
                    'source': self.policies[1]['source'] if len(self.policies) > 1 else self.policies[0]['source'],
                    'similarity': 0.6
                }
            ]
        
        logger.info(f"Fallback retrieval found {len(top_results)} matching policies")
        return top_results