"""
Text preprocessing utilities for cleaning and normalizing input text.
"""
import re
import logging
import unicodedata
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocess and normalize text input."""
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess and clean text input.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
            
        try:
            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Replace multiple whitespace with single space
            text = re.sub(r'\s+', ' ', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            # Return original text if preprocessing fails
            return text.strip() if text else ""
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1024) -> str:
        """
        Truncate text to a maximum length while preserving whole sentences.
        
        Args:
            text: Text to truncate
            max_length: Maximum length in characters
            
        Returns:
            Truncated text
        """
        if not text or len(text) <= max_length:
            return text
            
        try:
            # Find the last sentence boundary before max_length
            truncated = text[:max_length]
            
            # Try to end at sentence boundary (., !, ?)
            sentence_end = max(
                truncated.rfind('.'), 
                truncated.rfind('!'), 
                truncated.rfind('?')
            )
            
            if sentence_end > max_length * 0.5:  # If we found a sentence end in latter half
                truncated = text[:sentence_end + 1]
            else:
                # Otherwise try to end at a word boundary
                last_space = truncated.rfind(' ')
                if last_space > 0:
                    truncated = text[:last_space]
            
            return truncated
            
        except Exception as e:
            logger.error(f"Error truncating text: {e}")
            # Simple truncation if error occurs
            return text[:max_length]
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> list:
        """
        Extract important keywords from text for better retrieval.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        try:
            # Remove common punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Convert to lowercase and split into words
            words = text.lower().split()
            
            # Remove common stop words (simplified list)
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                          'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                          'from', 'of', 'as', 'this', 'that', 'these', 'those', 'it', 'its'}
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Count frequency
            word_counts = {}
            for word in filtered_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Sort by frequency and get top keywords
            keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return [word for word, count in keywords[:max_keywords]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []