"""
HateSpeechDetectionAgent: Responsible for classifying input text as
Hate, Toxic, Offensive, Neutral, or Ambiguous using Gemini API.
"""
import logging
from typing import Dict, Any, Tuple
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import GEMINI_API_KEY, CLASSIFICATION_LABELS, TEMPERATURE, MAX_TOKENS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Set Gemini API version to stable instead of beta
try:
    import google.api_core.client_options as client_options
    client_options.ClientOptions(api_endpoint="generativelanguage.googleapis.com")
except (ImportError, AttributeError):
    pass  # If this fails, we'll use the default endpoint

class HateSpeechDetectionAgent:
    """Agent to detect and classify potential hate speech."""
    
    def __init__(self):
        """Initialize the HateSpeechDetectionAgent."""
        try:
            # Try to use a model with potentially higher quota
            self.model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # Using flash model which typically has higher quotas
                google_api_key=GEMINI_API_KEY,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize primary model: {e}")
            try:
                # Fallback to a different model
                self.model = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-latest",
                    google_api_key=GEMINI_API_KEY,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                )
            except Exception as e:
                logger.error(f"Failed to initialize fallback model: {e}")
                self.model = None
        
        self.prompt_template = """
        You are a hate speech detection system. Your task is to analyze the following text and classify it into one of these categories:
        - Hate: Content that expresses, incites, or promotes hatred based on protected characteristics
        - Toxic: Content that is rude, disrespectful, or unreasonable that can make someone want to leave a discussion
        - Offensive: Content that may be considered inappropriate or hurtful but doesn't reach the level of hate speech
        - Neutral: Content that doesn't contain hate speech, toxic, or offensive elements
        - Ambiguous: Content where the intent is unclear or could be interpreted in multiple ways

        Text to analyze: "{text}"

        Provide your classification in a JSON format with two fields:
        - "classification": One of [Hate, Toxic, Offensive, Neutral, Ambiguous]
        - "brief_explanation": A short explanation (max 50 words) of why you classified it this way
        
        Only respond with the JSON object, nothing else.
        """
        
    def detect(self, text: str) -> Tuple[str, str]:
        """
        Detect and classify the input text.
        
        Args:
            text: The input text to classify
            
        Returns:
            Tuple containing (classification_label, brief_explanation)
        """
        try:
            logger.info(f"Analyzing text for hate speech detection")
            
            # LLM for classification
            if self.model:
                try:
                    prompt = self.prompt_template.format(text=text)
                    
                    # Get response from Gemini
                    response = self.model.invoke(prompt)
                    
                    # Parse the JSON response
                    import json
                    try:
                        # Extract content from AIMessage object
                        response_text = response.content if hasattr(response, 'content') else str(response)
                        
                        # Clean up markdown code blocks if present
                        if "```json" in response_text:
                            response_text = response_text.replace("```json", "").replace("```", "").strip()
                        
                        result = json.loads(response_text)
                        classification = result.get("classification", "Ambiguous")
                        explanation = result.get("brief_explanation", "No explanation provided")
                        
                        # Validate the classification is one of the allowed labels
                        if classification not in CLASSIFICATION_LABELS:
                            logger.warning(f"Got invalid classification: {classification}. Defaulting to 'Ambiguous'")
                            classification = "Ambiguous"
                            
                        return classification, explanation
                        
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON response: {response}")
                        
                except Exception as e:
                    logger.error(f"Error calling Gemini API: {e}")
                    # Continue to fallback
            
            # If we reach here, either the LLM call failed or we couldn't parse the result
            # Use a simple rule-based fallback approach
            return self._rule_based_fallback(text)
                
        except Exception as e:
            logger.error(f"Error in hate speech detection: {e}")
            return "Ambiguous", f"Error: {str(e)}"
            
    def _rule_based_fallback(self, text: str) -> Tuple[str, str]:
        """
        A simple rule-based fallback classifier when the API is unavailable.
        
        Args:
            text: The text to classify
            
        Returns:
            Tuple of (classification, explanation)
        """
        text = text.lower()
        
        # List of common offensive or toxic terms - this is extremely simplified
        hate_terms = ["hate", "kill", "murder", "die", "death", "exterminate", "eliminate"]
        offensive_terms = ["stupid", "idiot", "dumb", "fool", "shut up"]
        
        # Very basic rule-based classification
        for term in hate_terms:
            if term in text:
                return "Toxic", "The text contains potentially toxic language."
        
        for term in offensive_terms:
            if term in text:
                return "Offensive", "The text contains potentially offensive language."
        
        # Use character count to determine if neutral or ambiguous
        # Longer texts without obvious hate/offensive terms are more likely to be neutral
        if len(text) > 100:
            return "Neutral", "The text does not contain obvious toxic or offensive language."
        else:
            return "Ambiguous", "Unable to confidently classify this text without context."