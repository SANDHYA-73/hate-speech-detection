"""
ActionRecommenderAgent: Responsible for recommending moderation actions
based on the classification and additional factors.
"""
import logging
from typing import Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import GEMINI_API_KEY, ACTION_MAPPING, TEMPERATURE, MAX_TOKENS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionRecommenderAgent:
    """Agent to recommend appropriate moderation actions."""
    
    def __init__(self):
        """Initialize the ActionRecommenderAgent."""
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
                
        self.action_mapping = ACTION_MAPPING
        
        self.prompt_template = """
        You are a content moderation action advisor. Your task is to recommend a specific moderation action
        for the following content that has been classified as "{classification}".
        
        INPUT TEXT: "{text}"
        
        DETAILED REASONING: {detailed_reasoning}
        
        Based on the classification and the standard actions for this category:
        - {action_mapping}
        
        Please provide a specific moderation action recommendation and a brief justification.
        Your response should be in JSON format with two fields:
        - "action": The specific moderation action to take
        - "justification": A brief justification for this action (max 50 words)
        
        If there are special circumstances that might warrant deviating from the standard action for this classification, explain them.
        
        Only respond with the JSON object, nothing else.
        """
        
    def recommend_action(
        self, 
        text: str, 
        classification: str, 
        detailed_reasoning: str
    ) -> Tuple[str, str]:
        """
        Recommend a moderation action based on classification and reasoning.
        
        Args:
            text: The original input text
            classification: The classification label
            detailed_reasoning: Detailed reasoning from the PolicyReasoningAgent
            
        Returns:
            Tuple containing (recommended_action, justification)
        """
        try:
            logger.info(f"Recommending action for classification: {classification}")
            
            # Get standard action for this classification
            standard_action = self.action_mapping.get(classification, "Flag for human review")
            
            # Special case for short content, no LLM needed, or when LLM is unavailable
            if len(text.strip()) < 10 or classification == "Neutral" or not self.model:
                return standard_action, f"Standard action for {classification} content."
            
            # Format action mapping for prompt
            action_mapping_str = ""
            for label, action in self.action_mapping.items():
                action_mapping_str += f"{label}: {action}\n"
            
            # Create prompt
            prompt = self.prompt_template.format(
                text=text,
                classification=classification,
                detailed_reasoning=detailed_reasoning,
                action_mapping=action_mapping_str
            )
            
            try:
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
                    action = result.get("action", standard_action)
                    justification = result.get("justification", f"Standard action for {classification} content.")
                    
                    return action, justification
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {response}")
                    return standard_action, f"Standard action for {classification} content."
            except Exception as e:
                logger.error(f"Error calling Gemini API for action recommendation: {e}")
                return self._get_fallback_action(classification)
                
        except Exception as e:
            logger.error(f"Error recommending action: {e}")
            return "Flag for human review", f"Error in processing: {str(e)}"
            
    def _get_fallback_action(self, classification: str) -> Tuple[str, str]:
        """
        Get a fallback action recommendation based on classification.
        
        Args:
            classification: The classification label
            
        Returns:
            Tuple of (action, justification)
        """
        # Simply use the predefined mapping
        action = self.action_mapping.get(classification, "Flag for human review")
        justification = f"Standard action for {classification} content based on platform policies."
        
        return action, justification