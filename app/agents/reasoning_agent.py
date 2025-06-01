"""
PolicyReasoningAgent: Responsible for combining classification and retrieved
policy documents to provide a detailed justification for the decision.
"""
import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import GEMINI_API_KEY, TEMPERATURE, MAX_TOKENS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyReasoningAgent:
    """Agent to provide policy-based reasoning for hate speech classification."""
    
    def __init__(self):
        """Initialize the PolicyReasoningAgent."""
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
        You are a content moderation assistant that provides clear explanations for content classification decisions.
        
        INPUT TEXT: "{text}"
        
        CLASSIFICATION: {classification}
        
        BRIEF EXPLANATION: {brief_explanation}
        
        RELEVANT POLICIES:
        {policy_documents}
        
        Based on the classification and the relevant policies provided, explain in detail why the input text was classified as "{classification}".
        
        Your explanation should:
        1. Reference specific parts of the input text that led to this classification
        2. Cite specific policies or guidelines that apply to this content
        3. Be objective and educational in tone
        4. Be 50-100 words in length
        
        Provide your detailed reasoning:
        """
        
    def generate_reasoning(
        self, 
        text: str, 
        classification: str, 
        brief_explanation: str, 
        policy_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Generate detailed reasoning based on classification and policy documents.
        
        Args:
            text: The original input text
            classification: The classification label
            brief_explanation: Brief explanation from the detection agent
            policy_docs: List of relevant policy documents
            
        Returns:
            Detailed reasoning for the classification
        """
        try:
            logger.info(f"Generating detailed reasoning for classification: {classification}")
            
            # Check if we have a model available
            if not self.model:
                return self._fallback_reasoning(text, classification, brief_explanation, policy_docs)
                
            # Format the policy documents
            formatted_policies = ""
            for i, doc in enumerate(policy_docs, 1):
                formatted_policies += f"POLICY {i} (from {doc['source']}):\n{doc['content']}\n\n"
            
            if not formatted_policies:
                formatted_policies = "No specific policy documents were found for this content."
            
            # Create prompt with all information
            prompt = self.prompt_template.format(
                text=text,
                classification=classification,
                brief_explanation=brief_explanation,
                policy_documents=formatted_policies
            )
            
            try:
                # Get response from Gemini
                response = self.model.invoke(prompt)
                
                # Extract content correctly from AIMessage or other response type
                if hasattr(response, 'content'):
                    return response.content.strip()
                elif isinstance(response, str):
                    return response.strip()
                else:
                    return str(response).strip()
                    
            except Exception as e:
                logger.error(f"Error calling Gemini API for reasoning: {e}")
                return self._fallback_reasoning(text, classification, brief_explanation, policy_docs)
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Unable to generate detailed reasoning due to an error."
            
    def _fallback_reasoning(
        self,
        text: str,
        classification: str,
        brief_explanation: str,
        policy_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a fallback reasoning when the LLM is unavailable.
        
        Args:
            text: The original input text
            classification: The classification label
            brief_explanation: Brief explanation from the detection agent
            policy_docs: List of relevant policy documents
            
        Returns:
            Fallback reasoning text
        """
        # Create a template-based reasoning that doesn't require API calls
        reasoning_templates = {
            "Hate": "The content was classified as hate speech because it appears to target individuals or groups based on protected characteristics. Such content violates standard content policies that prohibit the promotion of hatred or violence against people based on attributes like race, ethnicity, gender, religion, or sexual orientation.",
            
            "Toxic": "The content was classified as toxic because it contains language that is likely to make others feel unwelcome or uncomfortable. While not necessarily hate speech, toxic content can be harmful to online communities and discussions.",
            
            "Offensive": "The content was classified as offensive because it contains language that may be inappropriate or hurtful, though it doesn't reach the threshold of hate speech. Offensive content often violates community standards for respectful communication.",
            
            "Neutral": "The content was classified as neutral because it doesn't contain language or themes that would be considered hateful, toxic, or offensive. The content appears to be within acceptable boundaries for online communication.",
            
            "Ambiguous": "The content was classified as ambiguous because its intent is unclear or it could be interpreted in multiple ways. Without additional context, it's difficult to determine whether this content violates community guidelines."
        }
        
        base_reasoning = reasoning_templates.get(classification, reasoning_templates["Ambiguous"])
        
        # Add policy reference if available
        policy_references = ""
        if policy_docs:
            policy_references = "\n\nRelevant policies include:"
            for i, doc in enumerate(policy_docs[:2], 1):  # Limit to 2 policies
                source = doc.get('source', 'Unknown policy')
                policy_references += f"\n- {source}"
        
        # Add the brief explanation
        if brief_explanation and "Error" not in brief_explanation:
            explanation_text = f"\n\nSpecific reason: {brief_explanation}"
        else:
            explanation_text = ""
            
        # Combine all parts
        full_reasoning = base_reasoning + policy_references + explanation_text
        
        return full_reasoning