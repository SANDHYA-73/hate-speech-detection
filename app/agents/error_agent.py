"""
ErrorHandlerAgent: Responsible for handling errors gracefully
and providing informative error messages.
"""
import logging
import traceback
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Node Execution → Exception Occurs → Error Agent Processes → State Updated with Error → Conditional Edge Routes to END
'''[Input] → [Preprocess] → [Classify] → [Retrieve] → [Reason] → [Recommend] → [End]
              ↓              ↓           ↓          ↓          ↓
           [Error?]      [Error?]    [Error?]   [Error?]   [Error?]
              ↓              ↓           ↓          ↓          ↓
            [End]         [End]       [Reason]   [Recommend] [End]'''
            
class ErrorHandlerAgent:
    """Agent to handle errors gracefully throughout the application."""
    
    def __init__(self):
        """Initialize the ErrorHandlerAgent."""
        self.error_messages = {
            "detection": "There was an issue analyzing your content. Please try again with different text.",
            "retrieval": "We couldn't retrieve relevant policy information. The system will continue with limited context.",
            "reasoning": "We couldn't generate a detailed explanation for this classification.",
            "action": "We couldn't determine the best action to take. A human moderator should review this content.",
            "audio": "There was an issue processing your audio file. Please ensure it's a supported format or try text input instead.",
            "export": "There was an issue exporting your results. Please try again later.",
            "general": "An unexpected error occurred. Please try again or contact support if the issue persists."
        }
    
    def handle_error(
        self, 
        error: Exception, 
        component: str = "general", 
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an error and return a user-friendly response.
        
        Args:
            error: The exception that was raised
            component: The component where the error occurred
            details: Additional details about the error context
            
        Returns:
            Dictionary with error information
        """
        # Log the error with full traceback
        logger.error(f"Error in {component}: {str(error)}")
        logger.debug(traceback.format_exc())
        
        # Get the user-friendly message for this component
        user_message = self.error_messages.get(component, self.error_messages["general"])
        
        # Prepare error details for internal use (not shown to users)
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "component": component,
            "traceback": traceback.format_exc(),
        }
        
        if details:
            error_details["context"] = details
        
        # Return user-friendly response
        return {
            "success": False,
            "message": user_message,
            "component": component,
            "error_details": error_details if logging.getLogger().level <= logging.DEBUG else None,
        }
    
    def format_for_display(self, error_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the error response for display in the UI.
        
        Args:
            error_response: The error response from handle_error
            
        Returns:
            Dictionary with user-friendly error information
        """
        return {
            "classification": "Error",
            "brief_explanation": error_response["message"],
            "detailed_reasoning": "The system encountered an error while processing your request.",
            "recommended_action": "Please try again or modify your input",
            "policy_references": [],
            "success": False,
            "component": error_response["component"]
        }