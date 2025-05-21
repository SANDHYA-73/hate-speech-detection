"""
LangGraph workflow for the hate speech detection process.
"""
import logging
from typing import Dict, Any, List

from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph

from ..agents.detection_agent import HateSpeechDetectionAgent
from ..agents.retriever_agent import HybridRetrieverAgent
from ..agents.reasoning_agent import PolicyReasoningAgent
from ..agents.action_agent import ActionRecommenderAgent
from ..agents.error_agent import ErrorHandlerAgent
from ..utils.preprocessing import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechDetectionWorkflow:
    """Orchestrate the workflow of hate speech detection using LangGraph."""
    
    def __init__(self):
        """Initialize the HateSpeechDetectionWorkflow."""
        # Initialize agents
        self.detection_agent = HateSpeechDetectionAgent()
        self.retriever_agent = HybridRetrieverAgent()
        self.reasoning_agent = PolicyReasoningAgent()
        self.action_agent = ActionRecommenderAgent()
        self.error_handler = ErrorHandlerAgent()
        self.text_preprocessor = TextPreprocessor()
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            The configured StateGraph
        """
        # Define the state object for the graph
        class State(dict):
            """State object for the workflow."""
            def __init__(self, input_text="", **kwargs):
                # Make sure we initialize with at least these keys
                initial_state = {
                    "input_text": input_text,
                    "preprocessed_text": "",
                    "classification": "",
                    "brief_explanation": "",
                    "policy_references": [],
                    "detailed_reasoning": "",
                    "recommended_action": "",
                    "action_justification": "",
                    "errors": {},
                }
                initial_state.update(kwargs)
                super().__init__(**initial_state)
        
        # Initialize the state graph
        workflow = StateGraph(State)
        
        # Define workflow nodes
        
        # 1. Preprocess text node
        def preprocess_text(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Make sure input_text exists and is not None
                if "input_text" not in state or state["input_text"] is None:
                    logger.error("Missing input_text in state")
                    raise ValueError("Input text is missing or None")
                
                input_text = state["input_text"]
                if not isinstance(input_text, str):
                    logger.warning(f"Input text is not a string, converting from {type(input_text)}")
                    input_text = str(input_text)
                    
                preprocessed_text = self.text_preprocessor.preprocess_text(input_text)
                preprocessed_text = self.text_preprocessor.truncate_text(
                    preprocessed_text, max_length=4096
                )
                return {"preprocessed_text": preprocessed_text}
            except Exception as e:
                logger.error(f"Error in preprocessing: {e}")
                error_response = self.error_handler.handle_error(e, "preprocessing")
                return {"errors": {"preprocessing": error_response}}
        
        # 2. Classification node
        def classify_text(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                text = state["preprocessed_text"]
                classification, brief_explanation = self.detection_agent.detect(text)
                return {
                    "classification": classification,
                    "brief_explanation": brief_explanation
                }
            except Exception as e:
                error_response = self.error_handler.handle_error(e, "detection")
                return {"errors": {"detection": error_response}}
        
        # 3. Policy retrieval node
        def retrieve_policies(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                text = state["preprocessed_text"]
                classification = state["classification"]
                policy_references = self.retriever_agent.retrieve(text, classification)
                return {"policy_references": policy_references}
            except Exception as e:
                error_response = self.error_handler.handle_error(e, "retrieval")
                return {"errors": {"retrieval": error_response}}
        
        # 4. Reasoning node
        def generate_reasoning(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                text = state["preprocessed_text"]
                classification = state["classification"]
                brief_explanation = state["brief_explanation"]
                policy_references = state["policy_references"]
                
                detailed_reasoning = self.reasoning_agent.generate_reasoning(
                    text, classification, brief_explanation, policy_references
                )
                return {"detailed_reasoning": detailed_reasoning}
            except Exception as e:
                error_response = self.error_handler.handle_error(e, "reasoning")
                return {"errors": {"reasoning": error_response}}
        
        # 5. Action recommendation node
        def recommend_action(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                text = state["preprocessed_text"]
                classification = state["classification"]
                detailed_reasoning = state["detailed_reasoning"]
                
                recommended_action, action_justification = self.action_agent.recommend_action(
                    text, classification, detailed_reasoning
                )
                return {
                    "recommended_action": recommended_action,
                    "action_justification": action_justification
                }
            except Exception as e:
                error_response = self.error_handler.handle_error(e, "action")
                return {"errors": {"action": error_response}}
        
        # Add nodes to the graph
        workflow.add_node("preprocess", preprocess_text)
        workflow.add_node("classify", classify_text)
        workflow.add_node("retrieve", retrieve_policies)
        workflow.add_node("reason", generate_reasoning)
        workflow.add_node("recommend", recommend_action)
        
        # Create an end node
        workflow.add_node("end", lambda x: x)
        
        # Define error checking conditionals
        def should_continue(state: Dict[str, Any]) -> str:
            # Check if there are any errors at the current stage
            if state.get("errors") and len(state["errors"]) > 0:
                return "error"
            return "continue"
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "preprocess",
            should_continue,
            {
                "error": "end",
                "continue": "classify"
            }
        )
        
        workflow.add_conditional_edges(
            "classify",
            should_continue,
            {
                "error": "end",
                "continue": "retrieve"
            }
        )
        
        workflow.add_conditional_edges(
            "retrieve",
            should_continue,
            {
                "error": "reason",  # Even if retrieval fails, we continue with reasoning
                "continue": "reason"
            }
        )
        
        workflow.add_conditional_edges(
            "reason",
            should_continue,
            {
                "error": "recommend",  # Even if reasoning fails, we recommend an action
                "continue": "recommend"
            }
        )
        
        # Connect the final node to the end
        workflow.add_edge("recommend", "end")
        
        # Define the entry point
        workflow.set_entry_point("preprocess")
        
        return workflow
    
    def process(self, input_text: str) -> Dict[str, Any]:
        """
        Process the input text through the workflow.
        
        Args:
            input_text: The text to process
            
        Returns:
            Dictionary with the complete analysis result
        """
        try:
            logger.info(f"Processing input text of length {len(input_text)}")
            
            # Create initial state
            initial_state = {"input_text": input_text}
            result = None
            
            # Directly use the manual execution since the other methods aren't working
            result = self._manual_execute_workflow(initial_state)
            
            # If result is None, something went wrong
            if result is None:
                logger.error("Workflow execution returned None result")
                error_response = self.error_handler.handle_error(
                    Exception("Workflow execution failed"), "general"
                )
                formatted_error = self.error_handler.format_for_display(error_response)
                formatted_error["input_text"] = input_text
                return formatted_error
            
            # Check for errors
            if result.get("errors") and len(result["errors"]) > 0:
                # Get the first error
                first_error_key = next(iter(result["errors"]))
                error_response = result["errors"][first_error_key]
                formatted_error = self.error_handler.format_for_display(error_response)
                
                # Add the input text to the result
                formatted_error["input_text"] = input_text
                return formatted_error
            
            # Format the successful result
            output = {
                "success": True,
                "input_text": input_text,
                "classification": result.get("classification", ""),
                "brief_explanation": result.get("brief_explanation", ""),
                "detailed_reasoning": result.get("detailed_reasoning", ""),
                "policy_references": result.get("policy_references", []),
                "recommended_action": result.get("recommended_action", ""),
                "action_justification": result.get("action_justification", "")
            }
            
            logger.info(f"Processing complete: {output['classification']}")
            return output
            
        except Exception as e:
            logger.error(f"Unexpected error in workflow: {e}")
            error_response = self.error_handler.handle_error(e, "general")
            formatted_error = self.error_handler.format_for_display(error_response)
            
            # Add the input text to the result
            formatted_error["input_text"] = input_text
            return formatted_error
    
    def _manual_execute_workflow(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manually execute the workflow when LangGraph execution methods fail.
        This is a fallback method that simulates the LangGraph execution.
        
        Args:
            initial_state: The initial state dictionary
            
        Returns:
            The final state after execution
        """
        logger.info("Using direct agent execution fallback")
        
        # Create a deep copy of the initial state to work with
        state = dict(initial_state)
        
        # Make sure input_text exists
        if "input_text" not in state or state["input_text"] is None:
            logger.error("Missing input_text in manual execution")
            state["errors"] = {
                "preprocessing": self.error_handler.handle_error(
                    ValueError("Input text is missing"), "preprocessing"
                )
            }
            return state
        
        input_text = state["input_text"]
        
        # Direct agent execution without using LangGraph nodes
        
        # 1. Preprocess text
        try:
            preprocessed_text = self.text_preprocessor.preprocess_text(input_text)
            preprocessed_text = self.text_preprocessor.truncate_text(
                preprocessed_text, max_length=4096
            )
            state["preprocessed_text"] = preprocessed_text
        except Exception as e:
            logger.error(f"Error in direct preprocessing: {e}")
            state["errors"] = {
                "preprocessing": self.error_handler.handle_error(e, "preprocessing")
            }
            return state
        
        # 2. Classify text
        try:
            classification, brief_explanation = self.detection_agent.detect(preprocessed_text)
            state["classification"] = classification
            state["brief_explanation"] = brief_explanation
        except Exception as e:
            logger.error(f"Error in direct classification: {e}")
            state["errors"] = {
                "detection": self.error_handler.handle_error(e, "detection")
            }
            return state
        
        # 3. Retrieve policies
        try:
            policy_references = self.retriever_agent.retrieve(preprocessed_text, classification)
            state["policy_references"] = policy_references
        except Exception as e:
            logger.error(f"Error in direct retrieval: {e}")
            state["policy_references"] = []
        
        # 4. Generate reasoning
        try:
            detailed_reasoning = self.reasoning_agent.generate_reasoning(
                preprocessed_text, classification, brief_explanation, state["policy_references"]
            )
            state["detailed_reasoning"] = detailed_reasoning
        except Exception as e:
            logger.error(f"Error in direct reasoning: {e}")
            state["detailed_reasoning"] = f"Classification based on content analysis. {brief_explanation}"
        
        # 5. Recommend action
        try:
            recommended_action, action_justification = self.action_agent.recommend_action(
                preprocessed_text, classification, state.get("detailed_reasoning", "")
            )
            state["recommended_action"] = recommended_action
            state["action_justification"] = action_justification
        except Exception as e:
            logger.error(f"Error in direct recommendation: {e}")
            state["recommended_action"] = "Flag for human review"
            state["action_justification"] = "Error occurred during processing"
        
        return state