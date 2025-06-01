"""
Interactive Hate Speech Detection Playground.

This module provides an interactive playground for testing multiple sentences of text
for hate speech detection with sentence-by-sentence analysis and recommendations.
"""
import streamlit as st
import pandas as pd
import time
import re
from typing import Dict, List, Any, Tuple

# Import project components
from .workflows.langgraph_workflow import HateSpeechDetectionWorkflow
from .utils.preprocessing import TextPreprocessor

# Initialize components
workflow = HateSpeechDetectionWorkflow()
text_preprocessor = TextPreprocessor()

# Define color mapping for classifications
CLASSIFICATION_COLORS = {
    "Hate": "#FFB3B3",       # Pastel Red
    "Toxic": "#FFB380",      # Pastel Orange
    "Offensive": "#FFE599",  # Pastel Amber/Yellow
    "Neutral": "#B3E6B3",    # Pastel Green
    "Ambiguous": "#E0E0E0"   # Light Gray
}

def get_severity_score(classification: str) -> int:
    """Map classification to severity score for sorting."""
    severity_map = {
        "Hate": 5,
        "Toxic": 4,
        "Offensive": 3,
        "Ambiguous": 2,
        "Neutral": 1
    }
    return severity_map.get(classification, 0)

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex pattern.
    
    Args:
        text: Text to split into sentences
        
    Returns:
        List of sentences
    """
    # Pattern matches sentence endings (period, question mark, exclamation point)
    # followed by whitespace and uppercase letter, or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])'

    
    # Split text using the pattern
    sentences = re.split(sentence_pattern, text)
    
    # Clean up sentences and filter out empty ones
    cleaned_sentences = [s.strip() for s in sentences if s.strip()]
    
    return cleaned_sentences

def analyze_sentence(sentence: str) -> Dict[str, Any]:
    """
    Analyze a single sentence.
    
    Args:
        sentence: A single sentence to analyze
        
    Returns:
        Dictionary with analysis results
    """
    if not sentence or sentence.strip() == "":
        return {
            "text": sentence,
            "classification": "Neutral",
            "explanation": "Empty or whitespace-only text.",
            "severity": 1
        }
    
    # Analyze the sentence with the workflow
    result = workflow.process(sentence)
    
    # Extract relevant information
    return {
        "text": sentence,
        "classification": result.get("classification", "Ambiguous"),
        "explanation": result.get("brief_explanation", "No explanation available."),
        "detailed_reasoning": result.get("detailed_reasoning", ""),
        "recommended_action": result.get("recommended_action", ""),
        "severity": get_severity_score(result.get("classification", "Ambiguous"))
    }

def generate_rephrase_suggestion(text: str, classification: str, explanation: str) -> str:
    """
    Generate a specific suggestion for rephrasing problematic text.
    
    Args:
        text: The original text
        classification: The classification of the text
        explanation: The explanation for the classification
        
    Returns:
        A specific suggestion for rephrasing the text
    """
    # Skip neutral content
    if classification == "Neutral":
        return "No rephrasing needed."
    
    try:
        # Use the reasoning agent to generate a rephrasing suggestion
        prompt = f"""
        The following text has been classified as {classification}:
        "{text}"
        
        Reason for classification: {explanation}
        
        Please rewrite this text in a more neutral, respectful and appropriate way.
        Your rewrite must:
        1. Be a complete, specific rephrasing (not just advice on how to change it)
        2. Maintain the core intent when possible, but remove any offensive elements
        3. Be approximately the same length as the original
        4. Use clear, direct language
        5. Be constructive and respectful
        
        Provide ONLY the rephrased text, with no explanations, no quotes, and no prefacing statements.
        Do not use phrases like "you could say" or "a better way to express this would be".
        Just provide the direct rephrasing.
        """
        
        # Use the same model as the reasoning agent
        response = workflow.reasoning_agent.model.invoke(prompt)
        
        # Extract the response content
        if hasattr(response, 'content'):
            suggestion = response.content.strip()
        else:
            suggestion = str(response).strip()
        
        # Clean up any markdown formatting, quotes, or prefacing statements
        if "```" in suggestion:
            suggestion = suggestion.replace("```", "").strip()
        
        # Remove quotes if present
        suggestion = suggestion.strip('"\'')
        
        # Remove common prefacing phrases
        prefacing_phrases = [
            "you could say ", "a better way to express this would be ", 
            "this could be rephrased as ", "consider saying ",
            "i would suggest: ", "suggested rephrasing: ",
            "rephrased version: ", "here's a rephrasing: "
        ]
        
        for phrase in prefacing_phrases:
            if suggestion.lower().startswith(phrase):
                suggestion = suggestion[len(phrase):].strip()
        
        # If the suggestion is too generic, use a specific fallback
        generic_suggestions = [
            "try to express this message more clearly",
            "consider rephrasing",
            "this should be rewritten",
            "express your point without"
        ]
        
        is_generic = any(gen in suggestion.lower() for gen in generic_suggestions)
        if is_generic or len(suggestion) < 10:
            # Use classification-specific detailed fallbacks
            return get_specific_fallback_suggestion(text, classification)
            
        return suggestion
    except Exception as e:
        # Use classification-specific detailed fallbacks
        return get_specific_fallback_suggestion(text, classification)

def get_specific_fallback_suggestion(text: str, classification: str) -> str:
    """
    Get a specific fallback suggestion based on classification.
    
    Args:
        text: The original text
        classification: The classification
        
    Returns:
        A specific rephrasing suggestion
    """
    # Extract subject if possible (basic extraction)
    words = text.split()
    subject = "this group" if len(words) < 3 else " ".join(words[0:2])
    
    if classification == "Hate":
        if "should" in text.lower():
            return f"I have concerns about certain policies related to {subject} that I'd like to discuss constructively."
        else:
            return f"I've observed some concerning patterns that I believe need addressing through respectful dialogue and understanding."
    
    elif classification == "Toxic":
        if "lazy" in text.lower() or "don't want to" in text.lower():
            return f"Some individuals may face challenges or barriers that aren't immediately apparent to others."
        else:
            return f"I find myself frustrated by certain behaviors, though I recognize there may be underlying factors I don't fully understand."
    
    elif classification == "Offensive":
        return f"I have a different perspective on this matter that I'd like to express respectfully."
    
    elif classification == "Ambiguous":
        if "they" in text.lower() or "them" in text.lower():
            return f"I've noticed certain patterns that concern me, though I'd like to better understand the specific context and factors involved."
        elif "special" in text.lower() or "treatment" in text.lower():
            return f"I wonder if there are historical or contextual factors that explain the different approaches being taken here."
        else:
            return f"I'd like to discuss this situation with more specific details so we can have a productive conversation about it."
            
    else:
        return f"I'd like to express my thoughts on this matter in a more constructive and specific way."

def analyze_paragraph(text: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Analyze text by breaking it into individual sentences.
    
    Args:
        text: Paragraph text to analyze
        
    Returns:
        Tuple of (list of sentence results, overall severity score)
    """
    # Split text into sentences
    sentences = split_into_sentences(text)
    results = []
    
    # Show progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Analyze each sentence
    total_sentences = len(sentences)
    for i, sentence in enumerate(sentences):
        if sentence.strip():  # Skip empty sentences
            progress_text.text(f"Analyzing sentence {i+1} of {total_sentences}")
            result = analyze_sentence(sentence)
            results.append(result)
        else:
            # Add empty sentences to maintain structure
            results.append({
                "text": sentence,
                "classification": "Neutral",
                "explanation": "",
                "severity": 0
            })
        progress_bar.progress((i + 1) / total_sentences)
        
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()
    
    # Calculate overall severity
    if results:
        overall_severity = sum(result.get("severity", 0) for result in results) / len(results)
    else:
        overall_severity = 0
    
    return results, overall_severity

def render_analyzed_text(results: List[Dict[str, Any]]) -> None:
    """
    Render the color-coded analyzed text with explanations.
    
    Args:
        results: List of analysis results for each sentence
    """
    # Display color legend
    st.subheader("Classification Legend")
    legend_cols = st.columns(5)
    for i, (classification, color) in enumerate(CLASSIFICATION_COLORS.items()):
        with legend_cols[i]:
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; color: {'white' if classification in ['Hate', 'Toxic'] else 'black'};'>{classification}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display color-coded results
    st.subheader("Analysis Results")
    
    # First, display the entire paragraph with colored spans for each sentence
    color_coded_paragraph = ""
    for result in results:
        classification = result.get("classification", "Ambiguous")
        color = CLASSIFICATION_COLORS.get(classification, "#CCCCCC")
        text = result.get("text", "")
        text_color = 'white' if classification in ['Hate', 'Toxic'] else 'black'
        
        # Add colored span for this sentence
        color_coded_paragraph += f"<span style='background-color: {color}; padding: 3px; border-radius: 3px; color: {text_color};'>{text}</span> "
    
    # Display the color-coded paragraph
    st.markdown(f"<div style='padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px;'>{color_coded_paragraph}</div>", unsafe_allow_html=True)
    
    # Then display each sentence individually with details
    st.subheader("Sentence-by-Sentence Analysis")
    
    for i, result in enumerate(results):
        if not result.get("text", "").strip():
            # Display empty line
            st.markdown("<br>", unsafe_allow_html=True)
            continue
            
        classification = result.get("classification", "Ambiguous")
        color = CLASSIFICATION_COLORS.get(classification, "#CCCCCC")
        text = result.get("text", "")
        
        # Text with colored background based on classification
        st.markdown(
            f"""<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: {'white' if classification in ['Hate', 'Toxic'] else 'black'};'>
            <strong>Sentence {i+1} - {classification}:</strong> {text}
            </div>""", 
            unsafe_allow_html=True
        )
        
        # Only show details for non-neutral content
        if classification != "Neutral":
            with st.expander(f"Details & Rephrasing Suggestion"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Explanation:**")
                    st.write(result.get("explanation", "No explanation available."))
                    
                    if "detailed_reasoning" in result and result["detailed_reasoning"]:
                        st.markdown("**Detailed Reasoning:**")
                        st.write(result.get("detailed_reasoning", ""))
                
                with col2:
                    st.markdown("**Recommended Action:**")
                    st.write(result.get("recommended_action", "No action recommended."))
                    
                    st.markdown("**Rephrasing Suggestion:**")
                    with st.spinner("Generating suggestion..."):
                        suggestion = generate_rephrase_suggestion(
                            text, 
                            classification,
                            result.get("explanation", "")
                        )
                        st.write(suggestion)

def playground_page():
    """Render the hate speech detection playground page."""
    # st.title("Hate Speech Detection Playground")
    st.markdown("""
    <style>
    .custom-title {
        font-size: 27px; /* Adjust size */
        line-height: 1.0; /* Adjust vertical height/spacing */
        font-weight: 500;
        margin-bottom: 20px;
    }
    </style>
    <h1 class="custom-title">Hate Speech Detection Playground</h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This playground allows you to test paragraphs of text for hate speech and harmful content.
    """)
    # Each sentence will be analyzed individually and color-coded based on its classification.
    # For problematic content, rephrasing suggestions will be provided.
    # Input text area
    text_input = st.text_area(
        "Enter text to analyze:",
        height=200,
        placeholder="Enter a paragraph to analyze."
    )
    # \n For example: I don't understand why they always get special treatment. Everyone should be treated equally, regardless of their background. Those people are just lazy and don't want to work. Maybe we should try to understand their situation before judging. They ruin everything wherever they go. Let's focus on building a more inclusive community.
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        analyze_button = st.button("Analyze Text", type="primary")
    
    with col2:
        st.markdown(
            "<p style='padding-top: 10px;'>Each sentence will be analyzed separately and color-coded based on severity.</p>",
            unsafe_allow_html=True
        )
    
    if analyze_button and text_input:
        # Analyze the text
        with st.spinner("Analyzing text sentence by sentence..."):
            results, overall_severity = analyze_paragraph(text_input)
        
        # Display an overall summary
        severity_percentage = min(100, max(0, overall_severity * 20))  # Convert to percentage (0-100)
        
        st.markdown("### Overall Content Analysis")
        
        # Create metrics for the summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sentences Analyzed", len(results))
        
        with col2:
            # Count problematic sentences
            problematic_count = sum(1 for r in results if r.get("classification") in ["Hate", "Toxic", "Offensive"])
            st.metric("Problematic Sentences", problematic_count)
            
        with col3:
            # Format severity as percentage
            severity_label = "Low" if severity_percentage < 30 else "Medium" if severity_percentage < 70 else "High"
            st.metric("Overall Severity", f"{severity_label} ({severity_percentage:.1f}%)")
        
        # Display color-coded results
        render_analyzed_text(results)

if __name__ == "__main__":
    playground_page()