"""
Main Streamlit application for the Hate Speech Detection Assistant.
"""
import os
import sys
import io
import base64
import streamlit as st
import pandas as pd
from typing import Dict, Any
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import project components
from app.workflows.langgraph_workflow import HateSpeechDetectionWorkflow
from app.utils.audio_processor import AudioProcessor
from app.utils.export import ResultExporter
from app.config import CLASSIFICATION_LABELS
from app.playground import playground_page
# Initialize session state if needed
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

# Initialize components
workflow = HateSpeechDetectionWorkflow()
audio_processor = AudioProcessor(model_size="base")
exporter = ResultExporter()

def process_input(input_text: str) -> Dict[str, Any]:
    """Process the input text and return the result."""
    if not input_text or input_text.strip() == "":
        return {
            "success": False,
            "message": "Please enter some text to analyze."
        }
    
    # Run the workflow
    result = workflow.process(input_text)
    
    # Add to history if successful
    if result.get("success", False):
        st.session_state.results_history.append(result)
    
    return result

def process_audio(audio_file) -> Dict[str, Any]:
    """Process the audio file and return the text and result."""
    if audio_file is None:
        return {
            "success": False,
            "message": "No audio file provided."
        }
    
    # Transcribe audio
    transcribed_text = audio_processor.transcribe_audio(audio_file)
    
    if not transcribed_text:
        return {
            "success": False,
            "message": "Could not transcribe the audio file. Please try again or use text input."
        }
    
    # Process the transcribed text
    result = workflow.process(transcribed_text)
    
    # Add transcribed text to the result
    result["transcribed_text"] = transcribed_text
    
    # Add to history if successful
    if result.get("success", False):
        st.session_state.results_history.append(result)
    
    return result

def export_results():
    """Export current results to CSV."""
    if len(st.session_state.results_history) == 0:
        st.warning("No results to export.")
        return None
    
    try:
        if len(st.session_state.results_history) == 1:
            # Export single result
            filepath = exporter.export_to_csv(st.session_state.results_history[0])
        else:
            # Export multiple results
            filepath = exporter.export_multiple_to_csv(st.session_state.results_history)
        
        return filepath
    except Exception as e:
        st.error(f"Error exporting results: {str(e)}")
        return None

def get_csv_download_link(filepath):
    """Generate a download link for the CSV file."""
    with open(filepath, 'rb') as f:
        csv_data = f.read()
    
    b64 = base64.b64encode(csv_data).decode()
    filename = os.path.basename(filepath)
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def get_classification_color(classification):
    """Return a color based on the classification."""
    colors = {
        "Hate": "red",
        "Toxic": "orange",
        "Offensive": "yellow",
        "Neutral": "green",
        "Ambiguous": "gray",
        "Error": "purple"
    }
    return colors.get(classification, "blue")

# Set page config
st.set_page_config(
    page_title="Hate Speech Detection Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🛡️ Hate Speech Detection Assistant")
# st.markdown("""Upload text or audio to get started.""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This application uses GenAI to detect hate speech and 
    toxic content. It retrieves relevant policy documents,
    explains the classification, and recommends moderation actions.
    """)
    st.header("Navigation")
    page = st.radio("Go to", ["Single Input Analysis", "Playground"])
    
    st.header("History")
    if len(st.session_state.results_history) > 0:
        if st.button("Export All Results to CSV"):
            filepath = export_results()
            if filepath:
                st.markdown(get_csv_download_link(filepath), unsafe_allow_html=True)
    
    st.header("Options")
    clear_history = st.button("Clear History")
    if clear_history:
        st.session_state.results_history = []
        st.success("History cleared.")
    
    # Display history count
    st.text(f"Results in history: {len(st.session_state.results_history)}")

# Create two tabs for text and audio input
# tab1, tab2 = st.tabs(["Text Input", "Audio Input"])

# Choose which page to display
if page == "Single Input Analysis":
    # Create two tabs for text and audio input
    tab1, tab2 = st.tabs(["Text Input", "Audio Input"])
    with tab1:
        st.header("Analyze Text")
        # Text input area
        input_text = st.text_area("Enter text to analyze:", height=150)
        
        # Process button
        if st.button("Analyze Text", key="analyze_text_btn"):
            with st.spinner("Analyzing text..."):
                # Store the text in session state to preserve it
                st.session_state.current_text = input_text
                result = process_input(input_text)
                
                # Store the result in session state
                st.session_state.last_result = result
                
                # Force a rerun to ensure UI updates
                st.rerun()
        
        # This code will run after the rerun
        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            if not result.get("success", False):
                st.error(result.get("message", "An error occurred during analysis."))
            else:
                # Display the result
                if 'last_result' in st.session_state:
                    result = st.session_state.last_result
                            
                # Clear the result from session state to avoid showing stale results
                if 'last_result' in st.session_state:
                    del st.session_state.last_result
                            
                col1, col2 = st.columns(2)
                        
                with col1:
                    st.subheader("Classification")
                    classification = result.get("classification", "Unknown")
                    color = get_classification_color(classification)
                    st.markdown(f"<h3 style='color:{color}'>{classification}</h3>", unsafe_allow_html=True)
                            
                    st.subheader("Brief Explanation")
                    st.write(result.get("brief_explanation", "No explanation available."))
                            
                    st.subheader("Recommended Action")
                    st.write(result.get("recommended_action", "No action recommended."))
                    st.write("**Justification:** " + result.get("action_justification", ""))
                        
                with col2:
                    st.subheader("Detailed Reasoning")
                    st.write(result.get("detailed_reasoning", "No detailed reasoning available."))
                            
                    st.subheader("Policy References")
                    policy_refs = result.get("policy_references", [])
                    if policy_refs:
                        for i, ref in enumerate(policy_refs, 1):
                            with st.expander(f"Policy {i}: {ref.get('source', 'Unknown')}"):
                                st.write(ref.get("content", "No content available."))
                    else:
                        st.write("No policy references found.")
                        
                # Export option for this result
                # if st.button("Export This Result to CSV"):
                #     filepath = exporter.export_to_csv(result)
                #     if filepath:
                #         st.markdown(get_csv_download_link(filepath), unsafe_allow_html=True)

    with tab2:
        st.header("Analyze Audio")
        uploaded_file = st.file_uploader("Upload an audio file (WAV format recommended)", type=["wav", "mp3", "m4a"])
        
        if uploaded_file is not None:
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Process button
            if st.button("Transcribe and Analyze", key="analyze_audio_btn"):
                with st.spinner("Transcribing and analyzing audio..."):
                    # result = process_audio(uploaded_file)
                    # Store the file in session state
                    st.session_state.current_audio = uploaded_file
                    
                    # Process the audio
                    result = process_audio(uploaded_file)
                    
                    # Store the result in session state
                    st.session_state.last_audio_result = result
                    
                    # Force a rerun to ensure UI updates
                    st.rerun()
        # This code will run after the rerun for audio
        if 'last_audio_result' in st.session_state:
            result = st.session_state.last_audio_result
            
                    
            if not result.get("success", False):
                st.error(result.get("message", "An error occurred during analysis."))
            else:
                # Display transcription
                st.subheader("Transcription")
                st.write(result.get("transcribed_text", "Transcription failed."))
                        
                # Display the result
                col1, col2 = st.columns(2)
                        
                with col1:
                    st.subheader("Classification")
                    classification = result.get("classification", "Unknown")
                    color = get_classification_color(classification)
                    st.markdown(f"<h3 style='color:{color}'>{classification}</h3>", unsafe_allow_html=True)
                            
                    st.subheader("Brief Explanation")
                    st.write(result.get("brief_explanation", "No explanation available."))
                            
                    st.subheader("Recommended Action")
                    st.write(result.get("recommended_action", "No action recommended."))
                    st.write("**Justification:** " + result.get("action_justification", ""))
                        
                with col2:
                    st.subheader("Detailed Reasoning")
                    st.write(result.get("detailed_reasoning", "No detailed reasoning available."))
                            
                    st.subheader("Policy References")
                    policy_refs = result.get("policy_references", [])
                    if policy_refs:
                        for i, ref in enumerate(policy_refs, 1):
                            with st.expander(f"Policy {i}: {ref.get('source', 'Unknown')}"):
                                st.write(ref.get("content", "No content available."))
                    else:
                        st.write("No policy references found.")
                        
                # Clear the result from session state to avoid showing stale results
                if 'last_audio_result' in st.session_state:
                    del st.session_state.last_audio_result
                        
                        
                # Export option for this result
                # if st.button("Export This Result to CSV"):
                #     filepath = exporter.export_to_csv(result)
                #     if filepath:
                #         st.markdown(get_csv_download_link(filepath), unsafe_allow_html=True)

    # Show history tab
    if len(st.session_state.results_history) > 0:
        st.header("Analysis History")
        history_data = [
            {
                "Input": result.get("transcribed_text", result.get("input_text", ""))[:50] + "...",
                "Classification": result.get("classification", ""),
                "Action": result.get("recommended_action", "")
            }
            for result in st.session_state.results_history
        ]
        
        history_df = pd.DataFrame(history_data, index=range(1, len(history_data) + 1))
        history_df.index.name = "Sr. No" 
        st.dataframe(history_df)

    # Footer
    st.markdown("---")
    st.markdown("| Hate Speech Detection Assistant |")
else:  # Display the playground
    playground_page()