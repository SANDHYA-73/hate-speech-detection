"""
FastAPI endpoint for the Hate Speech Detection Assistant.
"""
import tempfile
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .workflows.langgraph_workflow import HateSpeechDetectionWorkflow
from .utils.audio_processor import AudioProcessor

# Initialize components
workflow = HateSpeechDetectionWorkflow()
audio_processor = AudioProcessor(model_size="base")

# Define API models
class TextRequest(BaseModel):
    """Request model for text analysis."""
    text: str
    
class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    input_text: Optional[str] = None
    transcribed_text: Optional[str] = None
    classification: Optional[str] = None
    brief_explanation: Optional[str] = None
    detailed_reasoning: Optional[str] = None
    policy_references: Optional[list] = None
    recommended_action: Optional[str] = None
    action_justification: Optional[str] = None
    message: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="Hate Speech Detection API",
    description="API for detecting and analyzing hate speech in text and audio.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the Hate Speech Detection API",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest):
    """
    Analyze text for hate speech.
    
    Args:
        request: TextRequest with the text to analyze
        
    Returns:
        AnalysisResponse with the analysis results
    """
    if not request.text or len(request.text.strip()) == 0:
        return AnalysisResponse(
            success=False,
            message="Text cannot be empty."
        )
    
    try:
        # Process the text
        result = workflow.process(request.text)
        
        # Convert to response model
        return AnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text: {str(e)}"
        )

@app.post("/analyze/audio", response_model=AnalysisResponse)
async def analyze_audio(
    audio_file: UploadFile = File(...),
):
    """
    Analyze audio for hate speech.
    
    Args:
        audio_file: Audio file to transcribe and analyze
        
    Returns:
        AnalysisResponse with the analysis results
    """
    if not audio_file:
        return AnalysisResponse(
            success=False,
            message="Audio file is required."
        )
    
    try:
        # Read the audio file
        contents = await audio_file.read()
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp_path = temp.name
            temp.write(contents)
        
        # Process audio using a file-like object
        with open(temp_path, "rb") as audio_data:
            # Transcribe audio
            transcribed_text = audio_processor.transcribe_audio(audio_data)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        if not transcribed_text:
            return AnalysisResponse(
                success=False,
                message="Could not transcribe the audio file."
            )
        
        # Process the transcribed text
        result = workflow.process(transcribed_text)
        
        # Add transcribed text to the result
        result["transcribed_text"] = transcribed_text
        
        # Convert to response model
        return AnalysisResponse(**result)
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
                
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

# Run the API if executed directly
if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)