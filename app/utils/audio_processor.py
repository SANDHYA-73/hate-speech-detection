"""
Audio processing utility for transcribing audio inputs using Whisper.
"""
import os
import tempfile
import logging
from typing import BinaryIO, Optional
# import whisper
from faster_whisper import WhisperModel
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Process audio files and convert them to text using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the AudioProcessor.
        
        Args:
            model_size: Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        """
        try:
            logger.info(f"Loading Whisper model: {model_size}")
            self.model = WhisperModel(model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.model = None
    
    def transcribe_audio(self, audio_file: BinaryIO) -> Optional[str]:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_file: The audio file to transcribe
            
        Returns:
            Transcribed text or None if an error occurred
        """
        if self.model is None:
            logger.error("Whisper model not loaded")
            return None
            
        try:
            # Create a temporary file with .wav extension
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_path = temp_file.name
            
            # Write audio data to the temp file
            audio_file.seek(0)
            temp_file.write(audio_file.read())
            temp_file.close()  # Important: Close the file to ensure data is written
            
            logger.info(f"Temporary file created at {temp_path} ({os.path.getsize(temp_path)} bytes)")
            
            # Process the audio file with Faster Whisper
            logger.info(f"Transcribing audio file...")
            segments, info = self.model.transcribe(temp_path, beam_size=5)
            
            # Combine all segments into a single text
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
                logger.info(f"Temporary file {temp_path} removed")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
            
            logger.info(f"Audio transcription successful: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            # Clean up the temporary file if it exists
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                    logger.info(f"Temporary file {temp_path} removed during error handling")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove temporary file during error handling: {cleanup_error}")
            return None