import google.generativeai as genai
import logging
from config import GEMINI_API_KEY, GEMINI_MODEL, GENERATION_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google's Gemini API"""
    
    def __init__(self):
        """Initialize the Gemini client"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=GENERATION_CONFIG,
            )
            self.chat_session = self.model.start_chat(history=[])
            logger.info(f"Gemini client initialized with model: {GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise
    
    def send_message(self, message):
        """Send a message to the Gemini API and get a response"""
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            logger.error(f"Error sending message to Gemini: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def reset_chat(self):
        """Reset the chat session"""
        self.chat_session = self.model.start_chat(history=[])
        logger.info("Chat session reset")
        return "Chat session has been reset."
