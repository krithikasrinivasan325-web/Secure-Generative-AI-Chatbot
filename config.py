import os

# Gemini API configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"

# Generation parameters
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

# Security thresholds
DEEPFOOL_THRESHOLD = 0.8
JEM_ENERGY_THRESHOLD = 10.0

# Flask configuration
DEBUG = True
HOST = "0.0.0.0"
PORT = 5000
