import os
import logging
import torch
from flask import Flask, request, jsonify, render_template
import numpy as np
os.environ["GEMINI_API_KEY"] = "AIzaSyD7hec1NrQpaJ_9CAdICQY4db_cEXh4ilY"

# Import configuration
from config import DEBUG, HOST, PORT

# Import modules
from modules.gemini_client import GeminiClient
from modules.security.sase import SASESecurityHandler
from modules.security.jem import JointEnergyModel, JEMHandler
from modules.security.deepfool import DeepFoolDetector
from modules.utils import get_text_embedding, create_security_log, sanitize_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Check for API key
if "GEMINI_API_KEY" not in os.environ:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise EnvironmentError("GEMINI_API_KEY not set. Please set this environment variable.")

# Initialize components
try:
    # Initialize Gemini client
    gemini_client = GeminiClient()
    
    # Initialize security components
    sase_handler = SASESecurityHandler()
    
    # Initialize JEM model
    jem_model = JointEnergyModel()
    jem_handler = JEMHandler()
    
    # Initialize DeepFool detector (using JEM model as reference)
    deepfool_detector = DeepFoolDetector(reference_model=jem_model)
    
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

# Routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages with security checks"""
    try:
        # Get and sanitize user input
        user_input = request.json.get('message', '')
        user_input = sanitize_input(user_input)
        
        if not user_input:
            return jsonify({
                "response": "Please provide a valid message.",
                "alert": "Empty or invalid input received"
            })
        
        # Initialize security log
        log_entry = create_security_log(user_input)
        
        # 1. SASE Security Check
        sase_passed, sase_message = sase_handler.check_input(user_input)
        log_entry["security_checks"].append({
            "type": "SASE", 
            "result": sase_passed, 
            "message": sase_message
        })
        
        if not sase_passed:
            logger.warning(f"SASE check failed: {sase_message}")
            return jsonify({
                "response": "I cannot process this request due to security concerns.",
                "alert": sase_message,
                "security_log": log_entry
            })
        
        # 2. Generate text embedding for advanced checks
        try:
            text_embedding = get_text_embedding(user_input)
            
            # 3. JEM energy-based check
            jem_passed, energy_score, jem_message = jem_handler.check_input(text_embedding)
            log_entry["security_checks"].append({
                "type": "JEM", 
                "result": jem_passed,
                "energy_score": float(energy_score)
            })
            
            if not jem_passed:
                logger.warning(f"JEM check failed: {jem_message}")
                return jsonify({
                    "response": "Your input appears unusual. Could you provide more context?",
                    "alert": jem_message,
                    "security_log": log_entry
                })
            
            # 4. DeepFool adversarial detection
            deepfool_passed, perturbation, deepfool_message = deepfool_detector.detect(text_embedding)
            log_entry["security_checks"].append({
                "type": "DeepFool", 
                "result": deepfool_passed, 
                "perturbation": float(perturbation)
            })
            
            if not deepfool_passed:
                logger.warning(f"DeepFool check failed: {deepfool_message}")
                return jsonify({
                    "response": "Your input appears to be adversarial. Please rephrase your question.",
                    "alert": deepfool_message,
                    "security_log": log_entry
                })
                
        except Exception as e:
            logger.error(f"Error in security processing: {str(e)}")
            # Continue with Gemini anyway if security processing fails
            log_entry["security_checks"].append({
                "type": "ERROR", 
                "result": False,
                "message": f"Security processing error: {str(e)}"
            })
            
        # All checks passed or bypassed, send to Gemini
        response = gemini_client.send_message(user_input)
        
        log_entry["response_generated"] = True
        logger.info(f"Secure response generated for: {user_input[:50]}...")
        
        return jsonify({
            "response": response,
            "security_log": log_entry
        })
            
    except Exception as e:
        logger.error(f"General error in chat endpoint: {str(e)}")
        return jsonify({
            "response": "Sorry, I encountered an unexpected error.",
            "error": str(e)
        })

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    """Reset the chat session"""
    try:
        message = gemini_client.reset_chat()
        return jsonify({"message": message})
    except Exception as e:
        logger.error(f"Error resetting chat: {str(e)}")
        return jsonify({
            "message": "Error resetting chat session.",
            "error": str(e)
        })

# Run the app
if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Run Flask app
    logger.info(f"Starting Flask app on {HOST}:{PORT}")
    app.run(debug=DEBUG, host=HOST, port=PORT)
