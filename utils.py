import torch
from transformers import AutoTokenizer, AutoModel
import logging
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Initialize tokenizer only once for efficiency
_tokenizer = None
_bert_model = None

def get_text_embedding(text, max_length=128):
    """
    Get BERT embeddings for text input
    
    Args:
        text (str): Input text
        max_length (int): Maximum sequence length
        
    Returns:
        numpy.ndarray: Text embedding vector
    """
    global _tokenizer, _bert_model
    
    # Initialize models if not already done
    if _tokenizer is None or _bert_model is None:
        try:
            logger.info("Initializing BERT tokenizer and model")
            _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            _bert_model = AutoModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            logger.error(f"Failed to initialize BERT: {str(e)}")
            raise
    
    # Tokenize input
    inputs = _tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    )
    
    # Get embeddings
    with torch.no_grad():
        outputs = _bert_model(**inputs)
    
    # Use CLS token embedding as text representation
    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    
    logger.info(f"Generated embedding of shape {embedding.shape}")
    return embedding

def create_security_log(input_text, security_checks=None):
    """
    Create a security log entry
    
    Args:
        input_text (str): User input text
        security_checks (list): List of security check results
        
    Returns:
        dict: Security log entry
    """
    if security_checks is None:
        security_checks = []
        
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
        "security_checks": security_checks
    }
    
    return log_entry

def sanitize_input(text):
    """
    Basic input sanitization
    
    Args:
        text (str): Input text
        
    Returns:
        str: Sanitized text
    """
    if not isinstance(text, str):
        return ""
        
    # Limit length
    if len(text) > 5000:
        text = text[:5000]
    
    # Basic sanitization
    text = text.replace("<script>", "")
    text = text.replace("</script>", "")
    
    return text
