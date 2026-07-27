import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from config import JEM_ENERGY_THRESHOLD

# Configure logging
logger = logging.getLogger(__name__)

class JointEnergyModel(nn.Module):
    """
    Joint Energy-based Model (JEM) for detecting out-of-distribution inputs
    Implementation based on the JEM framework for anomaly detection
    """
    
    def __init__(self, input_dim=768, hidden_dims=[256, 64]):
        """
        Initialize the JEM model
        
        Args:
            input_dim (int): Dimension of input embeddings
            hidden_dims (list): Dimensions of hidden layers
        """
        super(JointEnergyModel, self).__init__()
        
        # Create sequential model with configurable hidden dimensions
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        # Final energy output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        logger.info(f"JEM model initialized with input dim {input_dim} and {len(hidden_dims)} hidden layers")
    
    def forward(self, x):
        """
        Calculate energy score for input
        Lower energy = more likely to be in-distribution
        
        Args:
            x: Input tensor of embeddings
            
        Returns:
            Energy score (scalar)
        """
        return self.model(x)

class JEMHandler:
    """Handler for JEM-based security checks"""
    
    def __init__(self, model_path=None):
        """
        Initialize JEM handler
        
        Args:
            model_path (str, optional): Path to pre-trained model weights
        """
        self.model = JointEnergyModel()
        
        # Load pre-trained weights if available
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded pre-trained JEM model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load JEM model weights: {str(e)}")
        
        # Set evaluation mode
        self.model.eval()
        logger.info("JEM handler initialized")
    
    def check_input(self, embedding):
        """
        Check if input is in-distribution using energy score
        
        Args:
            embedding: Tensor of input embeddings
            
        Returns:
            tuple: (is_safe, energy_score, message)
        """
        # Convert numpy array to tensor if needed
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Get energy score
        with torch.no_grad():
            energy_score = self.model(embedding).item()
        
        # Lower energy = more likely to be in-distribution
        is_safe = energy_score < JEM_ENERGY_THRESHOLD
        
        if not is_safe:
            logger.warning(f"JEM detected unusual input with energy score: {energy_score}")
            message = f"Unusual input pattern detected (energy score: {energy_score:.2f})"
        else:
            logger.info(f"JEM check passed with energy score: {energy_score:.2f}")
            message = "Input passed JEM energy check"
            
        return is_safe, energy_score, message
