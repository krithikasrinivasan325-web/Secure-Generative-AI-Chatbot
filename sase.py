import logging

# Configure logging
logger = logging.getLogger(__name__)

class SASESecurityHandler:
    """
    SASE (Secure Access Service Edge) implementation for text input security
    Provides pattern-based security checks for user inputs
    """
    
    def __init__(self):
        """Initialize with security patterns"""
        self.blocked_patterns = [
            # Common attack patterns
            "sql injection", "xss", "cross site", "<script>", 
            "rm -rf", "sudo", "system(", "exec(", "eval(",
            "drop table", "delete from", "--", "/*", 
            
            # Credential harvesting attempts
            "password:", "credential", "token:", "api key",
            
            # Code injection patterns
            "function()", "prototype", "constructor", "__proto__",
            
            # File system access attempts
            "readfile", "writefile", "unlink", "chmod",
            
            # Command execution
            "bash", "powershell", "cmd.exe", "/bin/sh"
        ]
        logger.info(f"SASE security handler initialized with {len(self.blocked_patterns)} patterns")
    
    def check_input(self, text):
        """
        Check if input contains suspicious patterns
        
        Args:
            text (str): User input to check
            
        Returns:
            tuple: (is_safe, message) where is_safe is a boolean and message is a string
        """
        if not text or not isinstance(text, str):
            return False, "Invalid input type"
            
        text_lower = text.lower()
        
        # Check against blocked patterns
        for pattern in self.blocked_patterns:
            if pattern in text_lower:
                logger.warning(f"SASE detected suspicious pattern: {pattern}")
                return False, f"Suspicious pattern detected: {pattern}"
        
        # Length check
        if len(text) > 5000:
            logger.warning("SASE detected unusually long input")
            return False, "Input exceeds maximum allowed length"
            
        logger.info("SASE security check passed")
        return True, "Input passed security check"
        
    def add_pattern(self, pattern):
        """Add a new pattern to the blocked list"""
        if pattern and pattern not in self.blocked_patterns:
            self.blocked_patterns.append(pattern)
            logger.info(f"Added new blocked pattern: {pattern}")
