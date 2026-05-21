import re
from typing import Tuple

class SMSValidator:
    """Validate and clean SMS messages"""
    
    @staticmethod
    def clean_message(message: str) -> str:
        """Remove extra whitespace and normalize"""
        return message.strip().lower()
    
    @staticmethod
    def is_valid_question(message: str) -> bool:
        """Check if message is a valid question"""
        if not message or len(message) < 3:
            return False
        if len(message) > 500:
            return False
        return True
    
    @staticmethod
    def get_error_message(error_type: str) -> str:
        """Return appropriate error message"""
        messages = {
            "empty": "Please ask a question about the school.",
            "too_short": "Your message is too short. Please ask a complete question.",
            "too_long": "Your message is too long (max 500 characters). Please shorten it.",
            "invalid": "I didn't understand that. Please ask a question about school schedules, attendance, events, etc.",
            "processing_error": "Sorry, I encountered an error. Please try again later."
        }
        return messages.get(error_type, messages["invalid"])

class SMSFormatter:
    """Format responses for SMS"""
    
    @staticmethod
    def truncate_response(response: str, max_chars: int = 160) -> str:
        """
        Truncate response to SMS length
        SMS limit is 160 chars, but we leave room for formatting
        """
        if len(response) <= max_chars:
            return response
        
        # Truncate and add ellipsis
        truncated = response[:max_chars-3] + "..."
        return truncated
    
    @staticmethod
    def format_for_sms(response: str) -> str:
        """
        Format Claude's response for SMS readability
        Remove markdown, extra newlines, etc.
        """
       
        response = response.replace("**", "").replace("##", "").replace("###", "")
        
        # Remove extra newlines
        response = re.sub(r'\n\n+', '\n', response)
        
        # Clean up
        response = response.strip()
        
        return response

class RateLimiter:
    """Simple rate limiting for SMS"""
    
    def __init__(self):
        self.requests = {}  # phone_number -> [timestamps]
    
    def is_allowed(self, phone_number: str, max_requests: int = 5, 
                   window_seconds: int = 60) -> Tuple[bool, str]:
        """
        Check if request is allowed
        Max 5 requests per minute per phone number
        """
        import time
        
        current_time = time.time()
        
        if phone_number not in self.requests:
            self.requests[phone_number] = []
        
        # Remove old timestamps outside the window
        self.requests[phone_number] = [
            ts for ts in self.requests[phone_number]
            if current_time - ts < window_seconds
        ]
        
        # Check limit
        if len(self.requests[phone_number]) >= max_requests:
            return False, "Too many requests. Please wait a minute."
        
        # Add current request
        self.requests[phone_number].append(current_time)
        return True, ""