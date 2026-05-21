import os
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from app.rag.rag_pipeline import RAGPipeline
from app.sms_utils import SMSValidator, SMSFormatter, RateLimiter

load_dotenv()

class SMSHandler:
    """This class handles SMS messages and generates responses"""
    
    def __init__(self):
        # Initializing Twilio client
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        
        self.client = Client(self.account_sid, self.auth_token)
        
        # Initializing RAG pipeline
        self.rag = RAGPipeline()
        
        # Initializing utilities
        self.validator = SMSValidator()
        self.formatter = SMSFormatter()
        self.rate_limiter = RateLimiter()
        
        print(" SMS Handler initialized")
    
    def handle_incoming_sms(self, from_number: str, message_body: str) -> str:
        """
        Handle incoming SMS message
        Returns TwiML response
        """
        print(f"\n SMS received from {from_number}: {message_body}")
        
        # Rate limiting
        is_allowed, rate_limit_msg = self.rate_limiter.is_allowed(from_number)
        if not is_allowed:
            return self._create_twiml_response(rate_limit_msg)
        
        # Validate message
        if not message_body or not message_body.strip():
            error_msg = self.validator.get_error_message("empty")
            return self._create_twiml_response(error_msg)
        
        cleaned_message = self.validator.clean_message(message_body)
        
        if not self.validator.is_valid_question(cleaned_message):
            if len(cleaned_message) < 3:
                error_msg = self.validator.get_error_message("too_short")
            elif len(cleaned_message) > 500:
                error_msg = self.validator.get_error_message("too_long")
            else:
                error_msg = self.validator.get_error_message("invalid")
            return self._create_twiml_response(error_msg)
        
        # Querying the RAG pipeline
        try:
            result = self.rag.query(cleaned_message)
            
            if result['status'] == 'success':
                answer = result['answer']
            else:
                answer = self.validator.get_error_message("processing_error")
                print(f"RAG error: {result.get('error')}")
        except Exception as e:
            answer = self.validator.get_error_message("processing_error")
            print(f"Exception: {e}")
        
        # Formatting for SMS
        formatted_answer = self.formatter.format_for_sms(answer)
        sms_response = self.formatter.truncate_response(formatted_answer)
        
        print(f" Response sent: {sms_response[:100]}...")
        
        return self._create_twiml_response(sms_response)
    
    def _create_twiml_response(self, message: str) -> str:
        """Create TwiML XML response for Twilio"""
        response = MessagingResponse()
        response.message(message)
        return str(response)
    
    def send_sms(self, to_number: str, message: str) -> bool:
        """
        Send SMS to a number (for notifications, etc.)
        """
        try:
            message_obj = self.client.messages.create(
                body=message,
                from_=self.twilio_phone,
                to=to_number
            )
            print(f" SMS sent to {to_number}: {message_obj.sid}")
            return True
        except Exception as e:
            print(f" Failed to send SMS to {to_number}: {e}")
            return False