from fastapi import FastAPI, Request
from fastapi.responses import Response
from dotenv import load_dotenv
import os
from app.api.sms_handler import SMSHandler

load_dotenv()

app = FastAPI(title="IntelliChat SMS API")

# Initialize SMS handler
sms_handler = SMSHandler()

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "IntelliChat SMS API is running",
        "version": "1.0"
    }

@app.post("/sms/webhook")
async def sms_webhook(req: Request):
    """
    Webhook endpoint for Twilio to send incoming SMS
    Twilio will POST to this endpoint when someone texts the number
    """
    # Get form data from Twilio
    form_data = await req.form()
    
    from_number = form_data.get("From", "")
    message_body = form_data.get("Body", "")
    message_sid = form_data.get("MessageSid", "")
    
    print(f"\n📨 Webhook received")
    print(f"   From: {from_number}")
    print(f"   Message: {message_body}")
    print(f"   SID: {message_sid}")
    
    # Handle the SMS
    twiml_response = sms_handler.handle_incoming_sms(from_number, message_body)
    
    # Return TwiML response to Twilio
    return Response(content=twiml_response, media_type="application/xml")

@app.post("/sms/send")
async def send_sms(phone_number: str, message: str):
    """
    Send SMS directly (for testing or admin use)
    """
    success = sms_handler.send_sms(phone_number, message)
    
    return {
        "status": "success" if success else "failed",
        "phone_number": phone_number,
        "message": message
    }

@app.get("/health")
def health_check():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "service": "intellichat-sms-api"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)