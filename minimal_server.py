#!/usr/bin/env python3
"""
Minimal FastAPI server for testing contact form
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Contact Form API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContactForm(BaseModel):
    name: str
    email: str
    subject: str
    message: str

@app.post("/contact")
async def send_contact_form(contact: ContactForm):
    """
    Send contact form message
    """
    try:
        # Import mailer
        from mailer import send_contact_email

        # Send email
        success = send_contact_email(
            name=contact.name,
            email=contact.email,
            subject=contact.subject,
            message=contact.message
        )

        if success:
            return {"success": True, "message": "Message sent successfully!"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")

    except Exception as e:
        print(f"Contact form error: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting minimal contact form server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)