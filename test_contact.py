#!/usr/bin/env python3
"""
Simple test script for contact form functionality
"""
import os
import sys
sys.path.append('.')

from mailer import send_contact_email

def test_contact_email():
    """Test the contact email functionality"""
    print("Testing contact email functionality...")

    try:
        result = send_contact_email(
            name="Test User",
            email="test@example.com",
            subject="Test Contact Form",
            message="This is a test message from the contact form."
        )

        if result:
            print("✅ Contact email sent successfully!")
            return True
        else:
            print("❌ Contact email failed to send")
            return False

    except Exception as e:
        print(f"❌ Error testing contact email: {e}")
        return False

if __name__ == "__main__":
    success = test_contact_email()
    sys.exit(0 if success else 1)