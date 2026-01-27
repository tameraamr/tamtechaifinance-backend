"""
Email Service using Resend API
Handles verification emails and transactional messages
"""
import os
import resend
from dotenv import load_dotenv

load_dotenv()

# Configure Resend API
resend.api_key = os.getenv("RESEND_API_KEY")

SENDER_EMAIL = "noreply@send.tamtech-finance.com"
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://www.tamtech-finance.com")


def send_verification_email(user_email: str, user_name: str, token: str) -> bool:
    """
    Send email verification link to user
    
    Args:
        user_email: Recipient email address
        user_name: User's first name for personalization
        token: Unique verification token
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    verification_url = f"{FRONTEND_URL}/verify-email?token={token}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Verify Your Email - Tamtech Finance</title>
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #0b1121;">
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color: #0b1121;">
            <tr>
                <td align="center" style="padding: 40px 20px;">
                    <!-- Main Container -->
                    <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0" style="max-width: 600px; background-color: #1a2332; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);">
                        
                        <!-- Header -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center;">
                                <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">
                                    🚀 Tamtech Finance
                                </h1>
                                <p style="margin: 10px 0 0 0; font-size: 14px; color: rgba(255, 255, 255, 0.9);">
                                    AI-Powered Stock Analysis Platform
                                </p>
                            </td>
                        </tr>
                        
                        <!-- Body -->
                        <tr>
                            <td style="padding: 50px 40px;">
                                <h2 style="margin: 0 0 20px 0; font-size: 24px; font-weight: 600; color: #e2e8f0;">
                                    Welcome, {user_name}! 👋
                                </h2>
                                
                                <p style="margin: 0 0 16px 0; font-size: 16px; line-height: 1.6; color: #cbd5e1;">
                                    Thank you for creating your Tamtech Finance account. You're one step away from accessing institutional-grade AI stock analysis.
                                </p>
                                
                                <p style="margin: 0 0 32px 0; font-size: 16px; line-height: 1.6; color: #cbd5e1;">
                                    Please verify your email address to unlock:
                                </p>
                                
                                <!-- Features List -->
                                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-bottom: 32px;">
                                    <tr>
                                        <td style="padding: 12px 0;">
                                            <span style="color: #10b981; font-size: 20px; margin-right: 10px;">✓</span>
                                            <span style="color: #cbd5e1; font-size: 15px;">3 Free AI Stock Analyses</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 12px 0;">
                                            <span style="color: #10b981; font-size: 20px; margin-right: 10px;">✓</span>
                                            <span style="color: #cbd5e1; font-size: 15px;">Real-time Market Data</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="padding: 12px 0;">
                                            <span style="color: #10b981; font-size: 20px; margin-right: 10px;">✓</span>
                                            <span style="color: #cbd5e1; font-size: 15px;">Advanced Financial Insights</span>
                                        </td>
                                    </tr>
                                </table>
                                
                                <!-- CTA Button -->
                                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                                    <tr>
                                        <td align="center" style="padding: 20px 0;">
                                            <a href="{verification_url}" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; font-size: 16px; font-weight: 600; text-decoration: none; padding: 16px 48px; border-radius: 8px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
                                                Verify My Account →
                                            </a>
                                        </td>
                                    </tr>
                                </table>
                                
                                <!-- Alternative Link -->
                                <p style="margin: 24px 0 0 0; font-size: 13px; line-height: 1.5; color: #94a3b8; text-align: center;">
                                    Button not working? Copy and paste this link:
                                </p>
                                <p style="margin: 8px 0 0 0; font-size: 13px; color: #667eea; word-break: break-all; text-align: center;">
                                    {verification_url}
                                </p>
                            </td>
                        </tr>
                        
                        <!-- Security Notice -->
                        <tr>
                            <td style="background-color: rgba(251, 191, 36, 0.1); padding: 24px 40px; border-top: 2px solid rgba(251, 191, 36, 0.3);">
                                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                                    <tr>
                                        <td style="padding-right: 15px; vertical-align: top;">
                                            <span style="font-size: 24px;">🔒</span>
                                        </td>
                                        <td>
                                            <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #fbbf24; font-weight: 600;">
                                                Security Notice
                                            </p>
                                            <p style="margin: 6px 0 0 0; font-size: 13px; line-height: 1.5; color: #cbd5e1;">
                                                This verification link will expire in <strong style="color: #fbbf24;">24 hours</strong>. If you didn't create this account, you can safely ignore this email.
                                            </p>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                        
                        <!-- Footer -->
                        <tr>
                            <td style="background-color: #0f1419; padding: 30px 40px; text-align: center;">
                                <p style="margin: 0 0 10px 0; font-size: 13px; color: #64748b;">
                                    © 2026 Tamtech Finance. All rights reserved.
                                </p>
                                <p style="margin: 0; font-size: 12px; color: #475569;">
                                    <a href="https://www.tamtech-finance.com/privacy" style="color: #667eea; text-decoration: none;">Privacy Policy</a>
                                    &nbsp;•&nbsp;
                                    <a href="https://www.tamtech-finance.com/terms" style="color: #667eea; text-decoration: none;">Terms of Service</a>
                                </p>
                            </td>
                        </tr>
                        
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    try:
        params = {
            "from": SENDER_EMAIL,
            "to": [user_email],
            "subject": "✅ Verify Your Email - Tamtech Finance",
            "html": html_content,
        }
        
        response = resend.Emails.send(params)
        print(f"✅ Verification email sent to {user_email}. ID: {response.get('id')}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email to {user_email}: {str(e)}")
        return False


def send_password_reset_email(user_email: str, user_name: str, token: str) -> bool:
    """
    Send password reset link to user (future implementation)
    """
    # Placeholder for future password reset functionality
    pass
