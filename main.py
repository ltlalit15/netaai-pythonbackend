from datetime import timedelta
# from fastapi import FastAPI, Depends, HTTPException
from fastapi import FastAPI, HTTPException, Depends, Request, Query
from sqlalchemy.orm import Session,selectinload
from database import SessionLocal, engine
import models, schemas, auth
from schemas import AdminProfile,AdminProfileUpdate,PromptOut,ProductTier,AdminSubscriptionCreate
import os
import smtplib
from typing import Optional
from typing import List

from fastapi import UploadFile
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request, Query



from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
import os
import stripe
import json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from contextlib import contextmanager
from sqlalchemy.exc import SQLAlchemyError
from typing import Generator
import uuid
import mimetypes
import aiofiles
from pathlib import Path
from stripe.error import StripeError
from jinja2 import Environment, FileSystemLoader
import aiosmtplib
import asyncio
from typing import Dict, Any
import httpx
from pydantic import BaseModel

app = FastAPI(root_path="/api")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "https://askneta.com,http://localhost:5174, http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure this in production
)

models.Base.metadata.create_all(bind=engine)


# Load environment variables
load_dotenv()


STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

stripe.api_key = STRIPE_SECRET_KEY

EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS") == "True"
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", "meatitd11@gmail.com")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", "vlolxfmlaysxgmqc")

ALLOWED_FILE_TYPES = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif'
}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

async def validate_file(file: UploadFile) -> str:
    if not file:
        return None
        
    # Check file size
    file_size = 0
    chunk_size = 1024
    while chunk := await file.read(chunk_size):
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
            )
    await file.seek(0)
    
    # Check file type using mimetypes
    content_type, _ = mimetypes.guess_type(file.filename)
    if not content_type or content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_FILE_TYPES.keys())}"
        )
    
    # Generate unique filename
    ext = ALLOWED_FILE_TYPES[content_type]
    filename = f"{uuid.uuid4()}{ext}"
    return filename

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error occurred")
    finally:
        db.close()

def get_db():
    with get_db_session() as db:
        yield db

SECRET_KEY = os.getenv("SECRET_KEY", "hh45h34brb##$67j#*chscbecyej")  # Fallback for development only
from datetime import datetime, timedelta



# Initialize Jinja2 environment
env = Environment(loader=FileSystemLoader('templates'))

# Uvicorn server startup block
# --------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)

class EmailError(Exception):
    pass

async def send_email_async(
    to_email: str,
    subject: str,
    template_name: str,
    template_data: Dict[str, Any]
) -> None:
    try:
        # Render email template
        template = env.get_template(f"{template_name}.html")
        html_content = template.render(**template_data)
        
        # Create message
        msg = MIMEMultipart()
        msg["From"] = EMAIL_HOST_USER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_content, "html"))
        
        # Send email asynchronously
        async with aiosmtplib.SMTP(
            hostname=EMAIL_HOST,
            port=EMAIL_PORT,
            use_tls=EMAIL_USE_TLS
        ) as smtp:
            await smtp.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
            await smtp.send_message(msg)
            
        logger.info(f"Email sent successfully to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        raise EmailError(f"Failed to send email: {str(e)}")

async def send_verification_email(email: str, user_id: int) -> None:
    """Send an email with a verification link."""
    token = generate_verification_token(user_id)
    verification_link = f"https://askneta.com/verify?token={token}"
    
    try:
        await send_email_async(
            to_email=email,
            subject="Verify Your Email - Neta Ai",
            template_name="verification_email",
            template_data={
                "verification_link": verification_link,
                "expiry_minutes": 60
            }
        )
    except EmailError as e:
        logger.error(f"Failed to send verification email: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to send verification email. Please try again later."
        )

async def send_reset_email(email: str, token: str) -> None:
    """Send a password reset email with a token link"""
    reset_link = f"https://askneta.com/reset-password?token={token}"
    
    try:
        await send_email_async(
            to_email=email,
            subject="Reset Your Password - Neta Ai",
            template_name="reset_password_email",
            template_data={
                "reset_link": reset_link,
                "expiry_minutes": 60
            }
        )
    except EmailError as e:
        logger.error(f"Failed to send reset email: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to send password reset email. Please try again later."
        )

# Update the endpoints to use async email sending
@app.post("/register", response_model=schemas.UserResponse)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = db.query(models.User).filter(models.User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = auth.hash_password(user.password)
        new_user = models.User(
            name=user.name,
            email=user.email,
            password=hashed_password,
            refferl=user.refferl,
            phone_number=user.phone_number,
            is_verified=False
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Create an empty profile for the new user
        db_profile = models.Profile(user_id=new_user.id)
        db.add(db_profile)
        db.commit()

        # Send verification email asynchronously
        asyncio.create_task(send_verification_email(new_user.email, new_user.id))

        return new_user
    except Exception as e:
        db.rollback()
        logger.error(f"Error in user registration: {e}")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")



def generate_reset_token(user_id: int):
    
    expiration_time = datetime.utcnow() + timedelta(minutes=60)
    payload = {"user_id": user_id, "exp": expiration_time}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

app = FastAPI()


def generate_verification_token(user_id: int):
    """Generate a JWT token with a 60-minute expiration"""
    expiration_time = datetime.utcnow() + timedelta(minutes=60)
    payload = {"user_id": user_id, "exp": expiration_time}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

app = FastAPI()

@app.post("/forgot-password")
async def forgot_password(email: str, db: Session = Depends(get_db)):
    """Send password reset link to user email"""
    try:
        user = db.query(models.User).filter(models.User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        token = generate_reset_token(user.id)
        await send_reset_email(email, token)
        
        return {"message": "Password reset link sent to your email"}
    except Exception as e:
        logger.error(f"Error in forgot password: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process password reset request. Please try again."
        )


# Create email templates directory and files
def create_email_templates():
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Verification email template
    verification_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .button { 
                display: inline-block;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }
            .footer { margin-top: 20px; font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Welcome to Neta Ai!</h2>
            <p>Thank you for registering. Please verify your email address by clicking the button below:</p>
            <p>
                <a href="{{ verification_link }}" class="button">Verify Email</a>
            </p>
            <p>This link will expire in {{ expiry_minutes }} minutes.</p>
            <p>If you didn't sign up for this account, please ignore this email.</p>
            <div class="footer">
                <p>This is an automated message, please do not reply.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Reset password email template
    reset_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .button { 
                display: inline-block;
                padding: 10px 20px;
                background-color: #2196F3;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }
            .footer { margin-top: 20px; font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Password Reset Request</h2>
            <p>You have requested to reset your password. Click the button below to proceed:</p>
            <p>
                <a href="{{ reset_link }}" class="button">Reset Password</a>
            </p>
            <p>This link will expire in {{ expiry_minutes }} minutes.</p>
            <p>If you didn't request a password reset, please ignore this email.</p>
            <div class="footer">
                <p>This is an automated message, please do not reply.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write templates to files
    (templates_dir / "verification_email.html").write_text(verification_template)
    (templates_dir / "reset_password_email.html").write_text(reset_template)

# Create email templates on startup
create_email_templates()

@app.get("/verify")
async def verify(token: str, db: Session = Depends(get_db)):
    """Verify user using JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload["user_id"]
        
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user.is_verified = True
        db.commit()
        
        return {"message": "User verified successfully!"}

    except ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Verification link has expired.")
    except InvalidTokenError:
        raise HTTPException(status_code=400, detail="Invalid verification token.")



# user Register end point 

@app.post("/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not auth.verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not db_user.is_verified:
        # Resend verification email if not verified
        try:
            send_verification_email(db_user.email, db_user.id)
            raise HTTPException(
                status_code=403, 
                detail="Account not verified. A new verification email has been sent."
            )
        except Exception as e:
            logger.error(f"Failed to resend verification email: {e}")
            raise HTTPException(
                status_code=403,
                detail="Account not verified. Please contact support for verification."
            )

    return {
        "message": "Login Successfully!",
        "user_id": db_user.id
    }

# user Forgot Password  with email  end point 
@app.post("/forgot-password")
def forgot_password(email: str, db: Session = Depends(get_db)):
    """Send password reset link to user email"""
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    token = generate_reset_token(user.id)
    send_reset_email(email, token)
    
    return {"message": "Password reset link sent to your email"}



#user Reset Password with Token 
@app.post("/reset-password")
def reset_password(token: str, new_password: str, db: Session = Depends(get_db)):
    """Reset password using a valid token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload["user_id"]
        
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Hash the new password and update it
        hashed_password = auth.hash_password(new_password)
        user.password = hashed_password
        db.commit()
        
        return {"message": "Password updated successfully!"}

    except ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Reset link has expired.")
    except InvalidTokenError:
        raise HTTPException(status_code=400, detail="Invalid reset token.")


# Get user 
@app.get("/user/{id}",response_model=schemas.UserUpdate)
def get_user(id:int,db:Session=Depends(get_db)):
    user =db.query(models.User).filter(models.User.id==id).first()
    if  not user:
        raise HTTPException(status_code=404,detail="user not found")
    return user

#Get user profile
# @app.get("/profile/{user_id}",response_model=schemas.ProfileUpdate)
# def get_profile(user_id:int,db:Session=Depends(get_db)):
#     profile = db.query(models.Profile).filter(models.Profile.user_id == user_id).first()
    
#     if not profile:
#         raise HTTPException(status_code=404,detail="profile not found")
#     return profile
from models import Profile

@app.get("/profile/{user_id}", response_model=schemas.ProfileOut)
def get_profile(user_id: int, db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id == user_id).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return profile

#update user profile
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
from fastapi import UploadFile, File, Form

# Helper function to safely parse optional integers
def parse_optional_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value and value.strip() != "" else None
    except ValueError:
        return None




@app.put("/user/update-password")
def update_password(payload: schemas.PasswordUpdate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not auth.verify_password(payload.old_password, user.password):
        raise HTTPException(status_code=401, detail="Old password is incorrect")

    # Hash new password
    hashed_password = auth.get_password_hash(payload.new_password)
    user.password = hashed_password

    db.commit()
    return {"message": "Password updated successfully"}

@app.put("/profile/update/{user_id}")
async def update_profile(
    user_id: int,
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    refferl: Optional[str] = Form(None),
    profile_img: Optional[UploadFile] = File(None),
    website: Optional[str] = Form(None),
    org_name: Optional[str] = Form(None),
    number_of_electricians: Optional[str] = Form(None),
    where_to_get_esupplies: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    license_number: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        profile = db.query(models.Profile).filter(models.Profile.user_id == user_id).first()
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Handle file upload
        if profile_img:
            filename = await validate_file(profile_img)
            if filename:
                upload_dir = Path("static/uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = upload_dir / filename
                async with aiofiles.open(file_path, 'wb') as f:
                    while chunk := await profile_img.read(8192):
                        await f.write(chunk)
                
                # Delete old profile image if exists
                if profile.profile_img:
                    old_path = Path(profile.profile_img)
                    if old_path.exists():
                        old_path.unlink()
                
                profile.profile_img = str(file_path)

        # Update other fields
        if website is not None:
            profile.website = website
        if org_name is not None:
            profile.org_name = org_name
        if number_of_electricians is not None:
            profile.number_of_electricians = parse_optional_int(number_of_electricians)
        if where_to_get_esupplies is not None:
            profile.where_to_get_esupplies = where_to_get_esupplies
        if address is not None:
            profile.address = address
        if license_number is not None:
            profile.license_number = license_number

        # Update user fields
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if user:
            if name is not None:
                user.name = name
            if email is not None:
                # Check if email is already taken
                existing_user = db.query(models.User).filter(
                    models.User.email == email,
                    models.User.id != user_id
                ).first()
                if existing_user:
                    raise HTTPException(status_code=400, detail="Email already taken")
                user.email = email
            if refferl is not None:
                user.refferl = refferl

        db.commit()
        db.refresh(profile)
        return {"message": "Profile updated successfully", "profile_id": profile.id}

    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(status_code=500, detail="Error updating profile")



from models import ChatSession
from schemas import ChatSessionCreate,ChatSessionOut
@app.post("/create_chat_session", response_model=ChatSessionOut)
def create_chat_session(payload: ChatSessionCreate, db: Session = Depends(get_db)):
    new_session = ChatSession(user_id=payload.user_id, session_name=payload.session_name)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return ChatSessionOut(
        session_id=new_session.id,
        session_name=new_session.session_name,
        created_at=new_session.created_at
    )


@app.get("/list_chat_sessions/{user_id}", response_model=List[ChatSessionOut])
def list_chat_sessions(user_id: int, db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).filter_by(user_id=user_id).order_by(ChatSession.created_at.desc()).all()
    return [
        ChatSessionOut(
            session_id=s.id,
            session_name=s.session_name,
            created_at=s.created_at
        )
        for s in sessions
    ]


@app.delete("/delete_chat_session/{session_id}")
def delete_chat_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    db.delete(session)
    db.commit()
    return {"message": "Chat session deleted"}



@app.get("/chat_history/")
def get_chat_history(
    user_id: int = Query(...),
    session_id: int = Query(...),
    db: Session = Depends(get_db)
):
    chat_session = db.query(models.ChatSession).filter_by(user_id=user_id, id=session_id).first()

    if not chat_session or not chat_session.chats:
        return {"message": "No chat history found", "history": []}

    return {
        "user_id": user_id,
        "session_id": session_id,
        "chat_history": chat_session.chats
    }


@app.get("/subscription/{user_id}")
def get_subscription_data(user_id: int, db: Session = Depends(get_db)):
    subscription = db.query(models.Subscription).filter_by(user_id=user_id).first()

    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    return {
        "user_id": user_id,
        "subscription_data": subscription.subscription_data
    }
# @app.get("/chat_history/")
# def get_chat_history(user_id: int = Query(...), db: Session = Depends(get_db)):
#     chat_history = db.query(models.ChatHistory).filter(models.ChatHistory.user_id == user_id).first()

#     if not chat_history or not chat_history.chats:
#         return {"message": "No chat history found", "history": []}

#     return {
#         "user_id": user_id,
#         "chat_history": chat_history.chats
#     }
# Create Payment Endpoint This endpoint will create a Stripe Checkout session.
# @app.post("/create-checkout-session")
# def create_checkout_session():
#     return {"checkout_url": "https://buy.stripe.com/test_dR614q9Sn7OJ6qY4gh"}

from fastapi import Query
from datetime import date, timedelta

@app.post("/Basic-plan")
def choose_free_or_basic_plan(
    user_id: int,
    plan_type: str = Query(..., enum=["Basic"]),
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Define start and end dates for the 1-month free trial
    today = date.today()
    one_month_later = today + timedelta(days=30)

    if plan_type == "Basic":
        subscription_data = {
            "product": "Basic",
            "status": "active",
            "source": "manual",
            "start_date": str(today),
            "end_date": str(one_month_later)
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid plan type")

    # Check if user already has a subscription
    existing_sub = db.query(models.Subscription).filter_by(user_id=user.id).first()
    if existing_sub:
        existing_sub.subscription_data = subscription_data
    else:
        new_sub = models.Subscription(user_id=user.id, subscription_data=subscription_data)
        db.add(new_sub)

    db.commit()

    return {
        "status": "success",
        "message": f"{plan_type} plan activated successfully with 1-month free trial",
        "subscription_data": subscription_data
    }



@app.post("/create-checkout-Student-Discount-tier")
def create_checkout_session_student(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check for existing subscription
    existing_sub = db.query(models.Subscription).filter(models.Subscription.user_id == user_id).order_by(models.Subscription.id.desc()).first()
    if existing_sub:
        product = existing_sub.subscription_data.get("product", "").lower()
        status = existing_sub.subscription_data.get("status", "").lower()

        if product in ["student discount tier", "pro tier"] and status == "active":
            raise HTTPException(status_code=400, detail=f"You already have a subscription to '{product}'")

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": "price_1RIjgWAkiLZQygvDHrDNfSzZ", "quantity": 1}],
            mode="subscription",
            success_url="https://askneta.com/payment-success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://askneta.com/dashboard/subscription",
            metadata={"user_id": str(user.id)}
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




# Add proper Stripe error handling and subscription management
class SubscriptionError(Exception):
    pass

async def handle_stripe_error(func):
    try:
        return await func()
    except StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

def update_subscription_status(db: Session, user_id: int, subscription_data: dict):
    try:
        subscription = db.query(models.Subscription).filter(
            models.Subscription.user_id == user_id
        ).first()
        
        if subscription:
            subscription.subscription_data = subscription_data
        else:
            subscription = models.Subscription(
                user_id=user_id,
                subscription_data=subscription_data
            )
            db.add(subscription)
        
        db.commit()
        return subscription
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating subscription: {e}")
        raise SubscriptionError("Failed to update subscription")

@app.post("/create-checkout-session")
async def create_checkout_session_pro(user_id: int, db: Session = Depends(get_db)):
    async def _create_session():
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check for existing subscription
        existing_sub = db.query(models.Subscription).filter(
            models.Subscription.user_id == user_id
        ).order_by(models.Subscription.id.desc()).first()
        
        if existing_sub:
            product = existing_sub.subscription_data.get("product", "").lower()
            status = existing_sub.subscription_data.get("status", "").lower()
            
            if product == "pro tier" and status == "active":
                raise HTTPException(
                    status_code=400,
                    detail="You already have an active Pro subscription"
                )

        try:
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": "price_1RIjg4AkiLZQygvDvqn9t6FY", "quantity": 1}],
                mode="subscription",
                success_url="https://askneta.com/payment-success?session_id={CHECKOUT_SESSION_ID}",
                cancel_url="https://askneta.com/dashboard/subscription",
                metadata={"user_id": str(user.id)},
                customer_email=user.email,  # Pre-fill customer email
                allow_promotion_codes=True,  # Allow promo codes
            )
            return {"checkout_url": session.url}
        except StripeError as e:
            logger.error(f"Stripe error creating checkout session: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    return await handle_stripe_error(_create_session)

@app.post("/stripe-webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        logger.error("Invalid payload")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error("Invalid signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            user_id = int(session.get("metadata", {}).get("user_id"))
            subscription_id = session.get("subscription")

            if not user_id or not subscription_id:
                raise HTTPException(status_code=400, detail="Missing user_id or subscription_id")

            # Update subscription metadata
            stripe.Subscription.modify(
                subscription_id,
                metadata={"user_id": str(user_id)}
            )

        elif event["type"] in ["invoice.paid", "invoice.payment_failed"]:
            invoice = event["data"]["object"]
            subscription_id = invoice.get("subscription")
            
            if subscription_id:
                subscription = stripe.Subscription.retrieve(subscription_id)
                user_id = int(subscription.metadata.get("user_id", 0))
                
                if user_id:
                    subscription_data = {
                        "subscription_id": subscription_id,
                        "status": subscription.status,
                        "current_period_end": datetime.fromtimestamp(
                            subscription.current_period_end
                        ).isoformat(),
                        "cancel_at_period_end": subscription.cancel_at_period_end,
                    }
                    
                    if event["type"] == "invoice.paid":
                        subscription_data["last_payment_status"] = "succeeded"
                    else:
                        subscription_data["last_payment_status"] = "failed"
                    
                    update_subscription_status(db, user_id, subscription_data)

        elif event["type"] == "customer.subscription.deleted":
            subscription = event["data"]["object"]
            user_id = int(subscription.metadata.get("user_id", 0))
            
            if user_id:
                subscription_data = {
                    "subscription_id": subscription.id,
                    "status": "canceled",
                    "canceled_at": datetime.fromtimestamp(
                        subscription.canceled_at
                    ).isoformat() if subscription.canceled_at else None,
                }
                update_subscription_status(db, user_id, subscription_data)

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cancel-subscription/{user_id}")
async def cancel_subscription(user_id: int, db: Session = Depends(get_db)):
    async def _cancel_subscription():
        subscription = db.query(models.Subscription).filter(
            models.Subscription.user_id == user_id
        ).first()
        
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")
        
        subscription_id = subscription.subscription_data.get("subscription_id")
        if not subscription_id:
            raise HTTPException(status_code=400, detail="Invalid subscription")
        
        try:
            # Cancel at period end
            stripe_sub = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True
            )
            
            subscription_data = subscription.subscription_data
            subscription_data.update({
                "status": stripe_sub.status,
                "cancel_at_period_end": True,
                "canceled_at": datetime.fromtimestamp(
                    stripe_sub.canceled_at
                ).isoformat() if stripe_sub.canceled_at else None,
            })
            
            update_subscription_status(db, user_id, subscription_data)
            return {"message": "Subscription will be canceled at the end of the billing period"}
            
        except StripeError as e:
            logger.error(f"Stripe error canceling subscription: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    return await handle_stripe_error(_cancel_subscription)





#admin api 


#admin Register


ALGORITHM = "HS256"


from datetime import datetime, timedelta  # ✅ Fix the import

@app.post("/admin-login")
def admin_login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not auth.verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not db_user.is_verified:
        raise HTTPException(status_code=403, detail="Account not verified")
    if not db_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    access_token = auth.create_access_token(
        {"user_id": db_user.id, "is_admin": db_user.is_admin},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {
        "message": "Login successful!",
        "access_token": access_token,
        "token_type": "bearer"
    }


# Get All user fron user table
@app.get("/admin/users", response_model=List[schemas.User])
def get_users(
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.get_admin),
):
    users = (
        db.query(models.User)
          .options(selectinload(models.User.subscriptions))  # plural
          .filter(models.User.is_admin == False)
          .all()
    )
    return users



from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

 #GET Admin Profile
@app.get("/admin-profile", response_model=schemas.AdminProfile)
def get_profile(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    return current_user



# UPDATE Admin Profile
from fastapi.responses import JSONResponse
@app.put("/admin/profile/update")
def update_profile(
    data: schemas.AdminProfileUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    user = db.query(models.User).filter(models.User.id == current_user.id).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    updated = False  # Track if anything was updated

    if data.name is not None and data.name != user.name:
        user.name = data.name
        updated = True
    if data.email is not None and data.email != user.email:
        user.email = data.email
        updated = True
    if data.phone_number is not None and data.phone_number != user.phone_number:
        user.phone_number = data.phone_number
        updated = True

    if not updated:
        return JSONResponse(status_code=400, content={"message": "No changes detected in profile."})

    db.commit()
    db.refresh(user)

    return {"message": "Profile updated successfully", "user": {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "phone_number": user.phone_number
    }}


@app.post("/create/new/users/", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(models.User).filter(models.User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(
        name=user.name,
        email=user.email,
        password=auth.hash_password(user.password),
        phone_number=user.phone_number,
        is_verified=True  # set directly here too
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user



from datetime import datetime
@app.get("/total/prompts/", response_model=list[int])
def get_prompt_counts(
    start_date: datetime = Query(None),
    end_date: datetime = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(models.Prompt.number_of_prompts)

    if start_date and end_date:
        # Include full day for end_date
        end_date += timedelta(days=1)
        query = query.filter(models.Prompt.created_at >= start_date, models.Prompt.created_at < end_date)

    results = query.all()

    # Flatten from [(7,), (2,), ...] to [7, 2, ...]
    return [r[0] for r in results]



@app.delete("/admin/user/{user_id}", status_code=200)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)  # Ensure admin token
):
    # Ensure the user performing the deletion is an admin
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="You are not authorized to perform this action")

    user = db.query(models.User).filter(models.User.id == user_id).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Optional: Prevent deleting other admins
    if user.is_admin:
        raise HTTPException(status_code=400, detail="Cannot delete another admin user")

    db.delete(user)
    db.commit()

    return {"message": f"User has been deleted successfully."}





import uuid


@app.post("/admin/subscription")
def create_subscription(
    data: schemas.AdminSubscriptionCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),  # Auth required
):
    # Optional: Check if current user is admin
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="You are not authorized to perform this action")

    existing = db.query(models.Subscription).filter(models.Subscription.user_id == data.user_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Subscription for this user already exists.")

    # Mock values for demonstration – Replace with Stripe logic if needed
    mock_customer_id = f"cus_{uuid.uuid4().hex[:12]}"
    mock_subscription_id = f"sub_{uuid.uuid4().hex[:12]}"

    subscription_data = {
        "status": "active",
        "product": data.product.value,
        "customer": mock_customer_id,
        "subscription_id": mock_subscription_id
    }

    new_subscription = models.Subscription(
        user_id=data.user_id,
        subscription_data=subscription_data
    )

    db.add(new_subscription)
    db.commit()
    db.refresh(new_subscription)

    return {
        "message": "Subscription created successfully",
        "user_id": data.user_id,
        "product": data.product.value,
        "subscription_data": subscription_data
    }





from models import User


@app.post("/submit")
def submit_form(
    form_data: schemas.SubmissionCreate,
    db: Session = Depends(get_db),
):
    # 1. Make sure the user exists
    user = db.query(models.User).filter(User.id == form_data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Create the submission
    submission = models.Submission(**form_data.dict())
    db.add(submission)
    db.commit()
    db.refresh(submission)

    # 3. Return success + data
    return JSONResponse(
        content={
            "message": "Your request has been submitted successfully",
            "submission": {
                "id": submission.id,
                "email": submission.email,
                "reason": submission.reason,
                "user_id": submission.user_id,
            },
        }
    )


def admin_required(current_user:models.User = Depends(auth.get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
@app.get("/admin/submissions", response_model=List[schemas.SubmissionData])
def list_submissions(
    db: Session = Depends(get_db),
    _:models.User = Depends(admin_required)
):
    return db.query(models.Submission).all()




@app.delete("/admin/submissions/{submission_id}", status_code=204)
def delete_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    _:models.User = Depends(admin_required)
):
    submission = db.query(models.Submission).filter(models.Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    db.delete(submission)
    db.commit()
    return JSONResponse(
        content={
            "message": "Delete successfully",
            
        }
    )






#Reports


@app.post("/reports",response_model=schemas.ReportResponse)
def create_report(

    report:schemas.ReportCreate,
    db: Session = Depends(get_db),
):
    User = db.query(models.User).filter(models.User.id  == report.user_id).first()
    if not User:
        raise HTTPException(status_code=404,detail="user not found")
    
    new_report=models.Report(
        user_id=report.user_id,
        report_data=report.report_data
    )
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    return new_report




#for admin

@app.get("/admin/reports",response_model=List[schemas.ReportResponse])
def get_all_reports(
    db:Session = Depends(get_db),
    _:User = Depends(admin_required)
):
    return db.query(models.Report).all()








# AI Model and RAG Integration
class ModelConfig:
    # Your existing model's endpoint
    EXISTING_MODEL_URL = os.getenv("EXISTING_MODEL_URL", "http://localhost:8000/model")
    # RAG system endpoint
    RAG_SYSTEM_URL = os.getenv("RAG_SYSTEM_URL", "http://localhost:8001/rag")

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

class RAGContext(BaseModel):
    query: str
    relevant_docs: List[Dict[str, Any]]
    confidence: float

class ModelRequest(BaseModel):
    message: str
    context: Optional[RAGContext] = None
    user_id: int
    session_id: int

class ModelResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None
    confidence: float
    model_used: str = "existing_model"  # To track which model was used

async def get_rag_context(query: str) -> RAGContext:
    """Get relevant context from RAG system for the query"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{ModelConfig.RAG_SYSTEM_URL}/retrieve",
                json={"query": query},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return RAGContext(**data)
        except httpx.HTTPError as e:
            logger.error(f"RAG system error: {e}")
            # Return empty context if RAG fails, but don't fail the request
            return RAGContext(
                query=query,
                relevant_docs=[],
                confidence=0.0
            )

async def get_model_response(request: ModelRequest) -> ModelResponse:
    """Get response from the existing trained model with RAG context"""
    async with httpx.AsyncClient() as client:
        try:
            # Prepare the request to your existing model
            model_request = {
                "message": request.message,
                "context": request.context.dict() if request.context else None,
                "user_id": request.user_id,
                "session_id": request.session_id
            }
            
            # Call your existing model
            response = await client.post(
                f"{ModelConfig.EXISTING_MODEL_URL}/predict",
                json=model_request,
                timeout=30.0
            )
            response.raise_for_status()
            return ModelResponse(**response.json())
            
        except httpx.HTTPError as e:
            logger.error(f"Model API error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error communicating with AI model"
            )

@app.post("/chat", response_model=ModelResponse)
async def chat_with_model(
    request: ModelRequest,
    db: Session = Depends(get_db)
):
    """Chat endpoint that integrates existing model with RAG system"""
    try:
        # Verify user and session
        user = db.query(models.User).filter(models.User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        session = db.query(models.ChatSession).filter(
            models.ChatSession.id == request.session_id,
            models.ChatSession.user_id == request.user_id
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Get context from RAG system
        rag_context = await get_rag_context(request.message)
        
        # Get response from existing model with RAG context
        model_response = await get_model_response(
            ModelRequest(
                message=request.message,
                context=rag_context,
                user_id=request.user_id,
                session_id=request.session_id
            )
        )

        # Store chat in database
        chat_data = {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if not session.chats:
            session.chats = []
            
        session.chats.append(chat_data)
        
        assistant_chat = {
            "role": "assistant",
            "content": model_response.response,
            "sources": model_response.sources,
            "confidence": model_response.confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        session.chats.append(assistant_chat)
        
        db.commit()

        return model_response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error processing chat request"
        )

@app.get("/model-status")
async def check_model_status():
    """Check if the existing model and RAG system are available"""
    try:
        async with httpx.AsyncClient() as client:
            # Check existing model status
            model_response = await client.get(f"{ModelConfig.EXISTING_MODEL_URL}/health")
            model_status = model_response.status_code == 200

            # Check RAG system status
            rag_response = await client.get(f"{ModelConfig.RAG_SYSTEM_URL}/health")
            rag_status = rag_response.status_code == 200

            return {
                "existing_model_status": "healthy" if model_status else "unhealthy",
                "rag_system_status": "healthy" if rag_status else "unhealthy",
                "overall_status": "healthy" if (model_status and rag_status) else "unhealthy"
            }
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        return {
            "existing_model_status": "unhealthy",
            "rag_system_status": "unhealthy",
            "overall_status": "unhealthy",
            "error": str(e)
        }

# Add to your .env file:
"""
EXISTING_MODEL_URL=http://your-existing-model-url
RAG_SYSTEM_URL=http://your-rag-system-url
"""
