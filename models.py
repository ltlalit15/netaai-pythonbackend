from sqlalchemy import JSON, Boolean ,Column, ForeignKey ,Integer,String,DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# class User(Base):
#     __tablename__='users'
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(255), nullable=False)
#     email = Column(String(255), unique=True, nullable=False)
#     password = Column(String(255), nullable=False)
#     phone_number = Column(String(20), nullable=False)
#     is_verified = Column(Boolean, default=False) 
#     first_login = Column(Boolean, default=True)
#     stripe_customer_id = Column(String, unique=True)

#     profile = relationship("Profile", back_populates="user", uselist=False)
#     subscription = relationship("Subscription", back_populates="user", uselist=False)
#     chat_history = relationship("ChatHistory", back_populates="user", uselist=False)
#     # chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    
    
class User(Base):
    __tablename__='users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    phone_number = Column(String(20), nullable=False)
    is_verified = Column(Boolean, default=False) 
    first_login = Column(Boolean, default=True)
    stripe_customer_id = Column(String(255), unique=True)
    is_admin=Column(Boolean,default=False)
    refferl = Column(String(255), nullable=True)
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    #profile = relationship("Profile", back_populates="user", uselist=False)
    profile = relationship("Profile", back_populates="user", cascade="all, delete-orphan")
    #subscription = relationship("Subscription", back_populates="user", uselist=False)
    subscriptions = relationship("Subscription",back_populates="user",cascade="all, delete-orphan",)
    chat_history = relationship("ChatHistory", back_populates="user", uselist=False)
    # chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    submissions = relationship("Submission", back_populates="user",cascade="all, delete-orphan",)
    reports = relationship("Report", back_populates="user", cascade="all, delete")       
    
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_name = Column(String(255), default="New Chat")  # Optional title per session
    chats = Column(JSON, default=[])  # List of {"user": "...", "bot": "..."}
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chat_sessions")       
    
class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    profile_img = Column(String(255))
    website = Column(String(255))
    org_name = Column(String(255))
    number_of_electricians = Column(Integer)
    where_to_get_esupplies = Column(String(255))
    address = Column(String(255))
    license_number=Column(String(255))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    user = relationship("User", back_populates="profile")    
    
    
class ChatHistory(Base):
    __tablename__ = "chat_histories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    chats = Column(JSON)  # Storing chats as a JSON object

    user = relationship("User", back_populates="chat_history") 
    
    
class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    subscription_data = Column(JSON)  # Store subscription details as JSON
    user = relationship("User",back_populates="subscriptions") 
    #user = relationship("User", back_populates="subscription")    

   



class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    number_of_prompts = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)




class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    user = relationship("User", back_populates="submissions")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    report_data = Column(JSON)  # Store report details as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="reports")
