from pydantic import BaseModel, EmailStr
from typing import Any, Optional, List
from uuid import UUID
from datetime import datetime

class SubmissionCreate(BaseModel):
    email: EmailStr
    reason: str
    user_id: int

class SubmissionResponse(BaseModel):
    id: int
    email: EmailStr
    reason: str
    user_id: int

    class Config:
        from_attributes = True  # Updated here

class SubmissionData(BaseModel):
    id: int
    email: EmailStr
    reason: str
    user_id: int

    class Config:
        from_attributes = True  # Updated here

class AdminProfile(BaseModel):
    id: int
    name: str
    email: EmailStr
    phone_number: str

    class Config:
        from_attributes = True  # Updated here

class AdminProfileUpdate(BaseModel):
    name: Optional[str]
    email: Optional[EmailStr]
    phone_number: Optional[str]

class PromptOut(BaseModel):
    id: int
    number_of_prompts: int
    created_at: datetime

    class Config:
        from_attributes = True  # Updated here
        
from enum import Enum
class ProductTier(str, Enum):
    pro = "Pro tier"
    student = "Student Discount tier"

class AdminSubscriptionCreate(BaseModel):
    user_id: int
    product: ProductTier        

class SubscriptionData(BaseModel):
    id: int
    subscription_data: dict

    class Config:
        from_attributes = True  # Updated here

class User(BaseModel):
    id: int
    name: str
    email: str
    phone_number: str
    is_verified: bool
    subscriptions: List[SubscriptionData] = []  # same name

    class Config:
        from_attributes = True  # Updated here

class ReportCreate(BaseModel):
    user_id: int
    report_data: Any

class ReportResponse(BaseModel):
    id: int
    user_id: int
    report_data: Any
    created_at: datetime

    class Config:
        from_attributes = True  # Updated here

# simple user         

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone_number: str
    refferl: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    phone_number: str
    refferl: Optional[str] = None
    is_verified: bool

    class Config:
        from_attributes = True  # Updated here
        
class UserUpdate(BaseModel):
    name: str
    email: EmailStr
    phone_number: str
    refferl: Optional[str] = None

class ProfileUserOut(BaseModel):
    name: str
    email: str
    refferl: Optional[str] = None
    class Config:
        from_attributes = True  # Updated here
        
class ProfileOut(BaseModel):
    profile_img: Optional[str] = None
    website: Optional[str] = None
    org_name: Optional[str] = None
    number_of_electricians: Optional[int] = None
    where_to_get_esupplies: Optional[str] = None
    address: Optional[str] = None
    license_number: Optional[str] = None
    user: ProfileUserOut

    class Config:
        from_attributes = True  # Already updated

class ProfileUpdate(BaseModel):
    profile_img: Optional[str] = None
    website: Optional[str] = None
    org_name: Optional[str] = None
    number_of_electricians: Optional[int] = None
    where_to_get_esupplies: Optional[str] = None
    address: Optional[str] = None
    license_number: Optional[str] = None

# for user pass update
class PasswordUpdate(BaseModel):
    email: str
    old_password: str
    new_password: str

from uuid import UUID
from datetime import datetime

class ChatSessionCreate(BaseModel):
    user_id: int
    session_name: str = "New Chat"

class ChatSessionOut(BaseModel):
    session_id: int
    session_name: str
    created_at: datetime

    class Config:
        from_attributes = True  # Updated here
