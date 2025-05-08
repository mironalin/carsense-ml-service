from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from jose import jwt
from jose.exceptions import JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.core.config import settings

# Security scheme for Swagger UI
security = HTTPBearer()

class TokenPayload(BaseModel):
    """Model representing JWT token payload."""
    sub: Optional[str] = None
    exp: Optional[int] = None
    role: Optional[str] = None
    permissions: Optional[list] = None

class TokenData(BaseModel):
    """Model representing extracted token data."""
    username: Optional[str] = None
    role: Optional[str] = None
    permissions: Optional[list] = None

def verify_token(token: str) -> TokenPayload:
    """Verify and decode JWT token."""
    try:
        # Use the same JWT secret as the main backend
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        # Check if token is expired
        if token_data.exp and datetime.fromtimestamp(token_data.exp) < datetime.now():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return token_data
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """Dependency to get current user from token."""
    token = credentials.credentials
    payload = verify_token(token)
    
    username: str = payload.sub
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract user data from token
    token_data = TokenData(
        username=username,
        role=payload.role,
        permissions=payload.permissions
    )
    
    return token_data

def check_permission(required_permission: str):
    """Dependency to check if user has specific permission."""
    def permission_checker(current_user: TokenData = Depends(get_current_user)) -> bool:
        if not current_user.permissions or required_permission not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {required_permission} required",
            )
        return True
    return permission_checker

def admin_required(current_user: TokenData = Depends(get_current_user)) -> bool:
    """Dependency to require admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return True 