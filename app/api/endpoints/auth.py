from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta
from app.core.config import settings
from typing import List, Optional

from app.core.security import get_current_user, TokenData, check_permission, admin_required

router = APIRouter()

class AuthTestResponse(BaseModel):
    """Response model for authentication test endpoints."""
    message: str
    username: str
    role: Optional[str] = None
    permissions: Optional[List[str]] = None

@router.get("/me")
async def get_user_info(current_user: TokenData = Depends(get_current_user)):
    """
    Test endpoint to verify authentication and return current user info.

    This endpoint requires a valid JWT token in the Authorization header.
    """
    return AuthTestResponse(
        message="Authentication successful",
        username=current_user.username,
        role=current_user.role,
        permissions=current_user.permissions
    )

@router.get("/admin-only")
async def admin_only_endpoint(_: bool = Depends(admin_required)):
    """
    Test endpoint that requires admin role.

    This endpoint requires a valid JWT token with admin role.
    """
    return {"message": "Admin access granted"}

@router.get("/permission-test/{permission}")
async def permission_test_endpoint(
    permission: str,
    _: bool = Depends(check_permission("read:data"))
):
    """
    Test endpoint that requires specific permission.

    This endpoint requires a valid JWT token with the 'read:data' permission.
    """
    return {"message": f"You have the required permission: read:data"}

@router.get("/protected-resource")
async def protected_resource(
    _: bool = Depends(check_permission("access:ml-models"))
):
    """
    Example endpoint for accessing a protected ML resource.

    Requires 'access:ml-models' permission in the JWT token.
    """
    return {"message": "You have access to protected ML resources"}

@router.get("/public")
async def public_endpoint():
    """
    Public endpoint that doesn't require authentication.
    """
    return {"message": "This is a public endpoint"}

# New endpoint for generating test tokens
class TestTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.get("/test-token", response_model=TestTokenResponse)
async def get_test_token(user_id: str = "test_user", role: str = "user", expires_delta_minutes: int = 60, permissions_str: Optional[str] = None):
    """
    Generate a test JWT token with a specified user_id, role, and optional permissions.
    Permissions should be a comma-separated string (e.g., "read:data,write:data").
    Useful for development and testing.
    """
    expire = datetime.utcnow() + timedelta(minutes=expires_delta_minutes)
    to_encode = {
        "sub": user_id,
        "role": role,
        "exp": expire
    }
    if permissions_str:
        to_encode["permissions"] = [p.strip() for p in permissions_str.split(',')]

    # Ensure JWT_SECRET_KEY is a string, not a Secret instance if using Pydantic's SecretStr
    # For this example, assuming it's directly usable as a string.
    secret_key = settings.JWT_SECRET_KEY.get_secret_value() if hasattr(settings.JWT_SECRET_KEY, 'get_secret_value') else settings.JWT_SECRET_KEY
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=settings.JWT_ALGORITHM)
    return TestTokenResponse(access_token=encoded_jwt)