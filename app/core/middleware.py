from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import time
from typing import Dict, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta

from app.core.config import settings
from app.core.security import get_current_user, TokenData

logger = logging.getLogger(__name__)

# In-memory store for rate limiting
# In production, you'd want to use Redis or another distributed cache
rate_limit_store: Dict[str, Tuple[int, datetime]] = {}

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API rate limiting based on client IP and user role.
    
    This implementation uses an in-memory store for simplicity.
    For production, consider using Redis or another distributed cache.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for certain paths
        if request.url.path.startswith("/api/v1/health"):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host
        
        # Check for Authorization header to identify user role
        auth_header = request.headers.get("Authorization")
        rate_limit = settings.RATE_LIMIT_PER_MINUTE
        
        # Try to extract user role from token if present
        user_role = None
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.replace("Bearer ", "")
                # This is a simplified approach - in a real implementation,
                # you'd want to use the proper token validation logic
                from app.core.security import verify_token
                payload = verify_token(token)
                user_role = payload.role
                
                # Admin users get higher rate limits
                if user_role == "admin":
                    rate_limit = settings.ADMIN_RATE_LIMIT_PER_MINUTE
            except Exception as e:
                # If token validation fails, use default rate limit
                logger.warning(f"Token validation failed in rate limiting: {str(e)}")
        
        # Rate limit key combines IP and role for more granular control
        rate_limit_key = f"{client_ip}:{user_role or 'anonymous'}"
        
        # Check if rate limited
        current_time = datetime.now()
        if rate_limit_key in rate_limit_store:
            count, window_start = rate_limit_store[rate_limit_key]
            
            # If window has expired, reset the counter
            window_size = timedelta(minutes=1)
            if current_time - window_start > window_size:
                rate_limit_store[rate_limit_key] = (1, current_time)
            # Otherwise increment and check limit
            else:
                if count >= rate_limit:
                    headers = {
                        "X-RateLimit-Limit": str(rate_limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int((window_start + window_size).timestamp())),
                        "Retry-After": str(int((window_start + window_size - current_time).total_seconds()))
                    }
                    return Response(
                        content="Rate limit exceeded",
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        headers=headers
                    )
                rate_limit_store[rate_limit_key] = (count + 1, window_start)
        else:
            rate_limit_store[rate_limit_key] = (1, current_time)
        
        # Add rate limit headers
        count, _ = rate_limit_store[rate_limit_key]
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, rate_limit - count))
        
        return response

def add_middleware(app: FastAPI) -> None:
    """Add all middleware to the FastAPI application."""
    app.add_middleware(RateLimitMiddleware) 