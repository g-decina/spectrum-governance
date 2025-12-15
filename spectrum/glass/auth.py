# src/spectrum/glass/auth.py (create this file)
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated, List

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User:
    username: str
    role: str  # "legal", "tech", "admin"

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    """Validates token and returns user info."""
    # TODO: Validate token against database/JWT
    # Placeholder:
    if token == "legal_token_123":
        return User(username="jane_legal", role="legal")
    elif token == "tech_token_456":
        return User(username="john_tech", role="tech")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials"
    )

def require_role(allowed_roles: List[str]):
    """Dependency factory for role-based access control."""
    async def check_role(user: Annotated[User, Depends(get_current_user)]) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}"
            )
        return user
    return check_role