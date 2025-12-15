from fastapi import HTTPException
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List

# In-memory rate limiter (replace with Redis for production)
rate_limit_storage: Dict[str, List[datetime]] = defaultdict(list)

def rate_limit(max_requests: int, window_hours: int):
    """Rate limiting dependency factory."""
    async def check_rate_limit(model_id: str, user: User):
        key = f"{user.username}:{model_id}"
        now = datetime.now()
        window_start = now - timedelta(hours=window_hours)

        # Remove old requests outside the window
        rate_limit_storage[key] = [
            ts for ts in rate_limit_storage[key] if ts > window_start
        ]

        # Check if limit exceeded
        if len(rate_limit_storage[key]) >= max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {max_requests} requests per {window_hours}h"
            )

        # Record this request
        rate_limit_storage[key].append(now)

    return check_rate_limit