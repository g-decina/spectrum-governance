import os
from functools import wraps

def _with_limited_threads(func):
    """
    Simple decorator that forces MKL_NUM_THREADS=1 during execution
    of the decorated function, then restores it afterward.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_threads = os.environ.get("MKL_NUM_THREADS")

        # MKL expects string values
        os.environ["MKL_NUM_THREADS"] = "1"

        try:
            # Preserve function arguments!
            return func(*args, **kwargs)
        finally:
            # Restore or remove variable
            if original_threads is not None:
                os.environ["MKL_NUM_THREADS"] = original_threads
            else:
                os.environ.pop("MKL_NUM_THREADS", None)

    return wrapper