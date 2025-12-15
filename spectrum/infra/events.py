from typing import Any, Dict

class InferenceEvent:
    """
    Audit Contract class.
    This structure defines what is immutable, sanitized
    and logged by the system.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def model_dump(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}