import sys
import uuid
import warnings
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from sklearn.base import BaseEstimator

from spectrum.infra.privacy import PIIPolicy

class RiskLevel(str, Enum):
    LOW = "Low"             # e.g. Marketing segmentation
    MEDIUM = "Medium"       # e.g. Fraud alert
    HIGH = "High"           # e.g. Credit denial
    CRITICAL = "Critical"   # e.g. Autonomous braking

class RiskProfile(BaseModel):
    """
    Immutable contract for risk tolerance.
    Encapsulating: P(Y in C(X) ≥ 1 - a)

    Includes privacy policy automatically determined by risk level:
    - LOW/MEDIUM: DETECT_ONLY (warn about PII but don't modify)
    - HIGH: PSEUDONYMIZE (hash PII fields)
    - CRITICAL: REDACT (remove PII fields entirely)
    """
    model_config = ConfigDict(frozen = True)

    level: RiskLevel
    alpha: float = Field(..., gt = 0.0, lt = 1.0, description = "Significance level (1 - Confidence)")
    pii_policy: Optional[PIIPolicy] = Field(default=None, description = "PII handling policy (auto-set based on risk level if not provided)")

    @field_validator("alpha")
    @classmethod
    def warn_if_loose(cls, v: float) -> float:
        if v < 0.01:
            warnings.warn(f"Alpha {v} is extremely low. Prediction sets may span the full domain.")
        return v

    @model_validator(mode = "after")
    def set_default_pii_policy(self) -> "RiskProfile":
        """Auto-set PII policy based on risk level if not explicitly provided."""
        if self.pii_policy is None:
            # Map risk levels to recommended PII policies
            policy_map = {
                RiskLevel.LOW: PIIPolicy.DETECT_ONLY,
                RiskLevel.MEDIUM: PIIPolicy.DETECT_ONLY,
                RiskLevel.HIGH: PIIPolicy.PSEUDONYMIZE,
                RiskLevel.CRITICAL: PIIPolicy.REDACT,
            }
            object.__setattr__(self, 'pii_policy', policy_map[self.level])
        return self
        
class InferenceEvent(BaseModel):
    """
    Audit Log Atom.
    Captures metadata immediately upon instantiation to avoid memory retention issues.
    """
    model_config = ConfigDict(arbitrary_types_allowed = True)
    
    # --- Identity ---
    event_id: uuid.UUID = Field(default_factory = uuid.uuid4)
    timestamp: datetime = Field(default_factory = datetime.now)
    model_version: str
    
    # --- Payload ---
    input_payload: Any = None
    output_payload: Any = None
    
    # --- Computed Metadata ---
    input_shape: Optional[List[int]] = None
    input_bytes: Optional[int] = None
    data_type: str = "unknown"
    
    @model_validator(mode = "after")
    def _compute_metadata(self) -> "InferenceEvent":
        data = self.input_payload
        
        if isinstance(data, np.ndarray):
            self.input_shape = list(data.shape)
            self.input_bytes = data.nbytes
            self.data_type = "numpy"
        
        elif isinstance(data, pd.DataFrame):
            self.input_shape = list(data.shape)
            self.input_bytes = int(data.memory_usage(deep = True).sum())
            self.data_type = "pandas"
            
        elif isinstance(data, torch.Tensor):
            self.input_shape = list(data.shape)
            self.input_bytes = data.element_size() * data.nelement()
            self.data_type = "torch"
            
        elif isinstance(data, str):
            self.input_shape = [len(data)]
            self.input_bytes = sys.getsizeof(data)
            self.data_type = "string"
            
        else:
            self.input_shape = [-1]
            self.input_bytes = sys.getsizeof(data)
            self.data_type = str(type(data))

        return self
    
class TargetModel(BaseModel):
    """
    A unified contract defining a model target. Enforces mutual exclusivity: 
    a model must be EITHER a local object OR an external API endpoint.
    """
    model_config = ConfigDict(arbitrary_types_allowed = True)
    
    # --- Tabular/Local Model Fields ---
    # Fitted Model Object Instance
    model_object: Optional[BaseEstimator] = Field(None, description = "The fitted model instance for local inspection.")
    # Metadata flag to quickly identify the type for the Wargame Runner
    is_tabular: bool = Field(False, description = "True if a local scikit-learn compatible model is provided.")
    
    # --- LLM/API Model Fields ---
    # Target endpoint for LLM/API attacks (e.g., garak)
    target_api_url: Optional[str] = Field(None, description = "The API URL for remote LLM inference/attack.")
    
    # --- MUTUAL EXCLUSIVITY VALIDATOR ---
    @model_validator(mode='after')
    def validate_exclusive_model_type(cls, data: Any) -> "TargetModel":
        is_object_set = data.model_object is not None
        is_url_set = data.target_api_url is not None
        
        if is_object_set and is_url_set:
            # Error Case 1: Both Object & API provided
            raise ValueError("TargetModel must be either a local model object OR an API URL, not both.")
        
        if not is_object_set and not is_url_set:
            # Error Case 2: Neither Object & API provided
            raise ValueError("TargetModel requires either a 'model_object' (sklearn) or a 'target_api_url' (LLM).")
            
        # Optional Cleanup: If model_object is set, enforce is_tabular = True
        if is_object_set:
            data.is_tabular = True
            
        return data