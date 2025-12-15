# src/spectrum/glass/schemas.py (create this file)
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from spectrum.infra.types import RiskLevel, RiskProfile
from spectrum.infra.privacy import PIIPolicy

# ------ DRIFT MONITORING ------
class DriftReport(BaseModel):
    """Response model matching DriftCheck output (cf blue.monitor)."""
    drift_detected: bool
    n_drifted_features: int
    feature_drift_scores: Dict[str, float]
    drifted_features: List[str]
    max_psi: float
    status: str
    alert_required: bool
    timestamp: datetime = Field(default_factory=datetime.now)

# ------ MODEL INFO ------
class ModelCoreInfo(BaseModel):
    """Core metadata about a registered model."""
    model_id: str
    version: str
    risk_profile: RiskProfile
    registered_at: datetime
    last_inference: Optional[datetime] = None
    total_inferences: int = 0

# ------ RED TEAM ------
class AdversarialTestRequest(BaseModel):
    """Parameters for triggering red team assessment."""
    attack_type: str = Field(..., description="Type of attack: 'extraction', 'poisoning', 'inference'")

class AdversarialTestResult(BaseModel):
    """Result of a red team assessment."""
    test_id: str
    model_id: str
    attack_type: str
    success_rate: float
    timestamp: datetime
    details: Dict[str, any]