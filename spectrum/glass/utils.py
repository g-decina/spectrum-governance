from pydantic import BaseModel

from spectrum.infra.types import RiskLevel

class ModelInfo(BaseModel):
    model_id: str
    version: str
    risk_level: RiskLevel