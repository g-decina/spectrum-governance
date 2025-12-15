from fastapi import Depends, FastAPI, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

import asyncio
from typing import List, Dict, Annotated

from spectrum.infra.privacy import PIIPolicy
from spectrum.infra.types import RiskProfile
from spectrum.glass.auth import get_current_user, require_role, User
from spectrum.glass.schemas import DriftReport, ModelCoreInfo
from spectrum.glass.utils import ModelInfo

# Q: Do you want active endpoints? Or just passive reads of pre-computed results?
# A: I want active endpoints so users can both check on their models *and* run elementary tests.

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/api/models")
async def models() -> List[ModelInfo]:
    # Purpose: List registered models
    # Response: List of models
    # Source: Loaded from artifact registry
    return None

@app.post("/api/models/{model_id}/drift/check")
async def trigger_drift_check(
    model_id: str,
    background_tasks: BackgroundTasks,
    token: Annotated[str, Depends(oauth2_scheme)]
):
    """Queues a drift check in the background. Returns immediately."""
    background_tasks.add_task(run_drift_check, model_id)
    return {"status": "queued", "model_id": model_id}

def run_drift_check(model_id: str):
    """Executes async, writes result to PostgreSQL."""
    ref_data = load_reference_data(model_id)
    curr_data = load_current_production_data(model_id)
    result = DriftCheck(ref_data, curr_data)
    save_drift_report(model_id, result)  # Save to DB

@app.get("/api/models/{model_id}", response_model=ModelCoreInfo)
async def get_model_core_info(model_id: str) -> ModelCoreInfo:
    # Load from PostgreSQL
    raise NotImplementedError("Core reporting not yet connected to database")

@app.get("/api/models/{model_id}/drift/latest", response_model=DriftReport)
async def get_latest_drift_report(model_id: str) -> DriftReport:
    # Load from PostgreSQL
    raise NotImplementedError("Drift reporting not yet connected to database")

@app.post("/api/privacy")
async def post_modify_privacy_settings(
    model_id: str,
    new_profile: RiskProfile,
    user: Annotated[User, Depends(require_role(["legal", "admin"]))]  # ✓ Enforced
):
    # Only legal/admin can reach here
    raise NotImplementedError("Privacy settings modification not yet implemented")

@app.post("/api/models/{model_id}/red-team/attack")
async def post_run_adversarial_test(
    model_id: str,
    user: Annotated[User, Depends(require_role(["tech", "admin"]))],
    _: None = Depends(rate_limit(max_requests=4, window_hours=24))  # ✓ Enforced
):
    # Trigger background task
    raise NotImplementedError("Adversarial testing request not yet implemented")