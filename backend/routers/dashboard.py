from fastapi import APIRouter, HTTPException
from backend.services.inference import model_service

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

@router.get("/scenarios")
def get_scenarios():
    """Returns a list of all available test scenarios."""
    try:
        scenarios = model_service.get_scenarios()
        return {"scenarios": scenarios}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scenario/{scenario_id}")
def get_scenario(scenario_id: int):
    """Returns the graph topology for a specific test scenario."""
    try:
        scenario = model_service.get_scenario(scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
        return scenario
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/{scenario_id}")
def predict_scenario(scenario_id: int):
    """
    Runs both GNN and SEIR inference on the given scenario.
    Both operate in the Origin-Known regime.
    """
    try:
        predictions = model_service.predict(scenario_id)
        if not predictions:
            raise HTTPException(status_code=404, detail="Scenario not found")
        return predictions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
