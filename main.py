from pathlib import Path 
import joblib 
import pandas as pd 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field 

# Run Code: python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload 

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "wqm_random_forest.joblib" 

model = joblib.load(MODEL_PATH) 

app = FastAPI(
    title = "Water Quality Monitoring ML API", 
    version = "1.0.0"
)

class WaterQualityInput(BaseModel):
    temp: float = Field(description="Suhu air dalam Celsius")
    ph : float = Field(description = "Nilai pH air")
    turbidity_cm : float = Field(
            description= "Turbidity dalam satuan cm"
        )

CLASS_LABELS = {
    0 : "Excellent", 
    1 : "Good", 
    2 : "Poor"
}

@app.get("/health")
def health_check():
    return {"status": "UP", "model_loaded": True} 

@app.post("/predict")
def predict_water_quality(data: WaterQualityInput):
    try:
        input_df = pd.DataFrame([{
            "Temp": data.temp, 
            "pH`": data.ph,
            "Turbidity (cm)": data.turbidity_cm
        }])

        prediction = int(model.predict(input_df)[0])

        return {
            "predicted_class": prediction, 
            "water_quality": CLASS_LABELS.get(
                prediction, "Unknown"
            )
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(exc)}"
        )

@app.get("/model-info")
def model_info():
    return {
        "features": list(model.feature_names_in_)
    }