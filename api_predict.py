
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn


model = joblib.load("XGBoost_model.joblib") 

app = FastAPI()

class InputData(BaseModel):
    vibration_level: float
    temperature_C: float
    pressure_PSI: float
    flow_rate_m3h: float
    hour: int
    day_of_week: int
    week_of_month: int
    vibration_level_rolling_mean: float
    temperature_C_rolling_mean: float
    pressure_PSI_rolling_mean: float
    flow_rate_m3h_rolling_mean: float
    vibration_trend: float
    temp_trend: float
    pressure_trend: float
    flow_trend: float

# Define the POST endpoint
@app.post("/predict")
def predict(data: InputData):

    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    return {
        "prediction": int(prediction[0]),
        "prediction_proba": float(prediction_proba[0])
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)