import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import CORS Middleware
from pydantic import BaseModel
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# --- ADD THIS SECTION ---
# This enables CORS and tells the browser that requests from your
# Vercel frontend are allowed.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------


# Define the structure of the input data using Pydantic
class UserFeatures(BaseModel):
    bank_transaction_average: float
    social_media_screentime: float
    ecommerce_screen_time: float
    cibil_score: int
    geographical_movement: float
    social_media_reach: int

# Load the trained model pipeline
pipeline = joblib.load('stacked_ensemble_pipeline.pkl')

@app.get("/")
def read_root():
    return {"message": "Credit Risk Analysis API is running."}


@app.post("/api/predict")
def predict_loan_status(features: UserFeatures):
    """
    Predicts loan approval based on user features.
    """
    # Convert the input data into a pandas DataFrame
    # The feature names must match exactly what the model was trained on.
    feature_cols = [
        'Bank transaction average(per month)', 'social media screentime',
        'e-commerce screen time', 'CIBIL score', 'geographical movement',
        'social media reach'
    ]
    
    df = pd.DataFrame([[
        features.bank_transaction_average,
        features.social_media_screentime,
        features.ecommerce_screen_time,
        features.cibil_score,
        features.geographical_movement,
        features.social_media_reach
    ]], columns=feature_cols)

    # Use the loaded pipeline to make predictions
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[:, 1][0]

    # Return the results in a JSON format
    return {
        "prediction": int(prediction),
        "probability_of_approval": float(probability)
    }

# This part is for local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

