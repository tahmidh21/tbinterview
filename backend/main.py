# Imports for FastAPI, Pydantic, SQLAlchemy, joblib, and standard library components
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import joblib
import json
from pathlib import Path
import os
import warnings

# Suppress standard sklearn warnings about feature names during prediction
warnings.filterwarnings("ignore", category=UserWarning)

# Set up paths relative to this file's location to be resilient to the CWD
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_DIR = CURRENT_DIR.parent / "model"
MODEL_PATH = MODEL_DIR / "tb_model.pkl"
FEATURES_PATH = MODEL_DIR / "features.json"
DB_PATH = CURRENT_DIR / "predictions.db"

# Load model and features once at startup
try:
    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model or features: {e}. Ensure you have trained the model first.")
    feature_cols = []
    model = None

# Set up the FastAPI application and CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLAlchemy Database Setup
engine = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Sanitize feature names for database columns and Pydantic fields
def sanitize_name(name: str) -> str:
    return name.replace(" ", "_")

sanitized_features = [sanitize_name(f) for f in feature_cols]
feature_mapping = dict(zip(feature_cols, sanitized_features))

# Define the SQLAlchemy Model for the predictions table dynamically based on features
class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    risk_label = Column(Integer)
    risk_name = Column(String)
    confidence = Column(Float)

# Dynamically add feature columns to the SQLAlchemy model
for san_feat in sanitized_features:
    setattr(PredictionRecord, san_feat, Column(Integer, default=0))

# Create tables if they do not exist
Base.metadata.create_all(bind=engine)

# Dynamically build the Pydantic BaseModel for incoming prediction requests
pydantic_fields = {san_feat: (int, 0) for san_feat in sanitized_features}
PredictRequest = create_model("PredictRequest", **pydantic_fields)

# Risk label mapping
RISK_MAPPING = {0: "Low", 1: "Medium", 2: "High"}

# Endpoint: POST /predict - to make a new prediction
@app.post("/predict")
def predict(request: PredictRequest):
    if model is None or not feature_cols:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Build the exact feature vector based on the ordered list from features.json
        req_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        vector = [req_dict[feature_mapping[feat]] for feat in feature_cols]
        
        # Predict using the trained RandomForest model
        prediction = model.predict([vector])[0]
        probabilities = model.predict_proba([vector])[0]
        
        risk_label = int(prediction)
        risk_name = RISK_MAPPING.get(risk_label, "Unknown")
        confidence = float(max(probabilities))
        
        # Save record to the database
        db = SessionLocal()
        try:
            record = PredictionRecord(
                risk_label=risk_label,
                risk_name=risk_name,
                confidence=confidence,
                timestamp=datetime.utcnow()
            )
            # Set all the feature values dynamically on the DB record
            for feat, san_feat in feature_mapping.items():
                setattr(record, san_feat, req_dict[san_feat])
            
            db.add(record)
            db.commit()
        finally:
            db.close()
            
        return {
            "risk_label": risk_label,
            "risk_name": risk_name,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint: GET /history - to get the recent 20 predictions
@app.get("/history")
def get_history():
    db = SessionLocal()
    try:
        records = db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(20).all()
        results = []
        for r in records:
            # Build the history dictionary
            rec_dict = {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "risk_label": r.risk_label,
                "risk_name": r.risk_name,
                "confidence": r.confidence
            }
            # Add all the sanitized feature columns
            for san_feat in sanitized_features:
                rec_dict[san_feat] = getattr(r, san_feat)
            results.append(rec_dict)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# Endpoint: DELETE /assessment-history - clears the prediction records
@app.delete("/assessment-history")
def clear_history():
    db = SessionLocal()
    try:
        db.query(PredictionRecord).delete()
        db.commit()
        return {"status": "success", "message": "Assessment history cleared"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# Endpoint: GET /feature-importance - return feature importance scores
@app.get("/feature-importance")
def get_feature_importance():
    importance_path = MODEL_DIR / "feature_importance.json"
    try:
        with open(importance_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="feature_importance.json not found. Retrain the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint: GET / - root health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "TB Risk Predictor API"}
