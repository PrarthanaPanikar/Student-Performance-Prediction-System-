"""
FastAPI Inference Service for Student Performance Prediction
Exposes RESTful endpoints for model predictions and explanations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student performance and providing risk assessments",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and metadata
model = None
model_metadata = None
prediction_log = []

# Pydantic models for request/response
class StudentFeatures(BaseModel):
    """Schema for student features input"""
    prior_gpa: float = Field(..., ge=0.0, le=4.0, description="Prior GPA (0-4)")
    attendance_pct: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage")
    study_hours_wk: float = Field(..., ge=0.0, le=50.0, description="Study hours per week")
    commute_min: float = Field(..., ge=0.0, le=180.0, description="Commute time in minutes")
    quiz_avg: float = Field(..., ge=0.0, le=100.0, description="Quiz average score")
    assign_avg: float = Field(..., ge=0.0, le=100.0, description="Assignment average score")
    midterm: float = Field(..., ge=0.0, le=100.0, description="Midterm exam score")
    on_time_submit_pct: float = Field(..., ge=0.0, le=100.0, description="On-time submission percentage")
    lms_logins_wk: float = Field(..., ge=0.0, le=50.0, description="LMS logins per week")
    forum_posts: float = Field(..., ge=0.0, le=100.0, description="Forum posts count")
    gender: str = Field(..., description="Gender (M/F)")
    school_type: str = Field(..., description="School type (Public/Private/Charter)")
    parent_edu: str = Field(..., description="Parent education level")

class BatchPredictionRequest(BaseModel):
    """Schema for batch predictions"""
    students: List[StudentFeatures]

class PredictionResponse(BaseModel):
    """Schema for single prediction response"""
    student_id: Optional[str] = None
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of passing")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    prediction: bool = Field(..., description="Will pass (True/False)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[PredictionResponse]
    summary: Dict[str, float] = Field(..., description="Batch summary statistics")

class ExplanationResponse(BaseModel):
    """Schema for explanation response"""
    student_id: Optional[str] = None
    top_factors: List[Dict[str, str]] = Field(..., description="Top contributing factors")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    risk_drivers: Dict[str, float] = Field(..., description="Risk driver scores")

class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    predictions_made: int

# Utility functions
def load_model():
    """Load the trained model"""
    global model, model_metadata
    
    try:
        model_path = "models/xgboost_calibrated.joblib"
        model = joblib.load(model_path)
        
        # Load model metadata if available
        metadata_path = "models/best_params.json"
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        
        logger.info("Model loaded successfully")
        return True
        
    except FileNotFoundError:
        logger.error("Model file not found")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def determine_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability >= 0.8:
        return "Low Risk"
    elif probability >= 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

def calculate_confidence(probability: float) -> float:
    """Calculate prediction confidence based on probability distance from 0.5"""
    return abs(probability - 0.5) * 2

def generate_recommendations(features: StudentFeatures, risk_level: str) -> List[str]:
    """Generate personalized recommendations based on features and risk level"""
    recommendations = []
    
    # Attendance-based recommendations
    if features.attendance_pct < 80:
        recommendations.append("Improve attendance - aim for >80% class participation")
        recommendations.append("Set up study buddy system for missed classes")
    
    # Study time recommendations
    if features.study_hours_wk < 10:
        recommendations.append("Increase study hours to at least 10 hours per week")
        recommendations.append("Create structured study schedule with breaks")
    
    # Performance-based recommendations
    if features.quiz_avg < 70:
        recommendations.append("Focus on quiz preparation - use practice questions")
        recommendations.append("Schedule regular review sessions for quizzes")
    
    if features.assign_avg < 70:
        recommendations.append("Improve assignment quality - seek feedback before submission")
        recommendations.append("Break down large assignments into smaller tasks")
    
    if features.midterm < 70:
        recommendations.append("Address midterm performance gaps with tutoring")
        recommendations.append("Review midterm topics for final exam preparation")
    
    # Engagement recommendations
    if features.lms_logins_wk < 3:
        recommendations.append("Increase LMS engagement - check platform daily")
        recommendations.append("Participate in online discussions and activities")
    
    if features.forum_posts < 2:
        recommendations.append("Engage more in class forums - ask questions and help others")
    
    # Risk-specific recommendations
    if risk_level == "High Risk":
        recommendations.append("URGENT: Schedule meeting with academic advisor")
        recommendations.append("Consider intensive tutoring support")
        recommendations.append("Implement daily progress tracking")
    elif risk_level == "Medium Risk":
        recommendations.append("Schedule regular check-ins with faculty")
        recommendations.append("Join study groups for peer support")
    
    # Add positive reinforcement for low risk
    if risk_level == "Low Risk":
        recommendations.append("Maintain current excellent performance")
        recommendations.append("Consider peer tutoring opportunities")
        recommendations.append("Prepare for advanced coursework")
    
    return recommendations[:5]  # Limit to top 5 recommendations

def get_top_risk_factors(features: StudentFeatures) -> List[Dict[str, str]]:
    """Identify top risk factors based on feature values"""
    risk_factors = []
    
    # Define risk thresholds and descriptions
    risk_checks = [
        (features.attendance_pct < 80, "attendance", f"Low attendance ({features.attendance_pct:.1f}%)"),
        (features.study_hours_wk < 10, "study_hours", f"Insufficient study time ({features.study_hours_wk:.1f} hrs/wk)"),
        (features.quiz_avg < 70, "quiz_performance", f"Poor quiz scores ({features.quiz_avg:.1f}%)"),
        (features.assign_avg < 70, "assignment_performance", f"Low assignment scores ({features.assign_avg:.1f}%)"),
        (features.midterm < 70, "midterm_performance", f"Weak midterm performance ({features.midterm:.1f}%)"),
        (features.on_time_submit_pct < 80, "submission_behavior", f"Late submissions ({features.on_time_submit_pct:.1f}% on-time)"),
        (features.lms_logins_wk < 3, "engagement", f"Low platform engagement ({features.lms_logins_wk:.1f} logins/wk)"),
        (features.forum_posts < 2, "participation", f"Minimal forum participation ({features.forum_posts:.0f} posts)"),
        (features.commute_min > 60, "commute", f"Long commute time ({features.commute_min:.0f} min)"),
    ]
    
    for is_risk, factor_name, description in risk_checks:
        if is_risk:
            risk_factors.append({
                "factor": factor_name,
                "description": description,
                "severity": "high" if factor_name in ["attendance", "study_hours"] else "medium"
            })
    
    # Sort by severity and return top 5
    risk_factors.sort(key=lambda x: (x["severity"] != "high", x["severity"] != "medium"))
    return risk_factors[:5]

def log_prediction(features: StudentFeatures, prediction: PredictionResponse):
    """Log prediction for monitoring"""
    global prediction_log
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "features": features.model_dump(),
        "prediction": prediction.model_dump()
    }
    
    prediction_log.append(log_entry)
    
    # Keep only last 1000 predictions in memory
    if len(prediction_log) > 1000:
        prediction_log.pop(0)

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")
    else:
        logger.info("API startup completed successfully")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Student Performance Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        predictions_made=len(prediction_log)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_student_performance(features: StudentFeatures, student_id: Optional[str] = None):
    """Predict performance for a single student"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        features_dict = features.model_dump()
        X = pd.DataFrame([features_dict])
        
        # Make prediction
        probability = float(model.predict_proba(X)[0, 1])  # Probability of passing
        prediction = bool(probability >= 0.5)
        risk_level = determine_risk_level(probability)
        confidence = calculate_confidence(probability)
        
        # Create response
        response = PredictionResponse(
            student_id=student_id,
            risk_probability=probability,
            risk_level=risk_level,
            prediction=prediction,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
        # Log prediction
        log_prediction(features, response)
        
        logger.info(f"Prediction made for student {student_id}: {risk_level} ({probability:.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_performance(request: BatchPredictionRequest):
    """Predict performance for multiple students"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for i, features in enumerate(request.students):
            # Convert features to DataFrame
            features_dict = features.model_dump()
            X = pd.DataFrame([features_dict])
            
            # Make prediction
            probability = float(model.predict_proba(X)[0, 1])
            prediction = bool(probability >= 0.5)
            risk_level = determine_risk_level(probability)
            confidence = calculate_confidence(probability)
            
            # Create response
            response = PredictionResponse(
                student_id=f"student_{i+1}",
                risk_probability=probability,
                risk_level=risk_level,
                prediction=prediction,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )
            
            predictions.append(response)
            log_prediction(features, response)
        
        # Calculate batch summary
        probabilities = [p.risk_probability for p in predictions]
        summary = {
            "total_students": len(predictions),
            "avg_risk_probability": np.mean(probabilities),
            "high_risk_count": sum(1 for p in predictions if p.risk_level == "High Risk"),
            "medium_risk_count": sum(1 for p in predictions if p.risk_level == "Medium Risk"),
            "low_risk_count": sum(1 for p in predictions if p.risk_level == "Low Risk"),
            "predicted_pass_count": sum(1 for p in predictions if p.prediction),
            "predicted_fail_count": sum(1 for p in predictions if not p.prediction)
        }
        
        logger.info(f"Batch prediction completed: {len(predictions)} students processed")
        
        return BatchPredictionResponse(predictions=predictions, summary=summary)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(features: StudentFeatures, student_id: Optional[str] = None):
    """Provide explanation and recommendations for a student"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # First get prediction
        features_dict = features.model_dump()
        X = pd.DataFrame([features_dict])
        probability = float(model.predict_proba(X)[0, 1])
        risk_level = determine_risk_level(probability)
        
        # Get top risk factors
        top_factors = get_top_risk_factors(features)
        
        # Generate recommendations
        recommendations = generate_recommendations(features, risk_level)
        
        # Calculate risk driver scores
        risk_drivers = {
            "attendance": max(0, (80 - features.attendance_pct) / 80),
            "study_time": max(0, (10 - features.study_hours_wk) / 10),
            "performance": max(0, (70 - (features.quiz_avg + features.assign_avg + features.midterm) / 3) / 70),
            "engagement": max(0, (3 - features.lms_logins_wk) / 3),
            "submission": max(0, (80 - features.on_time_submit_pct) / 80)
        }
        
        response = ExplanationResponse(
            student_id=student_id,
            top_factors=top_factors,
            recommendations=recommendations,
            risk_drivers=risk_drivers
        )
        
        logger.info(f"Explanation generated for student {student_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information and metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier with Calibration",
        "model_version": "1.0.0",
        "features": [
            "prior_gpa", "attendance_pct", "study_hours_wk", "commute_min",
            "quiz_avg", "assign_avg", "midterm", "on_time_submit_pct",
            "lms_logins_wk", "forum_posts", "gender", "school_type", "parent_edu"
        ],
        "target": "passed (binary classification)",
        "performance_metrics": {
            "note": "Metrics available in evaluation results",
            "files": ["outputs/evaluation_plots.png", "outputs/model_summary.csv"]
        },
        "metadata": model_metadata or {}
    }

@app.get("/predictions/stats")
async def get_prediction_stats():
    """Get prediction statistics"""
    if not prediction_log:
        return {"message": "No predictions made yet"}
    
    # Calculate statistics
    recent_predictions = prediction_log[-100:]  # Last 100 predictions
    probabilities = [p["prediction"]["risk_probability"] for p in recent_predictions]
    
    stats = {
        "total_predictions": len(prediction_log),
        "recent_predictions": len(recent_predictions),
        "avg_probability": np.mean(probabilities),
        "high_risk_percentage": sum(1 for p in recent_predictions if p["prediction"]["risk_level"] == "High Risk") / len(recent_predictions) * 100,
        "predictions_by_hour": {}
    }
    
    return stats

if __name__ == "__main__":
    import uvicorn
    
    # Load model before starting server
    if load_model():
        logger.info("Starting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        logger.error("Failed to start server - model not loaded")
