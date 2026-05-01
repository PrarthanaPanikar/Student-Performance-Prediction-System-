"""
Unit tests for FastAPI service
Tests API endpoints, validation, and error handling
"""

import pytest
import json
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.app import app

# Create test client
client = TestClient(app)

class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Student Performance Prediction API" in data["message"]
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
        assert "predictions_made" in data
    
    def test_model_info_endpoint(self):
        """Test model information endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "features" in data
        assert "target" in data
        assert len(data["features"]) > 0
    
    def test_predict_single_student_valid(self):
        """Test single student prediction with valid data"""
        valid_student = {
            "prior_gpa": 3.2,
            "attendance_pct": 85,
            "study_hours_wk": 12,
            "commute_min": 30,
            "quiz_avg": 75,
            "assign_avg": 78,
            "midterm": 72,
            "on_time_submit_pct": 90,
            "lms_logins_wk": 4,
            "forum_posts": 2,
            "gender": "M",
            "school_type": "Public",
            "parent_edu": "Bachelor"
        }
        
        response = client.post("/predict", json=valid_student)
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "risk_probability" in data
        assert "risk_level" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "timestamp" in data
        
        # Check data types and ranges
        assert isinstance(data["risk_probability"], (int, float))
        assert 0 <= data["risk_probability"] <= 1
        assert isinstance(data["prediction"], bool)
        assert 0 <= data["confidence"] <= 1
        assert data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
    
    def test_predict_single_student_invalid_gpa(self):
        """Test prediction with invalid GPA (out of range)"""
        invalid_student = {
            "prior_gpa": 5.0,  # Invalid: GPA should be 0-4
            "attendance_pct": 85,
            "study_hours_wk": 12,
            "commute_min": 30,
            "quiz_avg": 75,
            "assign_avg": 78,
            "midterm": 72,
            "on_time_submit_pct": 90,
            "lms_logins_wk": 4,
            "forum_posts": 2,
            "gender": "M",
            "school_type": "Public",
            "parent_edu": "Bachelor"
        }
        
        response = client.post("/predict", json=invalid_student)
        # Should still process but with validation warnings
        assert response.status_code == 200
    
    def test_predict_single_student_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_student = {
            "prior_gpa": 3.2,
            "attendance_pct": 85
            # Missing many required fields
        }
        
        response = client.post("/predict", json=incomplete_student)
        assert response.status_code == 422  # Validation error
    
    def test_predict_batch_valid(self):
        """Test batch prediction with valid data"""
        batch_data = {
            "students": [
                {
                    "prior_gpa": 3.2,
                    "attendance_pct": 85,
                    "study_hours_wk": 12,
                    "commute_min": 30,
                    "quiz_avg": 75,
                    "assign_avg": 78,
                    "midterm": 72,
                    "on_time_submit_pct": 90,
                    "lms_logins_wk": 4,
                    "forum_posts": 2,
                    "gender": "M",
                    "school_type": "Public",
                    "parent_edu": "Bachelor"
                },
                {
                    "prior_gpa": 2.8,
                    "attendance_pct": 70,
                    "study_hours_wk": 8,
                    "commute_min": 45,
                    "quiz_avg": 65,
                    "assign_avg": 68,
                    "midterm": 62,
                    "on_time_submit_pct": 75,
                    "lms_logins_wk": 2,
                    "forum_posts": 1,
                    "gender": "F",
                    "school_type": "Private",
                    "parent_edu": "High School"
                }
            ]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 2
        
        # Check summary statistics
        summary = data["summary"]
        assert "total_students" in summary
        assert "avg_risk_probability" in summary
        assert "high_risk_count" in summary
        assert "medium_risk_count" in summary
        assert "low_risk_count" in summary
    
    def test_explain_endpoint(self):
        """Test explanation endpoint"""
        student_data = {
            "prior_gpa": 2.5,
            "attendance_pct": 60,
            "study_hours_wk": 6,
            "commute_min": 60,
            "quiz_avg": 55,
            "assign_avg": 58,
            "midterm": 52,
            "on_time_submit_pct": 70,
            "lms_logins_wk": 2,
            "forum_posts": 0,
            "gender": "F",
            "school_type": "Public",
            "parent_edu": "High School"
        }
        
        response = client.post("/explain", json=student_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "top_factors" in data
        assert "recommendations" in data
        assert "risk_drivers" in data
        
        # Check structure of explanations
        assert isinstance(data["top_factors"], list)
        assert isinstance(data["recommendations"], list)
        assert isinstance(data["risk_drivers"], dict)
        
        if data["top_factors"]:
            factor = data["top_factors"][0]
            assert "factor" in factor
            assert "description" in factor
            assert "severity" in factor
    
    def test_predictions_stats_endpoint(self):
        """Test prediction statistics endpoint"""
        # First make some predictions
        student_data = {
            "prior_gpa": 3.0,
            "attendance_pct": 80,
            "study_hours_wk": 10,
            "commute_min": 30,
            "quiz_avg": 70,
            "assign_avg": 75,
            "midterm": 70,
            "on_time_submit_pct": 85,
            "lms_logins_wk": 3,
            "forum_posts": 2,
            "gender": "M",
            "school_type": "Public",
            "parent_edu": "Bachelor"
        }
        
        # Make a few predictions
        for _ in range(3):
            client.post("/predict", json=student_data)
        
        # Get stats
        response = client.get("/predictions/stats")
        assert response.status_code == 200
        data = response.json()
        
        assert "total_predictions" in data
        assert "recent_predictions" in data
        assert "avg_probability" in data
        assert "high_risk_percentage" in data

class TestDataValidation:
    """Test suite for data validation"""
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test minimum values
        min_student = {
            "prior_gpa": 0.0,
            "attendance_pct": 0.0,
            "study_hours_wk": 0.0,
            "commute_min": 0.0,
            "quiz_avg": 0.0,
            "assign_avg": 0.0,
            "midterm": 0.0,
            "on_time_submit_pct": 0.0,
            "lms_logins_wk": 0.0,
            "forum_posts": 0.0,
            "gender": "M",
            "school_type": "Public",
            "parent_edu": "High School"
        }
        
        response = client.post("/predict", json=min_student)
        assert response.status_code == 200
        
        # Test maximum values
        max_student = {
            "prior_gpa": 4.0,
            "attendance_pct": 100.0,
            "study_hours_wk": 50.0,
            "commute_min": 180.0,
            "quiz_avg": 100.0,
            "assign_avg": 100.0,
            "midterm": 100.0,
            "on_time_submit_pct": 100.0,
            "lms_logins_wk": 50.0,
            "forum_posts": 100.0,
            "gender": "F",
            "school_type": "Private",
            "parent_edu": "PhD"
        }
        
        response = client.post("/predict", json=max_student)
        assert response.status_code == 200
    
    def test_invalid_data_types(self):
        """Test with invalid data types"""
        invalid_student = {
            "prior_gpa": "invalid",  # Should be number
            "attendance_pct": 85,
            "study_hours_wk": 12,
            "commute_min": 30,
            "quiz_avg": 75,
            "assign_avg": 78,
            "midterm": 72,
            "on_time_submit_pct": 90,
            "lms_logins_wk": 4,
            "forum_posts": 2,
            "gender": "M",
            "school_type": "Public",
            "parent_edu": "Bachelor"
        }
        
        response = client.post("/predict", json=invalid_student)
        assert response.status_code == 422  # Validation error

class TestErrorHandling:
    """Test suite for error handling"""
    
    def test_model_not_loaded(self):
        """Test behavior when model is not loaded"""
        # This test would require mocking the model loading failure
        # For now, we assume model is loaded in test environment
        pass
    
    def test_malformed_json(self):
        """Test with malformed JSON"""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_empty_batch(self):
        """Test empty batch prediction"""
        response = client.post("/predict/batch", json={"students": []})
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 0
        assert data["summary"]["total_students"] == 0

class TestPerformance:
    """Test suite for performance"""
    
    def test_response_time(self):
        """Test API response time"""
        import time
        
        student_data = {
            "prior_gpa": 3.0,
            "attendance_pct": 80,
            "study_hours_wk": 10,
            "commute_min": 30,
            "quiz_avg": 70,
            "assign_avg": 75,
            "midterm": 70,
            "on_time_submit_pct": 85,
            "lms_logins_wk": 3,
            "forum_posts": 2,
            "gender": "M",
            "school_type": "Public",
            "parent_edu": "Bachelor"
        }
        
        start_time = time.time()
        response = client.post("/predict", json=student_data)
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
