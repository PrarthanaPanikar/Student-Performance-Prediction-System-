"""
Unit tests for ML pipeline components
Tests data processing, model training, and evaluation
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import create_preprocessing_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.train_baselines import ModelTrainer
from src.evaluate import ModelEvaluator

class TestDataGeneration:
    """Test suite for data generation"""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        from data.generate_synthetic_data import generate_student_data
        
        # Generate small dataset for testing
        df = generate_student_data(n_students=100, random_state=42)
        
        # Check basic properties
        assert len(df) == 100
        assert all(col in df.columns for col in ['student_id', 'passed', 'final_score'])
        
        # Check data types
        assert df['passed'].dtype in ['int64', 'bool']
        assert df['final_score'].dtype in ['float64']
        
        # Check value ranges
        assert df['prior_gpa'].between(0, 4).all()
        assert df['attendance_pct'].between(0, 100).all()
        assert df['final_score'].between(0, 100).all()
        
        # Check class balance
        pass_rate = df['passed'].mean()
        assert 0.3 <= pass_rate <= 0.9  # Reasonable pass rate

class TestDataIngestion:
    """Test suite for data ingestion"""
    
    def test_data_validation(self):
        """Test data validation pipeline"""
        from notebooks.01_ingest import validate_data, SCHEMA
        
        # Create test data
        test_data = pd.DataFrame({
            'student_id': ['STU_001', 'STU_002'],
            'gender': ['M', 'F'],
            'school_type': ['Public', 'Private'],
            'parent_edu': ['Bachelor', 'Master'],
            'prior_gpa': [3.2, 3.8],
            'attendance_pct': [85, 92],
            'study_hours_wk': [12, 15],
            'commute_min': [30, 20],
            'quiz_avg': [75, 88],
            'assign_avg': [78, 85],
            'midterm': [72, 80],
            'on_time_submit_pct': [90, 95],
            'lms_logins_wk': [4, 6],
            'forum_posts': [2, 3],
            'final_score': [78, 85],
            'final_grade_band': ['B', 'A'],
            'passed': [1, 1]
        })
        
        # Test validation
        validated_df = validate_data(test_data, SCHEMA)
        
        assert len(validated_df) == len(test_data)
        assert all(col in validated_df.columns for col in test_data.columns)

class TestPreprocessing:
    """Test suite for preprocessing pipeline"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.n_samples = 100
        
        self.test_data = pd.DataFrame({
            'prior_gpa': np.random.normal(3.0, 0.5, self.n_samples),
            'attendance_pct': np.random.normal(85, 10, self.n_samples),
            'study_hours_wk': np.random.gamma(2, 3, self.n_samples),
            'commute_min': np.random.exponential(25, self.n_samples),
            'quiz_avg': np.random.normal(75, 15, self.n_samples),
            'assign_avg': np.random.normal(78, 12, self.n_samples),
            'midterm': np.random.normal(72, 18, self.n_samples),
            'on_time_submit_pct': np.random.beta(8, 2, self.n_samples) * 100,
            'lms_logins_wk': np.random.poisson(4, self.n_samples) + 1,
            'forum_posts': np.random.poisson(2, self.n_samples),
            'gender': np.random.choice(['M', 'F'], self.n_samples),
            'school_type': np.random.choice(['Public', 'Private', 'Charter'], self.n_samples),
            'parent_edu': np.random.choice(['High School', 'Some College', 'Bachelor', 'Master', 'PhD'], self.n_samples),
            'passed': np.random.choice([0, 1], self.n_samples)
        })
        
        # Clip values to reasonable ranges
        self.test_data['prior_gpa'] = self.test_data['prior_gpa'].clip(0, 4)
        self.test_data['attendance_pct'] = self.test_data['attendance_pct'].clip(0, 100)
        self.test_data['study_hours_wk'] = self.test_data['study_hours_wk'].clip(0, 50)
        self.test_data['commute_min'] = self.test_data['commute_min'].clip(0, 180)
        self.test_data['quiz_avg'] = self.test_data['quiz_avg'].clip(0, 100)
        self.test_data['assign_avg'] = self.test_data['assign_avg'].clip(0, 100)
        self.test_data['midterm'] = self.test_data['midterm'].clip(0, 100)
        self.test_data['on_time_submit_pct'] = self.test_data['on_time_submit_pct'].clip(0, 100)
        self.test_data['lms_logins_wk'] = self.test_data['lms_logins_wk'].clip(0, 50)
        self.test_data['forum_posts'] = self.test_data['forum_posts'].clip(0, 100)
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline"""
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline()
        
        # Prepare features and target
        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = self.test_data[feature_cols]
        y = self.test_data['passed']
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)
        
        # Check transformation
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > len(feature_cols)  # Due to one-hot encoding
        
        # Check for NaN values
        assert not np.isnan(X_transformed).any()
        
        # Check for infinite values
        assert not np.isinf(X_transformed).any()
    
    def test_preprocessing_with_missing_values(self):
        """Test preprocessing with missing values"""
        # Introduce missing values
        data_with_missing = self.test_data.copy()
        data_with_missing.loc[::10, 'prior_gpa'] = np.nan
        data_with_missing.loc[::15, 'attendance_pct'] = np.nan
        data_with_missing.loc[::20, 'gender'] = np.nan
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline()
        
        # Prepare features
        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = data_with_missing[feature_cols]
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)
        
        # Should handle missing values
        assert X_transformed.shape[0] == len(X)
        assert not np.isnan(X_transformed).any()

class TestModelTraining:
    """Test suite for model training"""
    
    def setup_method(self):
        """Setup test data"""
        # Generate test data
        from data.generate_synthetic_data import generate_student_data
        
        self.df = generate_student_data(n_students=500, random_state=42)
        feature_cols = [col for col in self.df.columns if col not in ['final_score', 'final_grade_band', 'passed']]
        self.X = self.df[feature_cols]
        self.y = self.df['passed']
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer()
        
        assert trainer.preprocessor is not None
        assert len(trainer.models) == 0
        assert len(trainer.results) == 0
    
    def test_model_pipelines_creation(self):
        """Test creation of model pipelines"""
        trainer = ModelTrainer()
        trainer.create_model_pipelines()
        
        assert len(trainer.models) == 5
        assert 'Logistic Regression' in trainer.models
        assert 'Random Forest' in trainer.models
        assert 'Gradient Boosting' in trainer.models
        assert 'SVM' in trainer.models
        assert 'Naive Bayes' in trainer.models
    
    def test_cross_validation(self):
        """Test cross-validation evaluation"""
        trainer = ModelTrainer()
        trainer.create_model_pipelines()
        
        # Run CV evaluation
        results = trainer.evaluate_with_cv(self.X, self.y, cv_folds=3)  # Use fewer folds for speed
        
        assert len(results) == 5
        for model_name, model_results in results.items():
            assert 'f1' in model_results
            assert 'roc_auc' in model_results
            assert 'accuracy' in model_results
            assert 0 <= model_results['f1']['mean'] <= 1
            assert 0 <= model_results['roc_auc']['mean'] <= 1

class TestModelEvaluation:
    """Test suite for model evaluation"""
    
    def setup_method(self):
        """Setup test data and model"""
        # Generate test data
        from data.generate_synthetic_data import generate_student_data
        
        self.df = generate_student_data(n_students=200, random_state=42)
        feature_cols = [col for col in self.df.columns if col not in ['final_score', 'final_grade_band', 'passed']]
        self.X = self.df[feature_cols]
        self.y = self.df['passed']
        
        # Train a simple model for testing
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        from src.pipeline import create_preprocessing_pipeline
        
        self.model = Pipeline([
            ('preprocessor', create_preprocessing_pipeline()),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        self.model.fit(self.X, self.y)
    
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization"""
        evaluator = ModelEvaluator()
        
        assert evaluator.model_path == "models/xgboost_calibrated.joblib"
        assert evaluator.model is None
        assert evaluator.evaluation_results == {}
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        evaluator = ModelEvaluator()
        evaluator.model = self.model
        
        # Run evaluation
        results = evaluator.evaluate_model(self.X, self.y)
        
        assert 'predictions' in results
        assert 'probabilities' in results
        assert 'metrics' in results
        
        metrics = results['metrics']
        assert 'roc_auc' in metrics
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1

class TestFairnessAnalysis:
    """Test suite for fairness analysis"""
    
    def setup_method(self):
        """Setup test data"""
        from data.generate_synthetic_data import generate_student_data
        
        self.df = generate_student_data(n_students=300, random_state=42)
        feature_cols = [col for col in self.df.columns if col not in ['final_score', 'final_grade_band', 'passed']]
        self.X = self.df[feature_cols]
        self.y = self.df['passed']
    
    def test_fairness_metrics_calculation(self):
        """Test fairness metrics calculation"""
        evaluator = ModelEvaluator()
        
        # Create mock predictions for testing
        mock_predictions = np.random.choice([0, 1], size=len(self.y), p=[0.3, 0.7])
        mock_probabilities = np.random.random(len(self.y))
        
        evaluator.evaluation_results = {
            'predictions': mock_predictions,
            'probabilities': mock_probabilities
        }
        
        # Test fairness analysis
        fairness_results = evaluator.analyze_fairness(self.X, self.y, self.df)
        
        assert isinstance(fairness_results, dict)
        
        # Check if demographic groups are analyzed
        for demographic in ['gender', 'school_type', 'parent_edu']:
            if demographic in fairness_results:
                groups = fairness_results[demographic]
                assert isinstance(groups, dict)
                
                for group_name, group_metrics in groups.items():
                    assert 'size' in group_metrics
                    assert 'pass_rate' in group_metrics
                    assert 'roc_auc' in group_metrics

class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # Generate data
        from data.generate_synthetic_data import generate_student_data
        
        df = generate_student_data(n_students=200, random_state=42)
        feature_cols = [col for col in df.columns if col not in ['final_score', 'final_grade_band', 'passed']]
        X = df[feature_cols]
        y = df['passed']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train model
        trainer = ModelTrainer()
        trainer.create_model_pipelines()
        
        # Use just one model for speed
        model_name = 'Logistic Regression'
        pipeline = trainer.models[model_name]
        
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Basic sanity checks
        assert 0 <= f1 <= 1
        assert 0 <= roc_auc <= 1
        assert len(y_pred) == len(y_test)
        assert len(y_proba) == len(y_test)

class TestErrorHandling:
    """Test suite for error handling"""
    
    def test_empty_data(self):
        """Test handling of empty data"""
        preprocessor = create_preprocessing_pipeline()
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            preprocessor.fit_transform(empty_df)
    
    def test_invalid_features(self):
        """Test handling of invalid features"""
        preprocessor = create_preprocessing_pipeline()
        
        # DataFrame with wrong columns
        wrong_df = pd.DataFrame({
            'wrong_col1': [1, 2, 3],
            'wrong_col2': [4, 5, 6]
        })
        
        with pytest.raises(Exception):
            preprocessor.fit_transform(wrong_df)
    
    def test_mixed_data_types(self):
        """Test handling of mixed data types"""
        preprocessor = create_preprocessing_pipeline()
        
        # DataFrame with mixed types in numeric columns
        mixed_df = pd.DataFrame({
            'prior_gpa': [3.0, 'invalid', 2.5],
            'attendance_pct': [85, 90, 'invalid'],
            'gender': ['M', 'F', 'M'],
            'school_type': ['Public', 'Private', 'Charter'],
            'parent_edu': ['Bachelor', 'Master', 'PhD']
        })
        
        # Should handle or raise appropriate error
        try:
            result = preprocessor.fit_transform(mixed_df)
            # If successful, should not have NaN values
            assert not np.isnan(result).any()
        except Exception:
            # If it fails, that's also acceptable
            pass

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
