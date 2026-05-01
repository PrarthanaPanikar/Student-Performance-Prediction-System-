"""
Preprocessing Pipeline for Student Performance Prediction
Creates reproducible data transformations for training and inference
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Feature column definitions
NUMERIC_FEATURES = [
    "prior_gpa", "attendance_pct", "quiz_avg", "assign_avg", "midterm",
    "study_hours_wk", "on_time_submit_pct", "lms_logins_wk", "forum_posts", "commute_min"
]

CATEGORICAL_FEATURES = [
    "gender", "school_type", "parent_edu"
]

# Target columns (to be excluded from features)
TARGET_COLUMNS = ["final_score", "final_grade_band", "passed"]

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to select specific columns"""
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Handle both DataFrame and numpy array
        if hasattr(X, 'columns'):
            return X[self.columns]
        else:
            return X[:, self.columns]

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Custom transformer to clip outliers to reasonable ranges"""
    
    def __init__(self, clip_ranges=None):
        self.clip_ranges = clip_ranges or {}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        for col, (min_val, max_val) in self.clip_ranges.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].clip(min_val, max_val)
        
        return X_transformed

def create_preprocessing_pipeline():
    """
    Create a comprehensive preprocessing pipeline
    """
    
    # Define reasonable clipping ranges for outliers
    clip_ranges = {
        'prior_gpa': (0.0, 4.0),
        'attendance_pct': (0.0, 100.0),
        'quiz_avg': (0.0, 100.0),
        'assign_avg': (0.0, 100.0),
        'midterm': (0.0, 100.0),
        'study_hours_wk': (0.0, 50.0),
        'on_time_submit_pct': (0.0, 100.0),
        'lms_logins_wk': (0.0, 50.0),
        'forum_posts': (0.0, 100.0),
        'commute_min': (0.0, 180.0)
    }
    
    # Numeric preprocessing pipeline
    numeric_pipeline = Pipeline([
        ('clipper', OutlierClipper(clip_ranges)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combined preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor

def get_feature_names(preprocessor, categorical_features):
    """
    Get feature names after preprocessing (especially for one-hot encoded features)
    """
    # Get numeric feature names (they stay the same)
    numeric_features = NUMERIC_FEATURES
    
    # Get categorical feature names (after one-hot encoding)
    categorical_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    categorical_feature_names = []
    
    if hasattr(categorical_encoder, 'get_feature_names_out'):
        # For newer sklearn versions
        cat_names = categorical_encoder.get_feature_names_out(categorical_features)
        categorical_feature_names = list(cat_names)
    else:
        # For older sklearn versions
        for i, cat_feature in enumerate(categorical_features):
            categories = categorical_encoder.categories_[i]
            for category in categories:
                categorical_feature_names.append(f"{cat_feature}_{category}")
    
    # Combine all feature names
    all_feature_names = numeric_features + categorical_feature_names
    
    return all_feature_names

def create_feature_engineering_pipeline():
    """
    Create a pipeline that includes feature engineering
    """
    
    class FeatureEngineer(BaseEstimator, TransformerMixin):
        """Custom transformer for feature engineering"""
        
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            X_transformed = X.copy()
            
            # Create interaction features
            if 'attendance_pct' in X_transformed.columns and 'study_hours_wk' in X_transformed.columns:
                X_transformed['attendance_study_interaction'] = (
                    X_transformed['attendance_pct'] * X_transformed['study_hours_wk'] / 100
                )
            
            # Create performance consistency score
            performance_cols = ['quiz_avg', 'assign_avg', 'midterm']
            available_cols = [col for col in performance_cols if col in X_transformed.columns]
            
            if len(available_cols) >= 2:
                X_transformed['performance_consistency'] = X_transformed[available_cols].std(axis=1)
            
            # Create engagement score
            engagement_cols = ['lms_logins_wk', 'forum_posts', 'on_time_submit_pct']
            available_engagement = [col for col in engagement_cols if col in X_transformed.columns]
            
            if available_engagement:
                # Normalize and combine engagement metrics
                X_transformed['engagement_score'] = 0
                for col in available_engagement:
                    if col == 'on_time_submit_pct':
                        X_transformed['engagement_score'] += X_transformed[col] / 100
                    else:
                        # Normalize by approximate max values
                        max_val = 50 if col == 'lms_logins_wk' else 20
                        X_transformed['engagement_score'] += (X_transformed[col] / max_val).clip(0, 1)
                
                X_transformed['engagement_score'] = X_transformed['engagement_score'] / len(available_engagement)
            
            return X_transformed
    
    # Create the full pipeline
    feature_pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', create_preprocessing_pipeline())
    ])
    
    return feature_pipeline

def validate_pipeline(pipeline, X_sample):
    """
    Validate that the preprocessing pipeline works correctly
    """
    try:
        # Fit and transform sample data
        X_transformed = pipeline.fit_transform(X_sample)
        
        print(f"Pipeline validation successful!")
        print(f"Original shape: {X_sample.shape}")
        print(f"Transformed shape: {X_transformed.shape}")
        
        # Check for NaN values
        if np.isnan(X_transformed).any():
            print("WARNING: NaN values found in transformed data")
        else:
            print("No NaN values in transformed data")
        
        # Check for infinite values
        if np.isinf(X_transformed).any():
            print("WARNING: Infinite values found in transformed data")
        else:
            print("No infinite values in transformed data")
        
        return True
        
    except Exception as e:
        print(f"Pipeline validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Creating preprocessing pipeline...")
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'prior_gpa': np.random.normal(3.0, 0.5, 100),
        'attendance_pct': np.random.normal(85, 10, 100),
        'quiz_avg': np.random.normal(75, 15, 100),
        'assign_avg': np.random.normal(78, 12, 100),
        'midterm': np.random.normal(72, 18, 100),
        'study_hours_wk': np.random.gamma(2, 3, 100),
        'on_time_submit_pct': np.random.beta(8, 2, 100) * 100,
        'lms_logins_wk': np.random.poisson(4, 100) + 1,
        'forum_posts': np.random.poisson(2, 100),
        'commute_min': np.random.exponential(25, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'school_type': np.random.choice(['Public', 'Private', 'Charter'], 100),
        'parent_edu': np.random.choice(['High School', 'Some College', 'Bachelor', 'Master', 'PhD'], 100)
    })
    
    # Create and validate pipeline
    pipeline = create_preprocessing_pipeline()
    validate_pipeline(pipeline, sample_data)
    
    print("\nPreprocessing pipeline ready for use!")
