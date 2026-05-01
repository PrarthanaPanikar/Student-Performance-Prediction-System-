"""
Data Ingestion and Validation Pipeline
Loads student data, validates schema, and saves in optimized format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data schema definition
SCHEMA = {
    "student_id": "string",
    "gender": "category", 
    "school_type": "category",
    "parent_edu": "category",
    "prior_gpa": "float64",
    "attendance_pct": "float64",
    "study_hours_wk": "float64",
    "commute_min": "float64",
    "quiz_avg": "float64",
    "assign_avg": "float64",
    "midterm": "float64",
    "on_time_submit_pct": "float64",
    "lms_logins_wk": "float64",
    "forum_posts": "float64",
    "final_score": "float64",
    "final_grade_band": "category",
    "passed": "int64"
}

def validate_data(df, schema):
    """
    Validate data against schema and perform basic quality checks
    """
    logger.info("Validating data schema...")
    
    # Check required columns
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    type_mapping = {
        "string": "object",
        "category": "object", 
        "float64": "float64",
        "int64": "int64"
    }
    
    for col, expected_type in schema.items():
        if col in df.columns:
            target_type = type_mapping.get(expected_type, expected_type)
            try:
                df[col] = df[col].astype(target_type)
            except Exception as e:
                logger.warning(f"Could not convert {col} to {target_type}: {e}")
    
    return df

def perform_quality_checks(df):
    """
    Perform data quality checks and log issues
    """
    logger.info("Performing data quality checks...")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['student_id']).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate student IDs")
        df = df.drop_duplicates(subset=['student_id'], keep='first')
    
    # Check for missing values
    missing_summary = df.isna().mean().sort_values(ascending=False)
    high_missing = missing_summary[missing_summary > 0.1]
    
    if len(high_missing) > 0:
        logger.warning("Columns with >10% missing values:")
        for col, pct in high_missing.items():
            logger.warning(f"  {col}: {pct:.2%}")
    
    # Check value ranges
    range_issues = []
    
    # GPA should be between 0-4
    if (df['prior_gpa'] < 0).any() or (df['prior_gpa'] > 4).any():
        range_issues.append("prior_gpa outside 0-4 range")
    
    # Percentages should be 0-100
    pct_cols = ['attendance_pct', 'quiz_avg', 'assign_avg', 'midterm', 'on_time_submit_pct']
    for col in pct_cols:
        if (df[col] < 0).any() or (df[col] > 100).any():
            range_issues.append(f"{col} outside 0-100 range")
    
    # Positive numeric fields
    pos_cols = ['study_hours_wk', 'commute_min', 'lms_logins_wk', 'forum_posts']
    for col in pos_cols:
        if (df[col] < 0).any():
            range_issues.append(f"{col} has negative values")
    
    if range_issues:
        logger.warning("Range validation issues:")
        for issue in range_issues:
            logger.warning(f"  {issue}")
    
    return df

def load_and_validate_data(csv_path="data/students.csv", output_path="data/students.parquet"):
    """
    Main function to load, validate and save data
    """
    logger.info(f"Loading data from {csv_path}")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    except FileNotFoundError:
        logger.error(f"Data file not found: {csv_path}")
        raise
    
    # Validate schema
    df = validate_data(df, SCHEMA)
    
    # Quality checks
    df = perform_quality_checks(df)
    
    # Save optimized format
    logger.info(f"Saving validated data to {output_path}")
    df.to_parquet(output_path, index=False)
    
    # Log summary statistics
    logger.info("Data summary:")
    logger.info(f"  Total students: {len(df)}")
    logger.info(f"  Pass rate: {df['passed'].mean():.2%}")
    logger.info(f"  Grade distribution: {dict(df['final_grade_band'].value_counts().sort_index())}")
    logger.info(f"  Missing values: {df.isna().sum().sum()}")
    
    return df

if __name__ == "__main__":
    # Create data directory if needed
    Path("data").mkdir(exist_ok=True)
    
    # Load and validate data
    df = load_and_validate_data()
    
    print("Data ingestion completed successfully!")
    print(f"Final dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing value summary:")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("No missing values found")
