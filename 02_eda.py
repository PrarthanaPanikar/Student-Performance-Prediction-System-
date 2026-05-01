"""
Exploratory Data Analysis and Leakage Detection
Analyzes student data patterns and prevents target leakage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load validated data"""
    try:
        df = pd.read_parquet("data/students.parquet")
        logger.info(f"Loaded {len(df)} student records")
        return df
    except FileNotFoundError:
        logger.error("Validated data file not found. Run 01_ingest.py first")
        raise

def check_target_leakage(df):
    """
    Check for potential target leakage in features
    Target leakage occurs when features contain information that would not be available at prediction time
    """
    logger.info("Checking for target leakage...")
    
    # Define feature columns (exclude target variables)
    target_cols = ['final_score', 'final_grade_band', 'passed']
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    # Check for features that might contain target information
    leakage_warnings = []
    
    # Check if any feature is too highly correlated with target
    correlations = df[feature_cols + ['final_score']].corr()['final_score'].abs().sort_values(ascending=False)
    
    high_corr_features = correlations[correlations > 0.9].index.tolist()
    if high_corr_features:
        leakage_warnings.append(f"Features very highly correlated with target: {high_corr_features}")
    
    # Check for features that might be derived from target
    suspicious_features = []
    for col in feature_cols:
        if 'final' in col.lower() or 'grade' in col.lower():
            suspicious_features.append(col)
    
    if suspicious_features:
        leakage_warnings.append(f"Features that might contain target info: {suspicious_features}")
    
    # Log warnings
    if leakage_warnings:
        logger.warning("POTENTIAL TARGET LEAKAGE DETECTED:")
        for warning in leakage_warnings:
            logger.warning(f"  - {warning}")
    else:
        logger.info("No obvious target leakage detected")
    
    return feature_cols, target_cols, leakage_warnings

def analyze_class_balance(df):
    """Analyze target class balance"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pass/Fail distribution
    pass_counts = df['passed'].value_counts()
    axes[0].pie(pass_counts.values, labels=['Fail', 'Pass'], autopct='%1.1f%%', colors=['#ff6b6b', '#51cf66'])
    axes[0].set_title('Pass/Fail Distribution')
    
    # Grade band distribution
    grade_counts = df['final_grade_band'].value_counts().sort_index()
    axes[1].bar(grade_counts.index, grade_counts.values, color=['#51cf66', '#8ce99a', '#ffd43b', '#ff8787', '#ff6b6b'])
    axes[1].set_title('Grade Band Distribution')
    axes[1].set_xlabel('Grade')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('outputs/class_balance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    logger.info(f"Class balance - Pass rate: {df['passed'].mean():.2%}")
    logger.info(f"Grade distribution: {dict(grade_counts)}")

def analyze_feature_distributions(df, feature_cols):
    """Analyze feature distributions and relationships"""
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Numeric features analysis
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Distribution plots for numeric features
    n_numeric = len(numeric_cols)
    if n_numeric > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:6]):  # Plot first 6 numeric features
            axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(numeric_cols[:6]), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('outputs/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Categorical features analysis
    if categorical_cols:
        fig, axes = plt.subplots(1, len(categorical_cols), figsize=(15, 5))
        if len(categorical_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(categorical_cols):
            df[col].value_counts().plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{col} Distribution')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

def analyze_correlations(df, feature_cols):
    """Analyze feature correlations"""
    # Select numeric features for correlation matrix
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) > 1:
        # Correlation matrix
        corr_matrix = df[numeric_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                    )
        
        if high_corr_pairs:
            logger.info("Highly correlated feature pairs (>0.7):")
            for feat1, feat2, corr in high_corr_pairs:
                logger.info(f"  {feat1} - {feat2}: {corr:.3f}")

def analyze_feature_importance_by_target(df, feature_cols):
    """Analyze how features relate to the target"""
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_features:
        # Box plots for key features vs pass/fail
        key_features = ['prior_gpa', 'attendance_pct', 'quiz_avg', 'assign_avg', 'midterm', 'study_hours_wk']
        key_features = [f for f in key_features if f in numeric_features]
        
        if key_features:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, feature in enumerate(key_features[:6]):
                pass_data = df[df['passed'] == 1][feature]
                fail_data = df[df['passed'] == 0][feature]
                
                axes[i].boxplot([pass_data, fail_data], labels=['Pass', 'Fail'])
                axes[i].set_title(f'{feature} by Pass/Fail')
                axes[i].set_ylabel(feature)
            
            plt.tight_layout()
            plt.savefig('outputs/features_vs_target.png', dpi=300, bbox_inches='tight')
            plt.show()

def generate_eda_report(df, feature_cols, target_cols):
    """Generate comprehensive EDA report"""
    logger.info("Generating EDA report...")
    
    # Basic statistics
    logger.info("Dataset Overview:")
    logger.info(f"  Total students: {len(df)}")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Missing values: {df.isna().sum().sum()}")
    
    # Target statistics
    logger.info("Target Statistics:")
    logger.info(f"  Pass rate: {df['passed'].mean():.2%}")
    logger.info(f"  Average final score: {df['final_score'].mean():.2f}")
    logger.info(f"  Score std dev: {df['final_score'].std():.2f}")
    
    # Feature statistics
    numeric_features = df[feature_cols].select_dtypes(include=[np.number])
    if len(numeric_features.columns) > 0:
        logger.info("Feature Statistics (Numeric):")
        for col in numeric_features.columns[:5]:  # Show first 5
            logger.info(f"  {col}: mean={numeric_features[col].mean():.2f}, "
                       f"std={numeric_features[col].std():.2f}, "
                       f"range=[{numeric_features[col].min():.2f}, {numeric_features[col].max():.2f}]")

def main():
    """Main EDA pipeline"""
    logger.info("Starting Exploratory Data Analysis...")
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Check for target leakage
    feature_cols, target_cols, leakage_warnings = check_target_leakage(df)
    
    # Analyze class balance
    analyze_class_balance(df)
    
    # Analyze feature distributions
    analyze_feature_distributions(df, feature_cols)
    
    # Analyze correlations
    analyze_correlations(df, feature_cols)
    
    # Analyze feature importance vs target
    analyze_feature_importance_by_target(df, feature_cols)
    
    # Generate report
    generate_eda_report(df, feature_cols, target_cols)
    
    logger.info("EDA completed successfully!")
    logger.info(f"Visualizations saved to 'outputs/' directory")
    
    return df, feature_cols, target_cols

if __name__ == "__main__":
    df, feature_cols, target_cols = main()
    print("EDA completed! Check the outputs directory for visualizations.")
