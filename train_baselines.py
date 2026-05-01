"""
Baseline Model Training with Cross-Validation
Trains and evaluates multiple baseline models for student performance prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import joblib

# Import our preprocessing pipeline
from pipeline import create_preprocessing_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class to handle baseline model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.preprocessor = create_preprocessing_pipeline()
    
    def load_data(self):
        """Load and prepare data for training"""
        logger.info("Loading data for training...")
        
        try:
            df = pd.read_parquet("data/students.parquet")
            logger.info(f"Loaded {len(df)} student records")
        except FileNotFoundError:
            logger.error("Data file not found. Run data generation and ingestion first")
            raise
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in TARGET_COLUMNS]
        X = df[feature_cols].copy()
        y = df['passed'].copy()
        
        logger.info(f"Features: {len(feature_cols)}, Target: 'passed'")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def create_model_pipelines(self):
        """Create sklearn pipelines for different models"""
        
        self.models = {
            'Logistic Regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
            ]),
            
            'Random Forest': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=400, 
                    random_state=42, 
                    class_weight='balanced',
                    n_jobs=-1
                ))
            ]),
            
            'Gradient Boosting': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42
                ))
            ]),
            
            'SVM': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', SVC(
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                ))
            ]),
            
            'Naive Bayes': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', GaussianNB())
            ])
        }
        
        logger.info(f"Created {len(self.models)} baseline model pipelines")
    
    def evaluate_with_cv(self, X, y, cv_folds=5):
        """Evaluate models using stratified cross-validation"""
        
        logger.info(f"Evaluating models with {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring_metrics = ['f1', 'roc_auc', 'accuracy', 'precision', 'recall']
        
        for model_name, pipeline in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            model_results = {}
            
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
                    model_results[metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                    logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                except Exception as e:
                    logger.warning(f"  Could not evaluate {metric}: {e}")
                    model_results[metric] = {'mean': 0, 'std': 0, 'scores': []}
            
            self.results[model_name] = model_results
        
        return self.results
    
    def train_final_models(self, X, y):
        """Train final models on the full dataset"""
        
        logger.info("Training final models on full dataset...")
        
        trained_models = {}
        
        for model_name, pipeline in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                pipeline.fit(X, y)
                trained_models[model_name] = pipeline
                logger.info(f"  {model_name} trained successfully")
            except Exception as e:
                logger.error(f"  Error training {model_name}: {e}")
        
        return trained_models
    
    def evaluate_on_holdout(self, X, y, test_size=0.2):
        """Evaluate models on a holdout test set"""
        
        logger.info("Evaluating models on holdout test set...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        holdout_results = {}
        
        for model_name, pipeline in self.models.items():
            logger.info(f"Evaluating {model_name} on holdout set...")
            
            try:
                # Train on training set
                pipeline.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                holdout_results[model_name] = {
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_proba),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                logger.info(f"  F1: {holdout_results[model_name]['f1']:.4f}")
                logger.info(f"  ROC-AUC: {holdout_results[model_name]['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"  Error evaluating {model_name}: {e}")
        
        return holdout_results, (X_train, X_test, y_train, y_test)
    
    def plot_model_comparison(self):
        """Create comparison plots for model performance"""
        
        if not self.results:
            logger.warning("No results to plot. Run evaluation first.")
            return
        
        # Create output directory
        Path("outputs").mkdir(exist_ok=True)
        
        # Extract metrics for plotting
        metrics = ['f1', 'roc_auc', 'accuracy', 'precision', 'recall']
        model_names = list(self.results.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            means = [self.results[model][metric]['mean'] for model in model_names]
            stds = [self.results[model][metric]['std'] for model in model_names]
            
            axes[i].bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Score Comparison')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)
        
        # Hide unused subplot
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary table
        summary_data = []
        for model_name in model_names:
            row = {'Model': model_name}
            for metric in metrics:
                row[f'{metric}_mean'] = self.results[model_name][metric]['mean']
                row[f'{metric}_std'] = self.results[model_name][metric]['std']
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('outputs/model_summary.csv', index=False)
        
        logger.info("Model comparison plots saved to outputs/")
    
    def save_best_model(self, X, y):
        """Save the best performing model"""
        
        if not self.results:
            logger.warning("No results available. Run evaluation first.")
            return None
        
        # Find best model based on F1 score
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['f1']['mean'])
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"F1 Score: {self.results[best_model_name]['f1']['mean']:.4f}")
        
        # Train best model on full dataset
        best_pipeline = self.models[best_model_name]
        best_pipeline.fit(X, y)
        
        # Save model
        Path("models").mkdir(exist_ok=True)
        model_path = f"models/baseline_{best_model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(best_pipeline, model_path)
        
        logger.info(f"Best model saved to {model_path}")
        
        return best_pipeline, best_model_name
    
    def generate_report(self):
        """Generate a comprehensive training report"""
        
        if not self.results:
            logger.warning("No results to report. Run evaluation first.")
            return
        
        logger.info("=== BASELINE MODEL TRAINING REPORT ===")
        
        # Sort models by F1 score
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['f1']['mean'], 
                             reverse=True)
        
        logger.info("Model Rankings (by F1 Score):")
        for i, (model_name, results) in enumerate(sorted_models, 1):
            logger.info(f"{i}. {model_name}")
            logger.info(f"   F1: {results['f1']['mean']:.4f} (+/- {results['f1']['std'] * 2:.4f})")
            logger.info(f"   ROC-AUC: {results['roc_auc']['mean']:.4f} (+/- {results['roc_auc']['std'] * 2:.4f})")
            logger.info(f"   Accuracy: {results['accuracy']['mean']:.4f} (+/- {results['accuracy']['std'] * 2:.4f})")
        
        # Recommendations
        best_model = sorted_models[0][0]
        logger.info(f"\nRECOMMENDATION: Use {best_model} as the baseline model")
        logger.info("Next steps: Hyperparameter tuning with Optuna")

def main():
    """Main training pipeline"""
    
    logger.info("Starting baseline model training...")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    X, y, feature_cols = trainer.load_data()
    
    # Create model pipelines
    trainer.create_model_pipelines()
    
    # Evaluate with cross-validation
    cv_results = trainer.evaluate_with_cv(X, y)
    
    # Evaluate on holdout set
    holdout_results, splits = trainer.evaluate_on_holdout(X, y)
    
    # Plot comparisons
    trainer.plot_model_comparison()
    
    # Save best model
    best_model, best_name = trainer.save_best_model(X, y)
    
    # Generate report
    trainer.generate_report()
    
    logger.info("Baseline training completed successfully!")
    
    return trainer, best_model

if __name__ == "__main__":
    trainer, best_model = main()
    print("Baseline model training completed!")
    print("Check the outputs/ directory for visualizations and results.")
