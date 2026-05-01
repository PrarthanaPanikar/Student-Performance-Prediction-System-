"""
Hyperparameter Tuning with Optuna
Optimizes XGBoost model parameters and calibrates probabilities
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import joblib

# Import our preprocessing pipeline
from pipeline import create_preprocessing_pipeline, TARGET_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptunaTuner:
    """Class to handle hyperparameter tuning with Optuna"""
    
    def __init__(self, n_trials=50, random_state=42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.preprocessor = create_preprocessing_pipeline()
        self.best_params = None
        self.best_score = None
        self.study = None
        self.calibrated_model = None
    
    def load_data(self):
        """Load and prepare data for tuning"""
        logger.info("Loading data for hyperparameter tuning...")
        
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
        
        # Split for tuning
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Class distribution - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")
        
        return X_train, X_val, y_train, y_val
    
    def objective(self, trial, X_train, X_val, y_train, y_val):
        """Optuna objective function for XGBoost tuning"""
        
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # Create pipeline
        xgb_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', xgb.XGBClassifier(**params))
        ])
        
        try:
            # Train model
            xgb_pipeline.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = xgb_pipeline.predict(X_val)
            y_proba = xgb_pipeline.predict_proba(X_val)[:, 1]
            
            # Calculate F1 score (primary metric)
            f1 = f1_score(y_val, y_pred)
            
            # Calculate ROC-AUC as secondary metric
            roc_auc = roc_auc_score(y_val, y_proba)
            
            # Log metrics
            trial.set_user_attr('roc_auc', roc_auc)
            trial.set_user_attr('classification_report', classification_report(y_val, y_pred, output_dict=True))
            
            return f1
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    def tune_hyperparameters(self, X_train, X_val, y_train, y_val):
        """Run hyperparameter optimization with Optuna"""
        
        logger.info(f"Starting hyperparameter tuning with {self.n_trials} trials...")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, X_val, y_train, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Best F1 Score: {self.best_score:.4f}")
        logger.info("Best Parameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        return self.best_params, self.best_score
    
    def calibrate_model(self, X_train, X_val, y_train, y_val):
        """Train and calibrate the final model with best parameters"""
        
        if not self.best_params:
            logger.error("No best parameters found. Run tuning first.")
            return None
        
        logger.info("Training and calibrating final model...")
        
        # Create final pipeline with best parameters
        final_params = self.best_params.copy()
        final_params.update({
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'use_label_encoder': False
        })
        
        # Create base pipeline
        base_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', xgb.XGBClassifier(**final_params))
        ])
        
        # Calibrate using isotonic regression
        self.calibrated_model = CalibratedClassifierCV(
            base_pipeline,
            method='isotonic',
            cv=3
        )
        
        # Train calibrated model
        self.calibrated_model.fit(X_train, y_train)
        
        # Evaluate calibrated model
        y_pred = self.calibrated_model.predict(X_val)
        y_proba = self.calibrated_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        calibrated_f1 = f1_score(y_val, y_pred)
        calibrated_roc_auc = roc_auc_score(y_val, y_proba)
        
        logger.info(f"Calibrated Model Performance:")
        logger.info(f"  F1 Score: {calibrated_f1:.4f}")
        logger.info(f"  ROC-AUC: {calibrated_roc_auc:.4f}")
        
        return self.calibrated_model
    
    def plot_optimization_results(self):
        """Create visualizations of the optimization process"""
        
        if not self.study:
            logger.warning("No study results to plot. Run tuning first.")
            return
        
        # Create output directory
        Path("outputs").mkdir(exist_ok=True)
        
        # Plot optimization history
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=axes[0, 0])
        axes[0, 0].set_title('Optimization History')
        
        # Parameter importance
        try:
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[0, 1])
            axes[0, 1].set_title('Parameter Importance')
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Parameter importance plot failed:\n{str(e)}', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Parallel coordinate plot
        try:
            optuna.visualization.matplotlib.plot_parallel_coordinate(self.study, ax=axes[1, 0])
            axes[1, 0].set_title('Parallel Coordinate Plot')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Parallel coordinate plot failed:\n{str(e)}', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Slice plot for best parameters
        try:
            optuna.visualization.matplotlib.plot_slice(self.study, ax=axes[1, 1])
            axes[1, 1].set_title('Slice Plot')
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Slice plot failed:\n{str(e)}', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('outputs/optuna_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Optimization plots saved to outputs/optuna_results.png")
    
    def plot_calibration_curve(self, X_val, y_val):
        """Plot calibration curve for the calibrated model"""
        
        if not self.calibrated_model:
            logger.warning("No calibrated model to evaluate. Run calibration first.")
            return
        
        # Get probabilities
        y_proba = self.calibrated_model.predict_proba(X_val)[:, 1]
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10)
        
        # Plot calibration curve
        plt.figure(figsize=(10, 6))
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Model calibration
        plt.plot(prob_pred, prob_true, 'b-', linewidth=2, label='Model Calibration')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Calibration curve saved to outputs/calibration_curve.png")
    
    def save_model(self):
        """Save the calibrated model"""
        
        if not self.calibrated_model:
            logger.warning("No calibrated model to save. Run calibration first.")
            return
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Save model
        model_path = "models/xgboost_calibrated.joblib"
        joblib.dump(self.calibrated_model, model_path)
        
        # Save best parameters
        params_path = "models/best_params.json"
        import json
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        logger.info(f"Calibrated model saved to {model_path}")
        logger.info(f"Best parameters saved to {params_path}")
    
    def generate_tuning_report(self):
        """Generate a comprehensive tuning report"""
        
        if not self.study:
            logger.warning("No study results to report. Run tuning first.")
            return
        
        logger.info("=== HYPERPARAMETER TUNING REPORT ===")
        
        # Study summary
        logger.info(f"Study Summary:")
        logger.info(f"  Number of trials: {len(self.study.trials)}")
        logger.info(f"  Best F1 Score: {self.best_score:.4f}")
        logger.info(f"  Best trial: {self.study.best_trial.number}")
        
        # Best parameters
        logger.info(f"Best Parameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        # Parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(self.study)
            logger.info("Parameter Importance:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {param}: {imp:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
        
        # Recommendations
        logger.info("\nRECOMMENDATIONS:")
        logger.info("1. Use the calibrated XGBoost model for production")
        logger.info("2. Monitor calibration in production")
        logger.info("3. Consider feature importance analysis for insights")
        logger.info("4. Set up monitoring for model drift")

def main():
    """Main tuning pipeline"""
    
    logger.info("Starting hyperparameter tuning with Optuna...")
    
    # Initialize tuner
    tuner = OptunaTuner(n_trials=30)  # Reduced for faster execution
    
    # Load data
    X_train, X_val, y_train, y_val = tuner.load_data()
    
    # Tune hyperparameters
    best_params, best_score = tuner.tune_hyperparameters(X_train, X_val, y_train, y_val)
    
    # Calibrate model
    calibrated_model = tuner.calibrate_model(X_train, X_val, y_train, y_val)
    
    # Plot results
    tuner.plot_optimization_results()
    tuner.plot_calibration_curve(X_val, y_val)
    
    # Save model
    tuner.save_model()
    
    # Generate report
    tuner.generate_tuning_report()
    
    logger.info("Hyperparameter tuning completed successfully!")
    
    return tuner

if __name__ == "__main__":
    tuner = main()
    print("Hyperparameter tuning completed!")
    print("Check the outputs/ directory for visualizations.")
