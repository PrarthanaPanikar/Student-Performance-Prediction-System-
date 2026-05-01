"""
Model Evaluation, Explainability & Fairness Analysis
Comprehensive evaluation with SHAP explanations and fairness checks
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    brier_score_loss, precision_recall_curve, roc_curve
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from pathlib import Path
import logging
import joblib
import json

# Import our preprocessing pipeline
from pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class to handle comprehensive model evaluation"""
    
    def __init__(self, model_path="models/xgboost_calibrated.joblib"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.evaluation_results = {}
    
    def load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        
        return self.model
    
    def load_data(self):
        """Load evaluation data"""
        logger.info("Loading evaluation data...")
        
        try:
            df = pd.read_parquet("data/students.parquet")
            logger.info(f"Loaded {len(df)} student records")
        except FileNotFoundError:
            logger.error("Data file not found")
            raise
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in TARGET_COLUMNS]
        X = df[feature_cols].copy()
        y = df['passed'].copy()
        
        return X, y, df
    
    def get_feature_names(self):
        """Extract feature names after preprocessing"""
        
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        # Get the preprocessor from the calibrated model
        if hasattr(self.model, 'base_estimator'):
            base_model = self.model.base_estimator_
        else:
            base_model = self.model
        
        preprocessor = base_model.named_steps['preprocessor']
        
        # Get numeric feature names (they stay the same)
        numeric_features = NUMERIC_FEATURES
        
        # Get categorical feature names (after one-hot encoding)
        categorical_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        categorical_feature_names = []
        
        if hasattr(categorical_encoder, 'get_feature_names_out'):
            # For newer sklearn versions
            cat_names = categorical_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            categorical_feature_names = list(cat_names)
        else:
            # For older sklearn versions - fallback
            for i, cat_feature in enumerate(CATEGORICAL_FEATURES):
                categories = categorical_encoder.categories_[i]
                for category in categories:
                    categorical_feature_names.append(f"{cat_feature}_{category}")
        
        # Combine all feature names
        self.feature_names = numeric_features + categorical_feature_names
        
        logger.info(f"Extracted {len(self.feature_names)} feature names")
        
        return self.feature_names
    
    def evaluate_model(self, X, y):
        """Comprehensive model evaluation"""
        
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return
        
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'roc_auc': roc_auc_score(y, y_proba),
            'brier_score': brier_score_loss(y, y_proba),
            'accuracy': np.mean(y == y_pred),
            'precision': np.mean((y == 1) & (y_pred == 1)) / np.mean(y_pred == 1),
            'recall': np.mean((y == 1) & (y_pred == 1)) / np.mean(y == 1),
            'f1': 2 * (np.mean((y == 1) & (y_pred == 1)) / np.mean(y_pred == 1) * 
                      np.mean((y == 1) & (y_pred == 1)) / np.mean(y == 1)) / 
                   ((np.mean((y == 1) & (y_pred == 1)) / np.mean(y_pred == 1) + 
                    np.mean((y == 1) & (y_pred == 1)) / np.mean(y == 1)))
        }
        
        self.evaluation_results = {
            'predictions': y_pred,
            'probabilities': y_proba,
            'metrics': metrics
        }
        
        # Log results
        logger.info("Model Performance Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        return self.evaluation_results
    
    def plot_evaluation_results(self):
        """Create comprehensive evaluation plots"""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results to plot. Run evaluate_model() first.")
            return
        
        # Create output directory
        Path("outputs").mkdir(exist_ok=True)
        
        y_pred = self.evaluation_results['predictions']
        y_proba = self.evaluation_results['probabilities']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Confusion Matrix
        cm = self.evaluation_results['metrics']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        axes[0, 1].plot(fpr, tpr, linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve (AUC = {self.evaluation_results["metrics"]["roc_auc"]:.3f})')
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_proba)
        axes[0, 2].plot(recall, precision, linewidth=2)
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        
        # Probability Distribution
        axes[1, 0].hist(y_proba[y == 0], bins=30, alpha=0.7, label='Fail', density=True)
        axes[1, 0].hist(y_proba[y == 1], bins=30, alpha=0.7, label='Pass', density=True)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Probability Distribution by Class')
        axes[1, 0].legend()
        
        # Calibration Plot
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
        axes[1, 1].plot([0, 1], [0, 1], 'k--')
        axes[1, 1].plot(prob_pred, prob_true, 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Calibration Plot')
        
        # Threshold Analysis
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            f1 = 2 * np.sum((y == 1) & (y_pred_thresh == 1)) / (
                np.sum(y_pred_thresh == 1) + np.sum(y == 1)
            )
            f1_scores.append(f1)
        
        axes[1, 2].plot(thresholds, f1_scores, 'b-', linewidth=2)
        axes[1, 2].set_xlabel('Threshold')
        axes[1, 2].set_ylabel('F1 Score')
        axes[1, 2].set_title('F1 Score vs Threshold')
        axes[1, 2].axvline(x=0.5, color='r', linestyle='--', label='Default Threshold')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Evaluation plots saved to outputs/evaluation_plots.png")
    
    def generate_shap_explanations(self, X, sample_size=100):
        """Generate SHAP explanations for model interpretability"""
        
        if not self.model or not self.feature_names:
            logger.error("Model or feature names not available")
            return
        
        logger.info("Generating SHAP explanations...")
        
        # Sample data for SHAP analysis
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        # Get the base model (without calibration)
        if hasattr(self.model, 'base_estimator'):
            base_model = self.model.base_estimator_
        else:
            base_model = self.model
        
        # Transform the data
        X_transformed = base_model.named_steps['preprocessor'].transform(X_sample)
        
        # Get the classifier
        classifier = base_model.named_steps['classifier']
        
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(classifier)
            shap_values = explainer(X_transformed)
            
            # Create output directory
            Path("outputs").mkdir(exist_ok=True)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_transformed, feature_names=self.feature_names, 
                            plot_type="bar", show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig('outputs/shap_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Detailed summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_transformed, feature_names=self.feature_names, 
                            show=False)
            plt.title('SHAP Summary Plot')
            plt.tight_layout()
            plt.savefig('outputs/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Waterfall plot for a single prediction
            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title('SHAP Waterfall Plot - Single Prediction')
            plt.tight_layout()
            plt.savefig('outputs/shap_waterfall.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Extract feature importance
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Save feature importance
            importance_df.to_csv('outputs/shap_feature_importance.csv', index=False)
            
            logger.info("SHAP explanations generated and saved")
            logger.info("Top 5 most important features:")
            for _, row in importance_df.head().iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return shap_values, importance_df
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return None, None
    
    def analyze_fairness(self, X, y, df):
        """Analyze model fairness across different demographic groups"""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate_model() first.")
            return
        
        logger.info("Analyzing model fairness...")
        
        y_proba = self.evaluation_results['probabilities']
        
        # Analyze fairness by different demographic groups
        demographic_cols = ['gender', 'school_type', 'parent_edu']
        fairness_results = {}
        
        for col in demographic_cols:
            if col not in df.columns:
                continue
            
            logger.info(f"Analyzing fairness by {col}...")
            
            group_results = {}
            
            for group_value in df[col].unique():
                if pd.isna(group_value):
                    continue
                
                # Get indices for this group
                group_mask = df[col] == group_value
                group_y = y[group_mask]
                group_proba = y_proba[group_mask]
                
                if len(group_y) < 10:  # Skip groups with too few samples
                    continue
                
                # Calculate metrics for this group
                group_metrics = {
                    'size': len(group_y),
                    'pass_rate': group_y.mean(),
                    'mean_predicted_prob': group_proba.mean(),
                    'roc_auc': roc_auc_score(group_y, group_proba) if len(np.unique(group_y)) > 1 else 0,
                    'brier_score': brier_score_loss(group_y, group_proba)
                }
                
                group_results[str(group_value)] = group_metrics
                
                logger.info(f"  {group_value}: n={len(group_y)}, "
                           f"pass_rate={group_y.mean():.2%}, "
                           f"roc_auc={group_metrics['roc_auc']:.3f}")
            
            fairness_results[col] = group_results
        
        # Create fairness plots
        self.plot_fairness_analysis(fairness_results)
        
        # Save fairness results
        Path("outputs").mkdir(exist_ok=True)
        with open('outputs/fairness_analysis.json', 'w') as f:
            json.dump(fairness_results, f, indent=2)
        
        logger.info("Fairness analysis completed and saved")
        
        return fairness_results
    
    def plot_fairness_analysis(self, fairness_results):
        """Create fairness visualization plots"""
        
        if not fairness_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        for demographic, groups in fairness_results.items():
            if plot_idx >= len(axes):
                break
            
            # Prepare data for plotting
            group_names = list(groups.keys())
            pass_rates = [groups[g]['pass_rate'] for g in group_names]
            pred_probs = [groups[g]['mean_predicted_prob'] for g in group_names]
            roc_aucs = [groups[g]['roc_auc'] for g in group_names]
            
            # Create subplot
            x = np.arange(len(group_names))
            width = 0.25
            
            axes[plot_idx].bar(x - width, pass_rates, width, label='Actual Pass Rate', alpha=0.8)
            axes[plot_idx].bar(x, pred_probs, width, label='Mean Predicted Prob', alpha=0.8)
            axes[plot_idx].bar(x + width, roc_aucs, width, label='ROC-AUC', alpha=0.8)
            
            axes[plot_idx].set_xlabel(demographic.replace('_', ' ').title())
            axes[plot_idx].set_ylabel('Score')
            axes[plot_idx].set_title(f'Fairness Analysis by {demographic.replace("_", " ").title()}')
            axes[plot_idx].set_xticks(x)
            axes[plot_idx].set_xticklabels(group_names, rotation=45)
            axes[plot_idx].legend()
            axes[plot_idx].set_ylim(0, 1)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('outputs/fairness_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        
        if not self.evaluation_results:
            logger.warning("No evaluation results to report")
            return
        
        logger.info("=== MODEL EVALUATION REPORT ===")
        
        metrics = self.evaluation_results['metrics']
        
        logger.info("Overall Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        # Classification report details
        logger.info("\nDetailed Classification Report:")
        for class_label, class_metrics in metrics['classification_report'].items():
            if isinstance(class_metrics, dict):
                logger.info(f"  {class_label}:")
                for metric, value in class_metrics.items():
                    if isinstance(value, float):
                        logger.info(f"    {metric}: {value:.4f}")
        
        logger.info("\nRECOMMENDATIONS:")
        logger.info("1. Review SHAP feature importance for insights")
        logger.info("2. Monitor fairness metrics across demographic groups")
        logger.info("3. Consider threshold adjustment based on business needs")
        logger.info("4. Set up ongoing monitoring for model drift")

def main():
    """Main evaluation pipeline"""
    
    logger.info("Starting comprehensive model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model and data
    evaluator.load_model()
    evaluator.get_feature_names()
    X, y, df = evaluator.load_data()
    
    # Evaluate model
    results = evaluator.evaluate_model(X, y)
    
    # Create evaluation plots
    evaluator.plot_evaluation_results()
    
    # Generate SHAP explanations
    shap_values, importance_df = evaluator.generate_shap_explanations(X)
    
    # Analyze fairness
    fairness_results = evaluator.analyze_fairness(X, y, df)
    
    # Generate report
    evaluator.generate_evaluation_report()
    
    logger.info("Model evaluation completed successfully!")
    
    return evaluator

if __name__ == "__main__":
    evaluator = main()
    print("Model evaluation completed!")
    print("Check the outputs/ directory for all visualizations and reports.")
