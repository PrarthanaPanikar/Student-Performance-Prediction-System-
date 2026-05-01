"""
Data Drift Detection and Monitoring
Monitors feature distributions and model performance over time
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriftDetector:
    """Class for detecting data drift and model performance degradation"""
    
    def __init__(self, reference_data_path: str = "data/students.parquet", 
                 model_path: str = "models/xgboost_calibrated.joblib",
                 threshold: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            reference_data_path: Path to reference/training data
            model_path: Path to trained model
            threshold: Significance threshold for drift detection
        """
        self.reference_data_path = reference_data_path
        self.model_path = model_path
        self.threshold = threshold
        self.reference_data = None
        self.model = None
        self.feature_columns = None
        self.drift_results = {}
        
    def load_reference_data(self):
        """Load reference data for comparison"""
        try:
            self.reference_data = pd.read_parquet(self.reference_data_path)
            logger.info(f"Loaded reference data: {len(self.reference_data)} samples")
            
            # Identify feature columns (exclude targets)
            target_columns = ['final_score', 'final_grade_band', 'passed']
            self.feature_columns = [col for col in self.reference_data.columns 
                                   if col not in target_columns]
            
            return True
            
        except FileNotFoundError:
            logger.error(f"Reference data not found: {self.reference_data_path}")
            return False
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
            return True
            
        except FileNotFoundError:
            logger.error(f"Model not found: {self.model_path}")
            return False
    
    def detect_distribution_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect drift in feature distributions using KS test
        
        Args:
            current_data: Current data to compare against reference
            
        Returns:
            Dictionary with drift statistics for each feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded")
        
        drift_results = {}
        
        for feature in self.feature_columns:
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} not found in current data")
                continue
            
            # Get reference and current distributions
            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                logger.warning(f"Insufficient data for feature {feature}")
                continue
            
            # Perform KS test for numerical features
            if ref_values.dtype in ['float64', 'int64']:
                ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                
                drift_results[feature] = {
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < self.threshold,
                    'reference_mean': ref_values.mean(),
                    'current_mean': curr_values.mean(),
                    'reference_std': ref_values.std(),
                    'current_std': curr_values.std(),
                    'feature_type': 'numerical'
                }
            
            # Chi-square test for categorical features
            else:
                # Create contingency table
                all_categories = list(set(ref_values.unique()) | set(curr_values.unique()))
                
                ref_counts = [ref_values.value_counts().get(cat, 0) for cat in all_categories]
                curr_counts = [curr_values.value_counts().get(cat, 0) for cat in all_categories]
                
                # Perform chi-square test
                try:
                    chi2_stat, p_value, _, _ = stats.chi2_contingency([ref_counts, curr_counts])
                    
                    drift_results[feature] = {
                        'chi2_statistic': chi2_stat,
                        'p_value': p_value,
                        'drift_detected': p_value < self.threshold,
                        'reference_distribution': dict(zip(all_categories, ref_counts)),
                        'current_distribution': dict(zip(all_categories, curr_counts)),
                        'feature_type': 'categorical'
                    }
                except Exception as e:
                    logger.warning(f"Chi-square test failed for {feature}: {e}")
        
        return drift_results
    
    def detect_performance_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect drift in model performance
        
        Args:
            current_data: Current data with labels
            
        Returns:
            Dictionary with performance metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Check if labels are available
        if 'passed' not in current_data.columns:
            logger.warning("No labels available for performance drift detection")
            return {'error': 'No labels available'}
        
        # Prepare features
        feature_cols = [col for col in self.feature_columns if col in current_data.columns]
        X_current = current_data[feature_cols]
        y_current = current_data['passed']
        
        # Make predictions
        y_pred = self.model.predict(X_current)
        y_proba = self.model.predict_proba(X_current)[:, 1]
        
        # Calculate performance metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        performance = {
            'accuracy': accuracy_score(y_current, y_pred),
            'precision': precision_score(y_current, y_pred),
            'recall': recall_score(y_current, y_pred),
            'f1': f1_score(y_current, y_pred),
            'roc_auc': roc_auc_score(y_current, y_proba),
            'sample_size': len(current_data)
        }
        
        # Calculate baseline performance on reference data
        if self.reference_data is not None:
            X_ref = self.reference_data[feature_cols]
            y_ref = self.reference_data['passed']
            
            y_pred_ref = self.model.predict(X_ref)
            y_proba_ref = self.model.predict_proba(X_ref)[:, 1]
            
            baseline_performance = {
                'accuracy': accuracy_score(y_ref, y_pred_ref),
                'precision': precision_score(y_ref, y_pred_ref),
                'recall': recall_score(y_ref, y_pred_ref),
                'f1': f1_score(y_ref, y_pred_ref),
                'roc_auc': roc_auc_score(y_ref, y_proba_ref),
                'sample_size': len(self.reference_data)
            }
            
            # Calculate performance degradation
            performance['baseline'] = baseline_performance
            performance['accuracy_degradation'] = baseline_performance['accuracy'] - performance['accuracy']
            performance['f1_degradation'] = baseline_performance['f1'] - performance['f1']
            performance['roc_auc_degradation'] = baseline_performance['roc_auc'] - performance['roc_auc']
        
        return performance
    
    def generate_drift_report(self, current_data: pd.DataFrame, output_path: str = "outputs/drift_report.json"):
        """
        Generate comprehensive drift report
        
        Args:
            current_data: Current data for analysis
            output_path: Path to save report
        """
        logger.info("Generating drift report...")
        
        # Load reference data and model
        if not self.load_reference_data() or not self.load_model():
            return None
        
        # Detect distribution drift
        distribution_drift = self.detect_distribution_drift(current_data)
        
        # Detect performance drift
        performance_drift = self.detect_performance_drift(current_data)
        
        # Summary statistics
        total_features = len(distribution_drift)
        drifted_features = sum(1 for result in distribution_drift.values() if result.get('drift_detected', False))
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'reference_data_size': len(self.reference_data),
            'current_data_size': len(current_data),
            'total_features_analyzed': total_features,
            'features_with_drift': drifted_features,
            'drift_percentage': (drifted_features / total_features * 100) if total_features > 0 else 0,
            'distribution_drift': distribution_drift,
            'performance_drift': performance_drift,
            'summary': {
                'significant_drift': drifted_features > (total_features * 0.2),  # More than 20% features drifted
                'performance_degradation': performance_drift.get('f1_degradation', 0) < -0.05,  # More than 5% F1 degradation
                'needs_attention': (drifted_features > (total_features * 0.2)) or (performance_drift.get('f1_degradation', 0) < -0.05)
            }
        }
        
        # Save report
        Path("outputs").mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Drift report saved to {output_path}")
        logger.info(f"Features with drift: {drifted_features}/{total_features}")
        
        if report['summary']['needs_attention']:
            logger.warning("⚠️  Significant drift detected - model retraining recommended")
        
        return report
    
    def plot_drift_visualizations(self, current_data: pd.DataFrame, output_dir: str = "outputs"):
        """
        Create visualizations for drift analysis
        
        Args:
            current_data: Current data for analysis
            output_dir: Directory to save plots
        """
        if self.reference_data is None:
            logger.error("Reference data not loaded")
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get drift results
        drift_results = self.detect_distribution_drift(current_data)
        
        # Plot top drifted features
        drifted_features = [(feature, result) for feature, result in drift_results.items() 
                           if result.get('drift_detected', False)]
        
        if not drifted_features:
            logger.info("No significant drift detected")
            return
        
        # Sort by KS statistic (for numerical features) or p-value
        drifted_features.sort(key=lambda x: x[1].get('ks_statistic', 0) if x[1].get('feature_type') == 'numerical' 
                           else x[1].get('p_value', 1), reverse=True)
        
        # Plot top 10 drifted features
        top_features = drifted_features[:10]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, (feature, result) in enumerate(top_features):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            if result['feature_type'] == 'numerical':
                # Plot distributions
                ref_data = self.reference_data[feature].dropna()
                curr_data = current_data[feature].dropna()
                
                ax.hist(ref_data, bins=30, alpha=0.7, label='Reference', density=True)
                ax.hist(curr_data, bins=30, alpha=0.7, label='Current', density=True)
                ax.set_title(f'{feature}\nKS: {result["ks_statistic"]:.3f}')
                ax.legend()
            
            else:
                # Plot categorical distributions
                ref_dist = result['reference_distribution']
                curr_dist = result['current_distribution']
                
                categories = list(ref_dist.keys())
                ref_counts = list(ref_dist.values())
                curr_counts = list(curr_dist.values())
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax.bar(x - width/2, ref_counts, width, label='Reference', alpha=0.7)
                ax.bar(x + width/2, curr_counts, width, label='Current', alpha=0.7)
                ax.set_title(f'{feature}\nChi2: {result["chi2_statistic"]:.3f}')
                ax.set_xticks(x)
                ax.set_xticklabels(categories, rotation=45)
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/drift_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Drift visualizations saved to {output_dir}/drift_visualization.png")
    
    def setup_monitoring(self, check_interval_hours: int = 24):
        """
        Set up continuous monitoring (placeholder for production implementation)
        
        Args:
            check_interval_hours: How often to check for drift
        """
        logger.info(f"Setting up drift monitoring with {check_interval_hours}h interval")
        logger.info("Note: In production, this would be implemented with scheduled jobs")
        
        # This would be implemented with:
        # - Cron jobs
        # - Airflow DAGs
        # - Kubernetes CronJobs
        # - Cloud monitoring services

def simulate_drift_detection():
    """Simulate drift detection with synthetic data"""
    logger.info("Simulating drift detection...")
    
    # Initialize drift detector
    detector = DriftDetector()
    
    # Load reference data
    if not detector.load_reference_data():
        return
    
    # Generate current data with some drift
    from data.generate_synthetic_data import generate_student_data
    
    # Generate data with shifted parameters to simulate drift
    np.random.seed(123)  # Different seed for variation
    
    current_data = generate_student_data(n_students=1000, random_state=123)
    
    # Introduce some drift
    # Shift attendance down by 10%
    current_data['attendance_pct'] = np.clip(
        current_data['attendance_pct'] - 10, 0, 100
    )
    
    # Shift study hours down
    current_data['study_hours_wk'] = np.clip(
        current_data['study_hours_wk'] - 2, 0, 50
    )
    
    # Change school type distribution
    current_data.loc[current_data.sample(frac=0.3).index, 'school_type'] = 'Charter'
    
    # Generate drift report
    report = detector.generate_drift_report(current_data)
    
    # Create visualizations
    detector.plot_drift_visualizations(current_data)
    
    return report

def main():
    """Main function for drift monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data drift detection and monitoring")
    parser.add_argument('--simulate', action='store_true', help='Simulate drift detection')
    parser.add_argument('--current-data', type=str, help='Path to current data file')
    parser.add_argument('--threshold', type=float, default=0.05, help='Drift detection threshold')
    
    args = parser.parse_args()
    
    if args.simulate:
        # Run simulation
        report = simulate_drift_detection()
        
        if report:
            print("\n🔍 Drift Detection Summary:")
            print(f"Features analyzed: {report['total_features_analyzed']}")
            print(f"Features with drift: {report['features_with_drift']}")
            print(f"Drift percentage: {report['drift_percentage']:.1f}%")
            
            if report['summary']['needs_attention']:
                print("⚠️  Significant drift detected - model retraining recommended")
            else:
                print("✅ No significant drift detected")
    
    elif args.current_data:
        # Analyze provided data
        detector = DriftDetector(threshold=args.threshold)
        
        try:
            current_data = pd.read_parquet(args.current_data)
            report = detector.generate_drift_report(current_data)
            detector.plot_drift_visualizations(current_data)
            
            print(f"Drift analysis completed. Report saved to outputs/drift_report.json")
            
        except FileNotFoundError:
            print(f"Error: Current data file not found: {args.current_data}")
    
    else:
        print("Please specify --simulate or --current-data path")

if __name__ == "__main__":
    main()
