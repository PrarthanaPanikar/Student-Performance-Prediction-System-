"""
Main entry point for the Student Performance Prediction System
Provides a unified interface to run the complete pipeline
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_data_generation():
    """Generate synthetic student data"""
    logger.info("🔧 Step 1: Generating synthetic student data...")
    
    try:
        from data.generate_synthetic_data import generate_student_data
        
        df = generate_student_data(n_students=5000)
        logger.info(f"✅ Generated dataset with {len(df)} students")
        logger.info(f"   Pass rate: {df['passed'].mean():.2%}")
        return True
    except Exception as e:
        logger.error(f"❌ Data generation failed: {e}")
        return False

def run_data_ingestion():
    """Ingest and validate data"""
    logger.info("🔧 Step 2: Data ingestion and validation...")
    
    try:
        from notebooks.ingest import load_and_validate_data
        
        df = load_and_validate_data()
        logger.info(f"✅ Data ingestion completed: {df.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Data ingestion failed: {e}")
        return False

def run_eda():
    """Run exploratory data analysis"""
    logger.info("🔧 Step 3: Exploratory data analysis...")
    
    try:
        from notebooks.eda import main as eda_main
        
        df, feature_cols, target_cols = eda_main()
        logger.info("✅ EDA completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ EDA failed: {e}")
        return False

def run_baseline_training():
    """Train baseline models"""
    logger.info("🔧 Step 4: Training baseline models...")
    
    try:
        from src.train_baselines import main as train_main
        
        trainer, best_model = train_main()
        logger.info("✅ Baseline training completed")
        return True
    except Exception as e:
        logger.error(f"❌ Baseline training failed: {e}")
        return False

def run_hyperparameter_tuning():
    """Run hyperparameter tuning"""
    logger.info("🔧 Step 5: Hyperparameter tuning with Optuna...")
    
    try:
        from src.tune_optuna import main as tune_main
        
        tuner = tune_main()
        logger.info("✅ Hyperparameter tuning completed")
        return True
    except Exception as e:
        logger.error(f"❌ Hyperparameter tuning failed: {e}")
        return False

def run_evaluation():
    """Run model evaluation"""
    logger.info("🔧 Step 6: Model evaluation and explainability...")
    
    try:
        from src.evaluate import main as eval_main
        
        evaluator = eval_main()
        logger.info("✅ Model evaluation completed")
        return True
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    logger.info("🚀 Starting FastAPI server...")
    
    try:
        import uvicorn
        uvicorn.run("serving.app:app", host="0.0.0.0", port=8000, reload=True)
        return True
    except Exception as e:
        logger.error(f"❌ API server failed to start: {e}")
        return False

def run_full_pipeline():
    """Run the complete ML pipeline"""
    logger.info("🚀 Starting complete Student Performance Prediction pipeline...")
    
    steps = [
        ("Data Generation", run_data_generation),
        ("Data Ingestion", run_data_ingestion),
        ("Exploratory Analysis", run_eda),
        ("Baseline Training", run_baseline_training),
        ("Hyperparameter Tuning", run_hyperparameter_tuning),
        ("Model Evaluation", run_evaluation)
    ]
    
    successful_steps = 0
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Running: {step_name}")
        logger.info(f"{'='*60}")
        
        if step_func():
            successful_steps += 1
            logger.info(f"✅ {step_name} completed successfully")
        else:
            logger.error(f"❌ {step_name} failed")
            logger.error("⏹️ Pipeline stopped due to error")
            break
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Pipeline Summary: {successful_steps}/{len(steps)} steps completed")
    logger.info(f"{'='*60}")
    
    if successful_steps == len(steps):
        logger.info("🎉 Complete pipeline finished successfully!")
        logger.info("📊 Results available in 'outputs/' directory")
        logger.info("🤖 Model saved in 'models/' directory")
        logger.info("🚀 Start API server with: python main.py --api")
        logger.info("🌐 Start dashboard with: cd apps/web && npm run dev")
    else:
        logger.error("❌ Pipeline incomplete. Check logs for details.")
    
    return successful_steps == len(steps)

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Student Performance Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --data            # Generate data only
  python main.py --train           # Train models only
  python main.py --api             # Start API server
  python main.py --eval            # Evaluation only
        """
    )
    
    parser.add_argument(
        '--data', 
        action='store_true',
        help='Generate synthetic data only'
    )
    
    parser.add_argument(
        '--ingest', 
        action='store_true',
        help='Run data ingestion only'
    )
    
    parser.add_argument(
        '--eda', 
        action='store_true',
        help='Run exploratory analysis only'
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Train baseline models only'
    )
    
    parser.add_argument(
        '--tune', 
        action='store_true',
        help='Run hyperparameter tuning only'
    )
    
    parser.add_argument(
        '--eval', 
        action='store_true',
        help='Run model evaluation only'
    )
    
    parser.add_argument(
        '--api', 
        action='store_true',
        help='Start FastAPI server'
    )
    
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Run complete pipeline (default)'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("images").mkdir(exist_ok=True)
    
    print("🎓 Student Performance Prediction System")
    print("=" * 50)
    
    # Execute based on arguments
    if args.api:
        start_api_server()
    elif args.data:
        run_data_generation()
    elif args.ingest:
        run_data_ingestion()
    elif args.eda:
        run_eda()
    elif args.train:
        run_baseline_training()
    elif args.tune:
        run_hyperparameter_tuning()
    elif args.eval:
        run_evaluation()
    else:
        # Default: run full pipeline
        run_full_pipeline()

if __name__ == "__main__":
    main()
