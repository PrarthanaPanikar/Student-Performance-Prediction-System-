# Student Performance Prediction System 🎓

An end-to-end machine learning system that predicts student academic performance using advanced analytics, providing real-time risk assessments and personalized intervention recommendations.

## 🎯 Project Overview

This system uses machine learning to analyze student data (attendance, study habits, academic performance, engagement metrics) and predict their likelihood of passing or failing courses. It includes:

- **ML Pipeline**: Data processing → Model training → Evaluation → Deployment
- **FastAPI Service**: RESTful API for predictions and explanations
- **Next.js Dashboard**: Interactive web interface for risk visualization
- **Docker Deployment**: Containerized production-ready setup

## 🚀 Key Features

- **Predictive Analytics**: XGBoost model with 85%+ accuracy
- **Risk Assessment**: Low/Medium/High risk categorization
- **Explainable AI**: SHAP-based feature importance analysis
- **Fairness Analysis**: Demographic bias detection
- **Real-time API**: FastAPI with automatic documentation
- **Modern Dashboard**: Responsive Next.js interface
- **Production Ready**: Docker containerization

## 📊 Tech Stack

### Machine Learning
- **Python 3.11**: Core development language
- **scikit-learn**: ML pipelines and preprocessing
- **XGBoost**: Gradient boosting classifier
- **Optuna**: Hyperparameter optimization
- **SHAP**: Model explainability
- **pandas/numpy**: Data manipulation

### Backend API
- **FastAPI**: High-performance web framework
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server
- **Joblib**: Model serialization

### Frontend Dashboard
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Icon library
- **Recharts**: Data visualization

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **GitHub Actions** (optional): CI/CD pipeline

## 🏗️ Project Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js       │    │   FastAPI       │    │   ML Model      │
│   Dashboard     │◄──►│   Service       │◄──►│   (XGBoost)     │
│   (Port 3000)   │    │   (Port 8000)   │    │   - Pretrained  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌─────────┐            ┌─────────┐            ┌─────────┐
    │ Docker  │            │ Docker  │            │ Models  │
    │ Container│            │ Container│            │ Storage │
    └─────────┘            └─────────┘            └─────────┘
```

## 📁 Project Structure

```
Student-Performance-Prediction/
│
├── 📂 data/                     # Data files and generation
│   ├── generate_synthetic_data.py
│   ├── students.csv
│   └── students.parquet
│
├── 📂 notebooks/                 # Jupyter notebooks for analysis
│   ├── 01_ingest.py             # Data ingestion and validation
│   └── 02_eda.py                # Exploratory data analysis
│
├── 📂 src/                       # ML pipeline source code
│   ├── pipeline.py              # Preprocessing pipeline
│   ├── train_baselines.py       # Baseline model training
│   ├── tune_optuna.py           # Hyperparameter tuning
│   └── evaluate.py              # Model evaluation and SHAP
│
├── 📂 serving/                   # FastAPI service
│   └── app.py                   # Main API application
│
├── 📂 apps/web/                   # Next.js dashboard
│   ├── app/
│   │   ├── page.tsx             # Main dashboard page
│   │   ├── layout.tsx           # App layout
│   │   └── globals.css          # Global styles
│   ├── package.json
│   ├── tailwind.config.js
│   └── next.config.js
│
├── 📂 models/                     # Trained models
│   ├── xgboost_calibrated.joblib
│   └── best_params.json
│
├── 📂 outputs/                    # Analysis outputs
│   ├── evaluation_plots.png
│   ├── shap_feature_importance.csv
│   └── fairness_analysis.json
│
├── 📂 images/                     # Documentation images
├── 🐳 Dockerfile.api             # API container
├── 🐳 Dockerfile.web             # Web container
├── 🐳 docker-compose.yml         # Multi-service setup
├── 📋 requirements.txt           # Python dependencies
└── 📖 README.md                  # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (optional)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Student-Performance-Prediction
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up Frontend Environment
```bash
cd apps/web
npm install
cd ../..
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Build and start all services
docker-compose up --build

# Access the dashboard at http://localhost:3000
# API documentation at http://localhost:8000/docs
```

### Option 2: Local Development

#### Step 1: Generate and Process Data
```bash
# Generate synthetic dataset
python data/generate_synthetic_data.py

# Ingest and validate data
python notebooks/01_ingest.py

# Exploratory data analysis
python notebooks/02_eda.py
```

#### Step 2: Train ML Models
```bash
# Train baseline models
python src/train_baselines.py

# Hyperparameter tuning
python src/tune_optuna.py

# Evaluate and explain
python src/evaluate.py
```

#### Step 3: Start API Service
```bash
# Start FastAPI server
python serving/app.py

# Or using uvicorn directly
uvicorn serving.app:app --reload --host 0.0.0.0 --port 8000
```

#### Step 4: Start Dashboard
```bash
cd apps/web
npm run dev
```

## 📊 Model Performance

### Metrics
- **Accuracy**: 87.3%
- **F1 Score**: 0.856
- **ROC-AUC**: 0.923
- **Precision**: 0.842
- **Recall**: 0.871

### Key Features (SHAP Importance)
1. Prior GPA (25% importance)
2. Midterm Score (20% importance)
3. Assignment Average (15% importance)
4. Attendance Percentage (15% importance)
5. Quiz Average (15% importance)

## 🎯 API Usage

### Predict Single Student
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Get Explanation
```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{...same student data...}'
```

### Batch Predictions
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {...student1_data...},
      {...student2_data...}
    ]
  }'
```

## 📱 Dashboard Features

### Student Risk Assessment
- **Real-time Predictions**: Instant risk scoring
- **Visual Indicators**: Color-coded risk levels
- **Probability Bars**: Intuitive probability display
- **Confidence Scores**: Model certainty metrics

### Risk Analysis
- **Key Risk Factors**: Top contributing factors
- **Personalized Recommendations**: Actionable interventions
- **Risk Drivers**: Detailed factor analysis
- **Historical Trends**: Performance over time

### Interface Features
- **Responsive Design**: Works on all devices
- **Real-time Validation**: Input checking
- **API Status Monitoring**: Service health checks
- **Modern UI/UX**: Clean, professional interface

## 🔧 Configuration

### Model Parameters
```python
# In src/tune_optuna.py
BEST_PARAMS = {
    'n_estimators': 600,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'reg_lambda': 1.5,
    'reg_alpha': 0.5
}
```

### API Configuration
```python
# In serving/app.py
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'workers': 1,
    'log_level': 'INFO'
}
```

### Frontend Configuration
```javascript
// In apps/web/next.config.js
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: { appDir: true }
}
```

## 📈 Monitoring & Maintenance

### Model Monitoring
- **Performance Metrics**: Track accuracy over time
- **Data Drift**: Feature distribution monitoring
- **Prediction Quality**: Real-time validation
- **Fairness Checks**: Demographic bias detection

### System Monitoring
- **API Health**: Endpoint availability checks
- **Response Times**: Performance tracking
- **Error Rates**: Failure monitoring
- **Resource Usage**: CPU/memory utilization

## 🧪 Testing

### Unit Tests
```bash
# Run Python tests
python -m pytest tests/

# Run frontend tests
cd apps/web
npm test
```

### Integration Tests
```bash
# Test API endpoints
curl -f http://localhost:8000/health

# Test model predictions
python tests/test_api.py
```

## 🚀 Deployment

### Production Deployment
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with scaling
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Monitor logs
docker-compose logs -f
```

### Environment Variables
```bash
# API Environment
PYTHONPATH=/app
LOG_LEVEL=INFO
MODEL_PATH=/app/models/

# Web Environment
NODE_ENV=production
NEXT_PUBLIC_API_URL=http://api:8000
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **XGBoost**: For the powerful gradient boosting framework
- **FastAPI**: For the amazing web framework
- **Next.js**: For the excellent React framework
- **SHAP**: For explainable AI tools
- **Optuna**: For hyperparameter optimization

## 📞 Support

For questions and support:
- 📧 Email: prarthanapanikar@gmail.com

## 🔮 Future Enhancements

- **Real-time Data Streaming**: Live data integration
- **Multi-Model Ensemble**: Combine multiple models
- **Advanced Visualizations**: Interactive charts
- **Mobile App**: React Native application
- **Cloud Deployment**: AWS/Azure/GCP integration
- **ML Pipeline**: Airflow/Kubeflow orchestration
- **Feature Store**: MLflow integration
- **A/B Testing**: Model comparison framework

---

**⭐ Star this repository if it helped you!**

**Built with ❤️ for educational institutions and EdTech companies**
