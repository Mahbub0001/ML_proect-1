# HSC Result Predictor - Bangladesh Student Performance Analysis

A machine learning project that predicts Higher Secondary Certificate (HSC) results for Bangladeshi students based on various academic and socio-economic factors.

## 🎯 Project Overview

This project uses ensemble learning techniques to predict HSC results (GPA) with high accuracy. The model analyzes 13 different features including demographic information, parental education, family background, and previous academic performance to forecast student outcomes.

## 📊 Dataset

The project uses the **Bangladesh Student Performance Dataset** containing 2,018 student records with the following features:

### Features
- **Demographics**: gender, address (Urban/Rural), family size
- **Family Background**: parent status, mother's education, father's education, mother's job, father's job
- **Lifestyle**: relationship status, smoking habits, time spent with friends
- **Academic**: SSC result (previous GPA), tuition fees
- **Target Variable**: HSC result (predicted GPA)

### Dataset Statistics
- **Total Records**: 2,018 students
- **Numerical Features**: 5 (M_Edu, F_Edu, tuition_fee, time_friends, ssc_result)
- **Categorical Features**: 8 (gender, address, famsize, Pstatus, M_Job, F_Job, relationship, smoker)
- **Target Range**: 2.0 - 5.0 GPA

## 🚀 Model Performance

### Model Comparison Results
| Model | R² Score | MAE |
|-------|----------|-----|
| **Stacking Ensemble** | **0.958** | **0.101** |
| Gradient Boosting | 0.958 | 0.101 |
| Voting Ensemble | 0.958 | 0.101 |
| Random Forest | 0.951 | 0.109 |
| Linear Regression | 0.946 | 0.113 |

### Final Model
- **Algorithm**: Gradient Boosting Regressor (optimized)
- **R² Score**: 0.939
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 6
  - random_state: 42

## 🛠️ Technology Stack

### Core Libraries
- **pandas**: 3.0.0 - Data manipulation and analysis
- **numpy**: 2.4.1 - Numerical computations
- **scikit-learn**: 1.8.0 - Machine learning algorithms
- **gradio**: 6.5.1 - Web interface for ML models

### ML Algorithms Used
- Linear Regression
- Random Forest
- Gradient Boosting
- Voting Regressor
- Stacking Regressor

### Data Preprocessing
- One-Hot Encoding for categorical variables
- StandardScaler for numerical features
- SimpleImputer for missing values

## 📁 Project Structure

```
ML_project_1/
├── app.py                          # Gradio web application
├── recreate_model.py                # Model recreation script
├── ML_project1_hsc_result_pred.ipynb # Jupyter notebook with analysis
├── bangladesh_student_performance.csv # Dataset
├── gbr_model_fixed.pkl             # Trained model (compatible)
├── gbr_mode.pkl                   # Original model (deprecated)
├── requirements.txt                # Python dependencies
├── report (1).html                # EDA report
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML_project_1
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Unix/MacOS
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Launch the web interface**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Local URL: `http://127.0.0.1:7860`
   - The app will also generate a shareable public link

## 🎮 Web Interface Features

The Gradio-based web application provides:

- **Interactive Input Fields**:
  - Gender selection (Male/Female)
  - Address type (Urban/Rural)
  - Family size (GT3/LE3)
  - Parent status (Together/Apart)
  - Parental education sliders (0-4 scale)
  - Parental job dropdowns
  - Lifestyle choices
  - Academic inputs (tuition fees, SSC result)

- **Real-time Prediction**: Instant HSC result prediction
- **User-friendly Design**: Intuitive interface with clear labels
- **Result Formatting**: Predictions clipped to valid GPA range (0-5)

## 🔧 Model Development Process

### Data Preprocessing
1. **Data Cleaning**: Removed irrelevant features (date, age)
2. **Feature Engineering**: Converted categorical variables to strings
3. **Pipeline Creation**: Built preprocessing pipelines for numerical and categorical features
4. **Train-Test Split**: 67% training, 33% testing (random_state=42)

### Model Selection
1. **Baseline Models**: Tested multiple algorithms
2. **Ensemble Methods**: Implemented voting and stacking regressors
3. **Hyperparameter Tuning**: GridSearchCV for optimization
4. **Final Selection**: Chose best-performing model

### Key Insights
- **SSC Result**: Strongest predictor (correlation: 0.950)
- **Parental Education**: Positive correlation with HSC results
- **Time with Friends**: Negative correlation (-0.156)
- **Tuition Fees**: Slight positive correlation

## 🐛 Known Issues & Solutions

### StringDtype Compatibility Issue
- **Problem**: Original model had pandas StringDtype compatibility issues
- **Solution**: Created `recreate_model.py` with string conversion
- **Fixed Model**: `gbr_model_fixed.pkl` is compatible with current environment

## 📈 Model Evaluation

### Performance Metrics
- **R² Score**: 0.939 (93.9% variance explained)
- **MAE**: ~0.10 GPA points
- **Prediction Range**: 2.0 - 5.0 GPA

### Validation Approach
- **Cross-validation**: 5-fold CV during hyperparameter tuning
- **Holdout Test Set**: 33% of data for final evaluation
- **Multiple Metrics**: R², MAE for comprehensive assessment

## 🔮 Future Enhancements

1. **Feature Engineering**: Add more socio-economic indicators
2. **Model Optimization**: Try advanced algorithms (XGBoost, LightGBM)
3. **Deployment**: Docker containerization for production
4. **API Development**: RESTful API for integration
5. **Real-time Analytics**: Dashboard for monitoring predictions

## 📝 Usage Examples

### Python API Usage
```python
import pickle
import pandas as pd

# Load the model
with open('gbr_model_fixed.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare input data
input_data = pd.DataFrame([[
    'M', 'Urban', 'LE3', 'Together', 3, 2, 
    'Teacher', 'Services', 'No', 'No', 50000, 3, 4.0
]], columns=[
    'gender', 'address', 'famsize', 'Pstatus', 'M_Edu', 'F_Edu',
    'M_Job', 'F_Job', 'relationship', 'smoker', 'tuition_fee', 
    'time_friends', 'ssc_result'
])

# Make prediction
prediction = model.predict(input_data)[0]
print(f"Predicted HSC Result: {prediction:.2f}")
```

## 📄 License

This project is for educational and research purposes. Please cite if used in academic work.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or suggestions regarding this project, please open an issue in the repository.

---

**Note**: This model is trained on Bangladeshi student data and may not generalize well to other educational systems or populations. Use with appropriate caution and validation.
