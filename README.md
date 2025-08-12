# Loan Approval Prediction

A machine learning project that predicts loan approval status using applicant information. The model focuses on handling imbalanced data and provides comprehensive evaluation metrics for financial decision-making.

## Project Overview

This project implements a binary classification model to predict whether a loan application will be **Approved** or **Rejected** based on various applicant features including income, credit score, assets, and demographics.

### Key Features

- **Smart Preprocessing**: Handles missing values and encodes categorical features automatically
- **Imbalanced Data Handling**: Uses balanced logistic regression for fair predictions
- **Comprehensive Evaluation**: Focuses on precision, recall, and F1-score metrics
- **Visual Analytics**: Generates precision-recall curves and confusion matrices
- **Production Ready**: Clean, modular code with proper error handling

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd loan-approval-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

1. **Prepare your data**: Place your dataset as `loan_approval_dataset.csv` in the project directory

2. **Run the prediction model**:
```bash
python loan_prediction.py
```

3. **View results**: The script will display metrics and generate visualization plots

## Model Performance

Our model achieves excellent performance on the loan approval prediction task:

### Classification Metrics

| Metric | Class 0 (Rejected) | Class 1 (Approved) | Overall |
|--------|-------------------|-------------------|---------|
| **Precision** | 95% | 91% | 94% |
| **Recall** | 94% | 93% | 94% |
| **F1-Score** | 95% | 92% | 94% |
| **Support** | 531 | 323 | 854 |

### Key Performance Indicators

- **Overall Accuracy**: 94%
- **Macro Average F1**: 93%
- **Weighted Average F1**: 94%
- **Balanced Performance**: Both classes achieve >90% metrics

## Data Requirements

### Input Format
Your CSV file should contain the following types of columns:

| Column Type | Examples | Description |
|-------------|----------|-------------|
| **ID** | `loan_id` | Unique identifier for each application |
| **Target** | `loan_status` | Approval status ("Approved" or "Rejected") |
| **Numeric Features** | `income_annum`, `loan_amount`, `cibil_score` | Continuous variables |
| **Categorical Features** | `education`, `self_employed` | Discrete categories |

### Sample Data Structure
```csv
loan_id,education,self_employed,income_annum,loan_amount,cibil_score,loan_status
1,Graduate,No,9600000,29900000,778,Approved
2,Not Graduate,Yes,4100000,12200000,417,Rejected
```

## Technical Implementation

### Preprocessing Pipeline

1. **Missing Value Handling**:
   - Numeric features: Median imputation
   - Categorical features: Mode imputation

2. **Feature Encoding**:
   - Numeric features: StandardScaler normalization
   - Categorical features: One-hot encoding

3. **Data Splitting**: 80-20 train-test split with stratification

### Model Architecture

- **Algorithm**: Logistic Regression with balanced class weights
- **Solver**: SAGA (optimized for large datasets)
- **Regularization**: L2 regularization
- **Class Balancing**: Automatic weight adjustment for imbalanced data

### Evaluation Metrics

The model is evaluated using metrics specifically chosen for imbalanced classification:

- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases  
- **F1-Score**: Harmonic mean of precision and recall
- **PR-AUC**: Area under the Precision-Recall curve
- **Confusion Matrix**: Detailed classification breakdown

## Visualizations

The script generates two key visualizations:

### 1. Precision-Recall Curve
Shows the trade-off between precision and recall at different classification thresholds.

### 2. Confusion Matrix
Displays the detailed breakdown of correct and incorrect predictions for each class.

## Project Structure

```
loan-approval-prediction/
├── loan_prediction.py      # Main prediction script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── loan_approval_dataset.csv  # Input data
└── venv/                  # Virtual environment (created after setup)
```

## Model Insights

### Strengths
- **High Accuracy**: 94% overall prediction accuracy
- **Balanced Performance**: Works well for both approval and rejection cases
- **Robust Preprocessing**: Handles missing data and categorical variables
- **Fast Training**: Optimized for quick execution

### Use Cases
- **Financial Institutions**: Automated loan approval screening
- **Risk Assessment**: Credit risk evaluation
- **Decision Support**: Assist loan officers in decision-making
- **Compliance**: Consistent and auditable approval process

## Customization

### Adjusting Model Parameters
You can modify the model by editing these parameters in `loan_prediction.py`:

```python
# Model configuration
LogisticRegression(
    class_weight='balanced',    # Handle imbalanced data
    random_state=42,           # Reproducible results
    max_iter=200,              # Training iterations
    solver='saga'              # Optimization algorithm
)
```

### Adding New Features
To include additional features:
1. Add columns to your CSV file
2. The script automatically detects numeric vs categorical features
3. Re-run the training process

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical computing |
| scikit-learn | ≥1.2.0 | Machine learning |
| matplotlib | ≥3.7.0 | Plotting |
| seaborn | ≥0.12.0 | Statistical visualization |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions:

1. Check the existing issues in the repository
2. Create a new issue with a detailed description
3. Include your Python version and error messages

---

<div align="center">

**Star this repository if you found it helpful!**

Made with care for the machine learning community

</div> 