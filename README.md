# Loan Default Prediction

A machine learning web application that predicts whether a customer is likely to default on a loan.

## Features

- Logistic Regression model trained on loan default dataset
- Flask web application for predictions
- Interactive web interface

## Dataset

The model is trained on a dataset with the following features:
- Age, Income, Loan Amount
- Credit Score, Months Employed
- Number of Credit Lines, Interest Rate
- Loan Term, DTI Ratio
- Education, Employment Type
- Marital Status, Has Mortgage
- Has Dependents, Loan Purpose, Has Co-Signer

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ML
```

2. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install flask numpy scikit-learn
```

4. Train the model (if loan_model.pkl doesn't exist):
```bash
python train_model.py
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

3. Enter customer details and get prediction

## Model Performance

- Accuracy: ~88.5%
- The model predicts loan default based on customer financial and demographic information

## Files

- `app.py` - Flask web application
- `train_model.py` - Script to train and save the model
- `loan_model.pkl` - Trained model (generated after running train_model.py)
- `Loan_default.csv` - Training dataset
- `Loan Default.ipynb` - Jupyter notebook with EDA and model training
- `templates/index.html` - Web interface

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy

## License

MIT License
