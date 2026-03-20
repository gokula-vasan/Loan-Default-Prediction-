from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
import joblib
import numpy as np
import os
import json

app = Flask(__name__)
app.secret_key = 'loan_default_secret_key'

# Load models
model_path = 'models/model.joblib'
scaler_path = 'models/scaler.joblib'
features_path = 'models/features.joblib'

model = None
scaler = None
feature_names = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    print("Models loaded successfully")
else:
    print("WARNING: Run 'python train_save.py' first!")

def load_users():
    users_file = 'users.json'
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def check_login(username, password):
    users = load_users()
    return users.get(username) == password or (username == 'admin' and password == '1234')

@app.route('/', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session:
        return redirect(url_for('predict'))
    
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if check_login(username, password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('predict'))
        else:
            error = "Invalid credentials!"

    return render_template('spooky_login.html', error=error)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    username = session.get('username', 'User')
    prediction = None
    prob = None
    inputs = None
    
    if request.method == 'POST' and model:
        try:
            feature_map = {
                'Income': float(request.form['Income']),
                'CreditScore': float(request.form['CreditScore']),
                'LoanAmount': float(request.form['LoanAmount']),
                'InterestRate': float(request.form['InterestRate']),
                'LoanTerm': float(request.form['LoanTerm']),
                'DTIRatio': float(request.form['DTIRatio']),
                'Education': int(request.form['Education']),
                'EmploymentType': int(request.form['EmploymentType']),
                'MaritalStatus': int(request.form['MaritalStatus']),
                'LoanPurpose': int(request.form['LoanPurpose'])
            }
            
            full_input = np.zeros(len(feature_names))
            for feat, val in feature_map.items():
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    full_input[idx] = val
            
            full_input_scaled = scaler.transform(full_input.reshape(1, -1))
            pred = model.predict(full_input_scaled)[0]
            prob_val = model.predict_proba(full_input_scaled)[0][1]
            
            risk = "High Risk - Likely to Default!" if pred == 1 else "Low Risk - Approved!"
            prediction = risk
            prob = f"{prob_val:.2%}"
            inputs = feature_map
        except Exception as e:
            error = f"Prediction error: {str(e)}"
    
    return render_template('predict.html', username=username, prediction=prediction, prob=prob, inputs=inputs)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if len(username) < 3 or len(password) < 4:
            error = "Username min 3 chars, password min 4!"
        else:
            users = load_users()
            if username in users:
                error = "Username exists!"
            else:
                users[username] = password
                save_users(users)
                success = f"Account {username} created! Login now."
    
    return render_template('signup.html', error=error, success=success)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/Spooky Login Form/<path:filename>')
def serve_spooky_login(filename):
    return send_from_directory('Spooky Login Form', filename)

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, port=5001)

