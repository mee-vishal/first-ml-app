from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

# Load data and model
df = pd.read_csv("abc.csv")
X = df[['age', 'income']]
y = df['expenditure']

# Split and train again to calculate accuracy (optional: you can reuse saved model and save accuracy too)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load('model.pkl')

# Predict and calculate R²
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

@app.route('/')
def home():
    return render_template('index.html', prediction_text="", accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    income = float(request.form['income'])
    prediction = model.predict(np.array([[age, income]]))[0]
    return render_template('index.html',
                           prediction_text=f'Predicted Expenditure: ₹{prediction:.2f}',
                           accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
