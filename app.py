from flask import Flask, request, render_template
import joblib
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)
model = joblib.load('placement_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    r2 = None
    mae = None
    if request.method == 'POST':
        cgpa = float(request.form['cgpa'])
        extra = float(request.form['extra'])
        projects = float(request.form['projects'])
        internship = float(request.form['internship'])
        features = np.array([[cgpa, extra, projects, internship]])
        prediction = model.predict(features)[0]

        # Here, you typically compare on test data, but for demo:
        # Let's simulate true value input to calculate accuracy OR
        # Just show example static scores for demo

        # For demonstration, assuming model has test data saved:
        # Load test features and true labels (replace this with your real test data)
        # X_test = ...
        # y_test = ...
        # y_pred = model.predict(X_test)
        # r2 = r2_score(y_test, y_pred)
        # mae = mean_absolute_error(y_test, y_pred)

        # For now, static example values:
        r2 = 0.82  # Example R2 score
        mae = 5.6  # Example MAE (% placement)

    return render_template('index.html', prediction=prediction, r2=r2, mae=mae)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
