# save_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("abc.csv")
X = df[['age', 'income']]
y = df['expenditure']

# Train model
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')
