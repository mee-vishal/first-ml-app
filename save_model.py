import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv('placement_data.csv')

# Split data into features and target
X = df[['CGPA', 'ExtraCurriculars', 'Projects', 'Internship']]
y = df['PlacementPercent']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)
print(f"Model R^2 score: {r2:.3f}")

# Save the model
joblib.dump(model, 'placement_model.pkl')
