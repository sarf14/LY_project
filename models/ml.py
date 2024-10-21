import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Sample dataset
data = {
    'rainfall': [100, 150, 200, 250, 300, 350],
    'season': ['summer', 'winter', 'rainy', 'summer', 'winter', 'rainy'],
    'state': ['state1', 'state2', 'state3', 'state1', 'state2', 'state3'],
    'area': [1, 2, 3, 1, 2, 3],
    'crop': ['crop1', 'crop2', 'crop3', 'crop1', 'crop2', 'crop3']
}

# Create DataFrame
df = pd.DataFrame(data)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['season', 'state'])

# Split dataset
X = df.drop(columns='crop')
y = df['crop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and columns for use in the Flask app
joblib.dump(model, 'crop_prediction_model.pkl')
joblib.dump(X.columns, 'columns.pkl')  # Save the columns for future use

print(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test))}")
