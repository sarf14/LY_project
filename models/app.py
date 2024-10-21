from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model and the feature columns
model = joblib.load('crop_prediction_model.pkl')
columns = joblib.load('columns.pkl')  # Load the saved columns

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_crop():
    data = request.json
    rainfall = data['rainfall']
    season = data['season']
    state = data['state']
    area = data['area']

    # Prepare the input data
    input_data = pd.DataFrame({
        'rainfall': [rainfall],
        'area': [area]
    })

    # Dynamically add season and state columns with one-hot encoding
    for col in columns:
        if col.startswith('season_'):
            input_data[col] = 1 if f'season_{season}' == col else 0
        elif col.startswith('state_'):
            input_data[col] = 1 if f'state_{state}' == col else 0

    # Reindex input_data to match the model's expected input columns
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Make the prediction
    prediction = model.predict(input_data)

    # Return the predicted crop
    return jsonify({'predicted_crop': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
