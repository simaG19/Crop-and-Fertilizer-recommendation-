from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)
# Load the dataset
dataset = pd.read_csv('Crop_and_fertilizer dataset.csv')

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json()

    # Get the selected values from the request data
    soil_color = data['soil_color']
    nitrogen = float(data['nitrogen'])
    phosphorus = float(data['phosphorus'])
    potassium = float(data['potassium'])
    pH = float(data['pH'])

    input_data = pd.DataFrame(
        [[nitrogen, phosphorus, potassium, pH, soil_color]],
        columns=['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Soil_color']
    )

    # Perform one-hot encoding for District_Name and Soil_color columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(dataset[['Soil_color']])
    input_data_encoded = encoder.transform(input_data[['Soil_color']])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, dataset['Crop'], test_size=0.2, random_state=42)

    # Train the random forest model
    model_crop = RandomForestClassifier(n_estimators=100, random_state=42)
    model_crop.fit(X_train, y_train)

    # Make predictions
    predicted_crop = model_crop.predict(input_data_encoded)

    # Find the fertilizer associated with the recommended crop
    recommended_fertilizer = dataset[dataset['Crop'] == predicted_crop[0]]['Fertilizer'].values[0]

    response = {
        'recommended_crop': predicted_crop[0],
        'recommended_fertilizer': recommended_fertilizer
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()