from flask import Flask, request, render_template
import pandas as pd
import mlflow.sklearn

# Initialize Flask app
app = Flask(__name__)

# Set up MLflow tracking URI and experiment
mlflow.set_experiment("Rainfall")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Load the production model (adjust this to your production model and version)
model_version = 2
production_model_name = "Random Forest Model Data"  # Replace with your model name
prod_model_uri = f"models:/{production_model_name}@{model_version}"
loaded_model = mlflow.sklearn.load_model(prod_model_uri)

# Define the feature names (adjust these to your actual features)
feature_names = ['Humidity3pm', 'RainToday', 'WindGustSpeed', 'Humidity9am', 'Rainfall',
                 'WindSpeed9am', 'WindDir9am_NNW', 'WindDir9am_N', 'WindSpeed3pm',
                 'WindDir9am_NW', 'WindGustDir_NNE', 'Cloud9am', 'WindDir3pm', 'MaxTemp', 'Temp3pm']

@app.route('/')
def home():
    return render_template('index.html', prediction_result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        input_data = [
            float(request.form['Humidity3pm']),
            int(request.form['RainToday']),
            float(request.form['WindGustSpeed']),
            float(request.form['Humidity9am']),
            float(request.form['Rainfall']),
            float(request.form['WindSpeed9am']),
            bool(int(request.form['WindDir9am_NNW'])),  # Convert to boolean from int (checkbox)
            bool(int(request.form['WindDir9am_N'])),
            float(request.form['WindSpeed3pm']),
            bool(int(request.form['WindDir9am_NW'])),
            bool(int(request.form['WindGustDir_NNE'])),
            float(request.form['Cloud9am']),
            bool(int(request.form['WindDir3pm'])),
            float(request.form['MaxTemp']),
            float(request.form['Temp3pm']),
        ]

        # Create DataFrame for the input
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Predict using the loaded model
        prediction = loaded_model.predict(input_df)

        # Convert prediction to human-readable format
        result = "Rainfall" if prediction[0] == 1 else "No Rainfall"

        return render_template('index.html', prediction_result=result)
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)
