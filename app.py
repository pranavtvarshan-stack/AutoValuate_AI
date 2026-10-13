import os
import warnings
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn unpickling warnings if minor versions differ
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Load model pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'car_price_pipeline.joblib')
pipeline = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET'])
def index():
    # Load completely blank form with default select options
    return render_template('index.html', form_data=None, prediction_text=None, error_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate that dropdowns were selected
        required_dropdowns = ['brand_tier_encoded', 'drivewheel_rwd', 'fuelsystem_mpfi']
        for field in required_dropdowns:
            if not request.form.get(field):
                return render_template(
                    'index.html',
                    error_text=f"Please select a valid option for all dropdown menus.",
                    form_data=request.form
                )

        # Extract numerical inputs
        enginesize = float(request.form['enginesize'])
        curbweight = float(request.form['curbweight'])
        horsepower = float(request.form['horsepower'])
        carwidth = float(request.form['carwidth'])
        carlength = float(request.form['carlength'])
        wheelbase = float(request.form['wheelbase'])
        citympg = float(request.form['citympg'])
        highwaympg = float(request.form['highwaympg'])
        boreratio = float(request.form['boreratio'])
        
        # Categorical selections
        brand_tier_encoded = int(request.form['brand_tier_encoded'])
        drivewheel_rwd = int(request.form['drivewheel_rwd'])
        fuelsystem_mpfi = int(request.form['fuelsystem_mpfi'])

        # Calculate engineered metrics
        power_to_weight = horsepower / curbweight if curbweight > 0 else 0.0
        overall_mpg = (citympg + highwaympg) / 2.0

        # Construct DataFrame in exact feature order
        input_data = pd.DataFrame([{
            'enginesize': enginesize,
            'curbweight': curbweight,
            'horsepower': horsepower,
            'power_to_weight': power_to_weight,
            'carwidth': carwidth,
            'carlength': carlength,
            'wheelbase': wheelbase,
            'citympg': citympg,
            'highwaympg': highwaympg,
            'overall_mpg': overall_mpg,
            'brand_tier_encoded': brand_tier_encoded,
            'boreratio': boreratio,
            'drivewheel_rwd': drivewheel_rwd,
            'fuelsystem_mpfi': fuelsystem_mpfi
        }])

        # Generate price prediction
        prediction = pipeline.predict(input_data)[0]
        formatted_price = f"${prediction:,.2f}"

        return render_template(
            'index.html',
            prediction_text=formatted_price,
            form_data=request.form
        )

    except ValueError as ve:
        return render_template(
            'index.html',
            error_text="Invalid input value entered. Please verify all numerical fields are filled correctly.",
            form_data=request.form
        )
    except Exception as e:
        return render_template(
            'index.html',
            error_text=f"Error processing prediction: {str(e)}",
            form_data=request.form
        )

if __name__ == '__main__':
    app.run(debug=True, port=5000)