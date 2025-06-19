import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, request, render_template

## initialize Flask app
app = Flask(__name__)

## load the model
model = joblib.load(MODEL_OUTPUT_PATH)

## define the home route
@app.route('/', methods=['GET', 'POST']) 

def index():
    if request.method == 'POST':
        # get the form data
        lead_time  = int(request.form['lead_time'])
        no_of_special_request = int(request.form['no_of_special_request'])
        avg_price_per_room = float(request.form['avg_price_per_room'])
        arrival_month = int(request.form['arrival_month'])
        arrival_date = int(request.form['arrival_date'])
        market_segment_type = int(request.form['market_segment_type'])
        no_of_week_nights = int(request.form['no_of_week_nights'])
        no_of_weekend_nights = int(request.form['no_of_weekend_nights'])
        type_of_meal_plan = int(request.form['type_of_meal_plan'])
        room_type_reserved = int(request.form['room_type_reserved'])

        ## create a feature array from the form data
        features = np.array([[lead_time, no_of_special_request, avg_price_per_room, arrival_month, arrival_date, market_segment_type, no_of_week_nights, no_of_weekend_nights, type_of_meal_plan, room_type_reserved]])

        ## make a prediction using the model
        prediction = model.predict(features)

        return render_template('index.html', prediction=prediction[0])
    
    return render_template('index.html', prediction="Can't make prediction yet!")


## run the app
if __name__ == "__main__":
    app.run(host =  '0.0.0.0', port = 8080)