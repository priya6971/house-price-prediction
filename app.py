import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Use request.form to get data from the HTML form
    data = [float(x) for x in request.form.values()]
    
    # The rest of your code remains the same
    # The data needs to be reshaped for prediction
    final_input = np.array(data).reshape(1, -1)
    new_data = scaler.transform(final_input)
    output = model.predict(new_data)
    
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)



