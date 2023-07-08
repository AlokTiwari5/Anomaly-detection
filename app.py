from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    # Get the uploaded file
    file = request.files['data_file']

    # Load the data from the uploaded file
    data = pd.read_csv(file)

    # Select the relevant features for anomaly detection
    selected_features = ['speed', 'temperature', 'vibration']

    # Train the anomaly detection model
    model = IsolationForest(contamination=0.05)  # Adjust the contamination parameter as needed
    model.fit(data[selected_features])

    # Detect anomalies in the data
    predictions = model.predict(data[selected_features])
    anomalies = data[predictions == -1]

    return render_template('results.html', anomalies=anomalies)

if __name__ == '__main__':
    app.run(debug=True)
