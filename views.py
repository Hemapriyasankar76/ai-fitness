from flask import Flask, request, jsonify, render_template
import joblib
import json

app = Flask(__name__)

# Load your model at the start of the app
model = joblib.load("path_to_your_saved_model.pkl")

@app.route('/')
def hello_world():
    return render_template("login.html")

@app.route('/registration')
def registration():
    return render_template("register.html")

# Create a new route for model predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            dataSample = {
                "features": []  # Example: 10:00 AM, Tuesday, March
            }
            data = json.dumps(dataSample)
            #data = request.get_json()  # Get data posted as JSON
            # Assuming data is in the correct format directly for your model
            # You might need to process it before feeding into the model depending on how your model expects the data
            prediction = model.predict([data.values()])
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e), 'message': 'Error processing request'})

if __name__ == '__main__':
    app.run(debug=True)
