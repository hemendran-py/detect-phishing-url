from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("model/phishing_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get("url", "")
    
    # Vectorize the URL
    features = vectorizer.transform([url])
    
    # Predict phishing or legitimate
    prediction = model.predict(features)[0]
    return jsonify({"phishing": bool(prediction)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)