
# Phishing URL Detection

This project detects phishing URLs using a machine learning model. The project consists of three main components: training the model, running an API to serve the model, and testing the API.

### ⚡ Features
- **Model Training**: A machine learning model is trained on a dataset of URLs to predict whether a URL is phishing or legitimate.
- **Flask API**: The trained model is served via a Flask API, which predicts the URL type when a request is sent.
- **Testing the API**: A script sends requests to the API and receives the prediction (phishing or legitimate).



You can install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```
# 🚀 Setup and Usage

#### **1️⃣ Clone the Repository**

Start by cloning the project repository to your local machine:
```bash
git clone https://github.com/yourusername/phishing-url-detection.git
cd phishing-url-detection
```

#### **2️⃣ Train the Model**
Run the `train_model.py` script to train the phishing URL detection model using the provided dataset (`data/phishing_dataset.csv`).

```bash
python train_model.py
```
### **3️⃣ Run the API**
Once the model is trained, run the api.py script to start the Flask API server.
```bash
python api.py
```
The Flask API will start running at http://127.0.0.1:3000/predict. You can send POST requests to this endpoint with a URL to classify it.

### **4️⃣ Test the API**
Add the URL that u want to test in the `test_api.py` file and In another terminal window, run the testapi.py script to send test requests to the Flask API. This script will send test URLs to the /predict endpoint and check if the URL is phishing or legitimate.
```bash
python test_api.py
```
The `testapi.py` sends a URL as a JSON payload and receives a response indicating whether the URL is phishing or legitimate.


<img src="https://github.com/user-attachments/assets/e9251536-649c-4a75-8a26-7ab3fa9637b4" alt="image" height="150" />






