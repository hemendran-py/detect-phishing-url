print("imported and currently loading dataset")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
print("imported and currently loading dataset")
# Load dataset
data = pd.read_csv("data/phishing_dataset.csv")
# Separate features and labels
urls = data['url']
labels = data['type']
# Vectorize URLs
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(urls)
# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=70, random_state=42,max_depth=10)
model.fit(X_train, y_train)
# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Save model and vectorizer
joblib.dump(model, "app/model/phishing_model.pkl")
joblib.dump(vectorizer, "app/model/vectorizer.pkl")
