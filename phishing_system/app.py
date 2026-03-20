from flask import Flask
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("phishing_model_v1.pkl")
vectorizer = joblib.load("tfidf_vectorizer_v1.pkl")

# Dummy test route
@app.route("/")
def home():
    return "Phishing Detection System Running!"

# Test prediction route
@app.route("/test")
def test_prediction():
    
    # Example email
    subject = "URGENT: Verify your account now"
    body = "Click this link immediately to avoid suspension"

    # === SAME preprocessing as training ===
    text = (subject + " " + body).lower()

    # TF-IDF transformation
    X_tfidf = vectorizer.transform([text])

    # ⚠️ IMPORTANT: your model expects MORE features
    # For now, we will fake metadata (temporary)
    import numpy as np
    from scipy.sparse import hstack

    subject_length = len(subject)
    body_length = len(body)
    url_count = 1
    phishing_keyword_count = 2
    uppercase_count = sum(1 for w in text.split() if w.isupper())
    digit_count = sum(c.isdigit() for c in text)

    meta_features = np.array([[ 
        subject_length,
        body_length,
        url_count,
        phishing_keyword_count,
        uppercase_count,
        digit_count
    ]])

    # Combine features
    X_final = hstack([X_tfidf, meta_features])

    # Prediction
    prediction = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0]

    return f"""
    Prediction: {prediction} <br>
    Phishing Probability: {prob[1]:.4f} <br>
    Legitimate Probability: {prob[0]:.4f}
    """

# Run app
if __name__ == "__main__":
    app.run(debug=True)
