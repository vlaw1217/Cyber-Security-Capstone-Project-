from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("phishing_model_v1.pkl")
vectorizer = joblib.load("tfidf_vectorizer_v1.pkl")

# Dummy test route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    subject = request.form["subject"]
    body = request.form["body"]

    text = (subject + " " + body).lower()

    # TF-IDF
    X_tfidf = vectorizer.transform([text])

    # Metadata features
    import numpy as np
    from scipy.sparse import hstack

    subject_length = len(subject)
    body_length = len(body)
    url_count = text.count("http")
    phishing_keyword_count = sum(word in text for word in ["urgent", "verify", "account", "click"])
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

    X_final = hstack([X_tfidf, meta_features])

    prediction = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0]

    # return f"""
    # <h2>Result</h2>
    # Prediction: {prediction} <br>
    # Phishing Probability: {prob[1]:.4f} <br>
    # Legitimate Probability: {prob[0]:.4f} <br>
    # <br><a href="/">Try another</a>
    # """

    label = "Phishing" if prediction == 1 else "Legitimate"
    color = "red" if prediction == 1 else "green"

    return f"""
    <h2>Result</h2>
    <p><strong style="color:{color}; font-size:20px;">
    {label}
    </strong></p>

    <p>Phishing Probability: {prob[1]:.4f}</p>
    <p>Legitimate Probability: {prob[0]:.4f}</p>

    <br><a href="/">Try another</a>
    """

# Run app
if __name__ == "__main__":
    app.run(debug=True)
