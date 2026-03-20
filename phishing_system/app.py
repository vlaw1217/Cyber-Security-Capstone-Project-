from flask import Flask, render_template, request, redirect, session
import joblib
import sqlite3

# Initialize Flask app
app = Flask(__name__)

app.secret_key = "secret123"  # only for testing

# Load model and vectorizer
model = joblib.load("phishing_model_v1.pkl")
vectorizer = joblib.load("tfidf_vectorizer_v1.pkl")

# Dummy test route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():

    # 1️. Get input FIRST
    subject = request.form["subject"]
    body = request.form["body"]

    if not subject.strip() or not body.strip():
        return "Please enter subject and body"

    text = (subject + " " + body).lower()

    # 2️. TF-IDF
    X_tfidf = vectorizer.transform([text])

    # 3️. Metadata
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

    # 4️. Prediction
    prediction = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0]

    # Label logic
    if prob[1] > 0.8:
        label = "Phishing"
    elif prob[1] > 0.5:
        label = "Suspicious"
    else:
        label = "Legitimate"

    color = "red" if label == "Phishing" else "orange" if label == "Suspicious" else "green"

    # 5️. Save to DB (LAST)
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (user_id, subject, body, prediction, probability)
    VALUES (?, ?, ?, ?, ?)
    """, (
        session.get("user_id", 0),
        subject,
        body,
        label,
        float(prob[1])
    ))

    conn.commit()
    conn.close()

    # 6️. Return result
    return f"""
    <h2>Result</h2>
    <p style="color:{color}; font-size:20px;">
    <b>{label}</b>
    </p>

    Phishing Probability: {prob[1]:.4f}<br>
    Legitimate Probability: {prob[0]:.4f}<br>

    <br><a href="/dashboard">Back</a>
    """
@app.route("/register", methods=["GET", "POST"])

def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()

        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))

        conn.commit()
        conn.close()

        return redirect("/")

    return render_template("register.html")

@app.route("/login", methods=["POST"])

def login():
    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()

    conn.close()

    if user:
        session["user_id"] = user[0]
        return redirect("/dashboard")
    else:
        return "Login Failed"
    
@app.route("/dashboard")

def dashboard():
    if "user_id" not in session:
        return redirect("/")
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# Run app
if __name__ == "__main__":
    app.run(debug=True)
