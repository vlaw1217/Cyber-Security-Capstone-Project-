import joblib
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.sparse import hstack

# Initialize Flask application
app = Flask(__name__)

# Secret key used for session management (NOTE: replace in production)
app.secret_key = "secret123"

# Load trained ML model and TF-IDF vectorizer
model = joblib.load("phishing_model_v1.pkl")
vectorizer = joblib.load("tfidf_vectorizer_v1.pkl")


# ---------------------------
# HOME ROUTE
# ---------------------------
@app.route("/")
def home():
    # Render main page (login / input form)
    return render_template("index.html")


# ---------------------------
# PREDICTION ROUTE
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # 1. Retrieve user input from form
    subject = request.form["subject"]
    body = request.form["body"]

    # Basic validation to prevent empty input
    if not subject.strip() or not body.strip():
        return "Please enter subject and body"

    # Combine subject and body, normalize text
    text = (subject + " " + body).lower()

    # 2. Convert text to TF-IDF feature vector
    X_tfidf = vectorizer.transform([text])

    # 3. Generate additional metadata features
    subject_length = len(subject)                     # Length of subject
    body_length = len(body)                           # Length of email body
    url_count = text.count("http")                    # Number of URLs
    phishing_keyword_count = sum(
        word in text for word in ["urgent", "verify", "account", "click"]
    )                                                 # Presence of phishing keywords
    uppercase_count = sum(
        1 for w in (subject + " " + body).split() if w.isupper()
    )                                                 # Count of uppercase words
    digit_count = sum(c.isdigit() for c in text)      # Count of digits

    # Combine metadata into array
    meta_features = np.array([[ 
        subject_length,
        body_length,
        url_count,
        phishing_keyword_count,
        uppercase_count,
        digit_count
    ]])

    # Combine TF-IDF features with metadata features
    X_final = hstack([X_tfidf, meta_features])

    # 4. Perform prediction
    prediction = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0]

    # Assign label based on probability thresholds
    if prob[1] > 0.8:
        label = "Phishing"
    elif prob[1] > 0.5:
        label = "Suspicious"
    else:
        label = "Legitimate"

    # Color coding for UI display
    color = "red" if label == "Phishing" else "orange" if label == "Suspicious" else "green"

    # 5. Store prediction result in database
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (user_id, subject, body, prediction, probability)
    VALUES (?, ?, ?, ?, ?)
    """, (
        session.get("user_id", 0),   # Default to 0 if user not logged in
        subject,
        body,
        label,
        float(prob[1])
    ))

    conn.commit()
    conn.close()

    # 6. Return result to user (simple HTML response)
    return f"""
    <h2>Result</h2>
    <p style="color:{color}; font-size:20px;">
    <b>{label}</b>
    </p>

    Phishing Probability: {prob[1]:.4f}<br>
    Legitimate Probability: {prob[0]:.4f}<br>

    <br><a href="/dashboard">Back</a>
    """


# ---------------------------
# USER REGISTRATION
# ---------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        # Get user input
        username = request.form["username"]
        password = request.form["password"]

        # Hash password for secure storage
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()

        try:
            # Insert new user into database
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed_password)
            )
            conn.commit()

        # Handle duplicate username (UNIQUE constraint)
        except sqlite3.IntegrityError:
            conn.close()
            return "Username already exists"

        conn.close()
        return redirect("/")

    # Render registration page
    return render_template("register.html")


# ---------------------------
# USER LOGIN
# ---------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()

        conn.close()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            return redirect("/dashboard")
        else:
            return "Login Failed"

    # GET request → show login page
    return render_template("login.html")

# ---------------------------
# DASHBOARD
# ---------------------------
@app.route("/dashboard")
def dashboard():
    # Restrict access to logged-in users
    if "user_id" not in session:
        return redirect("/")

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Get current user's prediction history
    cursor.execute("""
    SELECT subject, prediction, probability
    FROM predictions
    WHERE user_id = ?
    ORDER BY id DESC
    """, (session["user_id"],))

    history = cursor.fetchall()
    conn.close()

    return render_template("dashboard.html", history=history)

# ---------------------------
# LOGOUT
# ---------------------------
@app.route("/logout")
def logout():

    # Clear session data
    session.clear()

    return redirect("/")


# ---------------------------
# RUN APPLICATION
# ---------------------------
if __name__ == "__main__":
    # Debug mode ON for development
    app.run(debug=True)