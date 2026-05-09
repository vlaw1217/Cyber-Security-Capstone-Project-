import joblib
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, session
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.sparse import hstack, csr_matrix
from database import get_all_predictions, get_user_predictions

# Initialize Flask application
app = Flask(__name__)

# Secret key used for session management (NOTE: replace in production)
app.secret_key = "secret123"

# Load trained ML model and TF-IDF vectorizer
model = joblib.load("phishing_model_v1.pkl")
vectorizer = joblib.load("tfidf_vectorizer_v1.pkl")

# ---------------------------
# DATABASE MIGRATION
# ---------------------------
# Ensure the users table has the is_blocked column.
# This prevents errors on other computers where app.db may not have the new column yet.
def ensure_is_blocked_column():
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]

    if "is_blocked" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN is_blocked INTEGER DEFAULT 0")
        conn.commit()

    conn.close()


# ---------------------------
# ADMIN ACCESS CONTROL
# ---------------------------
# This decorator protects admin-only routes.
# It first checks whether a user is logged in.
# Then it checks whether the logged-in user's role is "admin".
# If the user is not logged in, they are redirected to the login page.
# If the user is logged in but not an admin, they are redirected to the dashboard.
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/")

        if session.get("role") != "admin":
            return redirect("/dashboard")

        return f(*args, **kwargs)

    return decorated_function


# ---------------------------
# HOME ROUTE
# ---------------------------
@app.route("/")
def home():
    # Render main page (login / input form)
    return render_template("login.html")


# ---------------------------
# PREDICTION ROUTE
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return redirect("/")

    # 1. Retrieve user input from form
    subject = request.form["subject"]
    body = request.form["body"]

    # 2. Combine subject and body, then normalize text
    text = (subject + " " + body).lower().strip()

    # 3. Basic validation to prevent empty or too-short input
    if len(text) < 10:
        label = "Unable to classify"
        color = "gray"

        return f"""
        <h2>Result</h2>
        <p style="color:{color}; font-size:20px;">
        <b>{label}</b>
        </p>
        The email content is too short for reliable analysis.<br>
        <br><a href="/dashboard">Back</a>
        """

    # 4. Convert text to TF-IDF feature vector
    X_tfidf = vectorizer.transform([text])

    # 5. Generate additional metadata features
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

    # 6. Combine metadata into array
    meta_features = csr_matrix([[
    subject_length,
    body_length,
    url_count,
    phishing_keyword_count,
    uppercase_count,
    digit_count
    ]])

    # Combine TF-IDF features with metadata features
    X_final = hstack([X_tfidf, meta_features])

    # 7. Perform prediction
    prediction = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0]

    # 8. Label logic
    if prob[1] > 0.8:
        label = "Phishing"
    elif prob[1] > 0.5:
        label = "Suspicious"
    else:
        label = "Legitimate"

    #9. Color coding for UI display
    color = "red" if label == "Phishing" else "orange" if label == "Suspicious" else "green"

    # 10. Store prediction result in database
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (user_id, subject, body, prediction, probability)
    VALUES (?, ?, ?, ?, ?)
    """, (
        session["user_id"],   
        subject,
        body,
        label,
        float(prob[1])
    ))

    conn.commit()
    conn.close()

    # 11. Return result to user (simple HTML response)
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
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        confirm_password = request.form["confirm_password"].strip()

        # Check if passwords match

        if password != confirm_password:
            return """
            <h2 style="color:red;">Passwords do not match</h2>
            <p>Please re-enter your password carefully.</p>
            <a href="/register">Try Again</a>
            """
        
        # Hash password for secure storage
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()

        try:
            # Insert new user into database
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed_password, "user")
            )
            conn.commit()

        # Handle duplicate username (UNIQUE constraint)
        except sqlite3.IntegrityError:
            conn.close()
            return """
            <h2 style="color:red;">Username already exists</h2>
            <p>Please choose a different username, or go back to login if you already have an account.</p>
            <a href="/register">Try Another Username</a><br><br>
            <a href="/">Back to Login</a>
            """

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
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()

        conn.close()

        # Check whether the account has been blocked by an admin.
        # If is_blocked = 1, stop the login process and show an account blocked message.
        # This prevents blocked users from accessing the dashboard even with the correct password.
        if user and check_password_hash(user[2], password):
            # Check if account is blocked
            if len(user) > 4 and user[4] == 1:
                return """
                <h2 style="color:red;">Account Blocked</h2>
                <p>Your account has been blocked by an administrator.</p>
                <a href="/">Back to Login</a>
                """

            session["user_id"] = user[0]
            session["username"] = user[1]
            session["role"] = user[3]
            return redirect("/dashboard")
            
        else:
            return """
            <h2 style="color:red;">Login Failed</h2>
            <p>Invalid username or password.</p>
            <a href="/">Try Again</a>
            """

    # GET request → show login page
    return render_template("login.html")

# ---------------------------
# DASHBOARD
# ---------------------------
@app.route("/dashboard")
def dashboard():
    # Restrict dashboard access to logged-in users only.
    # If no user_id exists in the session, redirect back to the login page.
    if "user_id" not in session:
        return redirect("/")

    # Admin users can view all prediction records from every user.
    # Admin users also retrieve the full user list for user-management features.
    if session.get("role") == "admin":
        data = get_all_predictions()
        dashboard_title = "Admin Dashboard"

        # Retrieve all registered users so the admin can manage accounts.
        # is_blocked is used to show whether each account is active or blocked.
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, is_blocked FROM users ORDER BY id")
        users = cursor.fetchall()
        conn.close()
    
    # Regular users can only view their own prediction history.
    # users is set to an empty list because user management is admin-only.
    else:
        data = get_user_predictions(session["user_id"])
        dashboard_title = "User Dashboard"
        users = []

    # Send prediction data, user list, dashboard title, username, and role to dashboard.html.
    return render_template(
    "dashboard.html",
    data=data,
    users=users,
    dashboard_title=dashboard_title,
    username=session.get("username"),
    role=session.get("role")
)


# ---------------------------
# ADMIN: ADD USER
# ---------------------------
@app.route("/admin/add_user", methods=["POST"])
@admin_required
def admin_add_user():
    # Get new user information from the admin form.
    username = request.form["username"].strip()
    password = request.form["password"].strip()
    role = request.form["role"].strip()

    # Only allow valid roles.
    # If an invalid role is submitted, default to regular user.
    if role not in ["user", "admin"]:
        role = "user"

    # Store hashed password instead of plain text password.
    hashed_password = generate_password_hash(password)

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    try:
        # Create a new active account.
        # is_blocked = 0 means the account is active.
        cursor.execute(
            "INSERT INTO users (username, password, role, is_blocked) VALUES (?, ?, ?, ?)",
            (username, hashed_password, role, 0)
        )
        conn.commit()

    except sqlite3.IntegrityError:
        conn.close()
        return """
        <h2 style="color:red;">Username already exists</h2>
        <p>Please choose another username.</p>
        <a href="/dashboard">Back to Dashboard</a>
        """

    conn.close()
    return redirect("/dashboard")


# ---------------------------
# ADMIN: BLOCK USER
# ---------------------------
@app.route("/admin/block_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_block_user(user_id):
    # Prevent the current admin from blocking their own account.
    if user_id == session["user_id"]:
        return """
        <h2 style="color:red;">Action Not Allowed</h2>
        <p>You cannot block your own admin account.</p>
        <a href="/dashboard">Back to Dashboard</a>
        """

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # is_blocked = 1 means the user cannot log in.
    cursor.execute("UPDATE users SET is_blocked = 1 WHERE id = ?", (user_id,))

    conn.commit()
    conn.close()

    return redirect("/dashboard")


# ---------------------------
# ADMIN: UNBLOCK USER
# ---------------------------
@app.route("/admin/unblock_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_unblock_user(user_id):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # is_blocked = 0 means the user account is active again.
    cursor.execute("UPDATE users SET is_blocked = 0 WHERE id = ?", (user_id,))

    conn.commit()
    conn.close()

    return redirect("/dashboard")


# ---------------------------
# ADMIN: DELETE USER
# ---------------------------
@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    # Prevent the current admin from deleting their own account.
    if user_id == session["user_id"]:
        return """
        <h2 style="color:red;">Action Not Allowed</h2>
        <p>You cannot delete your own admin account.</p>
        <a href="/dashboard">Back to Dashboard</a>
        """

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Delete the selected user account.
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))

    conn.commit()
    conn.close()

    return redirect("/dashboard")


# ---------------------------
# VIEW PREDICTION DETAILS
# ---------------------------
@app.route("/prediction/<int:prediction_id>")
def prediction_detail(prediction_id):
    # Restrict access to logged-in users only.
    if "user_id" not in session:
        return redirect("/")

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Admin users can view any prediction record.
    if session.get("role") == "admin":
        cursor.execute("""
            SELECT
                predictions.id,
                users.username,
                predictions.subject,
                predictions.body,
                predictions.prediction,
                predictions.probability,
                predictions.timestamp
            FROM predictions
            JOIN users ON predictions.user_id = users.id
            WHERE predictions.id = ?
        """, (prediction_id,))

    # Regular users can only view their own prediction records.
    else:
        cursor.execute("""
            SELECT
                predictions.id,
                users.username,
                predictions.subject,
                predictions.body,
                predictions.prediction,
                predictions.probability,
                predictions.timestamp
            FROM predictions
            JOIN users ON predictions.user_id = users.id
            WHERE predictions.id = ?
            AND predictions.user_id = ?
        """, (prediction_id, session["user_id"]))

    prediction = cursor.fetchone()
    conn.close()

    if prediction is None:
        return """
        <h2 style="color:red;">Access Denied or Record Not Found</h2>
        <p>You do not have permission to view this prediction record.</p>
        <a href="/dashboard">Back to Dashboard</a>
        """

    return render_template("prediction_detail.html", prediction=prediction)


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
    # Ensure database has the required admin-management column before running the app.
    ensure_is_blocked_column()

    # Debug mode ON for development
    app.run(debug=True)