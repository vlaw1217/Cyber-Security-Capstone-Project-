import joblib
import sqlite3
import numpy as np
import os
import tempfile
import msal
import requests
import re
import html
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, session
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.sparse import hstack, csr_matrix
from database import (
    get_all_predictions, 
    get_user_predictions, 
    save_attachment_scan, 
    save_header_scan, 
    get_header_scan, 
    save_sandbox_scan, 
    update_sandbox_scan_result,
    get_attachment_scans_with_sandbox
)
from sandbox_api import (
    submit_file_to_hybrid_analysis,
    get_hybrid_analysis_overview,
    parse_hybrid_analysis_overview
)
from graph_attachments import get_email_attachments, download_attachment_bytes
from attachment_analyzer import analyze_attachment_metadata, calculate_sha256
from virustotal_checker import check_virustotal_hash
from header_analyzer import analyze_email_headers




# ---------------------------
# MICROSOFT GRAPH CONFIGURATION
# ---------------------------
# Load Microsoft Graph / Entra credentials from the .env file.
# These values are used for OAuth authentication and Graph API access.
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID", "common")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
REDIRECT_URI = os.getenv("REDIRECT_URI")

SCOPES = ["User.Read", "Mail.ReadWrite"]

# ---------------------------
# BUILD MSAL APPLICATION
# ---------------------------
# Create the Microsoft Authentication Library (MSAL) application object.
# This object is responsible for:
# - generating Microsoft login URLs
# - handling OAuth authentication
# - acquiring Graph API access tokens
def build_msal_app():
    return msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET
    )


# ---------------------------
# MOVE OUTLOOK EMAIL
# ---------------------------
# Moves an Outlook email to another folder using Microsoft Graph.
# Common destination folders:
# - deleteditems
# - junkemail
def move_outlook_email(graph_token, message_id, destination_folder):
    if not graph_token or not message_id or not destination_folder:
        return False, "Missing Graph token, message ID, or destination folder."

    headers = {
        "Authorization": "Bearer " + graph_token,
        "Content-Type": "application/json"
    }

    payload = {
        "destinationId": destination_folder
    }

    response = requests.post(
        f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/move",
        headers=headers,
        json=payload
    )

    if response.status_code == 201:
        return True, f"Email moved to {destination_folder}."

    return False, f"Graph API error {response.status_code}: {response.text[:500]}"


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
# ADMIN ACTIVITY LOG TABLE
# ---------------------------
# Create admin_logs table if it does not already exist.
# This table records important admin actions for accountability.
def ensure_admin_logs_table():
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER,
            admin_username TEXT,
            action TEXT,
            target_user_id INTEGER,
            target_username TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------
# ADMIN ACTIVITY LOGGER
# ---------------------------
# Insert an admin action into the admin_logs table.
def log_admin_action(action, target_user_id=None, target_username=None):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO admin_logs (
            admin_id,
            admin_username,
            action,
            target_user_id,
            target_username
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        session.get("user_id"),
        session.get("username"),
        action,
        target_user_id,
        target_username
    ))

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
# EMAIL BODY CLEANER
# ---------------------------
# Microsoft Graph returns full email bodies in HTML format.
# This function removes HTML tags and converts HTML entities
# so the model and UI receive readable text.
def clean_email_body(raw_body):

    # Handle empty email bodies safely.
    if not raw_body:
        return ""

    # Convert HTML entities into normal characters.
    # Example: &nbsp; becomes a regular space.
    clean_text = html.unescape(raw_body)

    # Remove HTML tags.
    clean_text = re.sub(r"<[^>]+>", " ", clean_text)

    # Replace non-breaking spaces with normal spaces.
    clean_text = clean_text.replace("\xa0", " ")

    # Remove extra spaces, tabs, and line breaks.
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return clean_text


# ---------------------------
# FORMAT EMAIL BODY FOR DISPLAY
# ---------------------------
# Converts plain image URLs in the saved email body into visible images.
# This is only for displaying the email body on prediction_detail.html.
def format_email_body_for_display(body):
    if not body:
        return ""

    # Escape the email body first so unsafe HTML is not rendered.
    safe_body = html.escape(body)

    # Detect image URLs, including ones inside square brackets.
    image_url_pattern = r"\[?(https?://[^\s\]]+\.(?:png|jpg|jpeg|gif|webp)(?:\?[^\s\]]*)?)\]?"

    # Replace image URLs with HTML image tags.
    formatted_body = re.sub(
        image_url_pattern,
        r'<br><img src="\1" alt="Email image" style="max-width:250px; height:auto; margin:10px 0;"><br>',
        safe_body
    )

    return formatted_body

# ---------------------------
# GET OUTLOOK EMAIL HEADERS
# ---------------------------
# Retrieve full internet message headers for one Outlook email.
# These headers are used for sender spoofing analysis.
def get_outlook_email_headers(graph_token, message_id):
    if not graph_token or not message_id:
        return []

    headers = {
        "Authorization": "Bearer " + graph_token
    }

    response = requests.get(
        f"https://graph.microsoft.com/v1.0/me/messages/{message_id}?$select=internetMessageHeaders",
        headers=headers
    )

    if response.status_code != 200:
        print("Header retrieval failed:", response.status_code)
        print(response.text[:1000])
        return []

    data = response.json()

    return data.get("internetMessageHeaders", [])


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
    # Retrieve and clean the email body text.
    # Outlook emails may contain raw HTML content,
    # so the body is cleaned before analysis.
    body = clean_email_body(request.form["body"])    
    source = request.form.get("source", "dashboard") # Used for back button after prediction is complete

    # Get the Microsoft Graph message ID from the Outlook email form.
    # This ID is needed to retrieve and save attachment scan results for the selected email.
    message_id = request.form.get("message_id")

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


    # ---------------------------
    # MODEL INPUT DEBUGGING
    # ---------------------------
    # Temporary debugging output used to inspect exactly
    # what information is being fed into the phishing model.

    print("\n================ MODEL INPUT DEBUG ================\n")

    print("SUBJECT:")
    print(subject)

    print("\nCLEANED EMAIL BODY:")
    print(body[:3000])  # Limit output length for readability

    print("\nCOMBINED TEXT FED TO TF-IDF:")
    print(text[:3000])

    print("\nMETADATA FEATURES:")
    print(f"Subject Length: {subject_length}")
    print(f"Body Length: {body_length}")
    print(f"URL Count: {url_count}")
    print(f"Phishing Keyword Count: {phishing_keyword_count}")
    print(f"Uppercase Word Count: {uppercase_count}")
    print(f"Digit Count: {digit_count}")

    print("\n===================================================\n")

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
    if prob[1] > 0.7:
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

    prediction_id = cursor.lastrowid

    conn.commit()
    conn.close()

    # ---------------------------------------------------
    # Save email header spoofing analysis for Outlook emails
    # ---------------------------------------------------
    # This runs only when the email came from Microsoft Graph
    # and a valid message_id is available.
    if source == "emails" and message_id and "graph_token" in session:

        internet_headers = get_outlook_email_headers(
            session["graph_token"],
            message_id
        )

        header_result = analyze_email_headers(internet_headers)

        risk_reason = " | ".join(header_result.get("warnings", []))

        save_header_scan(
            prediction_id=prediction_id,
            message_id=message_id,
            from_domain=header_result.get("from_domain"),
            return_path_domain=header_result.get("return_path_domain"),
            reply_to_domain=header_result.get("reply_to_domain"),
            spf_result=header_result.get("spf_result"),
            dkim_result=header_result.get("dkim_result"),
            dmarc_result=header_result.get("dmarc_result"),
            compauth_result=header_result.get("compauth_result"),
            dkim_domain=header_result.get("dkim_domain"),
            risk_score=header_result.get("risk_score"),
            risk_level=header_result.get("risk_level"),
            risk_reason=risk_reason
        )

    # ---------------------------------------------------
    # Save attachment analysis results for Outlook emails
    # ---------------------------------------------------
    # This runs only when the email came from Microsoft Graph
    # and a valid message_id is available.
    if source == "emails" and message_id and "graph_token" in session:

        # Retrieve attachment metadata again using the selected Outlook message ID.
        attachments = get_email_attachments(session["graph_token"], message_id)

        for attachment in attachments:
            # Get attachment ID and type from Microsoft Graph metadata.
            attachment_id = attachment.get("attachment_id")
            attachment_type = attachment.get("attachment_type")

            # Default SHA-256 hash is None until successfully calculated.
            sha256_hash = None

            # Download bytes only for Microsoft Graph file attachments.
            # The file is not opened, saved, or executed.
            if attachment_id and attachment_type == "#microsoft.graph.fileAttachment":
                file_bytes = download_attachment_bytes(
                    session["graph_token"],
                    message_id,
                    attachment_id
                )

                # Calculate SHA-256 hash from raw bytes.
                sha256_hash = calculate_sha256(file_bytes)

            # Run rule-based attachment metadata analysis.
            result = analyze_attachment_metadata(
                filename=attachment.get("name"),
                mime_type=attachment.get("content_type"),
                size_bytes=attachment.get("size")
            )

            # Add SHA-256 hash to the result.
            result["sha256_hash"] = sha256_hash

            # Check VirusTotal reputation using only the hash.
            virustotal_result = check_virustotal_hash(sha256_hash)

            # Prepare readable VirusTotal summary for database storage.
            virustotal_summary = (
                f"{virustotal_result.get('status')} - "
                f"{virustotal_result.get('message')}"
            )

            # If VirusTotal reports malicious detections, upgrade risk to High.
            if virustotal_result.get("status") == "Malicious":
                result["risk_level"] = "High"
                result["risk_reason"] += "; VirusTotal reported malicious detections"

            # If VirusTotal reports suspicious detections and current risk is Low,
            # upgrade risk to Medium.
            elif virustotal_result.get("status") == "Suspicious" and result["risk_level"] == "Low":
                result["risk_level"] = "Medium"
                result["risk_reason"] += "; VirusTotal reported suspicious detections"

            # Save the attachment scan result into SQLite and get the new attachment scan ID.
            # This ID will be used later to link sandbox analysis results to this attachment.
            attachment_scan_id = save_attachment_scan(
                prediction_id=prediction_id,
                message_id=message_id,
                filename=result.get("filename"),
                extension=result.get("extension"),
                mime_type=result.get("mime_type"),
                size_bytes=result.get("size_bytes"),
                sha256_hash=result.get("sha256_hash"),
                risk_level=result.get("risk_level"),
                risk_reason=result.get("risk_reason"),
                virustotal_result=virustotal_summary
            )

            # Decide whether this attachment should be submitted to the sandbox.
            # To reduce API usage, only medium/high risk attachments or dangerous file types are submitted.
            dangerous_extensions = [
                ".exe", ".js", ".vbs", ".scr", ".ps1", ".bat",
                ".cmd", ".jar", ".docm", ".xlsm", ".pptm", ".iso",
                ".img", ".lnk"
            ]

            extension = result.get("extension") or ""
            risk_level = result.get("risk_level") or ""

            should_submit_to_sandbox = (
                risk_level in ["Medium", "High"]
                or extension.lower() in dangerous_extensions
            )

                        # Submit suspicious attachments to Hybrid Analysis sandbox.
            # The attachment bytes are written to a temporary file only for API upload.
            # The temporary file is deleted after submission.
            if should_submit_to_sandbox and file_bytes:
                temp_file_path = None
                sandbox_scan_id = None

                try:
                    filename = result.get("filename") or "attachment.bin"

                    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                        temp_file.write(file_bytes)
                        temp_file_path = temp_file.name

                    sandbox_submission = submit_file_to_hybrid_analysis(
                        file_path=temp_file_path,
                        filename=filename
                    )

                    if sandbox_submission["success"]:
                        sandbox_task_id = (
                            sandbox_submission["data"].get("job_id")
                            or sandbox_submission["data"].get("id")
                            or sandbox_submission["data"].get("sha256")
                        )

                        sandbox_scan_id = save_sandbox_scan(
                            attachment_scan_id=attachment_scan_id,
                            sandbox_provider="Hybrid Analysis",
                            sandbox_task_id=sandbox_task_id,
                            sandbox_status="submitted",
                            raw_result=str(sandbox_submission["data"])
                        )

                        # Try to retrieve the overview/report using the SHA-256 hash.
                        overview_result = get_hybrid_analysis_overview(sha256_hash)

                        if overview_result["success"]:
                            parsed_result = parse_hybrid_analysis_overview(
                                overview_result["data"]
                            )

                            update_sandbox_scan_result(
                                sandbox_scan_id=sandbox_scan_id,
                                sandbox_status="completed",
                                sandbox_verdict=parsed_result["sandbox_verdict"],
                                threat_score=parsed_result["threat_score"],
                                behavior_summary=parsed_result["behavior_summary"],
                                network_indicators=parsed_result["network_indicators"],
                                file_indicators=parsed_result["file_indicators"],
                                report_url=parsed_result["report_url"],
                                raw_result=parsed_result["raw_result"]
                            )
                        else:
                            update_sandbox_scan_result(
                                sandbox_scan_id=sandbox_scan_id,
                                sandbox_status="report_pending",
                                behavior_summary=overview_result["message"],
                                raw_result=str(overview_result["data"])
                            )

                    else:
                        save_sandbox_scan(
                            attachment_scan_id=attachment_scan_id,
                            sandbox_provider="Hybrid Analysis",
                            sandbox_task_id=None,
                            sandbox_status="submission_failed",
                            behavior_summary=sandbox_submission["message"],
                            raw_result=str(sandbox_submission["data"])
                        )

                except Exception as sandbox_error:
                    save_sandbox_scan(
                        attachment_scan_id=attachment_scan_id,
                        sandbox_provider="Hybrid Analysis",
                        sandbox_task_id=None,
                        sandbox_status="error",
                        behavior_summary=str(sandbox_error),
                        raw_result=None
                    )

                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

    return redirect(f"/prediction/{prediction_id}?source={source}")


# ---------------------------
# MOVE EMAIL TO DELETED ITEMS
# ---------------------------
@app.route("/email/delete", methods=["POST"])
def delete_outlook_email():
    if "user_id" not in session:
        return redirect("/")

    if "graph_token" not in session:
        return redirect("/connect_outlook")

    message_id = request.form.get("message_id")

    success, message = move_outlook_email(
        graph_token=session["graph_token"],
        message_id=message_id,
        destination_folder="deleteditems"
    )

    if not success:
        return f"""
        <h2 style="color:red;">Unable to delete email</h2>
        <p>{message}</p>
        <a href="/dashboard#prediction-history">Back to Dashboard</a>
        """

    return redirect("/emails")


# ---------------------------
# MOVE EMAIL TO JUNK EMAIL
# ---------------------------
@app.route("/email/junk", methods=["POST"])
def junk_outlook_email():
    if "user_id" not in session:
        return redirect("/")

    if "graph_token" not in session:
        return redirect("/connect_outlook")

    message_id = request.form.get("message_id")

    success, message = move_outlook_email(
        graph_token=session["graph_token"],
        message_id=message_id,
        destination_folder="junkemail"
    )

    if not success:
        return f"""
        <h2 style="color:red;">Unable to move email to junk</h2>
        <p>{message}</p>
        <a href="/dashboard#prediction-history">Back to Dashboard</a>
        """

    return redirect("/emails")

# ---------------------------
# PASSWORD POLICY VALIDATION
# ---------------------------
def validate_password_policy(password):
    """
    Validate password strength for new user registration.

    Password requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number
    - At least one special character
    """

    if len(password) < 8:
        return False, "Password must be at least 8 characters long."

    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one uppercase letter."

    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one lowercase letter."

    if not re.search(r"\d", password):
        return False, "Password must include at least one number."

    if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\;/`~]", password):
        return False, "Password must include at least one special character."

    return True, "Password meets the security requirements."


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
        
        # Check password strength before creating the account.
        is_valid_password, password_message = validate_password_policy(password)

        if not is_valid_password:
            return f"""
            <h2 style="color:red;">Weak Password</h2>
            <p>{password_message}</p>
            <p>Password must be at least 8 characters and include uppercase, lowercase, number, and special character.</p>
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
    
    # Get selected prediction filter from URL query string.
    # Example: /dashboard?prediction_filter=Phishing
    prediction_filter = request.args.get("prediction_filter", "All")

    # Admin users can view all prediction records from every user.
    # Admin users also retrieve the full user list for user-management features.
    if session.get("role") == "admin":
        dashboard_title = "Admin Dashboard"

         # Admin can filter all prediction records by prediction type.
        if prediction_filter in ["Phishing", "Suspicious", "Legitimate"]:
            conn = sqlite3.connect("app.db")
            cursor = conn.cursor()
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
                WHERE predictions.prediction = ?
                ORDER BY predictions.timestamp DESC
            """, (prediction_filter,))
            data = cursor.fetchall()
            conn.close()
        else:
            data = get_all_predictions()

        # Retrieve all registered users so the admin can manage accounts.
        # is_blocked is used to show whether each account is active or blocked.
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, is_blocked FROM users ORDER BY id")
        users = cursor.fetchall()
        conn.close()

        # Retrieve high-risk scans for admin review.
        # High-risk means the email was classified as Phishing
        # or the phishing probability is 0.80 or higher.
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                predictions.id,
                users.username,
                predictions.subject,
                predictions.prediction,
                predictions.probability,
                predictions.timestamp
            FROM predictions
            JOIN users ON predictions.user_id = users.id
            WHERE (predictions.prediction = 'Phishing'
            OR predictions.probability >= 0.80)
            ORDER BY predictions.probability DESC, predictions.timestamp DESC
            LIMIT 5
        """)
        high_risk_scans = cursor.fetchall()
        conn.close()

        # Retrieve the latest admin activity logs.
        # This shows recent admin actions such as adding, blocking, unblocking, or deleting users.
        # LIMIT 10 keeps the dashboard readable by showing only the latest 10 actions.
        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                id,
                admin_username,
                action,
                target_username,
                timestamp
            FROM admin_logs
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        admin_logs = cursor.fetchall()
        conn.close()
    
    # Regular users can only view their own prediction history.
    # users is set to an empty list because user management is admin-only.
    else:
        dashboard_title = "User Dashboard"
        
        # Regular users can filter only their own prediction records.
        if prediction_filter in ["Phishing", "Suspicious", "Legitimate"]:
            conn = sqlite3.connect("app.db")
            cursor = conn.cursor()
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
                WHERE predictions.user_id = ?
                AND predictions.prediction = ?
                ORDER BY predictions.timestamp DESC
            """, (session["user_id"], prediction_filter))
            data = cursor.fetchall()
            conn.close()
        else:
            data = get_user_predictions(session["user_id"])

        users = []
        high_risk_scans = []
        admin_logs = []

    graph_connected = "graph_token" in session
    # Send prediction data, user list, dashboard title, username, and role to dashboard.html.
    return render_template(
    "dashboard.html",
    data=data,
    users=users,
    high_risk_scans=high_risk_scans,
    admin_logs=admin_logs,
    prediction_filter=prediction_filter,
    dashboard_title=dashboard_title,
    username=session.get("username"),
    role=session.get("role"),
    graph_connected=graph_connected,
    graph_email=session.get("graph_email")
)

# ---------------------------
# CONNECT OUTLOOK ACCOUNT
# ---------------------------
# Redirect the user to Microsoft's login page.
# After successful authentication, Microsoft redirects back
# to the /getAToken route with an authorization code.
@app.route("/connect_outlook")
def connect_outlook():

    # Ensure user is logged into local Flask app first
    if "user_id" not in session:
        return redirect("/")

    # Generate Microsoft OAuth login URL
    auth_url = build_msal_app().get_authorization_request_url(
        SCOPES,
        redirect_uri=REDIRECT_URI
    )

    # Redirect user to Microsoft login page
    return redirect(auth_url)


# ---------------------------
# MICROSOFT AUTH CALLBACK
# ---------------------------
# Microsoft redirects the user here after login.
# This route exchanges the authorization code for
# a Microsoft Graph access token.
@app.route("/getAToken")
def get_token():

    # Ensure user is logged into local Flask app
    if "user_id" not in session:
        return redirect("/")

    # Retrieve authorization code from Microsoft
    code = request.args.get("code")

    # Handle failed or cancelled login
    if not code:
        return "Microsoft login failed or was cancelled."

    # Exchange authorization code for access token
    result = build_msal_app().acquire_token_by_authorization_code(
        code,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    # Save access token into Flask session
    if "access_token" in result:

        session["graph_token"] = result["access_token"]

        headers = {
            "Authorization": "Bearer " + session["graph_token"],
            "Prefer": 'outlook.body-content-type="text"'
        }

        profile_response = requests.get(
            "https://graph.microsoft.com/v1.0/me",
            headers=headers
        )

        profile = profile_response.json()

        session["graph_email"] = (
            profile.get("mail")
            or profile.get("userPrincipalName")
            or "Connected account"
        )

    return redirect("/emails")


# ---------------------------
# LOAD OUTLOOK EMAILS
# ---------------------------
# Retrieve recent Outlook emails from Microsoft Graph API.
# Emails are displayed inside emails.html and can later
# be analyzed by the phishing detection model.
@app.route("/emails")
def emails():

    # Ensure user is logged into local Flask app
    if "user_id" not in session:
        return redirect("/")

    # Ensure Microsoft account is connected
    if "graph_token" not in session:
        return redirect("/connect_outlook")

    # Microsoft Graph authorization header
    # The Prefer header requests cleaner plain-text
    # email body content when available.
    headers = {
        "Authorization": "Bearer " + session["graph_token"],
        "Prefer": 'outlook.body-content-type="text"'
    }

    # Request recent emails from Microsoft Graph
    # Include id and hasAttachments because they are needed for attachment metadata retrieval.
    response = requests.get(
    "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages?$top=10&$select=id,subject,bodyPreview,body,from,receivedDateTime,hasAttachments",
    headers=headers
    )
    print("Graph email response status:", response.status_code)
    print("Graph email response:", response.text[:1000])    

    # Retrieve and analyze attachment metadata for each Outlook email.
    messages = response.json().get("value", [])

    # Retrieve and analyze attachment metadata for each Outlook email.
    # For each email, use its Microsoft Graph message ID to retrieve attachment metadata.
    for message in messages:
        message_id = message.get("id")

        if message_id:
            
            attachments = get_email_attachments(session["graph_token"], message_id)

            # Store attachment metadata so it can still be displayed in emails.html.
            message["attachments"] = attachments

            # Analyze each attachment using rule-based metadata checks.
            # This does not open or execute the attachment file.
            # File bytes may be downloaded only for SHA-256 hash calculation.
            attachment_analysis = []

            for attachment in attachments:
                # Get basic attachment fields returned by Microsoft Graph.
                attachment_id = attachment.get("attachment_id")
                attachment_type = attachment.get("attachment_type")

                # Default hash value is None until the file bytes are successfully downloaded.
                sha256_hash = None

                # Only file attachments should be downloaded for hashing.
                # Reference/cloud attachments or unsupported attachment types are skipped.
                if attachment_id and attachment_type == "#microsoft.graph.fileAttachment":
                    file_bytes = download_attachment_bytes(
                        session["graph_token"],
                        message_id,
                        attachment_id
                    )

                    # Calculate SHA-256 only from raw bytes.
                    # The file is not opened, saved, or executed.
                    sha256_hash = calculate_sha256(file_bytes)

                # Run the existing rule-based metadata risk analysis.
                result = analyze_attachment_metadata(
                    filename=attachment.get("name"),
                    mime_type=attachment.get("content_type"),
                    size_bytes=attachment.get("size")
                )

                # Add the SHA-256 hash into the analysis result.
                result["sha256_hash"] = sha256_hash

                # Check the SHA-256 hash reputation using VirusTotal.
                # This sends only the hash, not the actual file.
                virustotal_result = check_virustotal_hash(sha256_hash)

                # Add VirusTotal result into the analysis result.
                result["virustotal_status"] = virustotal_result.get("status")
                result["virustotal_message"] = virustotal_result.get("message")
                result["virustotal_malicious"] = virustotal_result.get("malicious")
                result["virustotal_suspicious"] = virustotal_result.get("suspicious")

                # If VirusTotal reports malicious detections, increase risk to High.
                if virustotal_result.get("status") == "Malicious":
                    result["risk_level"] = "High"
                    result["risk_reason"] += "; VirusTotal reported malicious detections"

                # If VirusTotal reports suspicious detections and current risk is Low,
                # increase risk to Medium.
                elif virustotal_result.get("status") == "Suspicious" and result["risk_level"] == "Low":
                    result["risk_level"] = "Medium"
                    result["risk_reason"] += "; VirusTotal reported suspicious detections"

                attachment_analysis.append(result)

            # Store the rule-based analysis result inside the message dictionary.
            # This allows emails.html to display the risk level and risk reason.
            message["attachment_analysis"] = attachment_analysis

            # Temporary terminal output for testing.
            # It only prints the number of attachments and their risk levels,
            # not the full private email content.
            risk_summary = [item["risk_level"] for item in attachment_analysis]
            print(
                f"Attachment metadata retrieved: {len(attachments)} attachment(s), "
                f"risk summary: {risk_summary}"
            )

    # Render email viewer page
    return render_template(
        "emails.html",
        messages=messages
    )


# ---------------------------
# DISCONNECT OUTLOOK ACCOUNT
# ---------------------------
@app.route("/disconnect_outlook")
def disconnect_outlook():

    # Remove Outlook session data
    session.pop("graph_token", None)
    session.pop("graph_email", None)

    return redirect("/dashboard")


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
        # Insert the new user account into the users table.
        # The password is already hashed, and is_blocked = 0 means the account is active.
        cursor.execute(
            "INSERT INTO users (username, password, role, is_blocked) VALUES (?, ?, ?, ?)",
            (username, hashed_password, role, 0)
        )

        # Store the new user's database ID.
        # This ID will be saved in the admin activity log.
        new_user_id = cursor.lastrowid

        conn.commit()
        conn.close()

        # Record this action in the admin activity log.
        # This helps show which admin created which account.
        log_admin_action(
            action=f"Added new {role} account",
            target_user_id=new_user_id,
            target_username=username
        )

        return redirect("/dashboard#user-management")

    except sqlite3.IntegrityError:
        conn.close()
        return """
        <h2 style="color:red;">Username already exists</h2>
        <p>Please choose another username.</p>
        <a href="/dashboard">Back to Dashboard</a>
        """


# ---------------------------
# ADMIN: BLOCK USER
# ---------------------------
@app.route("/admin/block_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_block_user(user_id):
    # Prevent the current admin from blocking their own account.
    # This avoids accidentally locking the administrator out of the system.
    if user_id == session["user_id"]:
        return """
        <h2 style="color:red;">Action Not Allowed</h2>
        <p>You cannot block your own admin account.</p>
        <a href="/dashboard">Back to Dashboard</a>
        """

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Get the target user's username before blocking the account.
    # This username will be saved in the admin activity log.
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()

    # Block the selected user account.
    # is_blocked = 1 means the user cannot log in.
    cursor.execute("UPDATE users SET is_blocked = 1 WHERE id = ?", (user_id,))

    conn.commit()
    conn.close()

    # Record this action in the admin activity log.
    # This helps show which admin blocked which user.
    if target_user:
        log_admin_action(
            action="Blocked user account",
            target_user_id=user_id,
            target_username=target_user[0]
        )

    return redirect("/dashboard#user-management")


# ---------------------------
# ADMIN: UNBLOCK USER
# ---------------------------
@app.route("/admin/unblock_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_unblock_user(user_id):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Get the target user's username before updating the account.
    # This username will be stored in the admin activity log.
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()

    # Unblock the selected user account.
    # is_blocked = 0 means the user can log in again.
    cursor.execute("UPDATE users SET is_blocked = 0 WHERE id = ?", (user_id,))

    conn.commit()
    conn.close()

    # Record this action in the admin activity log.
    # This helps show which admin unblocked which user.
    if target_user:
        log_admin_action(
            action="Unblocked user account",
            target_user_id=user_id,
            target_username=target_user[0]
        )

    return redirect("/dashboard#user-management")


# ---------------------------
# ADMIN: DELETE USER
# ---------------------------
@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    # Prevent the current admin from deleting their own account.
    # This avoids accidentally locking the administrator out of the system.
    if user_id == session["user_id"]:
        return """
        <h2 style="color:red;">Action Not Allowed</h2>
        <p>You cannot delete your own admin account.</p>
        <a href="/dashboard">Back to Dashboard</a>
        """

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Get the target user's username before deleting the account.
    # After deletion, the user record will no longer exist in the users table.
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()

    # Delete the selected user account from the users table.
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))

    conn.commit()
    conn.close()

    # Record this action in the admin activity log after deletion.
    # The username is stored as text so the log stays readable even after the account is removed.
    if target_user:
        log_admin_action(
            action="Deleted user account",
            target_user_id=user_id,
            target_username=target_user[0]
        )

    return redirect("/dashboard#user-management")


# ---------------------------
# ADMIN: CHANGE USER ROLE
# ---------------------------
@app.route("/admin/change_role/<int:user_id>", methods=["POST"])
@admin_required
def admin_change_role(user_id):
    # Prevent the current admin from changing their own role.
    # This avoids accidentally removing their own admin access.
    if user_id == session["user_id"]:
        return """
        <h2 style="color:red;">Action Not Allowed</h2>
        <p>You cannot change your own admin role.</p>
        <a href="/dashboard#user-management">Back to Dashboard</a>
        """

    # Get the new role from the submitted form.
    new_role = request.form["new_role"].strip()

    # Only allow valid role values.
    if new_role not in ["user", "admin"]:
        return """
        <h2 style="color:red;">Invalid Role</h2>
        <p>The selected role is not valid.</p>
        <a href="/dashboard#user-management">Back to Dashboard</a>
        """

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Get the target user's current username and role before updating.
    # This information will be used for the admin activity log.
    cursor.execute("SELECT username, role FROM users WHERE id = ?", (user_id,))
    target_user = cursor.fetchone()

    if target_user is None:
        conn.close()
        return """
        <h2 style="color:red;">User Not Found</h2>
        <p>The selected user does not exist.</p>
        <a href="/dashboard#user-management">Back to Dashboard</a>
        """

    old_role = target_user[1]

    # Update the selected user's role.
    cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))

    conn.commit()
    conn.close()

    # Record this action in the admin activity log.
    log_admin_action(
        action=f"Changed role from {old_role} to {new_role}",
        target_user_id=user_id,
        target_username=target_user[0]
    )

    return redirect("/dashboard#user-management")


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
    header_scan = get_header_scan(prediction_id)
    attachment_scans = get_attachment_scans_with_sandbox(prediction_id)

    if prediction is None:
        return """
        <h2 style="color:red;">Access Denied or Record Not Found</h2>
        <p>You do not have permission to view this prediction record.</p>
        <a href="/dashboard">Back to Dashboard</a>
        """

    source = request.args.get("source", "dashboard")
    formatted_body = format_email_body_for_display(prediction[3])


    return render_template(
        "prediction_detail.html",
        prediction=prediction,
        formatted_body=formatted_body,
        header_scan=header_scan,
        attachment_scans=attachment_scans,
        source=source
    )


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

    # Ensure admin activity log table exists before running the app.
    ensure_admin_logs_table()

    # Debug mode ON for development
    app.run(debug=True)