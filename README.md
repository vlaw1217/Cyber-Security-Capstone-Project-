# AI-Powered Phishing Detection System

A cybersecurity capstone project focused on developing an AI-powered phishing detection prototype that uses machine learning, email content analysis, attachment risk indicators, and dashboard monitoring to identify potentially malicious emails.

This project is being developed for **INFO49402 Cyber Security Capstone – Spring/Summer 2026** as a continuation of the Phase 1 proof-of-concept completed in **INFO36206**.

---

## Project Overview

The AI-Powered Phishing Detection System is designed to classify emails as phishing or legitimate based on email content and supporting risk indicators. In Phase 1, the project focused on building a proof-of-concept machine learning model using email subject and body text. The system used text preprocessing, TF-IDF feature extraction, supervised machine learning, and a basic Flask web interface for testing phishing predictions.

For Phase 2, the project is being revised and expanded into a more complete cybersecurity prototype. The system adds secure user access, role-based permissions, an administrative dashboard, Microsoft Graph API email integration, and attachment metadata analysis.

The goal is to demonstrate an end-to-end phishing analysis workflow from user login to email scanning, phishing classification, attachment risk checking, database storage, and dashboard result display.

---

## Project Status

**Current status:** Phase 2 system development in progress.

The project has moved from the Phase 1 machine learning proof-of-concept into the Phase 2 system implementation stage. The current system includes a Flask-based phishing detection web application with user authentication, role-based access control, prediction history storage, and an administrative dashboard.

Recent Phase 2 work focused on improving system structure, strengthening access control, building dashboard features for administrator monitoring, and integrating Microsoft Graph API for Outlook email retrieval. Users can now connect an Outlook account through Microsoft OAuth, retrieve recent Outlook emails, display email content in the system, and submit selected Outlook emails to the phishing detection model.

The system also includes the first stage of attachment analysis. Outlook attachment metadata can now be retrieved through Microsoft Graph API and displayed on the Outlook emails page, including filename, MIME type, file size, attachment type, and inline status. Full rule-based attachment risk scoring, SHA-256 hashing, database storage of attachment analysis results, and optional VirusTotal lookup are planned as next steps.

---

## Phase 1 Completed Work

The following work was completed during Phase 1:

- Collected and prepared phishing email dataset
- Cleaned and preprocessed email subject/body text
- Extracted TF-IDF text features
- Added basic engineered features where applicable
- Trained a supervised machine learning phishing detection model
- Saved the trained model and TF-IDF vectorizer
- Built a basic Flask web application
- Added manual email prediction functionality
- Displayed phishing classification results through the web interface
- Stored or prepared prediction history using a database workflow

---

## Phase 2 Current Enhancements

Phase 2 expands the original phishing detection proof-of-concept into a more complete cybersecurity prototype. The current focus is on secure system access, administrator monitoring, prediction history, dashboard review, and preparation for future Microsoft email integration and attachment analytics.

### 1. Phase 1 System Review and Refinement

Completed work:

- Reviewed the existing Flask phishing detection workflow
- Verified that the trained phishing model and TF-IDF vectorizer load correctly
- Tested manual email subject/body prediction
- Updated project files and repository structure
- Updated `.gitignore` to avoid pushing local database and virtual environment files
- Updated README documentation to better describe the project purpose, setup, and progress

### 2. Authentication and Role-Based Access Control

Completed work:

- Added user registration
- Added login and logout functionality
- Used hashed passwords for stored user credentials
- Added Flask session-based access control
- Added user and admin roles
- Restricted dashboard access to logged-in users only
- Added admin-only route protection
- Added blocked-user login prevention
- Tested login, registration, role-based access, and blocked-user behavior

### 3. Administration Dashboard

Completed work:

- Created an admin dashboard for monitoring scan results and system activity
- Added summary cards for total scans, phishing scans, suspicious scans, and legitimate scans
- Added admin alerts for phishing and suspicious email results
- Added prediction history table
- Updated prediction history to show usernames instead of only user IDs
- Added filtering by prediction type: All, Phishing, Suspicious, and Legitimate
- Added a View Details button for reviewing full email scan records
- Added a high-risk scans section for phishing or high-probability results
- Limited high-risk scan display to the top recent/highest-risk records
- Added UTC time labels for consistent audit-style timestamps

### 4. Admin User Management

Completed work:

- Added admin ability to create new users
- Added admin ability to create users with either regular user or admin role
- Added block and unblock user controls
- Added delete user control
- Added role-change controls to promote users to admin or change admins back to regular users
- Prevented the current admin from blocking, deleting, or demoting their own account
- Added confirmation prompts for sensitive admin actions

### 5. Admin Activity Logging

Completed work:

- Added an `admin_logs` database table
- Added logging for admin actions
- Logged user creation, blocking, unblocking, deletion, and role changes
- Displayed recent admin actions in an Admin Activity Log table
- Stored admin username, action type, target user, and timestamp
- Used UTC-style timestamps for consistency with security logging practices

### 6. Microsoft Graph API Email Integration

Completed work:

- Configured Microsoft Entra ID application registration for OAuth-based access
- Added Microsoft OAuth login flow using MSAL
- Added "Connect Outlook Email" functionality
- Stored Microsoft Graph access token in the Flask session
- Retrieved connected Outlook account information
- Retrieved recent Outlook emails through Microsoft Graph API
- Displayed Outlook email subject, sender, preview, and body content in the web application
- Allowed retrieved Outlook emails to be submitted to the phishing detection model
- Added "Disconnect Outlook" functionality
- Tested Outlook email retrieval using a connected testing email account

Current implementation:

- Outlook emails are retrieved from Microsoft Graph using the signed-in user's access token.
- Retrieved email content can be analyzed by the existing phishing detection model.
- Microsoft Graph integration is currently used for email retrieval and attachment metadata retrieval.

### 7. Attachment Analytics

Partially completed work:

- Created an `attachment_scans` SQLite table for future attachment scan result storage
- Added a Microsoft Graph attachment metadata retrieval module
- Retrieved attachment metadata from Outlook emails
- Displayed attachment information on the Outlook emails page
- Displayed filename, MIME type, file size, attachment type, and inline status
- Verified attachment retrieval using test emails with attachments

Planned next work:

- Add rule-based attachment risk detection
- Flag suspicious file extensions such as executable files, scripts, archives, and macro-enabled Office files
- Calculate SHA-256 hashes for file attachments
- Save attachment scan results into the `attachment_scans` table
- Combine email phishing prediction with attachment risk level
- Display attachment risk results in the scan detail page and admin dashboard
- Optionally integrate VirusTotal hash reputation lookup if API access is available

The attachment analytics component does not execute, open, or run attachment files. It focuses on safe static analysis using metadata, file type, file size, hash values, and optional reputation indicators.

---

## Expected Final Deliverables

By the end of Phase 2, the project is expected to include:

- Working phishing detection web application
- Reused or refined machine learning model and TF-IDF vectorizer
- Secure user registration and login system
- Password hashing and session-based authentication
- Role-based access control for regular users and administrators
- Functional administrative dashboard
- Admin user management features
- Prediction history with username display
- Prediction filtering and detailed scan review
- High-risk scan monitoring section
- Admin activity log for accountability
- Functional Microsoft Graph API integration for Outlook email retrieval
- Outlook email retrieval and submission to the phishing detection model
- Attachment metadata retrieval completed, with hash-based risk analysis planned as the next enhancement
- Updated SQLite database structure
- Clear GitHub README and setup documentation
- Weekly progress evidence
- Final presentation and demo materials
- End-to-end workflow demonstration

---

## Technology Stack

| Category | Tools / Technologies |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| Machine Learning | scikit-learn |
| Data Processing | pandas, NumPy |
| Feature Extraction | TF-IDF Vectorizer |
| Model Storage | joblib |
| Database | SQLite |
| Frontend | HTML, CSS, Bootstrap or basic styling |
| Email Integration | Microsoft Graph API, Microsoft Entra ID, OAuth 2.0 |
| Attachment Analysis | Python hashing libraries, metadata checks, optional VirusTotal API |
| Version Control | Git, GitHub |
| Development Environment | VS Code |

---

## Current Implemented Features

### User Features

- Register a new account
- Log in and log out
- Submit email subject and body for phishing analysis
- View personal prediction history
- Filter prediction history by prediction type
- View detailed scan results
- Connect Outlook account through Microsoft Graph API
- View recent Outlook emails inside the web application
- Submit retrieved Outlook emails for phishing analysis
- Disconnect Outlook account
- View attachment metadata for Outlook emails

### Admin Features

- View all prediction records
- View dashboard summary statistics
- Monitor phishing, suspicious, and legitimate scan counts
- Review high-risk scans
- Filter prediction history by prediction type
- View full scan details
- Add new users
- Block and unblock users
- Delete users
- Change user roles
- View admin activity logs

### Security Features

- Password hashing
- Session-based authentication
- Role-based access control
- Admin-only route protection
- Blocked-user login prevention
- Admin self-protection against blocking, deleting, or demoting own account
- UTC-style timestamps for audit consistency

### Email Integration and Attachment Metadata Features

- Microsoft OAuth-based Outlook connection
- Microsoft Graph API email retrieval
- Outlook email display inside the Flask application
- Attachment metadata retrieval through Microsoft Graph API
- Attachment filename, MIME type, size, attachment type, and inline status display
- Safe attachment handling approach that does not execute or open files

---

## Repository Structure

```text
Cyber-Security-Capstone-Project/
│
├── app.py
├── database.py
├── graph_attachments.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── phishing_model_v1.pkl
├── tfidf_vectorizer_v1.pkl
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── emails.html
│   └── prediction_detail.html
│
└── app.db              # Local SQLite database, ignored by Git
```

> Note: `app.db` is used locally for users, prediction history, blocked account status, and admin activity logs. It is ignored by Git and should not be pushed to GitHub.
