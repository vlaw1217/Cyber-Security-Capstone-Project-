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

The system also includes an expanded email security analysis workflow. Outlook attachment metadata can now be retrieved through Microsoft Graph API and displayed on the Outlook emails page, including filename, MIME type, file size, attachment type, inline status, SHA-256 hash, and attachment risk level. The system performs rule-based attachment risk detection, checks hash reputation through VirusTotal, and saves attachment analysis results into the SQLite `attachment_scans` table.

In addition, the system now supports email header spoofing analysis and sandbox-based attachment analysis. Email headers are retrieved through Microsoft Graph API and analyzed for SPF, DKIM, DMARC, Return-Path, Reply-To, and DKIM domain alignment indicators. Suspicious attachments can also be submitted to Hybrid Analysis / Falcon Sandbox for behavioral analysis. Sandbox verdicts and behavior summaries are saved into the SQLite `sandbox_scans` table and displayed on the Prediction Details page.

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

Completed work:

- Created an `attachment_scans` SQLite table for attachment scan result storage
- Added a Microsoft Graph attachment metadata retrieval module
- Retrieved attachment metadata from Outlook emails
- Displayed attachment information on the Outlook emails page
- Displayed filename, MIME type, file size, attachment type, and inline status
- Added rule-based attachment risk detection
- Flagged suspicious file extensions such as executable files, scripts, archives, macro-enabled Office files, and uncommon binary files
- Calculated SHA-256 hashes for file attachments when attachment bytes are available
- Integrated VirusTotal hash reputation lookup using SHA-256 hashes
- Displayed VirusTotal reputation status and message on the Outlook emails page
- Saved attachment scan results into the `attachment_scans` SQLite table after an Outlook email is analyzed
- Stored attachment metadata, SHA-256 hash, risk level, risk reason, VirusTotal result, and scan timestamp
- Verified attachment retrieval, risk detection, SHA-256 hashing, VirusTotal lookup, and SQLite storage using test Outlook emails with attachments

Current implementation:

- Attachment analysis is performed using safe static analysis.
- The system does not open, run, or execute attachment files.
- Attachment bytes are only retrieved when needed for SHA-256 hash calculation.
- VirusTotal integration checks only the hash value and does not upload attachment files.
- Attachment scan results are linked to the related prediction record through `prediction_id`.

### 8. Email Header Spoofing Analysis

Completed work:

* Added email header spoofing analysis for Outlook emails
* Retrieved Outlook internet message headers using Microsoft Graph API
* Analyzed SPF, DKIM, DMARC, Return-Path, Reply-To, and Microsoft composite authentication results
* Checked sender-domain alignment indicators such as visible From domain, Return-Path domain, Reply-To domain, and DKIM signing domain
* Calculated a header risk score and header risk level
* Saved header scan results into the SQLite `header_scans` table
* Displayed header risk score, risk level, authentication results, and explanation on the Prediction Details page

Current implementation:

* Header analysis is performed only for Outlook emails retrieved through Microsoft Graph API.
* The system analyzes authentication and domain-alignment indicators to help identify possible spoofing or sender impersonation.
* Header scan results are linked to the related prediction record through `prediction_id`.

### 9. Sandbox-Based Attachment Analysis

Completed work:

* Added Hybrid Analysis / Falcon Sandbox API configuration using environment variables
* Created a `sandbox_api.py` helper module for sandbox API communication
* Tested Hybrid Analysis API connection successfully
* Added file submission support for suspicious attachments
* Retrieved sandbox overview/report results using SHA-256 hash values
* Parsed sandbox verdicts and behavior summaries into simplified result fields
* Created a `sandbox_scans` SQLite table for sandbox result storage
* Saved sandbox provider, status, verdict, behavior summary, raw result, and timestamps into SQLite
* Linked sandbox scan results to attachment scan records through `attachment_scan_id`
* Displayed Hybrid Analysis sandbox results on the Prediction Details page

Current implementation:

* Suspicious attachments are submitted to Hybrid Analysis only when the attachment has a medium/high risk level or a risky file extension.
* The system writes attachment bytes to a temporary file only for API upload.
* Temporary files are deleted after sandbox submission.
* Sandbox results are stored in SQLite and displayed together with attachment metadata and risk results.
* VirusTotal is kept as a hash reputation comparison/fallback, while Hybrid Analysis provides deeper sandbox-based behavioral analysis.

Planned next work:

- Create overall risk calculation logic that combines email prediction risk, header risk, attachment risk, and sandbox results
- Display attachment and sandbox risk indicators in the admin dashboard
- Improve presentation of high-risk attachment results for administrator review
- Add optional refresh logic for sandbox reports that are still pending

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
- Attachment metadata retrieval, rule-based risk detection, SHA-256 hashing, VirusTotal hash lookup, Hybrid Analysis sandbox submission, sandbox verdict retrieval, and SQLite storage
- Updated SQLite database structure
- Clear GitHub README and setup documentation
- Weekly progress evidence
- Final presentation and demo materials
- End-to-end workflow demonstration
- Email header spoofing analysis using SPF, DKIM, DMARC, Return-Path, Reply-To, and DKIM domain alignment indicators
- Prediction Details page showing email body, model result, header spoofing analysis, attachment analysis, and sandbox analysis results

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
| Attachment Analysis | Python hashing libraries, rule-based metadata checks, SHA-256 hashing, VirusTotal hash lookup, Hybrid Analysis / Falcon Sandbox API |
| Version Control | Git, GitHub |
| Development Environment | VS Code |
| Email Header Analysis | Microsoft Graph internet message headers, SPF, DKIM, DMARC, Return-Path, Reply-To, DKIM domain alignment |

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
- View rule-based attachment risk level and risk reason
- View SHA-256 hash values for supported Outlook file attachments
- View VirusTotal hash reputation results when available
- View email header spoofing analysis results on the Prediction Details page
- View attachment scan results on the Prediction Details page
- View Hybrid Analysis sandbox verdict and behavior summary for submitted attachments
- Show/hide password option on login and registration pages

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
- Email header spoofing analysis using authentication and domain-alignment indicators
- Sandbox-based behavioral analysis for suspicious attachments
- Temporary attachment file handling for sandbox upload, with deletion after submission
- Password policy enforcement requiring minimum length, uppercase letter, lowercase letter, number, and special character
- Password visibility toggle on login and registration forms for better user usability

### Email Integration, Header Analysis, and Attachment Analysis Features

- Microsoft OAuth-based Outlook connection
- Microsoft Graph API email retrieval
- Outlook email display inside the Flask application
- Outlook internet message header retrieval through Microsoft Graph API
- Email header spoofing analysis using SPF, DKIM, DMARC, Return-Path, Reply-To, and DKIM domain alignment
- Header risk score, risk level, and explanation display on the Prediction Details page
- Attachment metadata retrieval through Microsoft Graph API
- Attachment filename, MIME type, size, attachment type, and inline status display
- Rule-based attachment risk detection using extension, MIME type, and file size
- SHA-256 hash calculation for supported file attachments
- VirusTotal hash reputation lookup using SHA-256 values
- Hybrid Analysis / Falcon Sandbox API integration for suspicious attachments
- Sandbox verdict and behavior summary retrieval
- SQLite storage of attachment scan results in the `attachment_scans` table
- SQLite storage of email header scan results in the `header_scans` table
- SQLite storage of sandbox results in the `sandbox_scans` table
- Prediction Details page display for header analysis, attachment analysis, and sandbox analysis results
- Safe attachment handling approach that does not manually open or execute files

---

## Repository Structure

```text
Cyber-Security-Capstone-Project/
│
├── app.py
├── database.py
├── graph_attachments.py
├── attachment_analyzer.py
├── sandbox_api.py
├── header_analyzer.py
├── virustotal_checker.py
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

> Note: `app.db` is used locally for users, prediction history, blocked account status, admin activity logs, and attachment scan result testing. It is ignored by Git and should not be pushed to GitHub.

> Note: `.env` stores local API credentials such as Microsoft Graph, VirusTotal, and Hybrid Analysis API keys. It must not be pushed to GitHub.

