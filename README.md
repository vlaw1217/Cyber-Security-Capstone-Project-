# AI-Powered Phishing Detection System

A cybersecurity capstone project focused on developing an AI-powered phishing detection prototype that uses machine learning, email content analysis, attachment risk indicators, and dashboard monitoring to identify potentially malicious emails.

This project is being developed for **INFO49402 Cyber Security Capstone – Spring/Summer 2026** as a continuation of the Phase 1 proof-of-concept completed in **INFO36206**.

---

## Project Overview

The AI-Powered Phishing Detection System is designed to classify emails as phishing or legitimate based on email content and supporting risk indicators. In Phase 1, the project focused on building a proof-of-concept machine learning model using email subject and body text. The system used text preprocessing, TF-IDF feature extraction, supervised machine learning, and a basic Flask web interface for testing phishing predictions.

For Phase 2, the project is being revised and expanded into a more complete cybersecurity prototype. The system will add secure user access, role-based permissions, an administrative dashboard, attempted Microsoft Graph API email integration, and attachment analytics.

The goal is to demonstrate an end-to-end phishing analysis workflow from user login to email scanning, phishing classification, attachment risk checking, database storage, and dashboard result display.

---

## Project Status

**Current status:** Phase 2 revised project planning and system refinement.

The project is currently being updated to align with the INFO49402 requirements. The revised plan includes weekly milestones, updated scope, risk mitigation, team responsibilities, GitHub evidence, and final deliverables.

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

## Phase 2 Planned Enhancements

Phase 2 will improve the system by adding the following components:

### 1. Phase 1 System Review and Refinement

The existing Phase 1 system will be reviewed, tested, and cleaned before new features are added.

Planned work:

- Test the existing trained model and vectorizer
- Verify the Flask prediction workflow
- Review database usage and prediction history
- Clean and organize the repository
- Update documentation and setup instructions

### 2. Authentication and Role-Based Access Control

The system will include secure access control for users and administrators.

Planned work:

- User registration
- Login and logout
- Password hashing
- Session management
- User and administrator roles
- Protected routes
- Admin-only access restrictions

### 3. Administration Dashboard

An administrative dashboard will provide visibility into system activity and phishing detection results.

Planned work:

- Display prediction history
- Display email scan results
- Show attachment scan results
- Show user activity where applicable
- Provide basic system status or summary statistics
- Allow administrators to review suspicious or phishing classifications

### 4. Microsoft Graph API Email Integration

The system will attempt to integrate with Microsoft Graph API to retrieve Outlook email data for automated phishing analysis.

Planned work:

- Configure Microsoft Entra ID application registration
- Implement OAuth-based authentication where possible
- Retrieve email subject, sender, body, and metadata
- Send retrieved email content to the phishing detection model
- Display classification results in the dashboard

If full API access is blocked by permissions, configuration, or account limitations, the system will use a controlled sample email ingestion workflow that follows the same processing structure.

### 5. Attachment Analytics

The attachment analytics component will not execute, open, or run attachment files. It will analyze attachment metadata, file extensions, file size, hash values, and optional reputation results from VirusTotal if API access is available.

Planned work:

- Detect whether an email includes attachments
- Extract attachment metadata such as filename, file type, and size
- Generate file hashes
- Check suspicious file extensions or patterns
- Integrate VirusTotal or another threat intelligence service if API access is available
- Display attachment risk results in the dashboard
- Store attachment scan metadata in the database

If external API access is limited, the system will still complete local attachment analysis using metadata, hash generation, and suspicious extension checks.

---

## Expected Final Deliverables

By the end of Phase 2, the project is expected to include:

- Working phishing detection web application
- Reused or refined machine learning model and TF-IDF vectorizer
- Secure authentication system
- Role-based access control
- Functional administrative dashboard
- Attempted Microsoft Graph API integration or fallback email ingestion workflow
- Attachment metadata and hash-based risk analysis
- Updated database structure
- Clear GitHub README and documentation
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

## Repository Structure

```text
Cyber-Security-Capstone-Project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── phishing_model_v1.pkl
│   └── tfidf_vectorizer_v1.pkl
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   └── admin_dashboard.html
│
├── static/
│   ├── css/
│   └── images/
│
├── database/
│   └── database_schema_notes.md
│
├── docs/
│   ├── project_plan.md
│   ├── weekly_updates/
│   ├── diagrams/
│   └── testing_notes.md
│
└── notebooks/
    └── model_training.ipynb
