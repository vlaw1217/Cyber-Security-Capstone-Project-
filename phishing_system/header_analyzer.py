# ---------------------------------------------------
# Email Header Spoofing Analyzer
# ---------------------------------------------------
# This file analyzes Microsoft Graph email headers to help detect
# sender spoofing and sender-domain mismatch.
#
# It checks:
# - From domain
# - Return-Path domain
# - Reply-To domain
# - SPF result
# - DKIM result
# - DMARC result
# - Microsoft composite authentication result
# - Third-party sending infrastructure indicators

import re
from email.utils import parseaddr


# Convert Microsoft Graph header list into dictionary
# ---------------------------------------------------
def headers_to_dict(internet_headers):
    """
    Convert Microsoft Graph internetMessageHeaders into a dictionary.

    Microsoft Graph returns headers like:
    [
        {"name": "From", "value": "..."},
        {"name": "Authentication-Results", "value": "..."}
    ]

    Some headers, like Received, can appear multiple times.
    Therefore, each key stores a list of values.
    """

    header_dict = {}

    if not internet_headers:
        return header_dict

    for header in internet_headers:
        name = header.get("name", "").lower().strip()
        value = header.get("value", "").strip()

        if not name:
            continue

        if name not in header_dict:
            header_dict[name] = []

        header_dict[name].append(value)

    return header_dict



# Extract domain from email header value
# ---------------------------------------------------
def extract_email_domain(header_value):
    """
    Extract domain from an email header value.

    Example:
    "Sheridan College <support@sheridancollege.ca>"
    becomes:
    "sheridancollege.ca"
    """

    if not header_value:
        return None

    name, email_address = parseaddr(header_value)

    if "@" not in email_address:
        return None

    return email_address.split("@")[-1].lower().strip()



# Extract authentication result
# ---------------------------------------------------
def get_auth_result(auth_headers, keyword):
    """
    Extract SPF, DKIM, DMARC, or compauth result from Authentication-Results.

    Examples:
    spf=pass
    dkim=fail
    dmarc=fail
    compauth=pass
    """

    combined_headers = " ".join(auth_headers).lower()

    match = re.search(rf"\b{keyword}=([a-z0-9_-]+)", combined_headers)

    if match:
        return match.group(1)

    return "unknown"



# Extract DKIM signing domain
# ---------------------------------------------------
def get_dkim_domain(auth_headers):
    """
    Extract DKIM signing domain from Authentication-Results.

    Example:
    header.d=sheridancollege.ca
    """

    combined_headers = " ".join(auth_headers).lower()

    match = re.search(r"\bheader\.d=([a-z0-9.-]+)", combined_headers)

    if match:
        return match.group(1).strip()

    return None



# Domain comparison helper
# ---------------------------------------------------
def domains_match_or_align(domain_one, domain_two):
    """
    Check whether two domains match or appear related.

    Exact match:
    sheridancollege.ca == sheridancollege.ca

    Subdomain match:
    mail.sheridancollege.ca aligns with sheridancollege.ca
    """

    if not domain_one or not domain_two:
        return False

    domain_one = domain_one.lower().strip()
    domain_two = domain_two.lower().strip()

    return (
        domain_one == domain_two
        or domain_one.endswith("." + domain_two)
        or domain_two.endswith("." + domain_one)
    )



# Main header analysis function
# ---------------------------------------------------
def analyze_email_headers(internet_headers):
    """
    Analyze email headers and return a spoofing risk summary.
    """

    header_dict = headers_to_dict(internet_headers)

    risk_score = 0
    warnings = []

    # Get important header values.
    from_header = header_dict.get("from", [""])[0]
    return_path_header = header_dict.get("return-path", [""])[0]
    reply_to_header = header_dict.get("reply-to", [""])[0]

    auth_results = header_dict.get("authentication-results", [])
    received_headers = header_dict.get("received", [])

    # Extract domains.
    from_domain = extract_email_domain(from_header)
    return_path_domain = extract_email_domain(return_path_header)
    reply_to_domain = extract_email_domain(reply_to_header)

    # Extract authentication results.
    spf_result = get_auth_result(auth_results, "spf")
    dkim_result = get_auth_result(auth_results, "dkim")
    dmarc_result = get_auth_result(auth_results, "dmarc")
    compauth_result = get_auth_result(auth_results, "compauth")
    dkim_domain = get_dkim_domain(auth_results)


    # SPF check
    # ---------------------------------------------------
    if spf_result in ["fail", "softfail"]:
        risk_score += 30
        warnings.append(
            f"SPF result is {spf_result}. The sending server may not be authorized for the sender domain."
        )
    elif spf_result in ["neutral", "none"]:
        risk_score += 10
        warnings.append(
            f"SPF result is {spf_result}. The email does not strongly prove the sender is authorized."
        )
    elif spf_result == "unknown":
        risk_score += 5
        warnings.append("SPF result was not found in the message headers.")


    # DKIM check
    # ---------------------------------------------------
    if dkim_result == "fail":
        risk_score += 25
        warnings.append(
            "DKIM failed. The email signature could not be verified."
        )
    elif dkim_result in ["none", "unknown"]:
        risk_score += 10
        warnings.append(
            "DKIM result was missing or unknown."
        )


    # DMARC check
    # ---------------------------------------------------
    if dmarc_result == "fail":
        risk_score += 35
        warnings.append(
            "DMARC failed. The visible From domain did not pass sender authentication alignment."
        )
    elif dmarc_result in ["none", "unknown"]:
        risk_score += 10
        warnings.append(
            "DMARC result was missing or unknown."
        )


    # Microsoft composite authentication check
    # ---------------------------------------------------
    if compauth_result == "fail":
        risk_score += 25
        warnings.append(
            "Microsoft composite authentication failed."
        )
    elif compauth_result in ["softpass", "none"]:
        risk_score += 10
        warnings.append(
            f"Microsoft composite authentication result is {compauth_result}."
        )


    # Return-Path alignment check
    # ---------------------------------------------------
    if from_domain and return_path_domain:
        if not domains_match_or_align(from_domain, return_path_domain):
            risk_score += 15
            warnings.append(
                f"Visible From domain is {from_domain}, but Return-Path domain is {return_path_domain}."
            )


    # Reply-To alignment check
    # ---------------------------------------------------
    if from_domain and reply_to_domain:
        if not domains_match_or_align(from_domain, reply_to_domain):
            risk_score += 20
            warnings.append(
                f"Visible From domain is {from_domain}, but Reply-To domain is {reply_to_domain}."
            )

  
    # DKIM signing domain alignment check
    # ---------------------------------------------------
    if from_domain and dkim_domain:
        if not domains_match_or_align(from_domain, dkim_domain):
            risk_score += 15
            warnings.append(
                f"Visible From domain is {from_domain}, but DKIM signing domain is {dkim_domain}."
            )


    # Third-party infrastructure check
    # ---------------------------------------------------
    received_text = " ".join(received_headers).lower()

    third_party_indicators = [
        "amazonaws.com",
        "amazonses.com",
        "sendgrid.net",
        "mailchimp",
        "constantcontact",
        "hubspotemail",
        "mandrillapp",
        "mailgun",
        "sparkpostmail"
    ]

    found_third_party = []

    for provider in third_party_indicators:
        if provider in received_text:
            found_third_party.append(provider)

    if found_third_party:
        risk_score += 10
        warnings.append(
            "Third-party sending infrastructure detected: "
            + ", ".join(found_third_party)
            + ". This is not automatically malicious, but it should align with authentication results."
        )


    # Final risk level
    # ---------------------------------------------------
    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        risk_level = "High"
    elif risk_score >= 35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if not warnings:
        warnings.append(
            "No major sender authentication or domain alignment issues were detected."
        )

    return {
        "from_domain": from_domain,
        "return_path_domain": return_path_domain,
        "reply_to_domain": reply_to_domain,
        "spf_result": spf_result,
        "dkim_result": dkim_result,
        "dmarc_result": dmarc_result,
        "compauth_result": compauth_result,
        "dkim_domain": dkim_domain,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "warnings": warnings
    }