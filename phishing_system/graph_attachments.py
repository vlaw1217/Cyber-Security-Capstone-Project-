import requests


def get_email_attachments(access_token, message_id):
    """
    Retrieve attachment metadata for one Outlook email using Microsoft Graph API.

    Parameters:
        access_token (str): Microsoft Graph access token for the signed-in user.
        message_id (str): Microsoft Graph message ID of the selected email.

    Returns:
        list: A list of attachment metadata dictionaries.
    """

    # Microsoft Graph endpoint for listing attachments of a specific email message
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/attachments"

    # Authorization header required by Microsoft Graph API
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    try:
        # Send GET request to Microsoft Graph
        response = requests.get(url, headers=headers)

        # If request fails, print the error and return empty list
        if response.status_code != 200:
            print("Failed to retrieve attachments")
            print("Status code:", response.status_code)
            print("Response:", response.text)
            return []

        # Convert response to JSON
        data = response.json()

        # Microsoft Graph returns attachments inside the "value" field
        attachments = data.get("value", [])

        # Store cleaned attachment metadata here
        attachment_metadata = []

        for attachment in attachments:
            attachment_info = {
                "attachment_id": attachment.get("id"),
                "name": attachment.get("name"),
                "content_type": attachment.get("contentType"),
                "size": attachment.get("size"),
                "attachment_type": attachment.get("@odata.type"),
                "is_inline": attachment.get("isInline", False)
            }

            attachment_metadata.append(attachment_info)

        return attachment_metadata

    except Exception as e:
        print("Error while retrieving attachments:", e)
        return []
    
def download_attachment_bytes(access_token, message_id, attachment_id):
    """
    Download raw attachment bytes from Microsoft Graph.

    This function is used only for SHA-256 hash calculation.
    It does not open, run, or execute the attachment file.

    Parameters:
        access_token (str): Microsoft Graph access token for the signed-in user.
        message_id (str): Microsoft Graph message ID of the email.
        attachment_id (str): Microsoft Graph attachment ID.

    Returns:
        bytes or None: Raw attachment bytes if successful, otherwise None.
    """

    # Microsoft Graph endpoint for retrieving raw attachment content.
    # The /$value path returns the raw bytes of a file attachment.
    url = (
        f"https://graph.microsoft.com/v1.0/me/messages/"
        f"{message_id}/attachments/{attachment_id}/$value"
    )

    # Authorization header required by Microsoft Graph API.
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    try:
        # Send request to download the attachment content.
        response = requests.get(url, headers=headers)

        # If download fails, print error information and return None.
        if response.status_code != 200:
            print("Failed to download attachment content")
            print("Status code:", response.status_code)
            return None

        # Return raw bytes only. Do not save or execute the file.
        return response.content

    except Exception as e:
        print("Error while downloading attachment content:", e)
        return None