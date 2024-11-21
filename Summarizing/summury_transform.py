'''
@Author: Samadhan Thube
@Date: 20/11/2024
@Last Modified by: Samadhan Thube
@Last Modified time: 21/11/2024
@Title: Summarize and Translate Emails to Hindi using Google Generative AI
'''

import os
import csv
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def initialize_model(model_name, config):
    """
    Initialize and configure the generative AI model.
    
    Parameters:
    model_name (str): Name of the generative AI model.
    config (dict): Configuration for the AI model.
    
    Returns:
    object: Configured GenerativeModel instance.
    """
    return genai.GenerativeModel(model_name=model_name, generation_config=config)


def summarize_email(body, model):
    """
    Generate a summary for the given email body using Google Generative AI.
    
    Parameters:
    body (str): The content of the email body.
    model (object): Configured generative AI model instance.
    
    Returns:
    str: A brief summary of the email content.
    """
    chat_session = model.start_chat(history=[])
    prompt = f"Summarize the following email in short:\n\n{body}\n"
    response = chat_session.send_message(prompt)
    return response.text.strip()


def translate_to_hindi(text, model):
    """
    Translate the given text to Hindi using Google Generative AI.
    
    Parameters:
    text (str): The text to be translated.
    model (object): Configured generative AI model instance.
    
    Returns:
    str: The translated text in Hindi.
    """
    chat_session = model.start_chat(history=[])
    prompt = f"Translate the following text to Hindi:\n\n{text}\n"
    response = chat_session.send_message(prompt)
    return response.text.strip()


def read_emails(file_path):
    """
    Read emails from a file and parse them into structured format.
    
    Parameters:
    file_path (str): Path to the input file containing email data.
    
    Returns:
    list: List of email data in structured format.
    """
    with open(file_path, 'r') as file:
        email_content = file.read().strip().split('END')

    emails = []
    for email in email_content:
        lines = email.strip().split('\n')
        if len(lines) < 4:
            continue  # Skip incomplete emails

        from_line = lines[0].replace("From: ", "").strip()
        to_line = lines[1].replace("To: ", "").strip()
        subject_line = lines[2].replace("Subject: ", "").strip()
        body = "\n".join(lines[4:]).replace("Body:\n", "").strip()

        emails.append({
            "from": from_line,
            "to": to_line,
            "subject": subject_line,
            "body": body
        })
    return emails


def save_to_csv(data, file_path, fieldnames):
    """
    Save processed data to a CSV file.
    
    Parameters:
    data (list): List of dictionaries containing processed email data.
    file_path (str): Path to save the CSV file.
    fieldnames (list): List of column names for the CSV file.
    """
    with open(file_path, mode='w', encoding='utf-8', errors='replace', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def process_emails(input_path, output_path):
    """
    Process emails: summarize and translate the content.
    
    Parameters:
    input_path (str): Path to the input file containing emails.
    output_path (str): Path to save the output CSV file.
    """
    # Model configuration
    summary_config = {
        "temperature": 0.7,
        "max_output_tokens": 512,
        "response_mime_type": "text/plain",
    }
    translation_config = {
        "temperature": 0.5,
        "max_output_tokens": 512,
        "response_mime_type": "text/plain",
    }

    # Initialize models
    summary_model = initialize_model("gemini-1.5-flash", summary_config)
    translation_model = initialize_model("gemini-1.5-flash", translation_config)

    # Read and process emails
    emails = read_emails(input_path)
    processed_emails = []
    for email in emails:
        summary = summarize_email(email["body"], summary_model)
        translated_summary = translate_to_hindi(summary, translation_model)
        processed_emails.append({
            "From": email["from"],
            "To": email["to"],
            "Subject": email["subject"],
            "Summary": summary,
            "Translated Summary": translated_summary
        })

    # Save to CSV
    save_to_csv(processed_emails, output_path, ["From", "To", "Subject", "Summary", "Translated Summary"])


def main():
    input_file_path = 'D:\\GenAi\\summerization\\sample_emails.txt'
    output_file_path = 'D:\\GenAi\\summerization\\emails_summary.csv'
    process_emails(input_file_path, output_file_path)
    print(f"Emails have been processed and saved to {output_file_path}")


if __name__ == '__main__':
    main()
