'''
    @Author: Samadhan Thube
    @Date: 21/11/2024
    @Last Modified by: Samadhan Thube
    @Last Modified time: 21/11/2024
    @Title: Process customer reviews, analyze sentiment, and save output to CSV
'''

import csv
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def analyze_sentiment_and_generate_message(body):
    """
    Description:
    Analyze the sentiment of a review and generate an appropriate response message.

    Parameters:
    body (str): The content of the review.

    Returns:
    tuple: Sentiment ('positive' or 'negative') and response message.
    """
    generation_config = {
        "temperature": 1,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(history=[])
    prompt = f"Analyze the sentiment of the following review. Respond with only 'positive' or 'negative':\n\n{body}\n"
    sentiment_response = chat_session.send_message(prompt).text.strip()

    if "positive" in sentiment_response.lower():
        message_prompt = f"Generate a short thank-you message for this positive review:\n\n{body}\n"
    else:
        message_prompt = f"Generate a short apology message for this negative review:\n\n{body}\n"

    response_message = chat_session.send_message(message_prompt).text.strip()
    return sentiment_response, response_message

def extract_item_and_company(body):
    """
    Description:
    Extract the purchased item and company name from the review.

    Parameters:
    body (str): The content of the review.

    Returns:
    tuple: Extracted item and company name.
    """
    generation_config = {
        "temperature": 0.7,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(history=[])
    item_prompt = f"Identify the purchased item in the following review. Respond with only the item name:\n\n{body}\n"
    item_response = chat_session.send_message(item_prompt).text.strip() or "unknown"

    company_prompt = f"Identify the company name in the following review. Respond with only the company name:\n\n{body}\n"
    company_response = chat_session.send_message(company_prompt).text.strip() or "unknown"

    return item_response, company_response

def read_reviews(file_path):
    """
    Description:
    Read reviews from the input file.

    Parameters:
    file_path (str): Path to the input file.

    Returns:
    list: List of review content strings.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip().split('END')

def process_reviews(review_content):
    """
    Description:
    Process each review to extract details and analyze sentiment.

    Parameters:
    review_content (list): List of review content strings.

    Returns:
    list: List of dictionaries containing review details.
    """
    reviews = []
    for review in review_content:
        body = review.strip()
        item, company = extract_item_and_company(body)
        sentiment, response_message = analyze_sentiment_and_generate_message(body)
        reviews.append({
            "item": item,
            "company": company,
            "sentiment": sentiment,
            "response_message": response_message
        })
        time.sleep(20)
    return reviews

def write_reviews_to_csv(reviews, output_file_path):
    """
    Description:
    Write processed reviews to a CSV file.

    Parameters:
    reviews (list): List of dictionaries containing review details.
    output_file_path (str): Path to the output CSV file.
    """
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Item', 'Company', 'Sentiment', 'Response Message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({
                'Item': review['item'],
                'Company': review['company'],
                'Sentiment': review['sentiment'],
                'Response Message': review['response_message']
            })

def main():
    input_file_path = 'D:\\GenAi\\inferring\\customer_reviews.txt'
    output_file_path = 'D:\\GenAi\\inferring\\review_analysis.csv'

    review_content = read_reviews(input_file_path)
    reviews = process_reviews(review_content)
    write_reviews_to_csv(reviews, output_file_path)
    print(f"Reviews have been processed and saved to {output_file_path}")

if __name__ == '__main__':
    main()
