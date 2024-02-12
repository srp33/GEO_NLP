from bs4 import BeautifulSoup
import datetime
import re

def print_time_stamp(message):
    print(f"{message} - {datetime.datetime.now()}")

def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ") # Remove newline characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'\W', ' ', str(text)) # Remove non-alphanumeric characters
    text = remove_html_tags(text)
    text = re.sub(r' +', r' ', text) # Remove consecutive spaces
    text = text.strip()

    return text

def remove_html_tags(text):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")

    # Extract text without HTML tags
    clean_text = soup.get_text(separator=" ")

    return clean_text
