from bs4 import BeautifulSoup
import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def print_time_stamp(message):
    print(f"{message} - {datetime.datetime.now()}")

def clean_text(text, convert_to_lower=True, remove_non_alnum=True):
    if convert_to_lower:
        text = text.lower()

    text = text.replace("\n", " ") # Remove newline characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    if remove_non_alnum:
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

def tokenize_and_remove_stop_words(text):
    nltk.download("punkt")
    nltk.download("stopwords")

    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))

    filtered_text = [word for word in words if not word.lower() in stop_words]

    return " ".join(filtered_text)
