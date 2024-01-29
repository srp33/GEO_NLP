from bs4 import BeautifulSoup
import datetime
import json
import os
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

def get_huggingface_list():
    return(['dmis-lab/biobert-large-cased-v1.1-squad', 'bert-base-uncased', "allenai/scibert_scivocab_uncased", "all-roberta-large-v1", "sentence-t5-xxl", "all-mpnet-base-v2"])

def get_model_types():
    return ["fasttext__cbow", "fasttext__skipgram", "en_core_web_lg", "en_core_sci_lg", "all-roberta-large-v1", "sentence-t5-xxl", "all-mpnet-base-v2", "dmis-lab/biobert-large-cased-v1.1-squad", "bert-base-uncased", "allenai/scibert_scivocab_uncased", "gpt2", "bioWordVec", "pretrained_fasttext_wiki", "pretrained_fasttext_wiki_subword", "pretrained_fasttext_crawl", "pretrained_fasttext_crawl_subword", 'BiomedRoberta', 'GEOBert']
