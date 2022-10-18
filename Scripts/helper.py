from bs4 import BeautifulSoup
from pathlib import Path
import datetime
import json
import os
import pke
import re
import zipfile
import nltk
from nltk.corpus import stopwords


all_geo_dict = {}
with open("/Data/AllGEO.json") as cache_file:
    all_geo_dict = json.loads(cache_file.read())

def print_time_stamp(message):
    print(f"{message} - {datetime.datetime.now()}")

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http\S+', '', text)
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    #text = text.replace('x', '')
    text = text.replace(',', ' ')
    text = re.sub(r'\n', r' ', text)
    text = re.sub(r'[n|N]o\.', r'number', text)
    text = re.sub(r' [0-9]+ ', r' ', text)
    text = re.sub(r' +', r' ', text)
    text = re.sub("  ", " ", text)
    #Double check with Professor piccolo TODO: do we want to remove more common words? Or do that just when making vectors?
    return text

def get_series_identifiers(query, file_name):
    file_path = f'/Data/{query}/{file_name}'

    with open(file_path, 'r+') as in_file:
        series_list = json.loads(in_file.read())
        return series_list 

def get_keyword_extractors():
    return {
        "KPMiner": pke.unsupervised.KPMiner(),
        "MultipartiteRank": pke.unsupervised.MultipartiteRank(),
        "PositionRank": pke.unsupervised.PositionRank(),
        "SingleRank": pke.unsupervised.SingleRank(),
        "TextRank": pke.unsupervised.TextRank(),
        "TfIdf": pke.unsupervised.TfIdf(),
        "TopicalPageRank": pke.unsupervised.TopicalPageRank(),
        "TopicRank": pke.unsupervised.TopicRank(),
        "YAKE": pke.unsupervised.YAKE()
    }

def get_list_extractors():
    extractor_list = list(get_keyword_extractors())
    extractor_list.append("Baseline")
    return(extractor_list)

def extract_keywords_baseline(text, num_keywords):
    keywords = text.split(" ")[:num_keywords]
    keywords = [k for k in keywords if len(k) > 2]
    unique_keywords = []

    for keyword in keywords:
        if keyword not in unique_keywords:
            unique_keywords.append(keyword)

            if len(unique_keywords) == num_keywords:
                break

    return unique_keywords

def remove_stop_words(text):
    text = text.split(" ")
    stops = set(stopwords.words('english'))
    new_text_list = []
    for word in text:
        if word not in stops:
            new_text_list.append(word)
    new_text = " ".join(new_text_list)
    return(new_text)

def get_keywords(keyword_extractor, num_keywords, series):
    if num_keywords == "full_text":
        with open(f"/Data/AllGEO.json", "r") as all_file:
            all_dict = json.loads(all_file.read())
            text = all_dict[series]
            new_text = remove_stop_words(text)
            return(new_text)
    if not os.path.exists(f"/Data/KeywordWeights/{series}"):
        extract_keywords(series, 32)
    with open(f"/Data/KeywordWeights/{series}", "r") as cache_file:
        keyword_dict = json.loads(cache_file.read())
        keywords = keyword_dict[keyword_extractor]
        
        if keywords == [] or keywords == [[]]:
            return("")
        elif num_keywords < len(keywords):
            keywords=keywords[:num_keywords]
        
        keyword_list = []
        for keyword_and_weight in keywords:
            keyword_list.append(keyword_and_weight[0])

        keyword_text = " ".join(keyword_list)
        return(keyword_text)

def get_model_types():
    return ["fasttext__cbow", "fasttext__skipgram", "en_core_web_lg", "en_core_sci_lg", "all-roberta-large-v1", "sentence-t5-xxl", "all-mpnet-base-v2", "dmis-lab/biobert-large-cased-v1.1-squad", "bert-base-uncased", "allenai/scibert_scivocab_uncased", "gpt2", "bioWordVec", "pretrained_fasttext_wiki", "pretrained_fasttext_wiki_subword", "pretrained_fasttext_crawl", "pretrained_fasttext_crawl_subword"]

def extract_keywords(geo_series_id, max_num_keywords=32):  
    # Check whether the specified path exists or not
    path = f"/Data/KeywordWeights/{geo_series_id}"
    print("In the function!")
    if not os.path.exists(path):
        print(f"Making {geo_series_id}")
        title_and_abstract = all_geo_dict[geo_series_id]
        extractor_dict = {}
        #Extracting Baseline
        unique_words = extract_keywords_baseline(title_and_abstract, max_num_keywords)
        word_list = []
        for i, word in enumerate(unique_words):
            word = [word, (max_num_keywords - i) / max_num_keywords]
            word_list.append(word)
        extractor_dict["Baseline"] = word_list
    
        #Using Keyword extraction methods
        for extractor_name, extraction in get_keyword_extractors().items():
            try:
            #use try: (code) except: (more code or word pass)
                extraction.load_document(input=title_and_abstract, language='en')
                extraction.candidate_selection()
                extraction.candidate_weighting()

                keywords = extraction.get_n_best(n=max_num_keywords)
                     
                extractor_dict[extractor_name] = keywords
                print("Got some keywords!")
            except:
                print("no keywords")
                extractor_dict[extractor_name] = []
    
        with open(f"/Data/KeywordWeights/{geo_series_id}", "w") as geo_file:
            geo_file.write(json.dumps(extractor_dict))

    return