from bs4 import BeautifulSoup
from pathlib import Path
import datetime
import json
import math
import os
import pke
import random
import re
import requests
#from scipy.spatial.distance import cosine
#from gensim.models.fasttext import FastText as FT_gensim
#from tester import tester
#import numpy as np

#def load_embedding(path):
#    return KeyedVectors.load_word2vec_format(path, binary = True)

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

def get_keywords(keyword_extractor, num_keywords, series):
    with open(f"/Data/KeywordWeights/{series}", "r") as cache_file:
        keyword_dict = json.loads(cache_file.read())
        keywords = keyword_dict[keyword_extractor]
        
        if keywords == [] or keywords == [[]]:
            return("")
        elif int(num_keywords) < len(keywords):
            keywords=keywords[:num_keywords]
        
        keyword_list = []
        for keyword_and_weight in keywords:
            keyword_list.append(keyword_and_weight[0])

        keyword_text = " ".join(keyword_list)
        return(keyword_text)


def get_model_types():
    return ["fasttext__cbow", "fasttext__skipgram"]

#def get_names_to_query(query_number):
#    file_path = f'/Data/Queries/q{query_number}/names_to_query.txt'
#
#    files = []
#
#    with open(file_path, 'r+') as in_file:
#        for line in in_file:
#            names = line.split(' ')
#            for name in names:
#                files.append(name.upper().strip())
#
#    return (files[0:len(files)-1])

#def get_score(file_path, query):
#    score_to_series = {}
#    with open(file_path, "r+") as in_file:
#        for line in in_file:
#            line_list = line.split('\t')
#            try:
#                score_to_series[line_list[1].strip()] = line_list[0].strip()
#            except:
#                print("Line List : {}".format(line_list))
#                print("Error was found in {}".format(file_path))
#    scores = []
#    to_test = [1,10,100]
#    for top_num in to_test:
#        try:
#            top_results = get_x_top_results(top_num, score_to_series)
#        except:
#            print("\n")
#            print(f"file_path : {file_path}")
#            print(f"QUERY : {query}")
#            print(f"Length of score_to_series : {len(score_to_series)}")
#            print(f"top num : {top_num}")
#        my_tester = tester()
#        my_tester.which_query(query)
#        score = my_tester.return_percent(top_results)
#        scores.append(score)
#    return scores

#def generate_results(result_path, model_name, best_combo, num_keywords):
#    outputFile = open(f"/Data/Results/{num_keywords}_{model_name}Output.txt", 'w+')
#    outputFile.write(f"{model_name} Results for {num_keywords} keywords\n")
#    outputFile.write("MODEL\tQUERY\t#\tSCORE\n")
#
#    for name in result_path:
#        for query in [1,2,3,4,5,6]:
#            path = f"/Data/Results/{model_name}/q{query}/{name}/{num_keywords}"
#            #path = "/Results/{}}/{}".format(model_name,name)
#            top_nums = [1, 10, 100]
#            scores = get_score(path + "/similarity.txt", query)
#            print(f"Got scores for the following path : {path}")
#            for i, score in enumerate(scores):
#                #if (top_nums[i] == 100):
#                    #best_combo.write("{}}\t{}\t{}\t{}\t{}\n".format(model_name, str(name).strip(), str(query).strip(), str(top_nums[i]).strip(), str(score).strip()))
#                strForFile = "{}\t{}\t{}\t{}\n".format(str(name).strip(), str(query).strip(), str(top_nums[i]).strip(), str(score).strip())
#                outputFile.write(strForFile)
#
#def evaluate_geo():
#    path_to_geo_queries = "/Data/GEO_Queries/"
#    path_to_queries = "/Data/Queries/"
#
#    query_list = ["q1_Family+History_Breast+Cancer.txt",  "q2_Liver+Damage_Hepatitis.txt",
#                  "q3_Monozygotic+Twins.txt",  "q4_Kidney_Tumor_Cell+Line.txt",
#                  "q5_Diabetes_Type+1.txt",  "q6_Osteosarcoma.txt"]
#
#    starGEO_datasets = (get_candidate_articles(100000)).keys()
#
#    results_dir_path = f"/Data/GEO_Queries/geo_results.txt"
#    with open(results_dir_path, 'w+') as out_file:
#        for top_n in [1,10,100]:
#            for path in query_list:
#                num_og = 0
#                geo_results = []
#                query_results = []
#                with open(path_to_geo_queries + path, 'r') as geo_file:
#                    superseries = False
#                    for line in geo_file:
#                        if line.startswith("(Submitter supplied) This SuperSeries is composed of the SubSeries listed below"):
#                            superseries = True
#                        if line.startswith("Series") and superseries:
#                            superseries = False
#                        if line.startswith("Series") and not superseries:
#                            num_og += 1
#                            split_sent = line.split()
#                            if split_sent[2] in starGEO_datasets:
#                                geo_results.append(split_sent[2])
#
#                with open(path_to_queries + f"q{path[1]}/names.txt", 'r') as query_file:
#                    for line in query_file:
#                        query_results = line.split()
#                num_relevant = 0
#                for series in geo_results[:top_n]:
#                    if series in query_results:
#                        num_relevant = num_relevant + 1
#
#                out_file.write(f"GEO\tq{path[1]}\t{top_n}\t {round(((num_relevant/len(query_results)) * 100),1)}\n")
#    print("Finished GEO evaluation")