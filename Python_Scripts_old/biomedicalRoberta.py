from pathlib import Path
from helper import *
from testingBert import *
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import sys
from transformers import *
from tokenizers import *

all_geo_file_path = sys.argv[1]
queries = sys.argv[2].split(",")
other_multiplication_rate_options = [int(x) for x in sys.argv[3].split(",")]
model = AutoModel.from_pretrained("allenai/biomed_roberta_base")
tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")

def find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name, averaging_method = "word_vector", word_method = 'cat'):
    results_dir_path = f"/Results/{query}/{model_name}/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
    name = averaging_method + "_" + word_method
    if os.path.exists(f'{results_dir_path}/{name}_similarity.tsv'):
        return()
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)
    training_vector_list = []
    for training_series_id in get_series_identifiers(query, "training_series"):
        keywords = get_keywords('Baseline', "full_text", training_series_id)
        if keywords == "":
            continue
        if averaging_method == "word_vector":
            training_vector_list.append(get_roberta_word_embedding(keywords, model, tokenizer, word_method))
        else:            
            training_vector_list.append(get_rob_sentence_embedding(keywords, model, tokenizer, word_method))
    average_training_vector = sum(training_vector_list) / len(training_vector_list)
    print(average_training_vector)

    list_of_ids = []
    for series in get_series_identifiers(f"{query}/other_series", other_multiplication_rate):
        if get_keywords(keyword_extractor_name, num_keywords, series) != "":
            list_of_ids.append(series)
    for series in get_series_identifiers(query, "testing_series"):
        if get_keywords(keyword_extractor_name, num_keywords, series) != "":
            list_of_ids.append(series)
    cos_sim_and_series_id_list = []
    #Finding vectors for each series to compare to training
    for testing_and_other_series_id in list_of_ids:
        keywords = get_keywords(keyword_extractor_name, num_keywords, testing_and_other_series_id)

        if averaging_method == "word_vector":
            testing_and_other_vector = get_roberta_word_embedding(keywords, model, tokenizer, word_method)
        else:
            testing_and_other_vector = get_rob_sentence_embedding(keywords, model, tokenizer, word_method)
        cos_sim = dot(average_training_vector[0], testing_and_other_vector[0])/(norm(average_training_vector[0])*norm(testing_and_other_vector[0]))
        cos_sim_and_series_id_list.append([cos_sim, testing_and_other_series_id])
        
    cos_sim_and_series_id_list.sort()
    cos_sim_and_series_id_list.reverse()
    with open(f'{results_dir_path}/{name}_similarity.tsv', 'w+') as out_file:
        print_time_stamp(f"Processing {results_dir_path}")
        out_file.write("Series ID\tSimilarity Score\tTest or Other Group\n")
        for series in cos_sim_and_series_id_list:
            test_or_other= ""
            if series[1] in get_series_identifiers(query, "testing_series"):
                test_or_other = "Test"
            else:
                test_or_other = "Other"
            out_file.write(f"{series[1]}\t{series[0]}\t{test_or_other}\n")
    return()
model_name = "BiomedRoberta"
num_keywords = 'full_text'
averaging_method = "word_vector"
keyword_extractor_name = 'Baseline'
for other_multiplication_rate in other_multiplication_rate_options:
    for word_method in ['cat', "sum"]:
        for averaging_method in ["word_vector", "sentence_vector"]:
            for query in queries: 
                print(f"I am running query: {query}, imbalance: {other_multiplication_rate}, averaging method: {averaging_method}, word method: {word_method}")
                find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name, averaging_method, word_method)