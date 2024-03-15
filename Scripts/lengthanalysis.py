import sys
import json
from helper import *

#We wanted to see if our method has a bias in favor of shorter or longer summary sections.
#This script pulls the test series' length in number of words and what position that series was ranked for similarity. 

all_geo_file_path = sys.argv[1]
queries = sys.argv[2].split(",")
other_multiplication_rate_options = [int(x) for x in sys.argv[3].split(",")]
other_multiplication_rate_options.append('all_star')

with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

def lengthAcc(file_path, query, model_name):
    threshold= len(get_series_identifiers(query, "testing_series"))
    with open(file_path, "r") as result_file:
        header = result_file.readline()
        all = result_file.readlines()
        to_record = []
        counter = 0
        for line in all:
            counter += 1
            #keep track of position
            line = line.rstrip("\n")
            line = line.split("\t")
            if line[2] == 'Test':
                length = len(all_dict[line[0]])
                to_record.append([line[0], length, counter])    
                #record only the test series lengths. 
    
        with open("/Results/length_retrieval.tsv", "a") as write_file:
            for info in to_record:
                write_file.write(f"{info[0]}\t{info[1]}\t{query}\t{info[2]}\t{model_name}\t{threshold}\n")
    return()

with open("/Results/length_retrieval.tsv", "w") as write_file:
    write_file.write(f"Series\tLength\tQuery\tPosition\tModel_Name\tThreshold\n")

averaging_method = "sentence_vector"
num_keywords = 'full_text'
keyword_extractor_name= 'Baseline'
for model_name in get_model_types():
    for query in queries:
        for other_multiplication_rate in other_multiplication_rate_options:
            results_dir_path = f"/Results/{query}/{model_name}/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}/{averaging_method}_similarity.tsv"
            lengthAcc(results_dir_path, query, model_name)