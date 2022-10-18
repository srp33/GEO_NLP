from helper import *
import sys
import os

num_keywords = [int(x) for x in sys.argv[1].split(",")]
model_types = get_model_types()
model_types.append("GEOBert")
num_keywords.append("full_text")

keyword_extraction_methods = get_list_extractors()
multiplication_rates = [int(x) for x in sys.argv[2].split(",")]
queries = sys.argv[3].split(",")

def calculate_accuracy(query, model, keywords, method, multiplication_rate, vector_average_method="sentence_vector"):
    if not os.path.exists(f"/Results/{query}/{model}/{keywords}/{method}/{multiplication_rate}/{vector_average_method}_similarity.tsv"):
        return()
    with open("Results/results.tsv", 'a') as results_file:
        with open(f"/Results/{query}/{model}/{keywords}/{method}/{multiplication_rate}/{vector_average_method}_similarity.tsv") as data_file:
            tmp = data_file.readline()
            list_test_series = []
            for series in get_series_identifiers(query, "testing_series"):
                if get_keywords(method, keywords, series) != "":
                    list_test_series.append(series)
            num_to_check_for = len(list_test_series)
            accuracy = 0
            for gse in range(num_to_check_for):
                line = data_file.readline()
                line = line.rstrip("\n")
                line = line.split("\t")
                if line[2] == "Test":
                    accuracy += 1
            accuracy = accuracy / num_to_check_for

            if model == "dmis-lab/biobert-large-cased-v1.1-squad":
                model = "BioBert"
            elif model == "bert-base-uncased":
                model = "Bert"
            elif model == "allenai/scibert_scivocab_uncased":
                model = "SciBert"
            elif model == "en_core_sci_lg":
                model = "SciSpacy"
            elif model == "en_core_web_lg":
                model = "Spacy"
            elif model == "all-roberta-large-v1":
                model = "Roberta"
            elif model == "sentence-t5-xxl":
                model = "T5"
                #T5 stands for Text-To-Text Transfer Transformer
            elif model == "all-mpnet-base-v2":
                model = "MPNet" #model made by microsoft
            results_file.write(f"{model}\t{multiplication_rate}\t{query}\t{vector_average_method}\t{accuracy}\n")
    return()
 
#creating a tsv file that contains all combination results.
with open("Results/results.tsv", 'w') as results_file:
    results_file.write("Model_Type\tMultiplication_Rate\tQuery\tVector_Method\tAccuracy\n")

print(queries)

for query in queries:
    for model in model_types:
        for method in ["Baseline"]:
        #for method in keyword_extraction_methods:
            for multiplication_rate in multiplication_rates:
                for keywords in ["full_text"]:
                    for vector_type in ["sentence_vector", "word_vector", "word_vector_cat", "word_vector_sum"]:
                        calculate_accuracy(query, model, keywords, method, multiplication_rate, vector_type)                         