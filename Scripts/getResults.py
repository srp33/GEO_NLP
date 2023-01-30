from helper import *
import sys
import os

#3 functions below. 
#The first creates a file for retrieval results
#The second creates a file only for 'all-star' multiplication rate (all of STARGEO). 
#The last script is for keyword extraction specifically. 

num_keywords = [int(x) for x in sys.argv[1].split(",")]
model_types = get_model_types()

keyword_extraction_methods = get_list_extractors()
multiplication_rates = [int(x) for x in sys.argv[2].split(",")]
queries = sys.argv[3].split(",")

bioPretrained = ["en_core_sci_lg","dmis-lab/biobert-large-cased-v1.1-squad","bioWordVec", "allenai/scibert_scivocab_uncased", "BiomedRoberta"]
customGeoTrained = ["fasttext__cbow", "fasttext__skipgram", "GEOBert"]
generalPretrained = ["pretrained_fasttext_wiki", "pretrained_fasttext_wiki_subword", "pretrained_fasttext_crawl", "en_core_web_lg", "pretrained_fasttext_crawl_subword", "all-roberta-large-v1", "sentence-t5-xxl","all-mpnet-base-v2", "bert-base-uncased", "gpt2"]

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
            if model in bioPretrained:
                category = "Pretrained-Bio"
            elif model in customGeoTrained:
                category = "Geo-Trained"
            elif model in generalPretrained:
                category = "Pretrained-General"
            else:
                category = "Unsure"
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
            
            results_file.write(f"{model}\t{multiplication_rate}\t{query}\t{vector_average_method}\t{category}\t{accuracy}\n")
    return()

#creating a tsv file that contains all results.
with open("Results/results.tsv", 'w') as results_file:
    results_file.write("Model_Type\tMultiplication_Rate\tQuery\tVector_Method\tCategory\tAccuracy\n")

method = 'Baseline'
keywords = 'full_text'
for query in queries:
     for model in model_types:
          for multiplication_rate in multiplication_rates:
                if model == 'BiomedRoberta':
                    vector_type = "sentence_vector_sum"
                calculate_accuracy(query, model, keywords, method, multiplication_rate, vector_type)                         

################################ ONLY ALL-STAR RESULTS :) ########################################
with open("Results/all_star.tsv", 'w') as write_file:
    write_file.write(f"Model_Type\tQuery\tCategory\tAccuracy\tWorst_Case\n")

def all_star_cal(model, query, category, vector_average_method = "word_vector", multiplication_rate = 'all_star'):
    if not os.path.exists(f"/Results/{query}/{model}/full_text/Baseline/{multiplication_rate}/{vector_average_method}_similarity.tsv"):
        print(f"no bueno: /Results/{query}/{model}/full_text/Baseline/{multiplication_rate}/{vector_average_method}_similarity.tsv")
        return()
    with open("Results/all_star.tsv", 'a') as results_file:
        with open(f"/Results/{query}/{model}/full_text/Baseline/{multiplication_rate}/{vector_average_method}_similarity.tsv") as data_file:
            keywords = 'full_text'
            worst_case = 0
            tmp = data_file.readline()
            list_test_series = []
            method = "Baseline"
            for series in get_series_identifiers(query, "testing_series"):
                if get_keywords(method, keywords, series) != "":
                    list_test_series.append(series)
            num_to_check_for = len(list_test_series)
            countdown = num_to_check_for
            accuracy = 0
            counter = 0
            for gse in range(num_to_check_for):
                line = data_file.readline()
                line = line.rstrip("\n")
                line = line.split("\t")
                counter += 1
                if line[2] == "Test":
                    accuracy += 1
                    countdown -= 1
                    if countdown == 0:
                        worst_case = counter
            accuracy = accuracy / num_to_check_for

            while(countdown != 0):
                line = data_file.readline()
                line = line.rstrip("\n")
                line = line.split("\t")
                counter += 1
                if line[2] == "Test":
                    countdown -= 1
                    if countdown == 0:
                        worst_case = counter
            print(worst_case)                
            results_file.write(f"{model}\t{query}\t{category}\t{accuracy}\t{worst_case}\n")

    return

multiplication_rate = "all_star"
vector= "word_vector"
for query in queries:
    for model in model_types:
        if model in bioPretrained:
            category = "Pretrained-Bio"
        elif model in customGeoTrained:
            category = "Geo-Trained"
        elif model in generalPretrained:
            category = "Pretrained-General"
        else:
            category = "Unsure"
        all_star_cal(model, query, category)


################################ KEYWORD EXTRACTION RESULTS ########################################

# def calc_keyword_accuracy(query, model, keywords, method, multiplication_rate, vector_average_method="sentence_vector"):
#     if not os.path.exists(f"/Results/{query}/{model}/{keywords}/{method}/{multiplication_rate}/{vector_average_method}_similarity.tsv"):
#         print(f"no bueno: /Results/{query}/{model}/{keywords}/{method}/{multiplication_rate}/{vector_average_method}_similarity.tsv")
#         return()
#     with open("Results/keyword_results.tsv", 'a') as results_file:
#         with open(f"/Results/{query}/{model}/{keywords}/{method}/{multiplication_rate}/{vector_average_method}_similarity.tsv") as data_file:
#             tmp = data_file.readline()
#             list_test_series = []
#             for series in get_series_identifiers(query, "testing_series"):
#                 if get_keywords(method, keywords, series) != "":
#                     list_test_series.append(series)
#             num_to_check_for = len(list_test_series)
#             accuracy = 0
#             for gse in range(num_to_check_for):
#                 line = data_file.readline()
#                 line = line.rstrip("\n")
#                 line = line.split("\t")
#                 if line[2] == "Test":
#                     accuracy += 1
#             accuracy = accuracy / num_to_check_for
#             if model in bioPretrained:
#                 category = "Pretrained-Bio"
#             elif model in customGeoTrained:
#                 category = "Geo-Trained"
#             elif model in generalPretrained:
#                 category = "Pretrained-General"
#             else:
#                 category = "Unsure"
#             if model == "dmis-lab/biobert-large-cased-v1.1-squad":
#                 model = "BioBert"
#             elif model == "bert-base-uncased":
#                 model = "Bert"
#             elif model == "allenai/scibert_scivocab_uncased":
#                 model = "SciBert"
#             elif model == "en_core_sci_lg":
#                 model = "SciSpacy"
#             elif model == "en_core_web_lg":
#                 model = "Spacy"
#             elif model == "all-roberta-large-v1":
#                 model = "Roberta"
#             elif model == "sentence-t5-xxl":
#                 model = "T5"
#                 #T5 stands for Text-To-Text Transfer Transformer
#             elif model == "all-mpnet-base-v2":
#                 model = "MPNet" #model made by microsoft
#             print("Made it this far!")
#             results_file.write(f"{model}\t{method}\t{keywords}\t{multiplication_rate}\t{query}\t{vector_average_method}\t{category}\t{accuracy}\n")
#     return()

############################################################################

# print(queries)

# with open("Results/keyword_results.tsv", 'w') as results_file:
#     results_file.write("Model_Type\tKeyword_Extractor\tNum_Keywords\tMultiplication_Rate\tQuery\tVector_Method\tCategory\tAccuracy\n")
# for multiplication_rate in multiplication_rates:
#      for model in model_types:
#          for method in keyword_extraction_methods:
#              for numKeywords in num_keywords:
#                  for query in queries:
#                      print(f"Making a {model} with {numKeywords} keywords of {method}")
#                      calc_keyword_accuracy(query, model, numKeywords, method, multiplication_rate, vector_type)

