import glob
import gzip
from helper import *
from sklearn.metrics import precision_recall_curve, auc
import sys
import os

similarities_dir_path = sys.argv[1]
metrics_dir_path = sys.argv[2]

metrics_file_path = f"{metrics_dir_path}/Results.tsv.gz"

with gzip.open(metrics_file_path, "w") as metrics_file:
    metrics_file.write((f"Query\tMethod\tMultiplication_Rate\tMetric\tValue\n").encode())

    for similarities_file_path in glob.glob(f"{similarities_dir_path}/*/*/*"):
        print(f"Calculating metrics for {similarities_file_path}")

        file_path_items = similarities_file_path.split("/")
        query = file_path_items[1]
        method = file_path_items[2]
        multiplication_rate = file_path_items[3].replace("rest_of_gemma_", "")

        groups_scores = []

        with open(similarities_file_path) as similarities_file:
            similarities_file.readline()

            for line in similarities_file:
                line_items = line.rstrip("\n").split("\t")
                #series = line_items[0]
                group = line_items[1]
                score = float(line_items[2])

                groups_scores.append([group, score])

        groups_scores = sorted(groups_scores, key=lambda x: x[1])
        groups = [x[0] for x in groups_scores]
        groups = [1 if group == "Testing" else 0 for group in groups]
        scores = [x[1] for x in groups_scores]

        # Calculate the area under the precision-recall curve
        precision, recall, _ = precision_recall_curve(groups, scores)
        auc_prc = auc(recall, precision)

        metrics_file.write((f"{query}\t{method}\t{multiplication_rate}\tAUC_PRC\t{auc_prc}\n").encode())

        # Calculate proportions in top N
        groups_scores = sorted(groups_scores, key=lambda x: x[1], reverse=True)
        num_testing = sum(groups)

        for n in [20, 50, 200, 500, 1000]:
            count = 0
            for group_score in groups_scores[:n]:
                if group_score[0] == "Testing":
                    count += 1

            if num_testing > n:
                recall = "NA"
            else:
                recall = f"{round(count / num_testing, 2)}"

            metrics_file.write((f"{query}\t{method}\t{multiplication_rate}\tRecall_Top_{n}\t{recall}\n").encode())

print(f"Saved metrics to {metrics_file_path}")

sys.exit(0)

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
vector = "word_vector"
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
