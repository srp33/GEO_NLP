from helper import *
import sys

model_types = get_model_types()
#model_types.append("Spacy")
#model_types.append("SciSpacy")
num_keywords = [int(x) for x in sys.argv[1].split(",")]
num_keywords.append("full_text")

keyword_extraction_methods = get_list_extractors()
multiplication_rates = [int(x) for x in sys.argv[2].split(",")][0:2]
queries = sys.argv[3].split(",")

#best_combo = "placeholder" #open('/home/jwengler/NLP_Paper/analysis/Data/Results/best_combo.txt', 'w+')
#for model in models:
#   for num_keywords in num_keywords:
#      generate_results(result_path, model, best_combo, num_keywords)



#creating a tsv file that contains all combination results.
with open("Results/results.tsv", 'w') as results_file:
    results_file.write("Extraction Method\tNumber of Keywords\tMultiplication Rate\tModel Type\tQuery\tAccuracy\n")
    #for query in queries:
    for query in ['q1', 'q2', 'q3', 'q4', 'q5']:
        for model in model_types:
            for method in ["KPMiner", "Baseline"]:
            #for method in keyword_extraction_methods:
                for multiplication_rate in multiplication_rates:
                    #for keywords in num_keywords:
                    for keywords in [2, 4, 'full_text']:
                        with open(f"/Results/{query}/{model}/{keywords}/{method}/{multiplication_rate}/similarity.tsv") as data_file:
                            num_to_check_for = len(get_series_identifiers(query, "testing_series"))
                            accuracy = 0
                            for gse in range(num_to_check_for):
                                line = data_file.readline()
                                line = line.rstrip("\n")
                                line = line.split("\t")
                                if line[2] == "Test":
                                    accuracy += 1
                            accuracy = accuracy / num_to_check_for
                            results_file.write(f"{method}\t{keywords}\t{multiplication_rate}\t{model}\t{query}\t{accuracy}\n")