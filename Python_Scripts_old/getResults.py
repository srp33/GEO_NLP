from helper import *
import sys as argv

model_types = argv[1]
num_keywords = [int(x) for x in argv[2].split(",")]
keyword_extraction_methods = list(argv[3].keys())
multiplication_rates = argv[4]
queries = argv[5]

result_path = ["MultipartiteRank", "PositionRank", "SingleRank", "TfIdf", "TextRank", "TopicalPageRank", "YAKE", "KPMiner", "TopicRank"]
models = ["FTSkipGram", "FTCBOW", "FTWiki", "BioWordVec", "SpaCy", "SciSpaCy"]
num_keywords = [10,20,30]
best_combo = "placeholder" #open('/home/jwengler/NLP_Paper/analysis/Data/Results/best_combo.txt', 'w+')

for model in models:
    for num_keywords in num_keywords:
        generate_results(result_path, model, best_combo, num_keywords)


#creating a tsv file that contains all combination results. keyword extraction method,
#num keywords, multiplication rate, type of model, and query to keep track of accuracy.

with open("results.tsv", 'w') as results_file:
    results_file.write("Extraction Method\tNumber of Keywords\tMultiplication Rate\tModel Type\tQuery\tAccuracy\n")
    for model in model_types:
        for keywords in num_keywords:
            for method in keyword_extraction_method:
                for multiplication_rate in multiplication_rates:
                    for query in queries:
                        #TODO: open the respective file and calculate its accuracy. Save that accuracy as a variable.
                        results_file.write(f"{method}\t{keywords}\t{multiplication_rate}\t{model}\t{query}\t\n")
