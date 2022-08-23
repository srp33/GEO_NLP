from helper import *
import fasttext
import multiprocessing
import numpy as np
from numpy import dot
from numpy.linalg import norm
import spacy
import sys
from sentence_transformers import SentenceTransformer

all_geo_file_path = sys.argv[1]
queries = sys.argv[2].split(",")
other_multiplication_rate_options = [int(x) for x in sys.argv[3].split(",")][:2]
num_keyword_options = [int(x) for x in sys.argv[4].split(",")]
num_keyword_options.append("full_text")

def find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name, averaging_method = "word_vector"):
    results_dir_path = f"/Results/{query}/{model_name}/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)

    training_vector_list = []
    if model_name == 'dmis-lab/biobert-large-cased-v1.1-squad' or model_name == "bert-base-uncased":
        model = SentenceTransformer(model_name)
    elif model_name == "en_core_sci_lg" or model_name == "en_core_web_lg":
        model = spacy.load(model_name)
    else:
        model = get_model(f"/Models/custom/{model_name}/{keyword_extractor_name}/{num_keywords}.bin")
                    
    #Finding training set vector average
    for training_series_id in get_series_identifiers(query, "training_series"):
        keywords = get_keywords(keyword_extractor_name, num_keywords, training_series_id)
        if keywords == "":
            continue
        if model_name == 'dmis-lab/biobert-large-cased-v1.1-squad' or model_name == 'bert-base-uncased':
            training_vector_list.append(model.encode(keywords))
        elif model_name == "en_core_sci_lg" or model_name == "en_core_web_lg":
            training_vector_list.append(model(keywords).vector)
        elif averaging_method != "word_vector":
            training_vector_list.append(model.get_sentence_vector(keywords))
        else:
            training_vector_list.append(get_keyword_embedding(keywords, model, 300))
    average_training_vector = sum(training_vector_list) / len(training_vector_list)
    
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
        if model_name == 'dmis-lab/biobert-large-cased-v1.1-squad' or model_name == 'bert-base-uncased':
            testing_and_other_vector = model.encode(keywords)
        elif model_name == "en_core_sci_lg" or model_name == "en_core_web_lg":
            testing_and_other_vector = model(keywords).vector
        elif averaging_method != "word_vector":
            testing_and_other_vector = model.get_sentence_vector(keywords)
        else:    
            testing_and_other_vector = get_keyword_embedding(keywords, model, 300)

        #calculate cos sim
        cos_sim = dot(average_training_vector, testing_and_other_vector)/(norm(average_training_vector)*norm(testing_and_other_vector))
        cos_sim_and_series_id_list.append([cos_sim, testing_and_other_series_id])

    cos_sim_and_series_id_list.sort()
    cos_sim_and_series_id_list.reverse()

    #recording findings
    with open(f'{results_dir_path}/{averaging_method}_similarity.tsv', 'w+') as out_file:
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

def get_model(model_path):
    model = fasttext.load_model(model_path)
    return model

def get_keyword_embedding(keywords, model, vector_size):
    doc_vec = np.zeros((vector_size))

    keywords = keywords.split(" ")
    for word in keywords:
        new_vec = model.get_word_vector(word)
        doc_vec = np.add(doc_vec, new_vec)
    avg_word_vector = doc_vec / len(keywords)

    return(avg_word_vector)

models = get_model_types()
models.append("en_core_sci_lg")
models.append("en_core_web_lg")
models.append("dmis-lab/biobert-large-cased-v1.1-squad")
models.append("bert-base-uncased")
with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

for keyword_extractor_name in ['KPMiner', 'Baseline']:
#for keyword_extractor_name in get_keyword_extractors():
    for query in queries:
        for num_keywords in num_keyword_options:
            for other_multiplication_rate in other_multiplication_rate_options:
                for model_name in ["en_core_sci_lg"]:
                    mp = multiprocessing.Process(target=find_similarity, args=(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name))
                    find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
                    if num_keywords == "full_text":
                        find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name, "sentence_vector")
                    mp.start()