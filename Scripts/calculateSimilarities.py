from gensim.models import KeyedVectors
#from gensim.models.fasttext import FastText as FT_gensim
from helper import *
import fasttext
import multiprocessing
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
import spacy
import sys
import time

all_geo_file_path = sys.argv[1]
queries = sys.argv[2].split(",")
other_multiplication_rate_options = [int(x) for x in sys.argv[3].split(",")][:2]
num_keyword_options = [int(x) for x in sys.argv[4].split(",")]
num_keyword_options.append("full_text")

# def find_similarity_average_at_end_sentence_vector(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name):
#     results_dir_path = f"/Results/{query}/{model_name}/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
#     Path(results_dir_path).mkdir(parents=True, exist_ok=True)
#     training_vector_list = []
#     model = get_model(f"/Models/custom/{model_name}/{keyword_extractor_name}/{num_keywords}.bin")
                    
#     #Finding training set vector averag e
#     for training_series_id in get_series_identifiers(query, "training_series"):
#         if num_keywords == "full_text":
#             keywords = all_dict[training_series_id]
#         else:
#             keywords = get_keywords(keyword_extractor_name, num_keywords, training_series_id)
#         training_vector_list.append(model.get_sentence_vector(keywords))
    
#     list_of_ids = get_series_identifiers(f"{query}/other_series", other_multiplication_rate)
#     list_of_ids.extend(get_series_identifiers(query, "testing_series"))
    
#     cos_sim_and_series_id_list = []
#     no_keywords_series_list = []
#     #Finding vectors for each series to compare to training
#     for testing_and_other_series_id in list_of_ids:
#         if num_keywords == "full_text":
#             keywords = all_dict[training_series_id]
#         else:
#             keywords = get_keywords(keyword_extractor_name, num_keywords, testing_and_other_series_id)    
#         testing_and_other_vector = model.get_sentence_vector(keywords)
        
#         #calculate cos sim
#         sum_similarity = 0
#         for train_vector in training_vector_list:
#             cos_sim = dot(train_vector, testing_and_other_vector)/(norm(train_vector)*norm(testing_and_other_vector))
#             sum_similarity += cos_sim
#         cos_sim = sum_similarity / len(training_vector_list)
#         if len(keywords) < 2:
#             no_keywords_series_list.append(testing_and_other_series_id)
#             continue
#         cos_sim_and_series_id_list.append([cos_sim, testing_and_other_series_id])

#     cos_sim_and_series_id_list.sort()
#     cos_sim_and_series_id_list.reverse()

#     with open(f'{results_dir_path}/similarity.tsv', 'w+') as out_file:
#         print_time_stamp(f"Processing {results_dir_path}")
#         out_file.write("Series ID\tSimilarity Score\tTest or Other Group\n")
#         for series in cos_sim_and_series_id_list:
#             test_or_other= ""
#             if series[1] in get_series_identifiers(query, "testing_series"):
#                 test_or_other = "Test"
#             else:
#                 test_or_other = "Other"
#             out_file.write(f"{series[1]}\t{series[0]}\t{test_or_other}\n")

#         for series in no_keywords_series_list:
#             test_or_other= ""
#             if series in get_series_identifiers(query, "testing_series"):
#                 test_or_other = "Test"
#             else:
#                 test_or_other = "Other"
#             out_file.write(f"{series}\tNA\t{test_or_other}\n")
            
#     return(cos_sim_and_series_id_list)



# def find_similarity_average_at_end(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name):
#     results_dir_path = f"/Results/{query}/{model_name}/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
#     Path(results_dir_path).mkdir(parents=True, exist_ok=True)

#     training_vector_list = []
#     model = get_model(f"/Models/custom/{model_name}/{keyword_extractor_name}/{num_keywords}.bin")
                    
#     #Finding training set vector average
#     for training_series_id in get_series_identifiers(query, "training_series"):
#         if num_keywords == "full_text":
#             keywords = all_dict[training_series_id]
#         else:
#             keywords = get_keywords(keyword_extractor_name, num_keywords, training_series_id)
#         training_vector_list.append(get_keyword_embedding(keywords, model, 300))
    
#     list_of_ids = get_series_identifiers(f"{query}/other_series", other_multiplication_rate)
#     list_of_ids.extend(get_series_identifiers(query, "testing_series"))
    
#     cos_sim_and_series_id_list = []
#     no_keywords_series_list = []
#     #Finding vectors for each series to compare to training
#     for testing_and_other_series_id in list_of_ids:
#         if num_keywords == "full_text":
#             keywords = all_dict[training_series_id]
#         else:
#             keywords = get_keywords(keyword_extractor_name, num_keywords, testing_and_other_series_id)    
#         testing_and_other_vector = get_keyword_embedding(keywords, model, 300)
        
#         #calculate cos sim
#         sum_similarity = 0
#         for train_vector in training_vector_list:
#             cos_sim = dot(train_vector, testing_and_other_vector)/(norm(train_vector)*norm(testing_and_other_vector))
#             sum_similarity += cos_sim
#         cos_sim = sum_similarity / len(training_vector_list)
        
#         if len(keywords) < 2:
#             no_keywords_series_list.append(testing_and_other_series_id)
#             continue
#         cos_sim_and_series_id_list.append([cos_sim, testing_and_other_series_id])

#     cos_sim_and_series_id_list.sort()
#     cos_sim_and_series_id_list.reverse()

#     with open(f'{results_dir_path}/similarity.tsv', 'w+') as out_file:
#         print_time_stamp(f"Processing {results_dir_path}")
#         out_file.write("Series ID\tSimilarity Score\tTest or Other Group\n")
#         for series in cos_sim_and_series_id_list:
#             test_or_other= ""
#             if series[1] in get_series_identifiers(query, "testing_series"):
#                 test_or_other = "Test"
#             else:
#                 test_or_other = "Other"
#             out_file.write(f"{series[1]}\t{series[0]}\t{test_or_other}\n")

#         for series in no_keywords_series_list:
#             test_or_other= ""
#             if series in get_series_identifiers(query, "testing_series"):
#                 test_or_other = "Test"
#             else:
#                 test_or_other = "Other"
#             out_file.write(f"{series}\tNA\t{test_or_other}\n")

    
#     return(cos_sim_and_series_id_list)


def find_similarity_sentence_vector(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name):
    results_dir_path = f"/Results/{query}/{model_name}/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)

    training_vector_list = []
    model = get_model(f"/Models/custom/{model_name}/{keyword_extractor_name}/{num_keywords}.bin")
                    
    #Finding training set vector average
    for training_series_id in get_series_identifiers(query, "training_series"):
        if num_keywords == "full_text":
            keywords = all_dict[training_series_id]
        else:
            keywords = get_keywords(keyword_extractor_name, num_keywords, training_series_id)
            if keywords == "":
                continue
        training_vector_list.append(model.get_sentence_vector(keywords))
    average_training_vector = sum(training_vector_list) / len(training_vector_list)
    
    list_of_ids = []
    for series in get_series_identifiers(f"{query}/other_series", other_multiplication_rate):
        if get_keywords(keyword_extractor_name, num_keywords, series) != "":
            list_of_ids.append(series)
    for series in get_series_identifiers(query, "testing_series"):
        if get_keywords(keyword_extractor_name, num_keywords, series) != "":
            list_of_ids.append(series)
    
    cos_sim_and_series_id_list = []
    no_keywords_series_list = []
    #Finding vectors for each series to compare to training
    for testing_and_other_series_id in list_of_ids:
        if num_keywords == "full_text":
            keywords = all_dict[training_series_id]
        else:
            keywords = get_keywords(keyword_extractor_name, num_keywords, testing_and_other_series_id)    
        testing_and_other_vector = model.get_sentence_vector(keywords)
        
        #calculate cos sim
        cos_sim = dot(average_training_vector, testing_and_other_vector)/(norm(average_training_vector)*norm(testing_and_other_vector))
        if len(keywords) < 2:
            no_keywords_series_list.append(testing_and_other_series_id)
            continue
        cos_sim_and_series_id_list.append([cos_sim, testing_and_other_series_id])

    cos_sim_and_series_id_list.sort()
    cos_sim_and_series_id_list.reverse()



    with open(f'{results_dir_path}/similarity.tsv', 'w+') as out_file:
        print_time_stamp(f"Processing {results_dir_path}")
        out_file.write("Series ID\tSimilarity Score\tTest or Other Group\n")
        for series in cos_sim_and_series_id_list:
            test_or_other= ""
            if series[1] in get_series_identifiers(query, "testing_series"):
                test_or_other = "Test"
            else:
                test_or_other = "Other"
            out_file.write(f"{series[1]}\t{series[0]}\t{test_or_other}\n")

        for series in no_keywords_series_list:
            test_or_other= ""
            if series in get_series_identifiers(query, "testing_series"):
                test_or_other = "Test"
            else:
                test_or_other = "Other"
            out_file.write(f"{series}\tNA\t{test_or_other}\n")
            
    return(cos_sim_and_series_id_list)

# def find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name):
#     results_dir_path = f"/Results/{query}/{model_name}/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
#     Path(results_dir_path).mkdir(parents=True, exist_ok=True)

#     training_vector_list = []
#     model = get_model(f"/Models/custom/{model_name}/{keyword_extractor_name}/{num_keywords}.bin")
                    
#     #Finding training set vector average
#     for training_series_id in get_series_identifiers(query, "training_series"):
#         if num_keywords == "full_text":
#             keywords = all_dict[training_series_id]
#         else:
#             keywords = get_keywords(keyword_extractor_name, num_keywords, training_series_id)
#         training_vector_list.append(get_keyword_embedding(keywords, model, 300))
#     average_training_vector = sum(training_vector_list) / len(training_vector_list)
    
#     list_of_ids = get_series_identifiers(f"{query}/other_series", other_multiplication_rate)
#     list_of_ids.extend(get_series_identifiers(query, "testing_series"))


    
#     cos_sim_and_series_id_list = []
#     no_keywords_series_list = []
#     #Finding vectors for each series to compare to training
#     for testing_and_other_series_id in list_of_ids:
#         if num_keywords == "full_text":
#             keywords = all_dict[training_series_id]
#         else:
#             keywords = get_keywords(keyword_extractor_name, num_keywords, testing_and_other_series_id)    
#         testing_and_other_vector = get_keyword_embedding(keywords, model, 300)
        
#         #calculate cos sim
#         cos_sim = dot(average_training_vector, testing_and_other_vector)/(norm(average_training_vector)*norm(testing_and_other_vector))
#         if len(keywords) < 2:
#             no_keywords_series_list.append(testing_and_other_series_id)
#             continue
#         cos_sim_and_series_id_list.append([cos_sim, testing_and_other_series_id])

#     cos_sim_and_series_id_list.sort()
#     cos_sim_and_series_id_list.reverse()

#     with open(f'{results_dir_path}/similarity.tsv', 'w+') as out_file:
#         print_time_stamp(f"Processing {results_dir_path}")
#         out_file.write("Series ID\tSimilarity Score\tTest or Other Group\n")
#         for series in cos_sim_and_series_id_list:
#             test_or_other= ""
#             if series[1] in get_series_identifiers(query, "testing_series"):
#                 test_or_other = "Test"
#             else:
#                 test_or_other = "Other"
#             out_file.write(f"{series[1]}\t{series[0]}\t{test_or_other}\n")

#         for series in no_keywords_series_list:
#             test_or_other= ""
#             if series in get_series_identifiers(query, "testing_series"):
#                 test_or_other = "Test"
#             else:
#                 test_or_other = "Other"
#             out_file.write(f"{series}\tNA\t{test_or_other}\n")

    
#     return(cos_sim_and_series_id_list)

def get_model(model_path):
    model = fasttext.load_model(model_path)
    return model
    #if model_name == "BioWordVec":
    #    print_time_stamp("Loading BioWordVec")
    #    return KeyedVectors.load_word2vec_format("/Models/BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary = True)
    #if model_name == "FastTextWiki":
    #if model_name == "SciSpaCy":
    #if model_name == "SpaCy":
    #This is what it recommended when I ran the installer: spacy.load('en_core_web_lg')

def get_keyword_embedding(keywords, model, vector_size):
    doc_vec = np.zeros((vector_size))

    keywords = keywords.split(" ")
    for word in keywords:
        new_vec = model.get_word_vector(word)
        doc_vec = np.add(doc_vec, new_vec)
    avg_word_vector = doc_vec / len(keywords)

    return(avg_word_vector)

# def remove_empty_extraction_methods(keyword_extractor_name, list_of_gse):
#     gse_with_valid_extraction = []
#     for gse_id in list_of_gse:
#         with open(f"Data/KeywordWeights/{gse}") as read_file:

#             gse_with_valid_extraction.append(gse_id)
        
#     return(gse_with_valid_extraction)

models = get_model_types()
with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

for keyword_extractor_name in ['KPMiner', 'Baseline']:
    for query in queries:
    #for keyword_extractor_name in get_keyword_extractors():
        for num_keywords in num_keyword_options:
            for other_multiplication_rate in other_multiplication_rate_options:
            #for model_name, vector_size in modelVectorSizeDict.items():
                for model_name in models:
                    find_similarity_sentence_vector(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
                    # with open('Results/comparing_average_cos_similarities.tsv', 'a+') as test_file:
                    # #     test_file.write('Sentence Vector averaged at the end\t\tSentence Vector with average trained\t\tWord Vector with average trained\t\tWord Vector averaged at the end\t\n')
                    # #mp = multiprocessing.Process(target=find_similarity_average_at_end_sentence_vector, args=(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name))
                        # sentence_end = find_similarity_average_at_end_sentence_vector(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
                        # sentence = find_similarity_sentence_vector(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
                        # word = find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
                        # word_end = find_similarity_average_at_end(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
                        # for i, score_and_series in enumerate(sentence_end):
                        #     test_file.write(f'{score_and_series[0]}\t{score_and_series[1]}\t')
                        #     test_file.write(f'{sentence[i][0]}\t{sentence[i][1]}\t')
                        #     test_file.write(f'{word[i][0]}\t{word[i][1]}\t')
                        #     test_file.write(f'{word_end[i][0]}\t{word_end[i][1]}\n')
                        # break
                    #mp.start()


# Finding each similarity individually, without keywords. Full_text
#Functions test similarity by word vectors averaged or by sentence vectors.
#Run full text with sentense and word vectors. 
#Functions also test similarity by average training vector and by averaging comparisons to each training vector.
