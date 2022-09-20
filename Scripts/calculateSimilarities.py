import tokenize
from helper import *
import fasttext
import os
import multiprocessing
import numpy as np
from numpy import dot
from numpy.linalg import norm
from gensim.models import KeyedVectors
import spacy
import sys
import zipfile
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import *
from gensim.models import Word2Vec

all_geo_file_path = sys.argv[1]
queries = sys.argv[2].split(",")
other_multiplication_rate_options = [int(x) for x in sys.argv[3].split(",")][:2]
num_keyword_options = [int(x) for x in sys.argv[4].split(",")]
num_keyword_options.append("full_text")
hugging_face_list = ['dmis-lab/biobert-large-cased-v1.1-squad', 'bert-base-uncased', "allenai/scibert_scivocab_uncased"]

def find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name, averaging_method = "word_vector"):
    results_dir_path = f"/Results/{query}/{model_name}/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
    if os.path.exists(f'{results_dir_path}/{averaging_method}_similarity.tsv'):
        return()
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)
    training_vector_list = []
    if model_name in hugging_face_list:
        model = SentenceTransformer(model_name)  
    elif model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained('gpt2')  # or any other checkpoint
        word_embeddings = model.transformer.wte.weight  # Word Token Embeddings 
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif model_name == "en_core_sci_lg" or model_name == "en_core_web_lg":
        model = spacy.load(model_name)
    elif model_name == "bioWordVec":
        model = KeyedVectors.load_word2vec_format("/Models/BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True)
    elif model_name.startswith("pretrained"):
        if model_name == "pretrained_fasttext_wiki":
            path = "/Models/wiki-news-300d-1M.vec"
        if model_name == "pretrained_fasttext_wiki_subword":
            path = "/Models/wiki-news-300d-1M-subword.vec"
        if model_name == "pretrained_fasttext_crawl":
            path = "/Models/crawl-300d-2M.vec"
        if model_name == "pretrained_fasttext_crawl_subword":
            path = "/Models/crawl-300d-2M-subword.vec"
        model = KeyedVectors.load_word2vec_format(path)
    else:
        model = fasttext.load_model(f"Models/custom/{model_name}/{keyword_extractor_name}/{num_keywords}.bin")

    #Finding training set vector average
    for training_series_id in get_series_identifiers(query, "training_series"):
        keywords = get_keywords(keyword_extractor_name, num_keywords, training_series_id)
        if keywords == "":
            continue
        if model_name in hugging_face_list:
            tokenize.add_special_tokens({'pad_token': '[PAD]'})
            training_vector_list.append(model.encode(keywords))
        if model_name == "gpt2":
            tmp_list = []
            for word in keywords:
                text_index = tokenizer.encode(word,add_prefix_space=True)
                vector = word_embeddings[text_index,:]
                vector = vector.detach().numpy()
                tmp_list.append(vector)
            training_vector_list.append(sum(tmp_list) / len(tmp_list))
        elif model_name == "scibert_scivocab_uncased":
            training_vector_list.append()
        elif model_name == "en_core_sci_lg" or model_name == "en_core_web_lg":
            training_vector_list.append(model(keywords).vector)
        elif averaging_method != "word_vector":
            training_vector_list.append(model.get_sentence_vector(keywords))
        elif model_name == "bioWordVec" or model_name.startswith("pretrained"):
            tmp_list = []
            for word in keywords:
                if word in model.wv.vocab: #is this why 1?
                    tmp_list.append(model[word])
            training_vector_list.append(sum(tmp_list) / len(tmp_list))
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
        if model_name in hugging_face_list:
            testing_and_other_vector = model.encode(keywords)
        elif model_name == "gpt2":
            tmp_list = []
            for word in keywords:
                text_index = tokenizer.encode(word,add_prefix_space=True)
                vector = word_embeddings[text_index,:]
                vector = vector.detach().numpy()
                tmp_list.append(vector)
                testing_and_other_vector = (sum(tmp_list) / len(tmp_list))
        elif model_name == "en_core_sci_lg" or model_name == "en_core_web_lg":
            testing_and_other_vector = model(keywords).vector
        elif model_name == "bioWordVec" or model_name.startswith("pretrained"):
            tmp_list = []
            for word in keywords:
                if word in model.wv.vocab:
                    tmp_list.append(model[word])
            testing_and_other_vector = (sum(tmp_list) / len(tmp_list))

        elif averaging_method != "word_vector":
            testing_and_other_vector = model.get_sentence_vector(keywords)
        else:    
            testing_and_other_vector = get_keyword_embedding(keywords, model, 300)
        #calculate cos sim
        cos_sim = dot(average_training_vector[0], testing_and_other_vector[0])/(norm(average_training_vector[0])*norm(testing_and_other_vector[0]))
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

with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

print("I have started calculate similarities script, about to enter the for loop") #TODO: delete later

keyword_extraction = get_list_extractors()
print(keyword_extraction)

for keyword_extractor_name in keyword_extraction:
    for query in queries:
        for num_keywords in num_keyword_options:
            for other_multiplication_rate in other_multiplication_rate_options:
                print("I am starting a new combination!") #TODO: delete later
                for model_name in ["fasttext_cbow"]:
                #for model_name in models:
                #for model_name in ["pretrained_fasttext_wiki", "pretrained_fasttext_wiki_subword", "pretrained_fasttext_crawl", "pretrained_fasttext_crawl_subword"]:
                    if model_name.startswith("fasttext") and num_keywords == "full_text":
                        find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name, "sentence_vector")
                    find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
                    #mp = multiprocessing.Process(target=find_similarity, args=(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name))
                    #find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
                    #mp.start()