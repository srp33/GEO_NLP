import os
os.environ['HF_HOME'] = "/Models/huggingface"

#import fasttext
#from gensim.models import KeyedVectors
#from gensim.models.doc2vec import Doc2Vec
import gzip
from helper import *
import joblib
import json
#import numpy as np
#from numpy import dot
#from numpy.linalg import norm
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
#import spacy
import sys
#from testingBert import *
#from transformers import pipeline
#from transformers import logging
#logging.set_verbosity_error()

#Translating text to numbers is known as encoding. Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.
#word-based tokenizers vs. character-based (better for languages like Chinese) vs. subword (best of both worlds)
#Batching is the act of sending multiple sentences through the model, all at once.
#Most models handle sequences of up to 512 or 1024 tokens. One option is to truncate them.

def save_encodings_for_series(checkpoint, this_series, gemma_list, all_dict, tmp_dir_path):
    Path(f"{tmp_dir_path}/{checkpoint}").mkdir(parents=True, exist_ok=True)

    tmp_file_path = f"{tmp_dir_path}/{checkpoint}/{this_series}"
    if os.path.exists(tmp_file_path):
        print(f"{tmp_file_path} already exists.")
        return

    with open(tmp_file_path, "w") as tmp_file:
        model = SentenceTransformer(checkpoint)

        this_series_vector = model.encode(all_dict[this_series])

        other_list = [x for x in gemma_list if x != this_series]
        other_series_sentences = [all_dict[x] for x in other_list]

        # FYI: I ran some tests, and this batched approach was slightly slower than processing them one at a time.
        #      other_series_vectors = model.encode(other_series_sentences)

        for i, other_series in enumerate(other_list):
            print(checkpoint, this_series, other_series)
            other_series_vector = model.encode(other_series_sentences[i])
            similarity = cos_sim(this_series_vector, other_series_vector)
            similarity = similarity.numpy()[0][0]

            tmp_file.write(f"{this_series}\t{other_series}\t{checkpoint}\t{similarity}\n")


gemma_json_file_path = sys.argv[1]
all_geo_json_file_path = sys.argv[2]
tmp_dir_path = sys.argv[3]
out_file_path = sys.argv[4]

if os.path.exists(out_file_path):
    print(f"{out_file_path} already exists.")
    sys.exit(0)

with gzip.open(gemma_json_file_path) as gemma_file:
    gemma_list = sorted(list(json.loads(gemma_file.read()).keys()))

with gzip.open(all_geo_json_file_path) as all_file:
    all_dict = json.loads(all_file.read())

#return(['dmis-lab/biobert-large-cased-v1.1-squad', 'bert-base-uncased', "allenai/scibert_scivocab_uncased", "all-roberta-large-v1", "sentence-t5-xxl", "all-mpnet-base-v2"])
#return ["fasttext__cbow", "fasttext__skipgram", "en_core_web_lg", "en_core_sci_lg", "all-roberta-large-v1", "sentence-t5-xxl", "all-mpnet-base-v2", "dmis-lab/biobert-large-cased-v1.1-squad", "bert-base-uncased", "allenai/scibert_scivocab_uncased", "gpt2", "bioWordVec", "pretrained_fasttext_wiki", "pretrained_fasttext_wiki_subword", "pretrained_fasttext_crawl", "pretrained_fasttext_crawl_subword", 'BiomedRoberta', 'GEOBert']
checkpoints = ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-roberta-large-v1"]

# FYI: This is the single-threaded way to iterate through all combinations.
#for checkpoint in checkpoints:
#    for this_series in gemma_list:
#        save_encodings_for_series(checkpoint, this_series, gemma_list, all_dict, tmp_dir_path)

# FYI: This is the multi-threaded way to iterate through all combinations.
combos = [[checkpoint, this_series] for checkpoint in checkpoints for this_series in gemma_list]
joblib.Parallel(n_jobs=32)(joblib.delayed(save_encodings_for_series)(combo[0], combo[1], gemma_list, all_dict, tmp_dir_path) for combo in combos)

#with gzip.open(out_file_path, "w") as out_file:
#    out_file.write("Series_A\tSeries_B\tMethod\tScore\n".encode())

sys.exit(1)

#model = KeyedVectors.load_word2vec_format("/Models/BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True)
#model = KeyedVectors.load_word2vec_format("/Models/wiki-news-300d-1M.vec")
#print(model.wv.vocab)
#model = Doc2Vec.load()

#This function has different if statements tailored to different models for their implementation and use.
def find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name, averaging_method = "sentence_vector", word_method = 'sum'):
    results_dir_path = f"/Results/{query}/{model_name}/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
    if os.path.exists(f'{results_dir_path}/{averaging_method}_similarity.tsv'):
        return()
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)
    training_vector_list = []

    #Load the model
    if model_name in hugging_face_list:
        model = SentenceTransformer(model_name)
    elif model_name == 'BiomedRoberta':
        model = AutoModel.from_pretrained("allenai/biomed_roberta_base")
        tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
    elif model_name == "GEOBert":
        model_path = "/Models/custom/bert"
        model = BertModel.from_pretrained(os.path.join(model_path, "checkpoint-3000"), output_hidden_states = True)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    elif model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        word_embeddings = model.transformer.wte.weight
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
            training_vector_list.append(model.encode(keywords))
        elif model_name == 'BiomedRoberta':
            training_vector_list.append(get_rob_sentence_embedding(keywords, model, tokenizer, word_method))
        elif model_name == "gpt2":
            tmp_list = []
            for word in keywords:
                text_index = tokenizer.encode(word,add_prefix_space=True)
                vector = word_embeddings[text_index,:]
                vector = vector.detach().numpy()
                tmp_list.append(vector)
            training_vector_list.append(sum(tmp_list) / len(tmp_list))
        elif model_name == "GEOBert":
            if averaging_method == "word_vector":
                training_vector_list.append(get_bert_word_embedding(keywords, model, tokenizer, word_method))
            else:
                training_vector_list.append(get_sentence_embedding(keywords, model, tokenizer))
        elif model_name == "scibert_scivocab_uncased":
            training_vector_list.append()
        elif model_name == "en_core_sci_lg" or model_name == "en_core_web_lg":
            training_vector_list.append(model(keywords).vector)
        elif averaging_method != "word_vector":
            training_vector_list.append(model.get_sentence_vector(keywords))
        elif model_name == "bioWordVec" or model_name.startswith("pretrained"):
            tmp_list = []
            for word in keywords:
                if word in model.wv.vocab:
                    tmp_list.append(model[word])
            training_vector_list.append(sum(tmp_list) / len(tmp_list))
        else:
            training_vector_list.append(get_keyword_embedding(keywords, model, 300))
    average_training_vector = sum(training_vector_list) / len(training_vector_list)

    #grab every series ID with the 'test' or 'other' label assigned for this query and multiplication rate
    list_of_ids = []
    for series in get_series_identifiers(f"{query}/other_series", other_multiplication_rate):
        if get_keywords(keyword_extractor_name, num_keywords, series) != "":
            list_of_ids.append(series)
    for series in get_series_identifiers(query, "testing_series"):
        if get_keywords(keyword_extractor_name, num_keywords, series) != "":

            list_of_ids.append(series)
    cos_sim_and_series_id_list = []
    na_list = []
    #Finding vectors for each series to compare to training
    for testing_and_other_series_id in list_of_ids:
        keywords = get_keywords(keyword_extractor_name, num_keywords, testing_and_other_series_id)
        if model_name in hugging_face_list:
            testing_and_other_vector = model.encode(keywords)
        elif model_name == 'BiomedRoberta':
            testing_and_other_vector = get_rob_sentence_embedding(keywords, model, tokenizer, word_method)
        elif model_name == "gpt2":
            tmp_list = []
            for word in keywords:
                text_index = tokenizer.encode(word,add_prefix_space=True)
                vector = word_embeddings[text_index,:]
                vector = vector.detach().numpy()
                tmp_list.append(vector)
            try:
                testing_and_other_vector = (sum(tmp_list) / len(tmp_list))
            except:
                cos_sim_and_series_id_list.append([0, testing_and_other_series_id])
                print("An exception occurred")
                continue
        elif model_name == "GEOBert":
            testing_and_other_vector = get_sentence_embedding(keywords, model, tokenizer)
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

        #calculate cosine similarity between series and averaged training vector
        if model_name == "gpt2" or model_name == 'BiomedRoberta':
            try:
                cos_sim = dot(average_training_vector[0], testing_and_other_vector[0])/(norm(average_training_vector[0])*norm(testing_and_other_vector[0]))
                cos_sim_and_series_id_list.append([cos_sim, testing_and_other_series_id])
            except:
                na_list.append(["NA", testing_and_other_series_id])
                print("An exception occurred")
        else:
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
        for series in na_list:
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

models = get_model_types()

with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

word_method = "sum"
#We found no significant difference between concatenation and sum methodologies for retrieving embeddings from top 4 layers.
#We used model default techniques to retrieve embeddings if functions were in place for their retrieval. Also we found no difference
#between sentence and word averaging methods. We default here to sentence vectors due the ease of SentenceTransformers.
keyword_extractor_name = "Baseline"
num_keywords= 'full_text'
#we only want to see full_text comparisons because we are not using keyword extraction here.

for model_name in models:
    for query in queries:
        for other_multiplication_rate in other_multiplication_rate_options:
            if model_name == "GEOBert":
                find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name, word_method)
            if model_name.startswith("fasttext") and num_keywords == "full_text":
                find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
            else:
                find_similarity(query, keyword_extractor_name, num_keywords, other_multiplication_rate, model_name)
