import os
os.environ['HF_HOME'] = "/Models/huggingface"

from chromadb.utils import embedding_functions
import fasttext
from gensim.models import KeyedVectors
import gzip
from helper import *
import joblib
import json
import numpy as np
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import sys

#Translating text to numbers is known as encoding. Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.
#word-based tokenizers vs. character-based (better for languages like Chinese) vs. subword (best of both worlds)
#Batching is the act of sending multiple sentences through the model, all at once.
#Most models handle sequences of up to 512 or 1024 tokens. One option is to truncate them.

def save_embeddings(checkpoint, series_list, all_dict, tmp_dir_path):
    model_root = os.path.dirname(checkpoint)
    model_name = os.path.basename(checkpoint)

    Path(f"{tmp_dir_path}/{checkpoint}").mkdir(parents=True, exist_ok=True)
    embeddings_file_path = f"{tmp_dir_path}/{checkpoint}/embeddings.gz"

    if os.path.exists(embeddings_file_path):
        print(f"{embeddings_file_path} already exists.")
        return

    # Got error: menadsa/S-Bio_ClinicalBERT
    if model_root in ["sentence-transformers", "hkunlp", "thenlper", "pritamdeka", "NeuML", "FremyCompany", "nuvocare", "allenai", "microsoft", "emilyalsentzer", "medicalai"]:
        model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=f"{model_root}/{model_name}", device="cuda")
    elif model_root in ["nomic-ai"]:
        model = SentenceTransformer(f"{model_root}/{model_name}", device="cuda", trust_remote_code=True)
    elif model_root == "openai":
        with open("Models/open_ai.key") as api_key_file:
            api_key = api_key_file.read().strip()

        model = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)
    elif model_root == "fasttext":
        if model_name == "cbow-wikinews":
            model = KeyedVectors.load_word2vec_format("/Models/wiki-news-300d-1M-subword.vec", binary=False)
        elif model_name == "cbow-commoncrawl":
            model = KeyedVectors.load_word2vec_format("/Models/crawl-300d-2M-subword.vec", binary=False)
    else:
        raise Exception(f"Unknown model root: {model_root}.")

    # We have to do this in smaller chunks because sometimes API server won't accept full list.
    series_embeddings = []
    series_per_chunk = 100
    for start_i in range(0, len(series_list), series_per_chunk):
        end_i = start_i + series_per_chunk
        print(f"Obtaining embeddings for {start_i} - {end_i}")

        text_list = [all_dict[series] for series in series_list[start_i:end_i]]

        if model_root == "fasttext":
            embeddings = []

            for text in text_list:
                words = tokenize_and_remove_stop_words(text)
                word_embeddings = []

                for word in words:
                    if model.has_index_for(word):
                        word_embeddings.append(model.get_vector(word, norm=True))

                embeddings.append(np.mean(word_embeddings, axis=0).tolist())
        elif model_root == "nomic-ai":
            embeddings = model.encode(text_list).tolist()
        else:
            embeddings = model(text_list)

        series_embeddings.extend(embeddings)

    embeddings_dict = {}
    for i, embedding in enumerate(series_embeddings):
        series = series_list[i]
        embeddings_dict[series] = embedding

    print(f"Saving to {embeddings_file_path}.")
    with gzip.open(embeddings_file_path, "w") as embeddings_file:
        embeddings_file.write(json.dumps(embeddings_dict).encode())

def save_similarities(checkpoint, tmp_dir_path):
    embeddings_file_path = f"{tmp_dir_path}/{checkpoint}/embeddings.gz"
    distances_file_path = f"{tmp_dir_path}/{checkpoint}/distances.gz"

    if os.path.exists(distances_file_path):
        print(f"{distances_file_path} already exists.")
        return

    with gzip.open(embeddings_file_path) as embeddings_file:
        embeddings_dict = json.loads(embeddings_file.read().decode())

    print(f"Saving to {distances_file_path}.")
    with gzip.open(distances_file_path, "w") as distances_file:
        distances_file.write("Series_A\tSeries_B\tMethod\tScore\n".encode())

        for series_A, series_A_embedding in embeddings_dict.items():
            print(f"Finding distances for {series_A}")
            for series_B, series_B_embedding in embeddings_dict.items():
                if series_A == series_B:
                    continue

                similarity = cos_sim(series_A_embedding, series_B_embedding)
                similarity = similarity.numpy()[0][0]

                distances_file.write((f"{series_A}\t{series_B}\t{checkpoint}\t{similarity}\n").encode())

gemma_json_file_path = sys.argv[1]
all_geo_json_file_path = sys.argv[2]
tmp_dir_path = sys.argv[3]

with gzip.open(gemma_json_file_path) as gemma_file:
    gemma_list = sorted(list(json.loads(gemma_file.read()).keys()))

with gzip.open(all_geo_json_file_path) as all_file:
    all_dict = json.loads(all_file.read())

# Built this list on February 26, 2024.
# FYI: PharMolix/BioMedGPT-LM-7B would not run because I did not have enough GPU memory.
checkpoints = ["fasttext/cbow-wikinews", "fasttext/cbow-commoncrawl", "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-roberta-large-v1", "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/msmarco-distilbert-base-v3", "sentence-transformers/sentence-t5-large", "sentence-transformers/paraphrase-TinyBERT-L6-v2", "hkunlp/instructor-xl", "thenlper/gte-large", "nomic-ai/nomic-embed-text-v1.5", "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb", "openai/text-embedding-ada-002", "openai/text-embedding-3-small", "openai/text-embedding-3-large", "NeuML/pubmedbert-base-embeddings", "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT", "FremyCompany/BioLORD-2023", "pritamdeka/S-BioBert-snli-multinli-stsb", "nuvocare/WikiMedical_sent_biobert", "sentence-transformers/average_word_embeddings_glove.6B.300d", "sentence-transformers/average_word_embeddings_glove.840B.300d", "allenai/scibert_scivocab_uncased", "allenai/biomed_roberta_base", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "emilyalsentzer/Bio_ClinicalBERT", "medicalai/ClinicalBERT"]

for checkpoint in checkpoints:
    save_embeddings(checkpoint, gemma_list, all_dict, tmp_dir_path)

joblib.Parallel(n_jobs=8)(joblib.delayed(save_similarities)(checkpoint, tmp_dir_path) for checkpoint in checkpoints)

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
