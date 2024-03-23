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

def save_embeddings(checkpoint, series_dict, tmp_dir_path):
    series_list = sorted(list(series_dict.keys()))

    model_root = os.path.dirname(checkpoint)
    model_name = os.path.basename(checkpoint)

    Path(f"{tmp_dir_path}/{checkpoint}").mkdir(parents=True, exist_ok=True)
    embeddings_file_path = f"{tmp_dir_path}/{checkpoint}/embeddings.gz"

    if os.path.exists(embeddings_file_path):
        print(f"{embeddings_file_path} already exists.")
        return

    # Got error: menadsa/S-Bio_ClinicalBERT
    if model_root in ["sentence-transformers", "hkunlp", "thenlper", "pritamdeka", "NeuML", "FremyCompany", "nuvocare", "allenai", "microsoft", "emilyalsentzer", "medicalai", "albert"]:
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

        text_list = [series_dict[series] for series in series_list[start_i:end_i]]

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
        embeddings_dict[series_list[i]] = embedding

    print(f"Saving to {embeddings_file_path}.")
    with gzip.open(embeddings_file_path, "w") as embeddings_file:
        embeddings_file.write(json.dumps(embeddings_dict).encode())

def save_similarities(checkpoint, embeddings1_file_path, embeddings2_file_path, tmp_dir_path):
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
            if series_A == "GSE10040":
                break

            for series_B, series_B_embedding in embeddings_dict.items():
                if series_A == series_B:
                    continue

                similarity = cos_sim(series_A_embedding, series_B_embedding)
                similarity = similarity.numpy()[0][0]

                distances_file.write((f"{series_A}\t{series_B}\t{checkpoint}\t{similarity}\n").encode())

series1_json_file_path = sys.argv[1]
series2_json_file_path = sys.argv[2]
tmp_dir_path = sys.argv[3]

with gzip.open(series1_json_file_path) as series1_file:
    series1_dict = json.loads(series1_file.read())

with gzip.open(series2_json_file_path) as series2_file:
    series2_dict = json.loads(series2_file.read())

# Built this list on February 27, 2024.
# FYI: PharMolix/BioMedGPT-LM-7B would not run because I did not have enough GPU memory.
#checkpoints = ["fasttext/cbow-wikinews", "fasttext/cbow-commoncrawl", "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-roberta-large-v1", "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/msmarco-distilbert-base-v3", "sentence-transformers/sentence-t5-large", "sentence-transformers/sentence-t5-xl", "sentence-transformers/paraphrase-TinyBERT-L6-v2", "hkunlp/instructor-xl", "thenlper/gte-large", "nomic-ai/nomic-embed-text-v1.5", "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb", "openai/text-embedding-ada-002", "openai/text-embedding-3-small", "openai/text-embedding-3-large", "NeuML/pubmedbert-base-embeddings", "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT", "FremyCompany/BioLORD-2023", "pritamdeka/S-BioBert-snli-multinli-stsb", "nuvocare/WikiMedical_sent_biobert", "sentence-transformers/average_word_embeddings_glove.6B.300d", "sentence-transformers/average_word_embeddings_glove.840B.300d", "allenai/scibert_scivocab_uncased", "allenai/biomed_roberta_base", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "emilyalsentzer/Bio_ClinicalBERT", "medicalai/ClinicalBERT", "albert/albert-base-v2", "albert/albert-xxlarge-v2"]
checkpoints = ["sentence-transformers/all-MiniLM-L6-v2"]

#TODO: Change how the embeddings file is named so we don't re-save it for both.
for checkpoint in checkpoints:
    save_embeddings(checkpoint, series1_dict, tmp_dir_path)
    save_embeddings(checkpoint, series2_dict, tmp_dir_path)


#joblib.Parallel(n_jobs=8)(joblib.delayed(save_similarities)(checkpoint, f"{tmp_dir_path}/{checkpoint}/embeddings.gz", f"{tmp_dir_path}/{checkpoint}/embeddings.gz", tmp_dir_path) for checkpoint in checkpoints)
