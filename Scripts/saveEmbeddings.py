import os
os.environ['HF_HOME'] = "/Models/huggingface"

from chromadb.utils import embedding_functions
import fasttext
#from gensim.models import KeyedVectors
import gzip
from helper import *
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys

def save_embeddings(checkpoint, series_dict, chunk_size, out_file_path):
    Path(os.path.dirname(out_file_path)).mkdir(parents=True, exist_ok=True)

    series_list = sorted(list(series_dict.keys()))

    model_root = os.path.dirname(checkpoint)
    model_name = os.path.basename(checkpoint)

    if os.path.exists(out_file_path):
        print(f"{out_file_path} already exists.")
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
            print("Loading word2vec vectors for cbow-wikinews", flush=True)
            model = KeyedVectors.load_word2vec_format("/Models/wiki-news-300d-1M-subword.vec", binary=False)
        elif model_name == "cbow-commoncrawl":
            print("Loading word2vec vectors for cbow-commoncrawl", flush=True)
            model = KeyedVectors.load_word2vec_format("/Models/crawl-300d-2M-subword.vec", binary=False)
    else:
        raise Exception(f"Unknown model root: {model_root}.")

    # We have to do this in smaller chunks because sometimes API server won't accept full list.
    series_embeddings_dict = {}

    # https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    for i, series in enumerate(series_list):
        if i % 100 == 0:
            print(f"Obtaining embeddings for {i} out of {len(series_list)} for {checkpoint}.", flush=True)

        series_text = series_dict[series]

        # We will split each text into chunks, even if we will later rejoin the chunks.
        text_chunks = text_splitter.split_text(series_text)

        if model_root == "fasttext":
            # For this type of model, we always do word-level analysis, so we don't use the split text.
            words = tokenize_and_remove_stop_words(series_text)

            word_embeddings = []
            for word in words:
                if model.has_index_for(word):
                    word_embeddings.append(model.get_vector(word, norm=True).tolist())

            series_embeddings_dict[series] = np.mean(word_embeddings, axis=0).tolist()
        elif model_root == "nomic-ai":
            series_embeddings_dict[series] = np.mean(model.encode(text_chunks), axis=0).tolist()
        else:
            series_embeddings_dict[series] = np.mean(model(text_chunks), axis=0).tolist()

    print(f"Saving to {out_file_path}.")
    with gzip.open(out_file_path, "w") as embeddings_file:
        embeddings_file.write(json.dumps(series_embeddings_dict).encode())

series_json_file_path = sys.argv[1]
checkpoints_file_path = sys.argv[2]
chunk_size = int(sys.argv[3])
out_dir_path = sys.argv[4]

with gzip.open(series_json_file_path) as series_file:
    series_dict = json.loads(series_file.read())

with open(checkpoints_file_path) as checkpoints_file:
    checkpoints = []
    for line in checkpoints_file:
        if line.startswith("#"):
            continue

        checkpoints.append(line.rstrip("\n"))

for checkpoint in checkpoints:
    save_embeddings(checkpoint, series_dict, chunk_size, f"{out_dir_path}/{checkpoint}.gz")
