import bm25s
import gzip
from helper import *
import json
import Stemmer
import sys

gemma_json_file_path = sys.argv[1]
plus = sys.argv[2] == "True"
out_file_path = sys.argv[3]

with gzip.open(gemma_json_file_path) as gemma_file:
    gemma_dict = json.loads(gemma_file.read())
    gemma_ids = sorted(list(gemma_dict.keys()))

gemma_texts = [gemma_dict[gemma_id] for gemma_id in gemma_ids]

if plus:
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(gemma_texts, stopwords="en", stemmer=stemmer)
else:
    tokens = bm25s.tokenize(gemma_texts)

retriever = bm25s.BM25()
retriever.index(tokens)

k = len(gemma_ids)

with gzip.open(out_file_path, "w") as out_file:
    out_file.write("Series_A\tSeries_B\tMethod\tScore\n".encode())

    for i, gemma_id in enumerate(gemma_ids):
        gemma_text = gemma_texts[i]
        print(gemma_id)
        #print(gemma_text)

        # results and scores have one row. They have one column per document.
        if plus:
            query_tokens = bm25s.tokenize(gemma_text, stopwords="en", stemmer=stemmer)
        else:
            query_tokens = bm25s.tokenize(gemma_text)


        results, scores = retriever.retrieve(query_tokens, k=k)

        for j in range(k):
            matching_index = results[0, j]
            matching_id = gemma_ids[matching_index]

            if gemma_id == matching_id:
                continue

            matching_text = gemma_texts[matching_index]
            score = scores[0, j]

            out_file.write((f"{gemma_id}\t{matching_id}\tBM25\t{score}\n").encode())
