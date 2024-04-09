import os
os.environ['HF_HOME'] = "/Models/huggingface"

import gzip
from helper import *
import joblib
import json
import os
from pathlib import Path
from sentence_transformers.util import cos_sim
import sys

def save_distances(checkpoint, series1_dict, series2_dict, embeddings_file_path, out_file_path):
    if os.path.exists(out_file_path):
        print(f"{out_file_path} already exists.")
        return

    Path(os.path.dirname(out_file_path)).mkdir(parents=True, exist_ok=True)

    with gzip.open(embeddings_file_path) as embeddings_file:
        embeddings_dict = json.loads(embeddings_file.read().decode())

    print(f"Saving to {out_file_path}.")
    with gzip.open(out_file_path, "w") as distances_file:
        distances_file.write("Series_A\tSeries_B\tMethod\tScore\n".encode())

        for series_A in sorted(series1_dict):
            print(f"Finding distances for {series_A} and {checkpoint}", flush=True)
            for series_B in sorted(series2_dict):
                if series_A == series_B:
                    continue

                distance = cos_sim(embeddings_dict[series_A], embeddings_dict[series_B])
                distance = distance.numpy()[0][0]

                distances_file.write((f"{series_A}\t{series_B}\t{checkpoint}\t{distance}\n").encode())

series1_json_file_path = sys.argv[1]
series2_json_file_path = sys.argv[2]
checkpoints_file_path = sys.argv[3]
embeddings_dir_path = sys.argv[4]
out_dir_path = sys.argv[5]

with gzip.open(series1_json_file_path) as series1_file:
    series1_dict = json.loads(series1_file.read())

with gzip.open(series2_json_file_path) as series2_file:
    series2_dict = json.loads(series2_file.read())

with open(checkpoints_file_path) as checkpoints_file:
    checkpoints = []
    for line in checkpoints_file:
        if line.startswith("#"):
            continue

        checkpoints.append(line.rstrip("\n"))

joblib.Parallel(n_jobs=8)(joblib.delayed(save_distances)(checkpoint, series1_dict, series2_dict, f"{embeddings_dir_path}/{checkpoint}.gz", f"{out_dir_path}/{checkpoint}.tsv.gz") for checkpoint in checkpoints)
