import glob
import gzip
from helper import *
import json
import math
import os
import sys

similarities_file_pattern = sys.argv[1]
gemma_json_file_path = sys.argv[2]
out_file_path = sys.argv[3]

with gzip.open(gemma_json_file_path) as the_file:
    gemma_dict = json.loads(the_file.read())

with gzip.open(out_file_path, "w") as out_file:
    out_file.write((f"Query\tMethod\tSeries\tTextLength\tDistance\n").encode())

    for similarities_file_path in glob.glob(similarities_file_pattern):
        print(f"Saving text-length vs. distance for {similarities_file_path}")

        file_path_items = similarities_file_path.split("/")
        query = file_path_items[1]
        method = file_path_items[2].replace("____", "/")

        with open(similarities_file_path) as similarities_file:
            similarities_file.readline()

            for line in similarities_file:
                line_items = line.rstrip("\n").split("\t")
                series = line_items[0]
                group = line_items[1]
                score = float(line_items[2])

                if group == "Testing":
                    out_file.write((f"{query}\t{method}\t{series}\t{len(gemma_dict[series])}\t{score}\n").encode())

print(f"Saved output to {out_file_path}")
