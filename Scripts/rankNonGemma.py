#import glob
import gzip
#import json
#import os
import sys

query_file_path = sys.argv[1]
non_gemma_file_path = sys.argv[2]

with open(query_file_path) as query_file:
    query_series = query_file.read().strip().split("\n")

with gzip.open(non_gemma_file_path) as non_gemma_file:
    non_gemma_series = non_gemma_file.read().decode().strip().split("\n")


