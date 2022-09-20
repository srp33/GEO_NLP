import multiprocessing
import os
import sys
import json
from helper import extract_keywords

all_geo_file_path = sys.argv[1]
queries = sys.argv[2].split(",")
multiplication_rate_options = [int(x) for x in sys.argv[3].split(",")]
max_keywords = max([int(x) for x in sys.argv[4].split(",")][:-1])

# Keys are series IDs, values are text from GEO
all_geo_dict = {}
with open(all_geo_file_path) as cache_file:
    all_geo_dict = json.loads(cache_file.read())

# for series in all_geo_dict:
#     extract_keywords(series)
print("about to start")
args=[(series, 32) for series in all_geo_dict]
pool = multiprocessing.Pool(processes = 12)
pool.starmap(extract_keywords, args)
pool.close()