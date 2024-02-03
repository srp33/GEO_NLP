import gzip
from helper import *
import json
import sys

gemma_json_file_path = sys.argv[1]
all_geo_json_file_path = sys.argv[2]
out_file_path = sys.argv[3]

star_list = []

with gzip.open(gemma_json_file_path) as gemma_file:
    gemma_list = sorted(list(json.loads(gemma_file.read()).keys()))

with gzip.open(all_geo_json_file_path) as all_file:
    all_dict = json.loads(all_file.read())

series_word_sets = {}
for series in gemma_list:
    text = clean_text(all_dict[series])
    words = set(text.split(" ")) - {''}
    series_word_sets[series] = words

with gzip.open(out_file_path, "w") as out_file:
    out_file.write("Series_A\tSeries_B\tProportion_Shared_Words\n".encode())

    for this_series in gemma_list:
        print(this_series)
        this_series_words = series_word_sets[this_series]

        for other_series in gemma_list:
            if this_series == other_series:
                continue

            other_series_words = series_word_sets[other_series]

            common_words = this_series_words & other_series_words
            all_words = this_series_words | other_series_words
            proportion_shared_words = len(common_words) / len(all_words)

            out_file.write((f"{this_series}\t{other_series}\t{proportion_shared_words}\n").encode())
