import gzip
from helper import *
import json
import sys

gemma_json_file_path = sys.argv[1]
out_file_path = sys.argv[2]

with gzip.open(gemma_json_file_path) as gemma_file:
    gemma_dict = json.loads(gemma_file.read())
    gemma_list = sorted(list(gemma_dict.keys()))

series_word_sets = {}
for series in gemma_list:
    text = gemma_dict[series]
    text = tokenize_and_remove_stop_words(text)
    words = set(text.split(" ")) - {''}
    series_word_sets[series] = words

with gzip.open(out_file_path, "w") as out_file:
    out_file.write("Series_A\tSeries_B\tMethod\tScore\n".encode())

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

            out_file.write((f"{this_series}\t{other_series}\tWord overlap\t{proportion_shared_words}\n").encode())
