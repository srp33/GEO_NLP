import gzip
from helper import *
import os
import sys

tsv_file_path = sys.argv[1]
json_file_path = sys.argv[2]

article_dict = {}

with gzip.open(tsv_file_path) as tsv_file:
    tsv_file.readline()

    for line in tsv_file:
        line_items = line.decode().rstrip("\n").split("\t")

        gse = line_items[0]
        title = line_items[1]
        summary = line_items[2]
        overall_design = line_items[3]
        all_species = line_items[6].lower().split("; ")
        all_taxon_id = line_items[7].split("; ")
        superseries_gse = line_items[8]

        # We remove series that are part of a superseries because including these
        #   could cause bias in the machine-learning analysis. Additionally,
        #   if users find a relevant SuperSeries, they will find the associated
        #   SubSeries.
        if "homo sapiens" in all_species and "9606" in all_taxon_id and superseries_gse == "":
            print(gse)
            article_dict[gse] = clean_text(f"{title} {summary} {overall_design}")

print(len(article_dict)) #75980
with gzip.open(json_file_path, 'w') as json_file:
    json_file.write(json.dumps(article_dict).encode())
