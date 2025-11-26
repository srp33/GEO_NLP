import gzip
from helper import *
import json
import os
import sys

tsv_file_path = sys.argv[1]
convert_to_lower = sys.argv[2] == "True"
json_file_path = sys.argv[3]

article_dict = {}

with gzip.open(tsv_file_path) as tsv_file:
    tsv_file.readline()

    experiment_types = set()

    for line in tsv_file:
        line_items = line.decode().rstrip("\n").split("\t")

        gse = line_items[0]
        title = line_items[1]
        summary = line_items[2]
        overall_design = line_items[3]
        experiment_type = line_items[4].lower().split("|")
        gpl = line_items[7].lower().split("|")
        gpl_title = line_items[8].lower()
        gpl_technology = line_items[9].lower()
        species = line_items[10].lower().split("|")
        taxon_id = line_items[11].split("|")
        superseries_gse = line_items[12]

        # We remove series that are part of a superseries because including these
        #   could cause bias in the machine-learning analysis. Additionally,
        #   if users find a relevant SuperSeries, they will find the associated
        #   SubSeries.
        if "homo sapiens" in species and "9606" in taxon_id and superseries_gse == "":
            if "expression profiling by array" in experiment_type:
                if "affymetrix" in gpl_title or "illumina" in gpl_title or "agilent" in gpl_title:
                    article_dict[gse] = clean_text(f"{title} {summary} {overall_design}", convert_to_lower=convert_to_lower)
            elif "expression profiling by high throughput sequencing" in experiment_type and "illumina" in gpl_title:
                article_dict[gse] = clean_text(f"{title} {summary} {overall_design}", convert_to_lower=convert_to_lower)

print(len(article_dict)) #48,893
with gzip.open(json_file_path, 'w') as json_file:
    json_file.write(json.dumps(article_dict).encode())
