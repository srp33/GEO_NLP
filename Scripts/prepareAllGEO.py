from helper import *
import os
import sys

tsv_file_path = sys.argv[1]
json_file_path = sys.argv[2]

#gse	title	summary
#GSE1	NHGRI_Melanoma_class	This series represents a group of cutaneous malignant melanomas and unrelated controls which were clustered based on correlation coefficients calculated through a comparison of gene expression;	profiles.;	Keywords: other

if not os.path.exists(json_file_path):
    article_dict = {}

    with open(tsv_file_path) as tsv_file:
        tsv_file.readline()

        for line in tsv_file:
            line_items = line.rstrip("\n").split("\t")

            gse = line_items[0]
            title = line_items[1]
            summary = line_items[2]

            text = clean_text(title + ' ' + summary)
            article_dict[gse] = text

    with open(json_file_path, 'w') as json_file:
        json_file.write(json.dumps(article_dict))
