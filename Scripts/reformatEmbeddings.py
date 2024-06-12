import gzip
import json
import sys

in_json_file_path = sys.argv[1]
out_tsv_file_path = sys.argv[2]

with gzip.open(in_json_file_path) as in_json_file:
    json_dict = json.loads(in_json_file.read())

with gzip.open(out_tsv_file_path, 'w') as out_tsv_file:
    for gse in sorted(json_dict):
        print(gse)
        out_list = [gse] + [str(x) for x in json_dict[gse]]
        out_tsv_file.write(("\t".join(out_list) + "\n").encode())
