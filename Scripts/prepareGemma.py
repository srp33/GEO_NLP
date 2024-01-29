import gzip
import json
import sys

all_geo_json_file_path = sys.argv[1]
gemma_txt_file_path = sys.argv[2]
gemma_json_file_path = sys.argv[3]

with gzip.open(all_geo_json_file_path) as the_file:
    all_geo_dict = json.loads(the_file.read())

with gzip.open(gemma_txt_file_path) as the_file:
    gemma_list = [line.decode().rstrip("\n") for line in the_file if line.rstrip(b"\n") != b""]

gemma_dict = {}
for series_id in gemma_list:
    gemma_dict[series_id] = all_geo_dict[series_id]

with gzip.open(gemma_json_file_path, "w") as the_file:
    the_file.write(json.dumps(gemma_dict).encode())
