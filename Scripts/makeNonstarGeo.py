import gzip
import json
import sys

star_geo_file_path = sys.argv[1]
all_geo_file_path = sys.argv[2]
non_star_geo_file_path = sys.argv[3]

with gzip.open(star_geo_file_path) as star_file:
    star_list = json.loads(star_file.read())
    star_set = set(star_list)

with gzip.open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

non_star_set = all_dict.keys() - star_set
non_star_list = sorted(list(non_star_set))

print(len(non_star_list)) # 71958
with gzip.open(non_star_geo_file_path, "w") as write_file:
    write_file.write(json.dumps(non_star_list).encode())
