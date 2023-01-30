import json
import sys

star_geo_file_path = sys.argv[1]
all_geo_file_path = sys.argv[2]

with open(star_geo_file_path) as star_file:
    star_list = json.loads(star_file.read())
    star_set = set(star_list)

with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

with open("/Data/NonstarGeo.csv", "w") as write_file:
    write_file.write("Series,Text\n")

for series, text in all_dict.items():
    if series not in star_set:
        with open("/Data/NonstarGeo.csv", "a") as write_file:
            write_file.write(f"{series},{all_dict[series]}\n")