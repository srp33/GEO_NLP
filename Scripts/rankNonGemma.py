import gzip
import os
import sys

distances_file_path = sys.argv[1]
query_descriptor = sys.argv[2]
method_descriptor = sys.argv[3]
queries_dir_path = sys.argv[4]
similarities_dir_path = sys.argv[5]

out_dir_path = f"{similarities_dir_path}/{query_descriptor}/{method_descriptor}"
out_file_path = f"{out_dir_path}/all"

if os.path.exists(out_file_path):
    print(f"{out_file_path} already exists.", flush=True)
    sys.exit(0)
else:
    print(f"Saving to {out_file_path}", flush=True)

os.makedirs(out_dir_path, exist_ok=True)

query_series_file_path = f"{queries_dir_path}/{query_descriptor}"

with open(query_series_file_path) as the_file:
    query_series = set([x.rstrip("\n") for x in the_file if len(x) > 0])

with gzip.open(distances_file_path) as distances_file:
    distances_file.readline()

    score_dict = {}

    for line_number, line in enumerate(distances_file):
        if line_number > 0 and line_number % 1000000 == 0:
            print(line_number, flush=True)

        line_items = line.decode().rstrip("\n").split("\t")
        series_A = line_items[0]
        series_B = line_items[1]
        overlap = float(line_items[3])

        if series_A in query_series:
            score_dict[series_B] = score_dict.setdefault(series_B, []) + [overlap]

mean_score_dict = {}
for series, overlaps in score_dict.items():
    mean = sum(overlaps) / len(overlaps)
    mean_score_dict[mean] = series

with open(out_file_path, "w") as out_file:
    out_file.write(f"Series\tMean_Score\n")

    for mean_score, series in sorted(mean_score_dict.items(), reverse=True):
        out_file.write((f"{series}\t{mean_score}\n"))
