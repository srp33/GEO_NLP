import glob
import gzip
import json
import os
import sys

gemma_overlap_file_path = sys.argv[1]
query_descriptor = sys.argv[2]
assignments_dir_path = sys.argv[3]
similarities_dir_path = sys.argv[4]

with open(f"{assignments_dir_path}/{query_descriptor}/training_series") as the_file:
    training_series = set(json.loads(the_file.read()))

with open(f"{assignments_dir_path}/{query_descriptor}/testing_series") as the_file:
    testing_series = set(json.loads(the_file.read()))

for assignments_file_path in sorted(glob.glob(f"{assignments_dir_path}/{query_descriptor}/rest_of_gemma_*")):
    with open(assignments_file_path) as the_file:
        other_series = set(json.loads(the_file.read()))

    os.makedirs(f"{similarities_dir_path}/{query_descriptor}/word_overlap", exist_ok=True)

    with gzip.open(gemma_overlap_file_path) as the_file:
        the_file.readline()

        out_file_path = f"{similarities_dir_path}/{query_descriptor}/word_overlap/{os.path.basename(assignments_file_path)}"
        print(f"Preparing to save to {out_file_path}")

        with open(out_file_path, "w") as out_file:
            out_file.write(f"Series\tGroup\tScore\n")

            score_dict = {}

            for line in the_file:
                line_items = line.decode().rstrip("\n").split("\t")
                series_A = line_items[0]
                series_B = line_items[1]
                overlap = float(line_items[3])

                if series_A in training_series and (series_B in testing_series or series_B in other_series):
                    score_dict[series_B] = score_dict.setdefault(series_B, []) + [overlap]

            non_series = (testing_series | other_series) - score_dict.keys()

            for series, overlaps in score_dict.items():
                group = "Testing" if series in testing_series else "Other"
                mean_overlap = sum(overlaps) / len(overlaps)

                out_file.write((f"{series}\t{group}\t{mean_overlap}\n"))
