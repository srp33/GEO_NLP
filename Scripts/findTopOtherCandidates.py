import glob
import gzip
import os
import sys

similarity_files_pattern = sys.argv[1]
out_file_path = sys.argv[2]

with gzip.open(out_file_path, "w") as out_file:
    out_file.write((f"Model\tQuery\tSeries\tScore\n").encode())

    for similarity_files_path in glob.glob(similarity_files_pattern):
        with open(similarity_files_path) as similarity_file:
            model = os.path.basename(os.path.dirname(similarity_files_path))
            query = os.path.basename(os.path.dirname(os.path.dirname(similarity_files_path)))

            #Series	Group	Score
            similarity_file.readline()

            similarity_dict = {}
            for line in similarity_file:
                line_items = line.rstrip("\n").split("\t")
                series = line_items[0]
                group = line_items[1]
                score = float(line_items[2])

                if group != "Other":
                    continue

                similarity_dict[score] = series

            count = 0
            for score, series in sorted(similarity_dict.items(), reverse=True):
                count += 1

                if count > 5:
                    break

                out_file.write((f"{model}\t{query}\t{series}\t{score}\n").encode())

print(f"Saved results to {out_file_path}")
