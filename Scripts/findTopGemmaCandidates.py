import glob
import gzip
import os
import sys

similarity_files_pattern = sys.argv[1]
all_geo_tsv_file_path = sys.argv[2]
out_file_path = sys.argv[3]

all_geo_dict = {}
with gzip.open(all_geo_tsv_file_path) as all_geo_tsv_file:
    all_geo_tsv_file.readline()

    for line in all_geo_tsv_file:
        line_items = line.rstrip(b"\n").split(b"\t")
        series = line_items[0].decode()
        title = line_items[1].decode()
        summary = line_items[2].decode()
        overall_design = line_items[3].decode()

        all_geo_dict[series] = [title, summary, overall_design]

with gzip.open(out_file_path, "w") as out_file:
    out_file.write((f"Model\tQuery\tSeries\tSeries_Title\tSeries_Summary\tSeries_Overall_Design\tScore\n").encode())

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

                title = all_geo_dict[series][0]
                summary = all_geo_dict[series][1]
                overall_design = all_geo_dict[series][2]

                out_file.write((f"{model}\t{query}\t{series}\t{title}\t{summary}\t{overall_design}\t{score}\n").encode())

print(f"Saved results to {out_file_path}")
