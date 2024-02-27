import glob
import gzip
from helper import *
from sklearn.metrics import precision_recall_curve, auc
import sys
import os

similarities_dir_path = sys.argv[1]
metrics_dir_path = sys.argv[2]

metrics_file_path = f"{metrics_dir_path}/Results.tsv.gz"

with gzip.open(metrics_file_path, "w") as metrics_file:
    metrics_file.write((f"Query\tMethod\tMultiplication_Rate\tMetric\tValue\n").encode())

    for similarities_file_path in glob.glob(f"{similarities_dir_path}/*/*/*"):
        print(f"Calculating metrics for {similarities_file_path}")

        file_path_items = similarities_file_path.split("/")
        query = file_path_items[1]
        method = file_path_items[2].replace("____", "/")
        multiplication_rate = file_path_items[3].replace("rest_of_gemma_", "")

        groups_scores = []

        with open(similarities_file_path) as similarities_file:
            similarities_file.readline()

            for line in similarities_file:
                line_items = line.rstrip("\n").split("\t")
                #series = line_items[0]
                group = line_items[1]
                score = float(line_items[2])

                groups_scores.append([group, score])

        groups_scores = sorted(groups_scores, key=lambda x: x[1])
        groups = [x[0] for x in groups_scores]
        groups = [1 if group == "Testing" else 0 for group in groups]
        scores = [x[1] for x in groups_scores]

        # Calculate the area under the precision-recall curve
        precision, recall, _ = precision_recall_curve(groups, scores)
        auc_prc = auc(recall, precision)

        metrics_file.write((f"{query}\t{method}\t{multiplication_rate}\tAUC_PRC\t{auc_prc}\n").encode())

        # Calculate proportions in top N
        groups_scores = sorted(groups_scores, key=lambda x: x[1], reverse=True)
        num_testing = sum(groups)

        for n in [20, 50, 200, 500, 1000]:
            count = 0
            for group_score in groups_scores[:n]:
                if group_score[0] == "Testing":
                    count += 1

            if num_testing > n:
                recall = "NA"
            else:
                recall = f"{round(count / num_testing, 2)}"

            metrics_file.write((f"{query}\t{method}\t{multiplication_rate}\tRecall_Top_{n}\t{recall}\n").encode())

print(f"Saved metrics to {metrics_file_path}")
