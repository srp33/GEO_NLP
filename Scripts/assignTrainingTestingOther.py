import gzip
from helper import *
import json
import os
import random
import sys

gemma_json_file_path = sys.argv[1]
query_descriptor = sys.argv[2]
query_dir_path = sys.argv[3]
assignments_dir_path = sys.argv[4]
other_multiplication_rates = [int(x) for x in sys.argv[5].split(",")]

with gzip.open(gemma_json_file_path) as gemma_file:
    gemma_list = sorted(list(json.loads(gemma_file.read()).keys()))

with open(f"{query_dir_path}/{query_descriptor}") as query_file:
    query_list = query_file.read().rstrip("\n").split("\n")

random.seed(0)
random.shuffle(query_list)

# If there is an odd number of series, this code ensures that
# there is one more in training than in testing.
if len(query_list) % 2 == 0:
    training_series = query_list[:int(len(query_list) / 2)]
    testing_series = query_list[int(len(query_list) / 2):]
else:
    training_series = query_list[:(int(len(query_list) / 2) + 1)]
    testing_series = query_list[(int(len(query_list) / 2) + 1):]

os.makedirs(f"{assignments_dir_path}/{query_descriptor}", exist_ok=True)

with open(f"{assignments_dir_path}/{query_descriptor}/training_series", "w") as training_file:
    training_file.write(json.dumps(sorted(training_series)))

with open(f"{assignments_dir_path}/{query_descriptor}/testing_series", "w") as testing_file:
    testing_file.write(json.dumps(sorted(testing_series)))

# Find series that are in Gemma but not used for training or testing
other_set = set(gemma_list) - set(query_list)

with open(f"{assignments_dir_path}/{query_descriptor}/rest_of_gemma_all", "w") as other_file:
    other_file.write(json.dumps(sorted(list(other_set))))

for other_multiplication_rate in other_multiplication_rates:
    other_list = list(other_set)
    random.shuffle(other_list)

    # Which series do we want in the other group?
    num_other = len(testing_series) * other_multiplication_rate
    other_series = other_list[:num_other]

    with open(f"/Assignments/{query_descriptor}/rest_of_gemma_{other_multiplication_rate}", "w") as other_file:
        other_file.write(json.dumps(sorted(other_series)))
