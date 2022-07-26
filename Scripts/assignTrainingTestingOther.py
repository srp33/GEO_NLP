from helper import *
import json
import random
import sys

star_file_path = sys.argv[1]
all_geo_file_path = sys.argv[2]
series = sys.argv[3].split(",")
query_id = sys.argv[4]
other_multiplication_rates = [int(x) for x in sys.argv[5].split(",")]

all_dict = {}
star_list = []

with open(star_file_path) as star_file:
    star_list = json.loads(star_file.read())

with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

PERCENT_SHARED_WORDS_THRESHOLD = .75 

unique_series = []
list_of_unique_sets = []

for i, s in enumerate(series):
    title_and_abstract = all_dict[s]
    title_and_abstract = clean_text(title_and_abstract)
    new_series_set = set(title_and_abstract.split(" "))
    new_series_set = new_series_set - {''}
        
    #The first series should be unique. 
    if i == 0:
        list_of_unique_sets.append(new_series_set)
        unique_series.append(s)
        continue

    percentages = []

    for i, series_set in enumerate(list_of_unique_sets):

        common_words = series_set & new_series_set
        all_words = series_set | new_series_set
        percent_shared_words = len(common_words) / len(all_words)
        percentages.append(percent_shared_words)

    if max(percentages) < PERCENT_SHARED_WORDS_THRESHOLD:
        list_of_unique_sets.append(new_series_set)
        unique_series.append(s)

random.seed(0)
random.shuffle(unique_series)

# If there is an odd number of series, this code ensures that
# there is one more in training than in testing.
if len(unique_series) % 2 == 0:
    training_series = unique_series[:int(len(unique_series) / 2)]
    testing_series = unique_series[int(len(unique_series) / 2):]
else:
    training_series = unique_series[:(int(len(unique_series) / 2) + 1)]
    testing_series = unique_series[(int(len(unique_series) / 2) + 1):]

with open(f"/Data/{query_id}/training_series", "w") as training_file:
    training_file.write(json.dumps(training_series))

with open(f"/Data/{query_id}/testing_series", "w") as testing_file:
    testing_file.write(json.dumps(testing_series))

# Find the series that are in STARGEO but not used for training or testing
other_candidates = set(star_list) - set(training_series) - set(testing_series)

for other_multiplication_rate in other_multiplication_rates:
    other_candidates_tmp = list(other_candidates)
    random.shuffle(other_candidates_tmp)

    # How many series do we want in the other group?
    num_other = len(testing_series) * other_multiplication_rate

    other_series = other_candidates_tmp[:num_other]

    out_dir_path = f"/Data/{query_id}/other_series"
    Path(out_dir_path).mkdir(parents=True, exist_ok=True)

    with open(f"{out_dir_path}/{other_multiplication_rate}", "w") as other_file:
        other_file.write(json.dumps(other_series))