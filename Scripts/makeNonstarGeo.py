import json
import sys

star_geo_file_path = sys.argv[1]
all_geo_file_path = sys.argv[2]

with open(star_geo_file_path) as star_file:
    star_list = json.loads(star_file.read())
    star_set = set(star_list)

with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())


with open("/Data/AllGeo.csv", "w") as write_file:
    write_file.write("Series,Text\n")


#making all star geo!
for series, text in all_dict.items():
    with open("/Data/AllGeo.csv", "a") as write_file:
        write_file.write(f"{series},{all_dict[series]}\n")






sys.exit()
# with open("/Data/NonstarGeo.csv", "w") as write_file:
#             write_file.write("Series,Text\n")

# for series, text in all_dict.items():
#     if series not in star_set:
#         with open("/Data/NonstarGeo.csv", "a") as write_file:
#             write_file.write(f"{series},{all_dict[series]}\n")



#make training doc as well!
train_df = []
valid_df = []
all_pairs = []
for query in ["q1", "q3", "q5"]:
    with open(f"Data/{query}/training_series", "r") as read_file:
        series_list = json.loads(read_file.read())
        for query in ["q2", "q4", "q6"]:
            with open(f"Data/{query}/training_series", "r") as other_file:
                other_series_list = json.loads(other_file.read())
                for i, series in enumerate(series_list):
                    #creating pairs of dissimilar abstracts
                    for other_series in other_series_list:
                        tmp_dict = {}
                        tmp_dict['sentence1'] = all_dict[series]
                        tmp_dict['sentence2'] = all_dict[other_series]
                        tmp_dict['similarity'] = 0
                            
                        if i % 2 == 0:
                            train_df.append(tmp_dict)
                        else:
                            valid_df.append(tmp_dict)
                    #creating pairs of similar abstracts        
                    if (i+1) < len(series_list):
                        for second_series in series_list[i:]:
                            tmp_dict = {}
                            tmp_dict['sentence1'] = all_dict[series]
                            tmp_dict['sentence2'] = all_dict[second_series]
                            tmp_dict['similarity'] = 1
                                
                            if i % 2 == 0:
                                train_df.append(tmp_dict)
                            else:
                                valid_df.append(tmp_dict)
                            all_pairs.append(tmp_dict)

print(all_pairs)
print(len(all_pairs))
# with open("/Data/STARpairs.json", "a") as write_file:
#     for dictionary in all_pairs:
#         write_file.write(json.dumps(dictionary))
