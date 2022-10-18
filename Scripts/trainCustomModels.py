#from gensim.models.fasttext import FastText as FT_gensim
#from gensim.summarization import keywords
from helper import *
import fasttext
import sys

star_geo_file_path = sys.argv[1]
all_geo_file_path = sys.argv[2]
num_keyword_options = [int(x) for x in sys.argv[3].split(",")]
num_keyword_options.append("full_text")

out_dir_path = sys.argv[4]

nonstar_text_list = []
corpus_file_path = "/Models/corpus_file.txt"

with open(star_geo_file_path) as star_file:
    star_list = json.loads(star_file.read())
    star_set = set(star_list)

with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

for series in all_dict:
    if series not in star_set:
        nonstar_text_list.append(series)

extraction_methods = get_list_extractors()

#for extraction_method in extraction_methods:
#    for num_keywords in num_keyword_options:
num_keywords = "full_text"
for extraction_method in extraction_methods:
    for model_type in ["fasttext__cbow", "fasttext__skipgram"]:
        if model_type.startswith("fasttext"):
            model_name = model_type.split("__")[1] 
        #don't want to remake this if it already exists. 
        new_path = f"Models/custom/{model_type}/{extraction_method}/{num_keywords}.bin"
        if not os.path.exists(new_path):
            with open(corpus_file_path, "w") as corpus_file: 
                counter = 0
                for series in nonstar_text_list:
                    if num_keywords == "full_text":
                        text = all_dict[series]
                        corpus_file.write(f"{text}/n")
                        continue
                    else:
                        counter += 1

                        keyword_text = get_keywords(extraction_method, num_keywords, series)

                    if counter > 1:
                        corpus_file.write("\n")
                        corpus_file.write(f"{keyword_text}")
                
            model = fasttext.train_unsupervised(corpus_file_path, model_name, dim=300)
            model.save_model(new_path)


                #     counter = 0
                #     for series in nonstar_text_list:
                #         text = all_dict[series]
                #         if counter > 1:
                #             corpus_file.write("\n")
                #         corpus_file.write(f"{text}")
                #         counter += 1
                # model = fasttext.train_unsupervised(corpus_file_path, model_name, dim=300)
                # model.save_model(new_path)

#When we had keyword extraction as a factor we used the code below. TODO: figure out if we should remove or keep this
        #https://github.com/facebookresearch/fastText/tree/main/python#train_unsupervised-parameters