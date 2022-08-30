from helper import *
import multiprocessing
import os
import sys

all_geo_file_path = sys.argv[1]
queries = sys.argv[2].split(",")
multiplication_rate_options = [int(x) for x in sys.argv[3].split(",")]
max_keywords = max([int(x) for x in sys.argv[4].split(",")][:-1])

# Keys are series IDs, values are text from GEO
all_geo_dict = {}
with open(all_geo_file_path) as cache_file:
    all_geo_dict = json.loads(cache_file.read())

def extract_keywords(geo_series_id, max_num_keywords=32):  
    # Check whether the specified path exists or not
    path = f"/Data/KeywordWeights/{geo_series_id}"
    if not os.path.exists(path):
        
        title_and_abstract = all_geo_dict[geo_series_id]
        extractor_dict = {}
    
        #Extracting Baseline
        unique_words = extract_keywords_baseline(title_and_abstract, max_num_keywords)
        word_list = []
        for i, word in enumerate(unique_words):
            word = [word, (max_num_keywords - i) / max_num_keywords]
            word_list.append(word)
        extractor_dict["Baseline"] = word_list
    
        #Using Keyword extraction methods
        for extractor_name, extraction in get_keyword_extractors().items():
            extraction.load_document(input=title_and_abstract, language='en')
            extraction.candidate_selection()
            extraction.candidate_weighting()

            keywords = extraction.get_n_best(n=max_num_keywords)
                     
            extractor_dict[extractor_name] = keywords
 
    
        with open(f"/Data/KeywordWeights/{geo_series_id}", "w") as geo_file:
            geo_file.write(json.dumps(extractor_dict))

    return

args=[(series, 32) for series in all_geo_dict]
pool = multiprocessing.Pool(processes = 12)
pool.starmap(extract_keywords, args)
pool.close()
