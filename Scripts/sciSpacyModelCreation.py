from gettext import npgettext
from helper import *
import multiprocessing
import scispacy
import spacy
import sys
import os

all_json_file_path = sys.argv[1]
with open(all_json_file_path) as all_file:
    all_dict = json.loads(all_file.read())

nlp = spacy.load("en_core_sci_lg")

def findSimilarity(series_1, list_training):
    score = 0
    for training in list_training:
        doc1 = nlp(series_1)
        doc2 = nlp(training)
        score += doc1.similarity(doc2)
    average_similarity = score / len(list_training)
    return(average_similarity)

def sciSpacy_similarity(num_keywords, keyword_extractor_name, query, other_multiplication_rate):
    results_dir_path = f"/Results/{query}/SciSpacy/{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)
    #if not os.path.exists(results_dir_path):
    if 1 == 1: #TODO: get rid of later, just didn't want to change indentation again temporarily
        with open(f'{results_dir_path}/similarity.tsv', "w") as result_file:
                
            training_series_list = get_series_identifiers(query, "training_series")
            training_list = []
            for series in training_series_list:
                if num_keywords == 'full_text':
                    keywords = all_dict[series]
                else:
                    keywords = get_keywords(keyword_extractor_name, num_keywords, series)
                if keywords == "":
                    continue
                training_list.append(keywords)


            list_of_ids = get_series_identifiers(f"{query}/other_series", other_multiplication_rate)
            list_of_ids.extend(get_series_identifiers(query, "testing_series"))
            list_of_test_ids = get_series_identifiers(query, "testing_series")

            failed_test = []
            gse_with_score = []
            for testing_and_other_series_id in list_of_ids:
                if num_keywords == 'full_text':
                    keywords = all_dict[testing_and_other_series_id]
                else:
                    keywords = get_keywords(keyword_extractor_name, num_keywords, testing_and_other_series_id)
                if keywords == "":
                    failed_test.append(testing_and_other_series_id)
                    continue
                similarity = findSimilarity(keywords, training_list)
                if similarity == "nan":
                    failed_test.append(testing_and_other_series_id)
                else:
                    gse_with_score.append([similarity, testing_and_other_series_id])
                
            gse_with_score.sort()
            gse_with_score.reverse()
            result_file.write("Series ID\tSimilarity Score\tTest or Other Group\n")
                    
            for series in gse_with_score:
                test_or_other= ""
                if series[1] in list_of_test_ids:
                    test_or_other = "Test"
                else:
                    test_or_other = "Other"
                result_file.write(f"{series[1]}\t{series[0]}\t{test_or_other}\n")

            for series in failed_test:
                test_or_other= ""
                if series in list_of_test_ids:
                    test_or_other = "Test"
                else:
                    test_or_other = "Other"
                result_file.write(f"{series[1]}\t{series[0]}\t{test_or_other}\n")
    return
#writing out my thought process here
#TODO: ask Piccolo if I need num keywords or if i can just comapre the whole thing. 
#For each query, numkeyword, multiplication rate, and extraction method:
#I need to get lists of the training, test, and other series
#I need to sum the cosine similarity between each other/test with training and divide by
#the number of training series
#I need to save that to a list of lists that I then sort/rank
#Save that to a file
#Make sure getResults.py can pick that up.

num_keywords_list = [2,4,8,16,32,'full_text']
for query in ["q1", "q2", "q3", "q4", "q5", "q6"]:
    for num_keywords in num_keywords_list:
        for keyword_extractor_name in ["KPMiner", "Baseline"]:
            for other_multiplication_rate in [1,2]:
                mp = multiprocessing.Process(target=sciSpacy_similarity, args=(num_keywords, keyword_extractor_name, query, other_multiplication_rate))
                mp.start()






# for query in range(1,7):
#     for keywordExtractor in [ "KPMiner", "Baseline"]:
#         mp = multiprocessing.Process(target=findSimilarity, args=(keywordExtractor, "SpaCy", model, candidate_articles, query, num_keywords, vector_size))
#         mp.start()
#         mp.join()
    #print(np.dot(doc1.vector, doc2.vector) / (np.linalg.norm(doc1.vector) * np.linalg.norm(doc2.vector)))




    #from article def cosine_distance_wordembedding_method(s1, s2):
    # import scipy
    # vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
    # vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
    # cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    # print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
