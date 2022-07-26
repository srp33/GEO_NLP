from Scripts.runModels import find_similarity
from helper import *
import multiprocessing
import spacy
import sys
import time

num_keywords = int(sys.argv[1])
vector_size = int(sys.argv[2])
query = sys.argv[3]

print_time_stamp("Getting candidate articles")
if reduced_set=='true':
    candidate_articles = get_candidate_articles(max_candidate_articles, True)
else:
    candidate_articles = get_candidate_articles(max_candidate_articles, False)

print_timestamp("Loading SciSpacy")
model = spacy.load("en_ner_bc5cdr_md")

start = time.time()

for query in range(6,7):
    for keyword_extractor in ["TopicRank", "TfIdf", "KPMiner", "YAKE", "TextRank", "SingleRank", "TopicalPageRank", "PositionRank", "MultipartiteRank"]:
        mp = multiprocessing.Process(target=find_similarity, args=(keyword_extractor, "SciSpaCy", model, candidate_articles, query, num_keywords, vector_size))
        mp.start()
        #mp.join()

end = time.time()
print('{:.4f} s'.format(end - start))
