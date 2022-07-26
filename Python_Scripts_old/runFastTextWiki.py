from helper import *
import multiprocessing
import fasttext.util
import sys
import time

num_keywords = int(sys.argv[1])
vector_size = int(sys.argv[2])
max_candidate_articles = int(sys.argv[3])
reduced_set = str(sys.argv[4])
print_timestamp("Getting candidate articles")
if reduced_set=='true':
    candidate_articles = get_candidate_articles(max_candidate_articles, True)
else:
    candidate_articles = get_candidate_articles(max_candidate_articles, False)
print_timestamp("Loading FasTextWiki")
fasttext.util.download_model('en', if_exists='ignore')
model = fasttext.load_model('cc.en.300.bin')

start = time.time()

for query in range(6,7):
    for keywordExtractor in ["TopicRank", "TfIdf", "KPMiner", "YAKE", "TextRank", "SingleRank", "TopicalPageRank", "PositionRank", "MultipartiteRank"]:
        mp = multiprocessing.Process(target=findSimilarity, args=(keywordExtractor, "FTWiki", model, candidate_articles, query, num_keywords, vector_size))
        mp.start()
        #mp.join()

end = time.time()
print('{:.4f} s'.format(end - start))


