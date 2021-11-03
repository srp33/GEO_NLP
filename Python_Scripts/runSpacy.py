from helper import *
import multiprocessing
import spacy
import sys
import time

numKeywords = int(sys.argv[1])
vectorSize = int(sys.argv[2])
maxCandidateArticles = int(sys.argv[3])
reducedSet = str(sys.argv[4])

printTimestamp("Getting candidate articles")
if reducedSet=='true':
    candidate_articles = getCandidateArticles(maxCandidateArticles, True)
else:
    candidate_articles = getCandidateArticles(maxCandidateArticles, False)

printTimestamp("Loading Spacy")
model = spacy.load("en_core_web_lg")

start = time.time()

for query in range(6,7):
    for keywordExtractor in ["TopicRank", "TfIdf", "KPMiner", "YAKE", "TextRank", "SingleRank", "TopicalPageRank", "PositionRank", "MultipartiteRank"]:
        mp = multiprocessing.Process(target=findSimilarity, args=(keywordExtractor, "SpaCy", model, candidate_articles, query, numKeywords, vectorSize))
        mp.start()
        #mp.join()

end = time.time()
print('{:.4f} s'.format(end - start))

