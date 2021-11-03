from helper import *
import sys

numKeywords = int(sys.argv[1])
reducedSet = str(sys.argv[2])

maxCandidateArticles = int(sys.argv[1])
printTimestamp("Getting candidate articles")
if reducedSet=='true':
    candidate_articles = getCandidateArticles(maxCandidateArticles, True)
else:
    candidate_articles = getCandidateArticles(maxCandidateArticles, False)
