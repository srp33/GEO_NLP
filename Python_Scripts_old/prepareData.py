from helper import *
import sys

num_keywords = int(sys.argv[1])
reduced_set = str(sys.argv[2])

max_candidate_articles = int(sys.argv[1])
print_timestamp("Getting candidate articles")
if reduced_set=='true':
    candidate_articles = get_candidate_articles(max_candidate_articles, True)
else:
    candidate_articles = get_candidate_articles(max_candidate_articles, False)
