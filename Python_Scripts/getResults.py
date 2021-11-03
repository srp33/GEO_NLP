from helper import *

resultPath = ["MultipartiteRank", "PositionRank", "SingleRank", "TfIdf", "TextRank", "TopicalPageRank", "YAKE", "KPMiner", "TopicRank"]
models = ["FTSkipGram", "FTCBOW", "FTWiki", "BioWordVec", "SpaCy", "SciSpaCy"]
numKeywords = [10,20,30]
bestCombo = "placeholder" #open('/home/jwengler/NLP_Paper/analysis/Data/Results/bestCombo.txt', 'w+')

for model in models:
    for numKeyword in numKeywords:
        generateResults(resultPath, model, bestCombo, numKeyword)

