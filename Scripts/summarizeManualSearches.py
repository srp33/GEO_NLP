import glob
import gzip
import json
import os
import re
import sys

baseDirPath = sys.argv[1]
gemma_json_file_path = sys.argv[2]
out_file_path = sys.argv[3]

with gzip.open(gemma_json_file_path) as gemma_file:
    gemmaSet = set(json.loads(gemma_file.read()).keys())

with gzip.open(out_file_path, "w") as out_file:
    out_file.write("Query\tSearch_Type\tNum_Top\tRecall\n".encode())

    for queryDirPath in glob.glob(f"{baseDirPath}/*"):
        query = os.path.basename(queryDirPath)

        queryFilePath = f"Queries/{query}"
        with open(queryFilePath) as queryFile:
            queryGeoIDs = queryFile.read().strip().split("\n")

        for searchNumFilePath in glob.glob(f"{queryDirPath}/*"):
            searchNum = os.path.basename(searchNumFilePath)

            if searchNum == "1":
                searchType = "Tag ontology term"
            elif searchNum == "2":
                searchType = "Tag ontology term plus synonyms"
            else:
                searchType = "MeSH term"

            searchResultsFilePath = f"{searchNumFilePath}/search_results.txt"

            with open(searchResultsFilePath) as searchFile:
                searchText = searchFile.read().strip()

                if len(searchText) > 0:
                    geoIDs = re.findall(r"Accession: (GSE\d+)", searchText)
                    geoIDsInGemma = [x for x in geoIDs if x in gemmaSet]

                    for numTop in [20, 50, 200, 500, 1000]:
                        if len(geoIDsInGemma) < numTop:
                            continue

                        matches = [x for x in geoIDsInGemma[:numTop] if x in queryGeoIDs]
                        numMatches = len(matches)
                        recall = numMatches / numTop

                        out_file.write((f"{query}\t{searchType}\t{numTop}\t{recall}\n").encode())

print(f"Saved to {out_file_path}")
