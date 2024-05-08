import glob
import gzip
import json
import os
import re
import sys

baseDirPath = sys.argv[1]
gemma_json_file_path = sys.argv[2]
out_summary_file_path = sys.argv[3]
out_items_file_path = sys.argv[4]

with gzip.open(gemma_json_file_path) as gemma_file:
    gemmaSet = set(json.loads(gemma_file.read()).keys())

with gzip.open(out_summary_file_path, "w") as out_summary_file:
    with gzip.open(out_items_file_path, "w") as out_items_file:
        out_summary_file.write("Query\tSearch_Type\tTop_Num\tMetric\tValue\n".encode())
        out_items_file.write("Query\tSearch_Type\tSeries_ID\n".encode())

        for queryDirPath in glob.glob(f"{baseDirPath}/*"):
            query = os.path.basename(queryDirPath)

            queryFilePath = f"Queries/{query}"
            with open(queryFilePath) as queryFile:
                queryGeoIDs = queryFile.read().strip().split("\n")
                numTagged = len(queryGeoIDs)

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
                        #geoIDs = [x for x in geoIDs if x in gemmaSet]

                        if len(geoIDs) > 0:
                            for numTop in [20, 50, 200, 500, 1000, len(geoIDs)]:
                                if len(geoIDs) < numTop:
                                    continue

                                matches = [x for x in geoIDs[:numTop] if x in queryGeoIDs]
                                numMatches = len(matches)

                                #precision = the fraction of retrieved documents that are relevant
                                precision = numMatches / numTop

                                #recall = num relevant documents in top n / total relevant documents
                                recall = numMatches / numTagged

                                if (precision + recall) == 0:
                                    f1 = 0.0
                                else:
                                    f1 = 2 * ((precision * recall) / (precision + recall))

                                out_summary_file.write((f"{query}\t{searchType}\t{numTop}\tRecall\t{recall:.2f}\n").encode())
                                out_summary_file.write((f"{query}\t{searchType}\t{numTop}\tPrecision\t{precision:.2f}\n").encode())
                                out_summary_file.write((f"{query}\t{searchType}\t{numTop}\tF1 score\t{f1:.2f}\n").encode())

                        for geoID in geoIDs:
                            out_items_file.write((f"{query}\t{searchType}\t{geoID}\n").encode())

print(f"Saved to {out_summary_file_path} and {out_items_file_path}")
