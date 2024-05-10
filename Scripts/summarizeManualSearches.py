import glob
import gzip
import json
import os
import re
import sys

baseDirPath = sys.argv[1]
all_geo_tsv_file_path = sys.argv[2]
gemma_json_file_path = sys.argv[3]
out_summary_file_path = sys.argv[4]
out_nongemma_nonsubseries_file_path = sys.argv[5]

all_geo_dict = {}
with gzip.open(all_geo_tsv_file_path) as all_geo_tsv_file:
    all_geo_tsv_file.readline()

    for line in all_geo_tsv_file:
        line_items = line.rstrip(b"\n").split(b"\t")
        series = line_items[0].decode()
        title = line_items[1].decode()
        summary = line_items[2].decode()
        overall_design = line_items[3].decode()

        all_geo_dict[series] = [title, summary, overall_design]

with gzip.open(gemma_json_file_path) as gemma_file:
    gemmaSet = set(json.loads(gemma_file.read()).keys())

with gzip.open(out_summary_file_path, "w") as out_summary_file:
    with gzip.open(out_nongemma_nonsubseries_file_path, "w") as out_nongemma_nonsubseries_file:
        out_summary_file.write("Query\tSearch_Type\tTop_Num\tMetric\tValue\n".encode())
        out_nongemma_nonsubseries_file.write("Query\tSearch_Type\tSeries_ID\tSeries_Title\tSeries_Summary\tSeries_Overall_Design\tIn_Gemma\n".encode())

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

                        if len(geoIDs) > 0:
                            for numTop in [20, 50, 200, 500, 1000, len(geoIDs)]:
                                if len(geoIDs) < numTop:
                                    continue

                                matches = [x for x in geoIDs[:numTop] if x in queryGeoIDs]
                                numMatches = len(matches)

                                #precision = the fraction of retrieved documents that are relevant
#                                precision = numMatches / numTop

                                #recall = num relevant documents in top n / total relevant documents
                                recall = numMatches / numTagged

#                                if (precision + recall) == 0:
#                                    f1 = 0.0
#                                else:
#                                    f1 = 2 * ((precision * recall) / (precision + recall))

                                out_summary_file.write((f"{query}\t{searchType}\t{numTop}\tRecall\t{recall:.2f}\n").encode())
#                                out_summary_file.write((f"{query}\t{searchType}\t{numTop}\tPrecision\t{precision:.2f}\n").encode())
#                                out_summary_file.write((f"{query}\t{searchType}\t{numTop}\tF1 score\t{f1:.2f}\n").encode())

                        for geoID in geoIDs:
                            if geoID in all_geo_dict:
                                title = all_geo_dict[geoID][0]
                                summary = all_geo_dict[geoID][1]
                                overall_design = all_geo_dict[geoID][2]
                                inGemma = "Yes" if geoID in gemmaSet else "No"

                                out_nongemma_nonsubseries_file.write((f"{query}\t{searchType}\t{geoID}\t{title}\t{summary}\t{overall_design}\t{inGemma}\n").encode())

print(f"Saved to {out_summary_file_path} and {out_nongemma_nonsubseries_file_path}")
