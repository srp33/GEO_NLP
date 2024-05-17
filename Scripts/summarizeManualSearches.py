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
out_nonsubseries_file_path = sys.argv[5]

all_geo_dict = {}
with gzip.open(all_geo_tsv_file_path) as all_geo_tsv_file:
    all_geo_tsv_file.readline()

    for line in all_geo_tsv_file:
        line_items = line.decode().rstrip("\n").split("\t")

        series = line_items[0]
        title = line_items[1]
        summary = line_items[2]
        overall_design = line_items[3]
        experiment_type = line_items[4].lower().split("|")
        gpl = line_items[5].lower().split("|")
        gpl_title = line_items[6].lower()
        gpl_technology = line_items[7].lower()
        species = line_items[8].lower().split("|")
        taxon_id = line_items[9].split("|")
        superseries_gse = line_items[10]

        # This repeats logic that we already had in our manual search.
        # But we do it to ensure consistency with the automated queries.
        if "homo sapiens" in species and "9606" in taxon_id and superseries_gse == "":
            if "expression profiling by array" in experiment_type:
                if "affymetrix" in gpl_title or "illumina" in gpl_title or "agilent" in gpl_title:
                    all_geo_dict[series] = [title, summary, overall_design]
            elif "expression profiling by high throughput sequencing" in experiment_type and "illumina" in gpl_title:
                all_geo_dict[series] = [title, summary, overall_design]

with gzip.open(gemma_json_file_path) as gemma_file:
    gemmaSet = set(json.loads(gemma_file.read()).keys())

with gzip.open(out_summary_file_path, "w") as out_summary_file:
    with gzip.open(out_nonsubseries_file_path, "w") as out_nonsubseries_file:
        out_summary_file.write("Query\tSearch_Type\tTop_Num\tMetric\tValue\n".encode())
        out_nonsubseries_file.write("Query\tSearch_Type\tRank\tSeries_ID\tSeries_Title\tSeries_Summary\tSeries_Overall_Design\tIn_Gemma\tMatches_Gemma\n".encode())

        for queryDirPath in glob.glob(f"{baseDirPath}/*"):
            query = os.path.basename(queryDirPath)

            testSeriesFilePath = f"Assignments/{query}/testing_series"
            with open(testSeriesFilePath) as testSeriesFile:
                testGeoIDs = json.loads(testSeriesFile.read())

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
                        geoIDs = [geoID for geoID in geoIDs if geoID in all_geo_dict]
                        geoIDsInGemma = [geoID for geoID in geoIDs if geoID in gemmaSet]

                        if len(geoIDsInGemma) > 0:
                            for numTop in [5, 10, 20, 50, 100, 200, 500, 1000]:
                                if len(geoIDsInGemma) < numTop:
                                    continue

                                matches = [x for x in geoIDsInGemma[:numTop] if x in testGeoIDs]
                                numMatches = len(matches)

                                recall = numMatches / len(testGeoIDs)

                                out_summary_file.write((f"{query}\t{searchType}\t{numTop}\tRecall\t{recall:.2f}\n").encode())

                        rank = 0
                        for geoID in geoIDs:
                            if geoID in all_geo_dict:
                                rank += 1
                                title = all_geo_dict[geoID][0]
                                summary = all_geo_dict[geoID][1]
                                overall_design = all_geo_dict[geoID][2]
                                inGemma = "Yes" if geoID in gemmaSet else "No"
                                matchesGemma = "Yes" if geoID in testGeoIDs else "No"

                                out_nonsubseries_file.write((f"{query}\t{searchType}\t{rank}\t{geoID}\t{title}\t{summary}\t{overall_design}\t{inGemma}\t{matchesGemma}\n").encode())

print(f"Saved to {out_summary_file_path} and {out_nonsubseries_file_path}")
