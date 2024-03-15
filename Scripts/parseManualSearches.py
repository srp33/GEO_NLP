import glob
import os
import re
import sys

baseDirPath = sys.argv[1]

for queryDirPath in glob.glob(f"{baseDirPath}/*"):
    for queryNumFilePath in glob.glob(f"{queryDirPath}/*"):
        queryNum = os.path.basename(queryNumFilePath)

        searchResultsFilePath = f"{queryNumFilePath}/search_results.txt"

        with open(searchResultsFilePath) as searchFile:
            searchText = searchFile.read().strip()

            if len(searchText) > 0:
                geoIDs = re.findall(r"Accession: GSE\d+", searchText)
                print(geoIDs)
                break
