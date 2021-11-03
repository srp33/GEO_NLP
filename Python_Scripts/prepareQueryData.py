from bs4 import BeautifulSoup
import fasttext
import json
import os
import pandas as pd
import random
import re
import requests
import sys

series = sys.argv[1].split(",")
queryID = sys.argv[2]

def cleanText(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http\S+', '', text)
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    text = text.replace(',', ' ')
    text = re.sub('\n', ' ', text)
    text = re.sub('[n|N]o\.', 'number', text)

    return text

namesFilePath = f"/Data/Queries/{queryID}/names.txt"

if os.path.exists(namesFilePath):
    sys.exit(0)

with open('/Data/allQueries.txt', 'a+') as allQueriesFile:
    for ser in series:
        strToWrite = f"{ser}\n"
        allQueriesFile.write(strToWrite)

with open(namesFilePath, "w") as nameFile:
    namesToQuery = []
    abstracts = []
    with open(f"/Data/starGEO.txt", 'a+') as corpusFile:
        for s in series:
            api_text = (f"http://stargeo.org/api/v2/series/{s}/")
            rq = requests.get(api_text).text
            data = json.loads(rq)
            temp_dict = data['attrs']
            name = data['gse_name']
            summary = temp_dict['summary']
            if summary not in abstracts:
                abstracts.append(summary)
                namesToQuery.append(name)

            abstracts.append(summary)
            title = temp_dict['title']
            summary = summary.replace("\n", " ")
            title = title.replace("\n", " ")
            corpusFile.write(cleanText(title))
            corpusFile.write(cleanText(summary))

            with open(f"/Data/Queries/{queryID}/{name}.txt", "w") as outFile:
                outFile.write(cleanText(title))
                outFile.write(' ')
                outFile.write(cleanText(summary))

    ranNames = random.choices(namesToQuery, k = int(len(series) / 2))
    with open(f"/Data/Queries/{queryID}/names_to_query.txt", "w") as nameQueryFile:
        for name in ranNames:
            nameQueryFile.write(name + ' ')

    for name in series:
        nameFile.write(name + ' ')
