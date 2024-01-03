# GEO_NLP

## Overview

In this project, we aim to perform a systematic benchmark comparison of various Natural Language Processing (NLP) models on a large public data repository: the Gene Expression Omnibus (GEO). Three categories of models are implemented:
1. Models pretrained on a large corpus of text that is not specifically related to biology.
2. Models pretrained on a large corpus of text that is specifically related to biology.
3. Models created from out-of-sample domain specific text. In this case, text not used for training or testing from the Gene Expression Omnibus website.

Without specific fine-tuning or further optimization, these models are used out of the box for a text ranking task. The embeddings of the titles and abstracts of GEO series are compared using cosine similarity (vector dot product method) to determine similarities. 

## Docker 
For user ease, a Dockerfile has been created. The docker is run with the following command:
```bash
sh run_docker.sh
```
It will execute all of the following steps, without the user needing to worry about package requirements. 

## 1. Download Titles and Abstracts from GEO

```bash
# Run Rscript to download Titles and Abstracts
Rscript getAllGEO.R "$all_geo_tsv_file_path"

# Run Python script to clean the data
python3 prepareAllGEO.py "$all_geo_tsv_file_path" "$all_geo_json_file_path"
```

## 2. Identify GEO Series annotated on STARGEO 
```bash
# Download the GSE IDs of series annotated on STARGEO
python3 prepareStarGEO.py "$star_geo_file_path" "$all_geo_json_file_path"
```

## 3. Train custom cbow and skipgram fasttext models
```bash
# Run Python script to train custom FastText models
python3 trainCustomModels.py "$star_geo_file_path" "$all_geo_json_file_path" "$num_keyword_options" "Models/custom"
```
## 4. Create BERT model from series information not in STARGEO
```bash
python3 makeNonstarGeo.py "$star_geo_file_path" "$all_geo_json_file_path"
python3 trainBert.py
```
## 5. Assign Training and Testing Groups for Queries
```bash
#For each query, given the IDs that are relevant, split them into Training and Testing groups
python3 assignTrainingTestingOther.py "$star_geo_file_path" "$all_geo_json_file_path" "GSE47860,GSE47861,GSE49481,GSE50567,GSE17072,GSE40115" q1 "$multiplication_rates"
```
## 6. Calculate the cosine similarities
```bash
# Run Python script to calculate cosine similarities
python3 calculateSimilarities.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"
```
## 7. Record Model Performance
```bash
# Record the performance of each model at each multiplication rate of imbalance for each query. 
python3 getResults.py "$num_keyword_options" "$multiplication_rates" "$queries"
```
## 8. Analyze Length Bias
```bash
# Run Python script to analyze the role of GEO Series Abstract length in ranking
python3 lengthanalysis.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"
```
