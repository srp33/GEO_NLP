#!/bin/bash

set -o errexit

if [ ! -f "/Models/BioWordVec_PubMed_MIMICIII_d200.vec.bin" ]
then
  wget -O "/Models/BioWordVec_PubMed_MIMICIII_d200.vec.bin" https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin
fi

if [ ! -f "/Models/wiki-news-300d-1M.vec" ]
then
  wget -O "/Models/wiki-news-300d-1M.vec.zip" https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
  cd /Models
  unzip wiki-news-300d-1M.vec.zip
  rm wiki-news-300d-1M.vec.zip
  cd -
fi

if [ ! -f "/Models/wiki-news-300d-1M-subword.vec" ]
then
  wget -O "/Models/wiki-news-300d-1M-subword.vec.zip" https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
  cd /Models
  unzip wiki-news-300d-1M-subword.vec.zip
  rm wiki-news-300d-1M-subword.vec.zip
  cd -
fi

if [ ! -f "/Models/crawl-300d-2M.vec" ]
then
  wget -O "/Models/crawl-300d-2M.vec.zip" https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
  cd /Models
  unzip crawl-300d-2M.vec.zip
  rm crawl-300d-2M.vec.zip
  cd -
fi

if [ ! -f "/Models/crawl-300d-2M-subword.vec" ]
then
  wget -O "/Models/crawl-300d-2M-subword.zip" https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
  cd /Models
  unzip crawl-300d-2M-subword.zip
  rm crawl-300d-2M-subword.zip
  cd -
fi

all_geo_tsv_file_path="/Data/AllGEO.tsv.gz"
all_geo_json_file_path="/Data/AllGEO.json.gz"
gemma_txt_file_path="/Data/Gemma.txt.gz"
non_gemma_txt_file_path="/Data/NonGemma.txt.gz"
gemma_json_file_path="/Data/Gemma.json.gz"
non_gemma_json_file_path="/Data/NonGemma.json.gz"
gemma_overlap_file_path="/Data/Gemma_word_overlap.tsv.gz"
multiplication_rates="1,2,5,10,100,300"

tmp_dir_path="/Data/tmp"
mkdir -p ${tmp_dir_path}

#python3 getAllGEO.py ${tmp_dir_path} "$all_geo_tsv_file_path"
# FYI: This excludes SubSeries
#python3 prepareAllGEO.py "$all_geo_tsv_file_path" "$all_geo_json_file_path"

#python getGemma.py "$all_geo_json_file_path" "$gemma_txt_file_path" "$non_gemma_txt_file_path"
#python3 prepareGemma.py "$all_geo_json_file_path" "$gemma_txt_file_path" "$gemma_json_file_path"
#python3 prepareGemma.py "$all_geo_json_file_path" "$non_gemma_txt_file_path" "$non_gemma_json_file_path"

# Save small queries
#python3 getQuerySeries.py 0005494 False "$all_geo_json_file_path" Queries/triple_negative_breast_carcinoma
#python3 getQuerySeries.py 0011429 False "$all_geo_json_file_path" Queries/juvenile_idiopathic_arthritis

# Save medium queries
#python3 getQuerySeries.py 0008608 False "$all_geo_json_file_path" Queries/down_syndrome
#python3 getQuerySeries.py 0004985,0000693 False "$all_geo_json_file_path" Queries/bipolar_disorder

# Save large queries
#python3 getQuerySeries.py 0005180 True "$all_geo_json_file_path" Queries/parkinson_disease
#python3 getQuerySeries.py 0005072 True "$all_geo_json_file_path" Queries/neuroblastoma

#python3 assignTrainingTestingOther.py "$gemma_json_file_path" triple_negative_breast_carcinoma Queries Assignments "$multiplication_rates"
#python3 assignTrainingTestingOther.py "$gemma_json_file_path" juvenile_idiopathic_arthritis Queries Assignments "$multiplication_rates"
#python3 assignTrainingTestingOther.py "$gemma_json_file_path" down_syndrome Queries Assignments "$multiplication_rates"
#python3 assignTrainingTestingOther.py "$gemma_json_file_path" bipolar_disorder Queries Assignments "$multiplication_rates"
#python3 assignTrainingTestingOther.py "$gemma_json_file_path" parkinson_disease Queries Assignments "$multiplication_rates"
#python3 assignTrainingTestingOther.py "$gemma_json_file_path" neuroblastoma Queries Assignments "$multiplication_rates"

#python3 calculateWordOverlap.py "$gemma_json_file_path" "$all_geo_json_file_path" "$gemma_overlap_file_path"

#python3 rankBasedOnWordOverlap.py "$gemma_overlap_file_path" triple_negative_breast_carcinoma word_overlap Assignments Similarities
#python3 rankBasedOnWordOverlap.py "$gemma_overlap_file_path" juvenile_idiopathic_arthritis word_overlap Assignments Similarities
#python3 rankBasedOnWordOverlap.py "$gemma_overlap_file_path" down_syndrome word_overlap Assignments Similarities
#python3 rankBasedOnWordOverlap.py "$gemma_overlap_file_path" bipolar_disorder word_overlap Assignments Similarities
#python3 rankBasedOnWordOverlap.py "$gemma_overlap_file_path" parkinson_disease word_overlap Assignments Similarities
#python3 rankBasedOnWordOverlap.py "$gemma_overlap_file_path" neuroblastoma word_overlap Assignments Similarities

#python3 trainCustomModels.py "$star_geo_file_path" "$all_geo_json_file_path" "$num_keyword_options" "Models/custom"
#python3 trainBert.py
#python3 calculateSimilarities.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"

#TODO: Clean up this script after getting the above code working.
#python3 calculateMetrics.py Similarities Metrics
#python3 getResults.py "$num_keyword_options" "$multiplication_rates" "$queries"

#python3 lengthanalysis.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"

# rm -rf ${tmp_dir_path}

#chmod 777 /Data -R
#chmod 777 /Results -R
