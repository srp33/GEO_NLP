#!/bin/bash

set -o errexit

#if [ ! -f "/Models/BioWordVec_PubMed_MIMICIII_d200.vec.bin" ]
#then
#  wget -O "/Models/BioWordVec_PubMed_MIMICIII_d200.vec.bin" https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin
#fi

#if [ ! -f "/Models/wiki-news-300d-1M.vec" ]
#then
#  wget -O "/Models/wiki-news-300d-1M.vec.zip" https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
#  cd /Models
#  unzip wiki-news-300d-1M.vec.zip
#  rm wiki-news-300d-1M.vec.zip
#  cd -
#fi

#if [ ! -f "/Models/wiki-news-300d-1M-subword.vec" ]
#then
#  wget -O "/Models/wiki-news-300d-1M-subword.vec.zip" https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
#  cd /Models
#  unzip wiki-news-300d-1M-subword.vec.zip
#  rm wiki-news-300d-1M-subword.vec.zip
#  cd -
#fi

#if [ ! -f "/Models/crawl-300d-2M.vec" ]
#then
#  wget -O "/Models/crawl-300d-2M.vec.zip" https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
#  cd /Models
#  unzip crawl-300d-2M.vec.zip
#  rm crawl-300d-2M.vec.zip
#  cd -
#fi

#if [ ! -f "/Models/crawl-300d-2M-subword.vec" ]
#then
#  wget -O "/Models/crawl-300d-2M-subword.zip" https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
#  cd /Models
#  unzip crawl-300d-2M-subword.zip
#  rm crawl-300d-2M-subword.zip
#  cd -
#fi

all_geo_tsv_file_path="/Data/AllGEO.tsv.gz"
all_geo_json_file_path="/Data/AllGEO.json.gz"
gemma_txt_file_path="/Data/Gemma.txt.gz"
non_gemma_txt_file_path="/Data/NonGemma.txt.gz"
gemma_json_file_path="/Data/Gemma.json.gz"
non_gemma_json_file_path="/Data/NonGemma.json.gz"
multiplication_rates="1,2,5,10,100,300"

tmp_dir_path="/Data/tmp"
mkdir -p ${tmp_dir_path}/word_overlap

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

overlap_scores_file_path="${tmp_dir_path}/word_overlap/scores.tsv.gz"
#python3 findWordOverlap.py "$gemma_json_file_path" "$all_geo_json_file_path" "$overlap_scores_file_path"

#for tag in triple_negative_breast_carcinoma juvenile_idiopathic_arthritis down_syndrome bipolar_disorder parkinson_disease neuroblastoma
#do
#  python3 rankTestingOther.py "$overlap_scores_file_path" $tag word_overlap Assignments Similarities
#do

#TODO: Add other checkpoints, etc.
#python3 findVectorDistances.py "$gemma_json_file_path" "$all_geo_json_file_path" "${tmp_dir_path}"

#for d in ${tmp_dir_path}/sentence-transformers/*
#do
#    method_descriptor=sentence-transformers____$(basename $d)
#
#    for tag in triple_negative_breast_carcinoma juvenile_idiopathic_arthritis down_syndrome bipolar_disorder parkinson_disease neuroblastoma
#    do
#        python3 rankTestingOther.py "$d/distances.gz" $tag ${method_descriptor} Assignments Similarities &
#    done
#    wait
#done

#TODO: Remove this script when done with above.
#python3 calculateSimilarities.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"
#python3 trainCustomModels.py "$star_geo_file_path" "$all_geo_json_file_path" "$num_keyword_options" "Models/custom"
#python3 trainBert.py

#TODO: Remove extra code at the end of this script after getting the other working.
python3 calculateMetrics.py Similarities Metrics

#TODO: Add this to calculateMetrics.py?
#python3 lengthanalysis.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"

# rm -rf ${tmp_dir_path}

#TODO: Add more
#chmod 777 /Data -R
#chmod 777 /Results -R
