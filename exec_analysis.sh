#!/bin/bash

set -o errexit

if [ ! -f "/Models/wiki-news-300d-1M-subword.vec" ]
then
  wget -O "/Models/wiki-news-300d-1M-subword.vec.zip" https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
  cd /Models
  unzip wiki-news-300d-1M-subword.vec.zip
  rm wiki-news-300d-1M-subword.vec.zip
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
multiplication_rates="1,2,5,10,100,300"

tmp_dir_path="/Data/tmp"
mkdir -p ${tmp_dir_path}/word_overlap

#python3 getAllGEO.py ${tmp_dir_path} "$all_geo_tsv_file_path"
#python3 prepareAllGEO.py "$all_geo_tsv_file_path" "$all_geo_json_file_path"

# This sometimes gives an error message, but it seems to work properly.
#python3 getGemma.py "$all_geo_json_file_path" "$gemma_txt_file_path" "$non_gemma_txt_file_path"
#python3 prepareGemma.py "$all_geo_json_file_path" "$gemma_txt_file_path" "$gemma_json_file_path"
#python3 prepareGemma.py "$all_geo_json_file_path" "$non_gemma_txt_file_path" "$non_gemma_json_file_path"

# Save query series on April 19, 2024.
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

#python3 summarizeManualSearches.py Manual_Searches "$gemma_json_file_path" Results/Manual_Searches.tsv.gz

overlap_scores_file_path="${tmp_dir_path}/word_overlap/scores.tsv.gz"
#python3 findWordOverlap.py "$gemma_json_file_path" "$overlap_scores_file_path"

#for tag in triple_negative_breast_carcinoma juvenile_idiopathic_arthritis down_syndrome bipolar_disorder parkinson_disease neuroblastoma
#do
#  python3 rankTestingOther.py "$overlap_scores_file_path" $tag word_overlap Assignments Similarities &
#done
#wait

# We are using an extremely large chunk size so it doesn't split the texts into chunks.
#python3 saveEmbeddings.py "$all_geo_json_file_path" checkpoints.txt 100000000 "${tmp_dir_path}/Embeddings"
#python3 findDistances.py "$gemma_json_file_path" "$gemma_json_file_path" checkpoints.txt "${tmp_dir_path}/Embeddings" "${tmp_dir_path}/Distances"

# Two two smallest embedding sizes are GloVe (size = 300) and all-MiniLM-L6-v2 (size = 384).
# That description for all-MiniLM-L6-v2 says, "By default, input text longer than 256 word pieces is truncated."
python3 saveEmbeddings.py "$all_geo_json_file_path" checkpoints.txt 256 "${tmp_dir_path}/Embeddings_Chunks"
python3 findDistances.py "$gemma_json_file_path" "$gemma_json_file_path" checkpoints.txt "${tmp_dir_path}/Embeddings_Chunks" "${tmp_dir_path}/Distances_Chunks"

#for f in ${tmp_dir_path}/Distances/*/*.tsv.gz
#do
#    model_root=$(dirname $f)
#    model_root=$(basename $model_root)
#
#    model_name=$(basename $f)
#    model_name=${model_name/\.tsv\.gz/}
#
#    method_descriptor=${model_root}____${model_name}
#
#    for tag in triple_negative_breast_carcinoma juvenile_idiopathic_arthritis down_syndrome bipolar_disorder parkinson_disease neuroblastoma
#    do
#        python3 rankTestingOther.py "$f" $tag ${method_descriptor} Assignments Similarities &
#    done
#    wait
#done

#for f in ${tmp_dir_path}/Distances_Chunks/*/*.tsv.gz
#do
#    model_root=$(dirname $f)
#    model_root=$(basename $model_root)
#
#    model_name=$(basename $f)
#    model_name=${model_name/\.tsv\.gz/}
#
#    method_descriptor=${model_root}____${model_name}
#
#    for tag in triple_negative_breast_carcinoma juvenile_idiopathic_arthritis down_syndrome bipolar_disorder parkinson_disease neuroblastoma
#    do
#        python3 rankTestingOther.py "$f" $tag ${method_descriptor} Assignments Similarities_Chunks &
#    done
#    wait
#done

#python3 calculateMetrics.py Similarities Results/Metrics.tsv.gz
#python3 calculateMetrics.py Similarities_Chunks Results/Metrics_Chunks.tsv.gz

#python3 findTopOtherCandidates.py "Similarities/*/*/rest_of_gemma_all" Results/Top_Other_Candidates.tsv.gz
#python3 findTopOtherCandidates.py "Similarities_Chunks/*/*/rest_of_gemma_all" Results/Top_Other_Candidates_Chunks.tsv.gz

# Process non-Gemma series.
#python3 findDistances.py "$gemma_json_file_path" "$non_gemma_json_file_path" checkpoints2.txt "${tmp_dir_path}/Embeddings" "${tmp_dir_path}/Distances_NonGemma"

#for f in ${tmp_dir_path}/Distances_NonGemma/*/*.tsv.gz
#do
#    echo $f
#    model_root=$(dirname $f)
#    model_root=$(basename $model_root)
#
#    model_name=$(basename $f)
#    model_name=${model_name/\.tsv\.gz/}
#
#    method_descriptor=${model_root}____${model_name}
#
#    for tag in triple_negative_breast_carcinoma juvenile_idiopathic_arthritis down_syndrome bipolar_disorder parkinson_disease neuroblastoma
#    do
#        python3 rankNonGemma.py "$f" $tag ${method_descriptor} Queries Similarities_NonGemma
#    done
#done

#python3 findTopNonGemmaCandidates.py "Similarities_NonGemma/*/*/all" ${all_geo_tsv_file_path} Results/Top_NonGemma_Candidates.tsv.gz

#python3 saveCheckpointMetadata.py "Data/tmp/*/*/embeddings.gz" Results/Embedding_Sizes.tsv.gz Results/Checkpoint_Metadata.tsv.gz


#TODO: Add this to calculateMetrics.py?
# Checks for bias in favor of shorter or longer summary sections.
#python3 lengthanalysis.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"

#python3 calculateSimilarities.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"
#python3 trainCustomModels.py "$star_geo_file_path" "$all_geo_json_file_path" "$num_keyword_options" "Models/custom"
#python3 trainBert.py

# rm -rf ${tmp_dir_path}

#TODO: Add more
#chmod 777 /Data -R
#chmod 777 /Results -R
