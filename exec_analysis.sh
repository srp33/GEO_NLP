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
multiplication_rates="1,2,5,10,100,300"
num_keyword_options="2,4,8,16,32"

tmp_dir_path="/Data/tmp"
mkdir -p ${tmp_dir_path}

#python3 getAllGEO.py ${tmp_dir_path} "$all_geo_tsv_file_path"
#python3 prepareAllGEO.py "$all_geo_tsv_file_path" "$all_geo_json_file_path"

#python getGemma.py "$all_geo_json_file_path" "$gemma_txt_file_path" "$non_gemma_txt_file_path"
#python3 prepareGemma.py "$all_geo_json_file_path" "$gemma_txt_file_path" "$gemma_json_file_path"
#python3 prepareGemma.py "$all_geo_json_file_path" "$non_gemma_txt_file_path" "$non_gemma_json_file_path"

# Save small queries
#python getQuerySeries.py 0005494 False Queries/triple_negative_breast_carcinoma
#python getQuerySeries.py 0011429 False Queries/juvenile_idiopathic_arthritis

# Save medium queries
#python getQuerySeries.py 0008608 False Queries/down_syndrome
#python getQuerySeries.py 0004985,0000693 False Queries/bipolar_disorder

# Save large queries
#python getQuerySeries.py 0005180 True Queries/parkinson_disease
#python getQuerySeries.py 0005072 True Queries/neuroblastoma

#python3 prepareStarGEO.py "$star_geo_file_path" "$all_geo_json_file_path" 
#python3 makeNonstarGeo.py "$star_geo_file_path" "$all_geo_json_file_path" "$non_star_geo_file_path"


#python3 assignTrainingTestingOther.py "$star_geo_file_path" "$all_geo_json_file_path" "GSE47860,GSE47861,GSE49481,GSE50567,GSE17072,GSE40115,GSE166819" q1 "$multiplication_rates"
exit
python3 assignTrainingTestingOther.py "$star_geo_file_path" "$all_geo_json_file_path" "GSE7741,GSE98324,GSE74000,GSE28619,GSE57726,GSE57725,GSE24667,GSE87028,GSE123584" q2 "$multiplication_rates"
python3 assignTrainingTestingOther.py "$star_geo_file_path" "$all_geo_json_file_path" "GSE80612,GSE7036,GSE74937,GSE7624,GSE4187,GSE51284,GSE7486,GSE33476,GSE53822,GSE24060,GSE16059,GSE16461,GSE19519,GSE928,GSE73142,GSE85452,GSE62199,GSE51056,GSE86331,GSE33321,GSE7821,GSE45736,GSE46449,GSE21421,GSE37659,GSE151454,GSE172464" q3 "$multiplication_rates"
python3 assignTrainingTestingOther.py "$star_geo_file_path" "$all_geo_json_file_path" "GSE18058,GSE23926,GSE54635,GSE78179,GSE68606,GSE68950,GSE127996,GSE76948,GSE36157,GSE123709,GSE25493,GSE51536,GSE841,GSE54709,GSE23655,GSE5117" q4 "$multiplication_rates" #GSE148795 not found
python3 assignTrainingTestingOther.py "$star_geo_file_path" "$all_geo_json_file_path" "GSE146338,GSE44313,GSE44314,GSE17635,GSE33440,GSE72377,GSE72376,GSE182494,GSE89022,GSE154609,GSE30210,GSE102234,GSE43488,GSE141193,GSE70901,GSE70752,GSE29142,GSE38396,GSE15653,GSE37025,GSE55098,GSE72492,GSE71730,GSE80569,GSE77350,GSE29908,GSE35711,GSE30208,GSE30209,GSE35712,GSE35716,GSE35725,GSE52724,GSE54876,GSE20966,GSE25462,GSE68049,GSE9006,GSE105167,GSE21340,GSE111154,GSE41767,GSE84934,GSE146615,GSE51546,GSE85226,GSE50800,GSE20247,GSE42094,GSE36084,GSE75062,GSE164338,GSE66413,GSE78840,GSE35713,GSE14368,GSE14503,GSE15543,GSE24147,GSE28038,GSE30211,GSE40496,GSE54849,GSE60360,GSE52376,GSE60803,GSE104190,GSE143690,GSE4117,GSE35411,GSE84908,GSE136277,GSE109022,GSE8157,GSE59363,GSE125590,GSE46263,GSE62761,GSE4901,GSE24290,GSE24555,GSE83452,GSE49566,GSE95675,GSE95674,GSE11907,GSE11908,GSE67279,GSE72462,GSE118230,GSE156993,GSE78891,GSE113873,GSE60760,GSE8908,GSE71099,GSE71102,GSE132187,GSE9017" q5 "$multiplication_rates"
python3 assignTrainingTestingOther.py "$star_geo_file_path" "$all_geo_json_file_path" "GSE30699,GSE28424,GSE24401,GSE42903,GSE5045,GSE16088,GSE16091,GSE36001,GSE11127,GSE14359,GSE21257,GSE38133,GSE52089,GSE55958,GSE19913,GSE32395,GSE42351,GSE48281,GSE53155,GSE12512,GSE55957,GSE85537,GSE76535,GSE16089,GSE19276,GSE32981,GSE33382,GSE3362,GSE38134,GSE8079,GSE42572,GSE11414,GSE12865,GSE14789,GSE14827,GSE22970,GSE28252,GSE30807,GSE37552,GSE39055,GSE39057,GSE56001,GSE9508,GSE73166,GSE73422,GSE86109,GSE89074,GSE89370,GSE73120,GSE87437,GSE96892,GSE115590,GSE119975,GSE1153,GSE16066,GSE16070,GSE16080,GSE16082,GSE16085,GSE18947,GSE27900,GSE29634,GSE33458,GSE35512,GSE39072,GSE42352,GSE44713,GSE5796,GSE58209,GSE6711,GSE50527,GSE57203,GSE66673,GSE68950,GSE65065,GSE70719,GSE81892,GSE63390,GSE97572,GSE129091,GSE134603,GSE143556,GSE176373,GSE1000,GSE11115,GSE13504,GSE19060,GSE21751,GSE24170,GSE26244,GSE26857,GSE28256,GSE28912,GSE32182,GSE46448,GSE46493,GSE46549,GSE49003,GSE50988,GSE5117,GSE51349,GSE54267,GSE54820,GSE54942,GSE9854,GSE55749,GSE61928,GSE70414,GSE65401,GSE87363,GSE94805,GSE107855,GSE107836,GSE118129" q6 "$multiplication_rates"

##Training models 
#python3 trainCustomModels.py "$star_geo_file_path" "$all_geo_json_file_path" "$num_keyword_options" "Models/custom"
#python3 trainBert.py


python3 calculateSimilarities.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"
python3 getResults.py "$num_keyword_options" "$multiplication_rates" "$queries"
python3 lengthanalysis.py "$all_geo_json_file_path" "$queries" "$multiplication_rates" "$num_keyword_options"

# rm -rf ${tmp_dir_path}
exit

chmod 777 /Data -R
chmod 777 /Results -R
