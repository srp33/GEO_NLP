FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

RUN apt-get -y update \
    && apt-get install -y software-properties-common curl wget zip python3 python3-pip \
    && apt-get -y update \
    && add-apt-repository universe

RUN mkdir -p /huggingface \
 && chmod 777 /huggingface -R

RUN pip3 install --upgrade pip setuptools wheel \
 && pip3 install geofetch==0.12.5 bs4==0.0.2 gemmapy==0.0.2 scikit-learn==1.4.0 \
                 transformers==4.37.2 datasets==2.16.1 sentence-transformers==2.4.0 tensorflow[and-cuda]==2.15.0.post1 \
                 chromadb==0.4.22 openai==1.12.0 fasttext==0.9.2 gensim==4.3.2 einops==0.7.0 \
                 joblib==1.3.2 nltk==3.8.1 langchain-text-splitters==0.0.1

RUN mkdir /nltk_data \
 && chmod 777 /nltk_data

COPY checkpoints*.txt /

COPY exec_analysis.sh /
COPY save_for_webapp.sh /

COPY Scripts/* /
