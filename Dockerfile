FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
#FROM python:3.11.7-bullseye

RUN apt-get -y update \
    && apt-get install -y software-properties-common curl python3 python3-pip \
    && apt-get -y update \
    && add-apt-repository universe

RUN mkdir -p /huggingface \
 && chmod 777 /huggingface -R

RUN pip3 install --upgrade pip setuptools wheel \
 && pip3 install geofetch==0.12.5 bs4==0.0.2 gemmapy==0.0.2 scikit-learn==1.4.0 \
                 transformers==4.37.2 datasets==2.16.1 sentence-transformers==2.3.1 tensorflow[and-cuda]==2.15.0.post1 chromadb==0.4.22

# RUN python3 -m spacy download en_core_web_lg # This line was throwing an error that exited the file 9/2023

#SciSpacy
#RUN pip install scispacy
#RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz

#SciBert
#RUN wget -O scibert_uncased.tar https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar
#RUN tar -xvf scibert_uncased.tar

#Need this for stopwords.
#COPY ImportNLTK.py /
#RUN python3 /ImportNLTK.py
#RUN pip install -U scikit-activeml

COPY exec_analysis.sh /

COPY Scripts/* /
