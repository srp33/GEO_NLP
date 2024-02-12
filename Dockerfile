FROM python:3.11.7-bullseye

RUN pip install --upgrade pip setuptools wheel
RUN pip install geofetch==0.12.5 joblib==1.3.2 bs4==0.0.2 gemmapy==0.0.2 scikit-learn==1.4.0
#RUN pip install gensim==4.3.2
RUN mkdir -p /huggingface \
 && chmod 777 /huggingface -R
RUN pip install transformers==4.37.2 datasets==2.16.1 sentence-transformers==2.3.1 tensorflow==2.15.0.post1 

#COPY requirements.txt /
#RUN pip install -r requirements.txt

# RUN python3 -m spacy download en_core_web_lg # This line was throwing an error that exited the file 9/2023

#SciSpacy
#RUN pip install scispacy
#RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz

#SciBert
#RUN wget -O scibert_uncased.tar https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar
#RUN tar -xvf scibert_uncased.tar

#Need this for stopwords!
#COPY ImportNLTK.py /
#RUN python3 /ImportNLTK.py
#RUN pip install -U scikit-activeml

COPY exec_analysis.sh /

COPY Scripts/* /
