#FROM python:3.8.5
FROM bioconductor/bioconductor_docker:RELEASE_3_15

RUN R -e 'BiocManager::install("GEOmetadb", force = TRUE)'

RUN pip install --upgrade pip setuptools wheel

RUN pip install git+https://github.com/boudinfl/pke.git

COPY requirements.txt /
RUN pip install -r requirements.txt

# RUN python3 -m spacy download en_core_web_lg # This line was throwing an error that exited the file 9/2023

#SciSpacy
RUN pip install scispacy
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz

#SciBert
RUN wget -O scibert_uncased.tar https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar
RUN tar -xvf scibert_uncased.tar

#Need this for stopwords!
COPY ImportNLTK.py /
RUN python3 /ImportNLTK.py
RUN pip install -U scikit-activeml

COPY exec_analysis.sh /

COPY Scripts/* /
