#FROM python:3.8.5
FROM bioconductor/bioconductor_docker:RELEASE_3_14


RUN R -e 'BiocManager::install("GEOmetadb", force = TRUE)'


RUN pip install --upgrade pip setuptools wheel

RUN pip install git+https://github.com/boudinfl/pke.git

COPY requirements.txt /
RUN pip install -r requirements.txt

RUN python3 -m spacy download en_core_web_lg

#SciSpacy
RUN pip install scispacy
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz

#SciBert
# RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_scibert-0.5.0.tar.gz
# RUN python3 -m scispacy download en_core_sci_lg

# COPY ImportNLTK.py /
# RUN python3 /ImportNLTK.py


COPY exec_analysis.sh /

COPY Scripts/* /
