#FROM python:3.8.5
FROM bioconductor/bioconductor_docker:RELEASE_3_14

#RUN pip install --upgrade pip
RUN python3 -m pip install urllib3

RUN pip install git+https://github.com/boudinfl/pke.git

COPY requirements.txt /
RUN pip install -r requirements.txt

RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bc5cdr_md-0.2.5.tar.gz
RUN python3 -m spacy download en_core_web_lg

#SciBert
# RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_scibert-0.5.0.tar.gz
# RUN python3 -m scispacy download en_core_sci_lg

RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz
RUN pip install scispacy


# RUN pip install https://files.pythonhosted.org/packages/d2/f0/f5bd3fd4a0bcef4d85e5e82347ae73d376d68dc8086afde75838ba0473a2/biobert-embedding-0.1.2.tar.gz
# RUN pip install biobert-embedding


COPY ImportNLTK.py /
RUN python3 /ImportNLTK.py

RUN R -e 'BiocManager::install("GEOmetadb", force = TRUE)'

COPY exec_analysis.sh /

COPY Scripts/* /
