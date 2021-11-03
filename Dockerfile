FROM python:3.8.5

RUN pip install --upgrade pip
RUN python -m pip install urllib3

RUN pip install git+https://github.com/boudinfl/pke.git

COPY requirements.txt /
RUN pip install -r requirements.txt

RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bc5cdr_md-0.2.5.tar.gz
RUN python -m spacy download en_core_web_lg

COPY Python_Scripts/* /
COPY Bash_Scripts/* /

RUN python3 /NLTKImport.py
