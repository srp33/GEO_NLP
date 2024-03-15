import tokenize
import fasttext
import os
import multiprocessing
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm
from gensim.models import KeyedVectors
import spacy
import sys
import zipfile
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel
from transformers import *
from tokenizers import *
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

model = SentenceTransformer("bert-base-uncased")
cancer = model.encode("cancer")
tumor = model.encode('tumor')
print(cancer)
print(tumor)
print(dot(cancer, tumor) / (norm(cancer) * norm(tumor)))
print(len(tumor))