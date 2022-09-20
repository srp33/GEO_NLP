from cgi import test
from helper import * #TODO: is this needed?
from numpy import dot
import sys
from numpy.linalg import norm
from gensim.models import KeyedVectors
import gensim.downloader as api
w2v_model = KeyedVectors.load_word2vec_format("/Models/BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True)
average_training_vector = w2v_model['biology']
testing_and_other_vector = w2v_model['science']
cos_sim = dot(average_training_vector, testing_and_other_vector)/(norm(average_training_vector)*norm(testing_and_other_vector))
print(cos_sim)


training_vector_list = []
keywords = "This trial better work perfectly"
tmp_list = []
for word in keywords:
    if word in w2v_model.wv.vocab:
        tmp_list.append(w2v_model[word])
training_vector_list.append(sum(tmp_list) / len(tmp_list))
keywords = "Flawless testing"
tmp_list = []
for word in keywords:
    if word in w2v_model.wv.vocab:
        tmp_list.append(w2v_model[word])
training_vector_list.append(sum(tmp_list) / len(tmp_list))

print(training_vector_list)
average_training_vector = sum(training_vector_list) / len(training_vector_list)
print(average_training_vector)

tmp_list = []
testing_and_other_vector = []
keywords = "Awful failure"
for word in keywords:
    if word in w2v_model.wv.vocab:
        tmp_list.append(w2v_model[word])
testing_and_other_vector.append(sum(tmp_list) / len(tmp_list))
tmp_list = []
keywords = "Successful exam"
for word in keywords:
    if word in w2v_model.wv.vocab:
        tmp_list.append(w2v_model[word])
testing_and_other_vector.append(sum(tmp_list) / len(tmp_list))

#calculate cos sim
for words in testing_and_other_vector:
    cos_sim = dot(average_training_vector, words)/(norm(average_training_vector)*norm(words))
    print(cos_sim)