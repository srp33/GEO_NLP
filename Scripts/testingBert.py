from datasets import *
from transformers import *
from tokenizers import *
import os
import torch
import json
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine
import sys
# code from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#why-bert-embeddings
all_geo_file_path = sys.argv[1]
all_dict = {}
with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())

print('begin_build')

def get_sentence_embedding(text, model, tokenizer):
      
    marked_text = marked_text = "[CLS] " + text + " [SEP]"

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    # `hidden_states` is a Python list.
    print('      Type of hidden_states: ', type(hidden_states))

    # Each layer in the list is a torch tensor.
    print('Tensor shape for each layer: ', hidden_states[0].size())

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    token_embeddings.size()
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    token_embeddings.size()
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    token_embeddings.size()
    #sentence vector!
    # `hidden_states` has shape [13 x 1 x 22 x 768]

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return(sentence_embedding)

def get_rob_sentence_embedding(text, model, tokenizer, word_method):
      
    marked_text = marked_text = "[CLS] " + text + " [SEP]"

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    if word_method == 'cat':
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
        return(outputs[1])    
    else:
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            lastHiddenState = outputs[0]
            print(lastHiddenState)
            print(lastHiddenState.size())
        token_embeddings = lastHiddenState.permute(1,0,2)
        print(token_embeddings)
        token_embeddings.size()
        print(token_embeddings.size())
        return(token_embeddings[2])

def get_bert_word_embedding(text, model, tokenizer, method):
    marked_text = marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        print(outputs)
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)

    token_embeddings.size()
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    token_embeddings.size()
    token_embeddings = token_embeddings.permute(1,0,2)

    token_embeddings.size()
    
    if method == "cat":
        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []
        #Word vector from second to last layer!
        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:

        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs_cat.append(cat_vec)
        total = 0
        for vec in token_vecs_cat:
            total += vec
        avg_cat = total / len(token_vecs_cat)
        return(avg_cat)

    elif method == "sum":
        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []
        #Word sum vector from last 4 layers
        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:

        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)

        # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)   
        total = 0 
        for vec in token_vecs_sum:
            total += vec
        avg_sum = total / len(token_vecs_sum)
        return(avg_sum)

def get_roberta_word_embedding(text, model, tokenizer, method):
    marked_text = marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    if method == "cat":
    
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        model.eval()
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            print(outputs)
            lastHiddenState = outputs[1]
            print(lastHiddenState)
            print(lastHiddenState.size())
        return(lastHiddenState)
    else:
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        model.eval()
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            print(outputs)
            lastHiddenState = outputs[0]
            print(lastHiddenState)
            print(lastHiddenState.size())
        token_embeddings = lastHiddenState.permute(1,0,2)
        print(token_embeddings)
        token_embeddings.size()
        print(token_embeddings.size())
        return(token_embeddings[2])