from transformers import BertTokenizer, BertForPreTraining
import torch
import random
from numpy import dot
from numpy.linalg import norm
from helper import *
import sys
from sentence_transformers import SentenceTransformer
import json
import os
## Separate thing! Using tutorial from https://towardsdatascience.com/how-to-train-bert-aaad00533168
star_geo_file_path = sys.argv[1]
all_geo_file_path = sys.argv[2]

corpus_file_path = "/Models/corpus_file.txt"

with open(star_geo_file_path) as star_file:
    star_list = json.loads(star_file.read())
    star_set = set(star_list)

counter = 0
with open("Data/AllGEO.tsv", 'r') as all_file:
    for line in all_file:
        line = line.split("\t")
        if line[0] in star_set:
            continue
        else:
            with open(corpus_file_path, 'a') as corpus_file:
                corpus_file.write(f"{line[1]}. ")
                corpus_file.write(line[2])
                corpus_file.write("\n")
                counter += 1
        if counter >= 100:
            break



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

with open(corpus_file_path, 'r') as fp:
    text = fp.read().split('\n')

#bag = [item for sentence in text for item in sentence.split('.') if item != '']
#bag_size = len(bag)
sentence_a = []
sentence_b = []
label = []

for paragraph in text:
    print(paragraph)
    sentences = [
        sentence for sentence in paragraph.split('.') if sentence != ''
    ]
    print(sentences)
    num_sentences = len(sentences)
    print(num_sentences)
    if num_sentences > 1:
        start = random.randint(0, num_sentences-2)
        # 50/50 whether is IsNextSentence or NotNextSentence
        if random.random() >= 0.5:
            # this is IsNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(sentences[start+1])
            label.append(0)
        else:
            index = random.randint(0, bag_size-1)
            # this is NotNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(bag[index])
            label.append(1)

print(sentence_a)
print(sentence_b)
print(label)
inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
inputs['next_sentence_label'] = torch.LongTensor([label]).T
# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)
selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )
for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
dataset = OurDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

from tqdm import tqdm  # for our progress bar

epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        print(batch)
        # initialize calculated gradients (from prev step)
        #torch.optim.Optimizer.zero_grad(batch['self'])
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to('cpu')
        token_type_ids = batch['token_type_ids'].to('cpu')
        attention_mask = batch['attention_mask'].to('cpu')
        next_sentence_label = batch['next_sentence_label'].to('cpu')
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        next_sentence_label=next_sentence_label,)
        # extract loss
        loss = outputs.loss
        if loss != None:
            loss.backward()
        print(loss)
        # calculate loss for every parameter that needs grad update
        # update parameters
        #torch.optim.Optimizer.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        if loss != None:
            loop.set_postfix(loss=loss.item())
model.save_pretrained ("Models/custom/geoBert")
#torch.save(model.module.state_dict(), PATH)
# query = 'q1'
# keyword_extractor_name = 'KPMiner'
# num_keywords = 2
# other_multiplication_rate = 2
# averaging_method = "sentence_vector"
# model_name = "edited_Bert"
# results_dir_path = f"/Results/{query}//{num_keywords}/{keyword_extractor_name}/{other_multiplication_rate}"
# Path(results_dir_path).mkdir(parents=True, exist_ok=True)

# training_vector_list = []
# model = SentenceTransformer(tokenizer = BertTokenizer.from_pretrained('Fine_tune_BERT/')

# )

# #Finding training set vector average
# for training_series_id in get_series_identifiers(query, "training_series"):
#     keywords = get_keywords(keyword_extractor_name, num_keywords, training_series_id)
#     if keywords == "":
#         continue
#     training_vector_list.append(model.encode(keywords))
# average_training_vector = sum(training_vector_list) / len(training_vector_list)
    
# list_of_ids = []
# for series in get_series_identifiers(f"{query}/other_series", other_multiplication_rate):
#     if get_keywords(keyword_extractor_name, num_keywords, series) != "":
#         list_of_ids.append(series)
# for series in get_series_identifiers(query, "testing_series"):
#     if get_keywords(keyword_extractor_name, num_keywords, series) != "":
#         list_of_ids.append(series)
    
# cos_sim_and_series_id_list = []
# #Finding vectors for each series to compare to training
# for testing_and_other_series_id in list_of_ids:
#     keywords = get_keywords(keyword_extractor_name, num_keywords, testing_and_other_series_id)
#     if keywords == "":
#         continue
#     testing_and_other_vector = model.encode(keywords)
        
#     #calculate cos sim
#     cos_sim = dot(average_training_vector, testing_and_other_vector)/(norm(average_training_vector)*norm(testing_and_other_vector))
#     cos_sim_and_series_id_list.append([cos_sim, testing_and_other_series_id])

# cos_sim_and_series_id_list.sort()
# cos_sim_and_series_id_list.reverse()

# #recording findings
# with open(f'{results_dir_path}/{averaging_method}_similarity.tsv', 'w+') as out_file:
#     print_time_stamp(f"Processing {results_dir_path}")
#     out_file.write("Series ID\tSimilarity Score\tTest or Other Group\n")
#     for series in cos_sim_and_series_id_list:
#         test_or_other= ""
#         if series[1] in get_series_identifiers(query, "testing_series"):
#             test_or_other = "Test"
#         else:
#             test_or_other = "Other"
#         out_file.write(f"{series[1]}\t{series[0]}\t{test_or_other}\n")