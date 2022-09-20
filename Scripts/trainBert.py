import sentencepiece
from datasets import *
from transformers import *
from tokenizers import *
import os
import json
from tqdm.auto import tqdm
import sys
#Could we reformat to corpus file for the nonStar Geo series and go straight to tokenizing? 
dataset = load_dataset('json', data_files = "https://huggingface.co/datasets/spiccolo/gene_expression_omnibus_nlp/resolve/main/NonstarGeo.json")
print(dataset)

text_data = []
series_count = 0

print(dataset["train"]["GSE4"])
# for gse in tqdm(dataset["train"].features):
#     print(dataset["train"][f"{gse}"])
#     break
sys.exit()
# get set of GSE IDs
ids = dataset["train"].features
print(ids)
for series in tqdm(ids):
    print(series)
    print(dataset["train"][series])
    sys.exit()
    # text_data.append(data)
    # series_count += 1
    # if series_count == 10:
    #   print("\n".join(text_data))
    break

  # series = series["GSE4"].replace('\n', ' ')
  # text_data.append(series)
  # print("\n".join(text_data))

# for t in dataset["train"]["features"][:3]:
#     print(t)