from chromadb.utils import embedding_functions
import os
# Change this path to a different path.
# It is where some temporary files will be stored.
os.environ['HF_HOME'] = "/Models/huggingface"

# This is the search text that the user would enter.
gse_text = "color cancer cell lines from female patients"

model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="thenlper/gte-large")
embeddings = model(gse_text)

# We get back a list, so we retrieve the single element from that list.
embedding = embeddings[0]

print(embedding)
print(len(embedding))
