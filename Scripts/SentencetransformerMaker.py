#https://huggingface.co/blog/how-to-train-sentence-transformers#:~:text=Understand%20how%20Sentence%20Transformers%20models%20work%20by%20creating,Share%20your%20model%20to%20the%20Hugging%20Face%20Hub.

from sentence_transformers import SentenceTransformer, models
## Step 1: use an existing language model
word_embedding_model = models.Transformer('distilroberta-base')
## Step 2: use a pool function over the token embeddings
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
## Join steps 1 and 2 using the modules argument
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)
