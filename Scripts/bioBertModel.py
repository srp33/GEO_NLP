from biobert_embedding.embedding import BiobertEmbedding
biobert = BiobertEmbedding()
print(biobert.sentence_vector("She has breast cancer."))