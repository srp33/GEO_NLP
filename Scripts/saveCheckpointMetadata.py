import glob
import gzip
import json
import os
import sys

embeddings_file_pattern = sys.argv[1]
out_file_path = sys.argv[2]

#def get_data_source_type(model_name):
#    model_name = model_name.lower()
#
#    if "biomed" in model_name or "pubmed" in model_name or "biolord" in model_name "biobert" in model_name:
#        return "Biomedical"
#    elif "scibert" in model_name:
#        return "Scientific"
#    return "General"
#
#def get_model_type(model_root, model_name):
#    model_name = model_name.lower()
#
#    if model_name.startswith("cbow"):
#        return "Word2Vec"
#    elif "glove" in model_name:
#        return "GloVe"
#    elif model_root == "openai":
#        return "Generative Pre-trained Transformer"
#    elif
#    return "Transformer"
#Masked and Permuted Pre-training for Language Understanding (MPNet)
#Bidirectional Encoder Representations from Transformers (BERT)
#Lite version of BERT (ALBERT, DistilBERT)
#Robustly Optimized BERT approach (RoBERTa)
#Transformer with Extra Long context (Transformer-XL)
#Self-supervised language representation learning (ELECTRA)
#Self-supervised contrastive learning (MiniLM-L6-v2)

info = [["Checkpoint", "Data_Source_Type", "Model_Category", "Fine_Tuning"],
        ["fasttext/cbow-wikinews", "General", "Continuous Bag of Words (Word2Vec)", "NA"],
        ["fasttext/cbow-commoncrawl", "General", "Continuous Bag of Words (Word2Vec)", "NA"],
        ["sentence-transformers/all-mpnet-base-v2", "General", "Self-supervised contrastive learning", "A 1B sentence pairs dataset"],
        ["sentence-transformers/all-roberta-large-v1", "General", "Self-supervised contrastive learning", "A 1B sentence pairs dataset"],
        ["sentence-transformers/all-MiniLM-L6-v2", "General", "Self-supervised contrastive learning", "A 1B sentence pairs dataset"],
        ["sentence-transformers/msmarco-distilbert-base-v3", "", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["sentence-transformers/sentence-t5-large", "", "", ""],
        ["sentence-transformers/sentence-t5-xl", "", "", ""],
        ["sentence-transformers/paraphrase-TinyBERT-L6-v2", "", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["hkunlp/instructor-xl", "", "", ""],
        ["thenlper/gte-large", "", "", ""],
        ["nomic-ai/nomic-embed-text-v1.5", "", "Matryoshka Representation Learning", ""],
        ["pritamdeka/S-Biomed-Roberta-snli-multinli-stsb", "", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["openai/text-embedding-ada-002", "General", "Generative Pre-trained Transformer (GPT)", ""],
        ["openai/text-embedding-3-small", "General", "Generative Pre-trained Transformer (GPT)", ""],
        ["openai/text-embedding-3-large", "General", "Generative Pre-trained Transformer (GPT)", ""],
        ["NeuML/pubmedbert-base-embeddings", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["FremyCompany/BioLORD-2023", "Biomedical", "", ""],
        ["pritamdeka/S-BioBert-snli-multinli-stsb", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["nuvocare/WikiMedical_sent_biobert", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["sentence-transformers/average_word_embeddings_glove.6B.300d", "General", "Global Vectors for Word Representation (GloVe)", ""],
        ["sentence-transformers/average_word_embeddings_glove.840B.300d", "General", "Global Vectors for Word Representation (GloVe)", ""],
        ["allenai/scibert_scivocab_uncased", "Scientific", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["allenai/biomed_roberta_base", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["emilyalsentzer/Bio_ClinicalBERT", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["medicalai/ClinicalBERT", "Clinical", "Bidirectional Encoder Representations from Transformers (BERT)", ""],
        ["google/electra-base-discriminator", "General", "Self-supervised language representation learning", ""],
        ["google/electra-small-discriminator", "General", "Self-supervised language representation learning", ""],
        ["albert/albert-base-v2", "General", "Masked language modeling (MLM)", ""],
        ["albert/albert-xxlarge-v2", "General", "Masked language modeling (MLM)", ""]]

lengths_dict = {}

for embeddings_file_path in glob.glob(embeddings_file_pattern):
    with gzip.open(embeddings_file_path) as embeddings_file:
        print(f"Getting embedding size for {embeddings_file_path}")
        model_root = os.path.basename(os.path.dirname(os.path.dirname(embeddings_file_path)))
        model_name = os.path.basename(os.path.dirname(embeddings_file_path))

        embeddings_dict = json.loads(embeddings_file.read().decode())
        example_series = sorted(embeddings_dict.keys())[0]
        example_embedding = embeddings_dict[example_series]

        lengths_dict[(model_root, model_name)] = len(example_embedding)

with open(out_file_path) as out_file:
    def write(items):
        out_file.write("\t".join(items) + "\n")

    write(["Checkpoint", "Embedding_Size", "Data_Source_Type", "Model_Type"])
    write(["word_overlap", "NA", "NA", "Basic"])

    for model_root_name, length in sorted(lengths_dict.items()):
        write([f"{model_root_name[0]}/{model_root_name[1]}", length, get_data_source_type(model_name), get_model_type(model_root, model_name)])

