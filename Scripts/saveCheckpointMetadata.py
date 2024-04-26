import glob
import gzip
import json
import os
import sys

embeddings_file_pattern = sys.argv[1]
out_sizes_file_path = sys.argv[2]
out_metadata_file_path = sys.argv[3]

size_dict = {}

for embeddings_file_path in glob.glob(embeddings_file_pattern):
    with gzip.open(embeddings_file_path) as embeddings_file:
        print(f"Getting embedding size for {embeddings_file_path}")
        model_root = os.path.basename(os.path.dirname(os.path.dirname(embeddings_file_path)))
        model_name = os.path.basename(os.path.dirname(embeddings_file_path))

        embeddings_dict = json.loads(embeddings_file.read().decode())
        example_series = sorted(embeddings_dict.keys())[0]
        example_embedding = embeddings_dict[example_series]

        size_dict[(model_root, model_name)] = len(example_embedding)

with gzip.open(out_sizes_file_path, "w") as out_file:
    out_file.write("Checkpoint\tEmbedding_Size\n".encode())
    out_file.write("word_overlap\tNA\n".encode())

    for model_root_name, size in sorted(size_dict.items()):
        out_file.write((f"{model_root_name[0]}/{model_root_name[1]}\t{size}\n").encode())

print(f"Saved to {out_sizes_file_path}")
sys.exit(0)

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



# Data Source Type: General, Biological, Medical
# Model Category:
#   Continuous Bag of Words (Word2Vec)
#   Self-supervised contrastive learning
#   Bidirectional Encoder Representations from Transformers (BERT)
#   Text-to-text transformers (T5)
#   Matryoshka Representation Learning
#   Generative Pre-trained Transformer (GPT)
#   Global Vectors for Word Representation (GloVe)
#   Masked language modeling (MLM)
#   Instruction-finetuned text embedding model

# The Fine Tuning column indicates whether a different checkpoint was fine-tuned to construct a given checkpoint.

info = [["Checkpoint", "Data_Source_Type", "Model_Category", "Fine_Tuning"],
        ["fasttext/cbow-wikinews", "General", "Continuous Bag of Words (Word2Vec)", "NA"],
        ["fasttext/cbow-commoncrawl", "General", "Continuous Bag of Words (Word2Vec)", "NA"],
        ["sentence-transformers/all-mpnet-base-v2", "General", "Self-supervised contrastive learning", "\"We used the pretrained microsoft/mpnet-base model and fine-tuned in on a 1B sentence pairs dataset.\""],
        ["sentence-transformers/all-roberta-large-v1", "General", "Self-supervised contrastive learning", "\"We used the pretrained roberta-large model and fine-tuned in on a 1B sentence pairs dataset.\""],
        ["sentence-transformers/all-MiniLM-L6-v2", "General", "Self-supervised contrastive learning", "\"We used the pretrained nreimers/MiniLM-L6-H384-uncased model and fine-tuned in on a 1B sentence pairs dataset.\""],
        ["sentence-transformers/msmarco-distilbert-base-v3", "", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["sentence-transformers/sentence-t5-large", "General", "Text-to-text transformers (T5)", "NA"],
        ["sentence-transformers/sentence-t5-xl", "General", "Text-to-text transformers (T5)", "NA"],
        ["sentence-transformers/paraphrase-TinyBERT-L6-v2", "General", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["hkunlp/instructor-xl", "General", "Instruction-finetuned text embedding model", "NA"],
        ["thenlper/gte-large", "General", "General Text Embeddings (GTE)", "NA"],
        ["nomic-ai/nomic-embed-text-v1.5", "General", "Matryoshka Representation Learning", "NA"],
        ["pritamdeka/S-Biomed-Roberta-snli-multinli-stsb", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", "\"The base model used is allenai/biomed_roberta_base which has been fine-tuned for sentence similarity.\""],
        ["openai/text-embedding-ada-002", "General", "Generative Pre-trained Transformer (GPT)", "NA"],
        ["openai/text-embedding-3-small", "General", "Generative Pre-trained Transformer (GPT)", "NA"],
        ["openai/text-embedding-3-large", "General", "Generative Pre-trained Transformer (GPT)", "NA"],
        ["NeuML/pubmedbert-base-embeddings", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["FremyCompany/BioLORD-2023", "Biomedical", "Ontology-driven concept representations", "\"This model is based on sentence-transformers/all-mpnet-base-v2 and was further finetuned on the BioLORD-Dataset and LLM-generated definitions from the Automatic Glossary of Clinical Terminology (AGCT).\""],
        ["pritamdeka/S-BioBert-snli-multinli-stsb", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["nuvocare/WikiMedical_sent_biobert", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", "\"Based on the dmis-lab/biobert-base-cased-v1.2 backbone and has been trained on the WikiMedical_sentence_simialrity dataset.\""],
        ["sentence-transformers/average_word_embeddings_glove.6B.300d", "General", "Global Vectors for Word Representation (GloVe)", "NA"],
        ["sentence-transformers/average_word_embeddings_glove.840B.300d", "General", "Global Vectors for Word Representation (GloVe)", "NA"],
        ["allenai/scibert_scivocab_uncased", "Scientific", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["allenai/biomed_roberta_base", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["emilyalsentzer/Bio_ClinicalBERT", "Biomedical", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["medicalai/ClinicalBERT", "Clinical", "Bidirectional Encoder Representations from Transformers (BERT)", "NA"],
        ["google/electra-base-discriminator", "General", "Self-supervised language representation learning", "NA"],
        ["google/electra-small-discriminator", "General", "Self-supervised language representation learning", "This checkpoint ostentibly is a fine-tuning of google/electra-base-discriminator."],
        ["albert/albert-base-v2", "General", "Masked language modeling (MLM)", "NA"],
        ["albert/albert-xxlarge-v2", "General", "Masked language modeling (MLM)", "This checkpoint ostentibly is a fine-tuning of albert/albert-xxlarge-v2."]]

with gzip.open(out_metadata_file_path) as out_file:
    def write(items):
        out_file.write("\t".join(items) + "\n")

    for row in info:
        write(row)
