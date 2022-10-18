#It appears it will be much simpler to format the data to whatever object/class they have set this up for
#rather than modify the rest of the code to fit our data format. Ideas below...

#all from https://keras.io/examples/nlp/semantic_similarity_with_bert/ seeing if it works still!
import numpy as np
import sys
import pandas as pd
import tensorflow as tf
from datasets import *
from transformers import *
from tokenizers import *
from torch.utils.data import DataLoader
import json
from tqdm.auto import tqdm
from sentence_transformers import InputExample
from sentence_transformers import losses, models, SentenceTransformer

word_embedding_model = models.Transformer('distilroberta-base')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

dataset = load_dataset('json', data_files = "https://huggingface.co/datasets/spiccolo/gene_expression_omnibus_nlp/resolve/main/STARpairs.json", split="train")

d = dataset.train_test_split(test_size=0.1)
print(d["train"])
print(d["test"])

print(dataset)

train_examples = []
train_data = dataset['train']['set']
# For agility we only 1/2 of our available data
n_examples = dataset['train'].num_rows // 2

for i in range(n_examples):
  example = train_data[i]
  train_examples.append(InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)


train_loss = losses.ContrastiveLoss(model=model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10) 



# max_length = 128  # Maximum length of input sentence to the model.
# batch_size = 32
# epochs = 2
# labels = [0,1]

# all_geo_file_path = sys.argv[1]



# # Shape of the data
# print(f"Total train samples : {len(train_df)}")
# print(f"Total validation samples: {len(valid_df)}")
# print(f"Sentence1: {train_df[0]['sentence1']}") #what is the format of this? a list of dictionaries?
# print(f"Sentence2: {train_df[0]['sentence2']}")
# print(f"Similarity: {train_df[0]['similarity']}")



# train_df["label"] = train_df["similarity"].apply(
#     lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
# )
# y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)

# valid_df["label"] = valid_df["similarity"].apply(
#     lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
# )
# y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=3)

# class BertSemanticDataGenerator(tf.keras.utils.Sequence):
#     """Generates batches of data.

#     Args:
#         sentence_pairs: Array of premise and hypothesis input sentences.
#         labels: Array of labels.
#         batch_size: Integer batch size.
#         shuffle: boolean, whether to shuffle the data.
#         include_targets: boolean, whether to incude the labels.

#     Returns:
#         Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
#         (or just `[input_ids, attention_mask, `token_type_ids]`
#          if `include_targets=False`)
#     """

#     def __init__(
#         self,
#         sentence_pairs,
#         labels,
#         batch_size=batch_size,
#         shuffle=True,
#         include_targets=True,
#     ):
#         self.sentence_pairs = sentence_pairs
#         self.labels = labels
#         self.shuffle = shuffle
#         self.batch_size = batch_size
#         self.include_targets = include_targets
#         # Load our BERT Tokenizer to encode the text.
#         # We will use base-base-uncased pretrained model.
#         self.tokenizer = transformers.BertTokenizer.from_pretrained(
#             "bert-base-uncased", do_lower_case=True
#         )
#         self.indexes = np.arange(len(self.sentence_pairs))
#         self.on_epoch_end()

#     def __len__(self):
#         # Denotes the number of batches per epoch.
#         return len(self.sentence_pairs) // self.batch_size

#     def __getitem__(self, idx):
#         # Retrieves the batch of index.
#         indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
#         sentence_pairs = self.sentence_pairs[indexes]

#         # With BERT tokenizer's batch_encode_plus batch of both the sentences are
#         # encoded together and separated by [SEP] token.
#         encoded = self.tokenizer.batch_encode_plus(
#             sentence_pairs.tolist(),
#             add_special_tokens=True,
#             max_length=max_length,
#             return_attention_mask=True,
#             return_token_type_ids=True,
#             pad_to_max_length=True,
#             return_tensors="tf",
#         )

#         # Convert batch of encoded features to numpy array.
#         input_ids = np.array(encoded["input_ids"], dtype="int32")
#         attention_masks = np.array(encoded["attention_mask"], dtype="int32")
#         token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

#         # Set to true if data generator is used for training/validation.
#         if self.include_targets:
#             labels = np.array(self.labels[indexes], dtype="int32")
#             return [input_ids, attention_masks, token_type_ids], labels
#         else:
#             return [input_ids, attention_masks, token_type_ids]

#     def on_epoch_end(self):
#         # Shuffle indexes after each epoch if shuffle is set to True.
#         if self.shuffle:
#             np.random.RandomState(42).shuffle(self.indexes)

# # Create the model under a distribution strategy scope.
# strategy = tf.distribute.MirroredStrategy()

# with strategy.scope():
#     # Encoded token ids from BERT tokenizer.
#     input_ids = tf.keras.layers.Input(
#         shape=(max_length,), dtype=tf.int32, name="input_ids"
#     )
#     # Attention masks indicates to the model which tokens should be attended to.
#     attention_masks = tf.keras.layers.Input(
#         shape=(max_length,), dtype=tf.int32, name="attention_masks"
#     )
#     # Token type ids are binary masks identifying different sequences in the model.
#     token_type_ids = tf.keras.layers.Input(
#         shape=(max_length,), dtype=tf.int32, name="token_type_ids"
#     )
#     # Loading pretrained BERT model.
#     bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
#     # Freeze the BERT model to reuse the pretrained features without modifying them.
#     bert_model.trainable = False

#     bert_output = bert_model(
#         input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
#     )
#     sequence_output = bert_output.last_hidden_state
#     pooled_output = bert_output.pooler_output
#     # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
#     bi_lstm = tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(64, return_sequences=True)
#     )(sequence_output)
#     # Applying hybrid pooling approach to bi_lstm sequence output.
#     avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
#     max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
#     concat = tf.keras.layers.concatenate([avg_pool, max_pool])
#     dropout = tf.keras.layers.Dropout(0.3)(concat)
#     output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
#     model = tf.keras.models.Model(
#         inputs=[input_ids, attention_masks, token_type_ids], outputs=output
#     )

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss="categorical_crossentropy",
#         metrics=["acc"],
#     )


# print(f"Strategy: {strategy}")
# model.summary()
# train_data = BertSemanticDataGenerator(
#     train_df[["sentence1", "sentence2"]].values.astype("str"),
#     y_train,
#     batch_size=batch_size,
#     shuffle=True,
# )
# valid_data = BertSemanticDataGenerator(
#     valid_df[["sentence1", "sentence2"]].values.astype("str"),
#     y_val,
#     batch_size=batch_size,
#     shuffle=False,
# )
# history = model.fit(
#     train_data,
#     validation_data=valid_data,
#     epochs=epochs,
#     use_multiprocessing=True,
#     workers=-1,
# )
# # Unfreeze the bert_model.
# bert_model.trainable = True
# # Recompile the model to make the change effective.
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-5),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )
# model.summary()
# history = model.fit(
#     train_data,
#     validation_data=valid_data,
#     epochs=epochs,
#     use_multiprocessing=True,
#     workers=-1,
# )