#all from https://keras.io/examples/nlp/semantic_similarity_with_bert/ seeing if it works still!
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import sys

star_geo_file_path = sys.argv[1]
all_geo_file_path = sys.argv[2]

nonstar_text_list = []
corpus_file_path = "/Models/corpus_file.txt"

with open(star_geo_file_path) as star_file:
    star_list = json.loads(star_file.read())
    star_set = set(star_list)

with open(all_geo_file_path) as all_file:
    all_dict = json.loads(all_file.read())
train_df = []
for series in all_dict:
    if series in star_set:
        continue
    else:
        train_df.append(all_dict[series])
#pass it the list instead of one huge text corpus

max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2

class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    #metaclass that is a class that creates other classes. 

    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """
    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        # self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        # indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs,
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding="longest",
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        # if self.include_targets:
        #     labels = np.array(self.labels[indexes], dtype="int32")
        #     return [input_ids, attention_masks, token_type_ids], labels
        # else:
        #     return [input_ids, attention_masks, token_type_ids]

    # def on_epoch_end(self):
    #     # Shuffle indexes after each epoch if shuffle is set to True.
    #     if self.shuffle:
    #         np.random.RandomState(42).shuffle(self.indexes)

# Create the model under a distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    bert_output = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output
    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )


print(f"Strategy: {strategy}")
model.summary()
train_data = BertSemanticDataGenerator(
    train_df,
    "y_train",
    batch_size=batch_size,
    shuffle=True,
)
print("I made a bertsemanticdatagenerator instance! Now I am going to fit the model.")
history = model.fit(
    train_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
print("I fit the model, now I am going to unfreeze and train it!")
# Unfreeze the bert_model.
bert_model.trainable = True
# Recompile the model to make the change effective.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
history = model.fit(
    train_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)

print("SUCCESS!!!")

# #Is this class needed???
# class BertSemanticDataGenerator(tf.keras.utils.Sequence):
#     """Generates batches of data.

#     Args:
#         sentence_pairs: Array of premise and hypothesis input sentences.

#     Returns:
#     """

#     def __init__(
#         self,
#         sentence_pairs,
#         batch_size=batch_size,
#         shuffle=True,
#         include_targets=True,
#     ):
#         self.sentence_pairs = sentence_pairs
#         self.shuffle = shuffle
#         self.batch_size = batch_size
#         self.include_targets = include_targets
#         # Load our BERT Tokenizer to encode the text.
#         # We will use biobert-large pretrained model.
#         self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
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
#     #bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
#     bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")

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
#     train_data = train_df
#     model = tf.keras.models.Model(train_df)
#     # history = model.fit(train_data)
#     # Unfreeze the bert_model.
#     bert_model.trainable = True

#     # Recompile the model to make the change effective.
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(1e-5),
#         loss="categorical_crossentropy",
#         metrics=["accuracy"],
#     )
# print(model)
# model.save(f"Models/custom/BERT.bin")