import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, concatenate, Conv1D, LayerNormalization

class AttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, last_conv_filters, dropout=0.1, **kwargs):
        super(AttentionEncoder, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.GlorotNormal(seed=42)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.last_conv_filters = last_conv_filters

        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=head_size, value_dim=head_size,
                                                      num_heads=num_heads, dropout=dropout,
                                                      kernel_initializer=self.initializer)
        self.conv1 = Conv1D(filters=ff_dim, kernel_size=1, activation="relu", kernel_initializer=self.initializer)
        self.conv2 = Conv1D(filters=last_conv_filters, kernel_size=1, kernel_initializer=self.initializer)
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)

        self.dp1 = Dropout(self.dropout)

    def call(self, inputs, query=None):
        # Attention and Normalization
        if query == None:
            query = inputs
        x = self.mha(query, inputs)
        x = self.dp1(x)
        res = self.ln1(x + inputs)

        # Feed Forward Part
        x = self.conv1(res)
        x = self.dp1(x)
        x = self.conv2(x)
        out = self.ln2(x + res)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'head_size': self.head_size,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'last_conv_filters': self.last_conv_filters,
        })
        return config
    

class CategoricalDense(tf.keras.layers.Layer):
    def __init__(self, vocab_size_dict, output_units=16, dense_layers=[64, 32], dropout=0.25, **kwargs):

        super(CategoricalDense, self).__init__(**kwargs)
        self.vocab_size_dict = vocab_size_dict
        self.output_units = output_units
        self.dense_layers = dense_layers
        self.dropout = dropout

        self.initializer = tf.keras.initializers.GlorotNormal(seed=42)
        self.embeddings = [
            Embedding(input_dim=vocab_size, output_dim=int(vocab_size ** .25) + 1)
            for vocab_size in vocab_size_dict.values()
        ]
        self.denses = [
            Sequential([Dense(dim, activation='relu', kernel_initializer=self.initializer), Dropout(dropout)])
            for dim in dense_layers
        ]
        self.outputs = Dense(output_units, activation='relu', kernel_initializer=self.initializer)

    def call(self, inputs):

        embedding_outputs = []
        for idx, embedding_layer in enumerate(self.embeddings):
            embedding_outputs.append(Flatten()(embedding_layer(inputs[..., idx])))
        x = concatenate(embedding_outputs)
        for dense in self.denses:
            x = dense(x)
        outputs = self.outputs(x)

        return outputs

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'vocab_size_dict': self.vocab_size_dict,
            'output_units': self.output_units,
            'dense_layers': self.dense_layers,
            'dropout': self.dropout
        })
        return config
    

class TransformerEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, layer_num, ff_dim, **kwargs):
        super(TransformerEmbeddingLayer, self).__init__(**kwargs)
        self.head_size, self.num_heads, self.layer_num, self.ff_dim = head_size, num_heads, layer_num, ff_dim
        self.encoders = [
            AttentionEncoder(head_size=head_size, num_heads=num_heads,
                             ff_dim=ff_dim, last_conv_filters=ff_dim)
            for _ in range(layer_num)
        ]

    def call(self, inputs, query=None):
        x = inputs
        for encoder in self.encoders:
            x = encoder(x, query)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'head_size': self.head_size,
            'num_heads': self.num_heads,
            'layer_num': self.layer_num,
            'ff_dim': self.ff_dim
        })
        return config