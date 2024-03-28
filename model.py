import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LayerNormalization
from bagelcode.ml_model.model import TransformerEmbeddingLayer, CategoricalDense

EMBEDDING_SIZE = 24

class TemporalConvBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, stride=1, res_conv=False):
        super(TemporalConvBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.stride = stride
        self.res_conv = res_conv
        self.conv1 = Conv1D(embedding_dim/4, kernel_size=10, strides=stride, padding='valid')
        self.ln1 = LayerNormalization(epsilon=1e-6)
        
        self.conv2 = Conv1D(embedding_dim/2, kernel_size=10, strides=stride, padding='valid')
        self.ln2 = LayerNormalization(epsilon=1e-6)

        self.conv3 = Conv1D(embedding_dim, kernel_size=10, strides=stride, padding='valid')
        self.ln3 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.ln1(x)
        x = tfa.layers.GELU()(x)
    
        x = self.conv2(x)
        x = self.ln2(x)

        x = self.conv3(x)
        x = self.ln3(x)

        out = tfa.layers.GELU()(x)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embedding_dim' : self.embedding_dim,
            'stride' : self.stride,
            'res_conv' : self.res_conv
        })
        return config


class EncoderIncubator(tf.keras.Model):
    def __init__(self, window_length, feature_length, **kwargs):
        super().__init__(**kwargs)
        self.window_length = window_length
        self.feature_length = feature_length
        self.embedding_size = EMBEDDING_SIZE
        
        hidden_size = self.embedding_size * 2
        dense_dims = [
            self.embedding_size * 2**i for i in range(1, 3)
            if self.embedding_size * 2**i < window_length * feature_length
        ] + [window_length * feature_length]
        
        self.encoder = TemporalConvBlock(hidden_size)
        self.decoder = Sequential([Dense(dim, activation='relu') for dim in dense_dims])

    def call(self, inputs):
        x = self.encoder(inputs)
        z_mean = x[... , :int(self.embedding_size)]
        z_log_var = x[... , int(self.embedding_size):]
        z = z_mean

        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss = K.mean(kl_loss)
        self.add_loss(kl_loss)

        x = self.decoder(z)
        return x


class TransForeCaster(tf.keras.Model):
    def __init__(
        self,
        behavior_length,
        portrait_length,
        window_length,
        target_day,
        vocab_size_dict,
        encoding_layers,
        behavior_category_indice,
        portrait_category_indice,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.behavior_length = behavior_length
        self.portrait_length = portrait_length
        self.window_length = window_length
        self.target_day = target_day
        self.vocab_size_dict = vocab_size_dict
        self.encoding_layers = encoding_layers
        self.behavior_category_indice = behavior_category_indice
        self.portrait_category_indice = portrait_category_indice

        embedding_dim = EMBEDDING_SIZE
        mha_num, head_size, num_heads, ff_dim = 2, embedding_dim, 8, embedding_dim
        user_info_embed_ff_dims = [embedding_dim]
        status_embed_ff_dims = [64]
        output_ff_dims = [1024, 512, 256]

        self.inter_category_norm = LayerNormalization(epsilon=1e-6)
        self.inter_category_dense = Dense(units=embedding_dim, activation='relu', use_bias=False)
        self.trans_enc = TransformerEmbeddingLayer(
            head_size=head_size,
            num_heads=num_heads,
            layer_num=mha_num,
            ff_dim=ff_dim,
            name='trans_enc'
        )
        self.context_norm = LayerNormalization(epsilon=1e-6)
        self.context_dense = Dense(units=embedding_dim, activation='relu', use_bias=False)
        self.user_info_embed = CategoricalDense(
            self.vocab_size_dict,
            output_units=embedding_dim,
            dense_layers=user_info_embed_ff_dims,
            dropout=0.2,
            name='user_info_embed'
        )
        self.status_embed = Sequential(
            [Dense(dim, activation='relu') for dim in status_embed_ff_dims + [embedding_dim]],
            name='status_embed'
        )
        self.output_head = Sequential(
            [Dense(dim) for dim in output_ff_dims] + [Dense(self.target_day, activation='relu')],
            name='output_head'
        )

    def process_inputs(self, inputs):
        user_input, behavior_input, portrait_input = inputs[0], inputs[1], inputs[2]
        all_inputs = [
            tf.gather(behavior_input, behavior_idx, axis=-1) for behavior_idx in self.behavior_category_indice
        ] + [
            tf.gather(portrait_input, portrait_idx, axis=-1) for portrait_idx in self.portrait_category_indice
        ]
        return user_input, portrait_input, all_inputs

    def get_inter_category_embedding(self, all_inputs):
        category_encoder_outputs = [
            tf.squeeze(encoder(inputs), axis=1) for encoder, inputs in zip(self.encoding_layers, all_inputs)
        ]
        category_embed = tf.stack(category_encoder_outputs, axis=1)

        embedding_size = category_embed.shape[-1]//2
        z_mean = category_embed[..., :embedding_size]
        category_embed = z_mean

        category_embed = self.inter_category_dense(category_embed)
        category_embed = self.inter_category_norm(category_embed)
        return category_embed

    def get_cross_category_embedding(self, user_input, portrait_input, inter_category_embed):
        user_info_embed = self.user_info_embed(user_input)
        status_embed = self.status_embed(portrait_input[:,-1,:])
        static_embed = user_info_embed + status_embed
        query = tf.expand_dims(static_embed, axis=1)

        contexts = self.trans_enc(inter_category_embed, query)
        context = tf.math.reduce_mean(contexts, axis=1)
        context = self.context_norm(context)
        output = self.context_dense(context)
        return output

    def call(self, inputs):
        user_input, portrait_input, all_inputs = self.process_inputs(inputs)
        inter_category_embed = self.get_inter_category_embedding(all_inputs)
        output = self.get_cross_category_embedding(user_input, portrait_input, inter_category_embed)
        output = self.output_head(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'behavior_length': self.behavior_length,
            'portrait_length': self.portrait_length,
            'window_length': self.window_length,
            'target_day': self.target_day,
            'vocab_size_dict': self.vocab_size_dict,
            'encoding_layers': self.encoding_layers,
            'behavior_category_indice': self.behavior_category_indice,
            'portrait_category_indice': self.portrait_category_indice,
        })
        return config
