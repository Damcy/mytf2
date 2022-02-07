import tensorflow as tf

from mytf2.layer.embedding import OnDeviceEmbedding
from mytf2.layer.embedding import PositionEmbedding
from mytf2.layer.transformer import TransformerEncoder
from mytf2.layer.attention import SelfAttentionMask
from mytf2.layer.activation import get_activation, serialize_activation


@tf.keras.utils.register_keras_serializable(package="mytf2")
class BertEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 max_position_embeddings=512,
                 intermediate_size=3072,
                 type_vocab_size=2,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 **kwargs):
        super(BertEncoder, self).__init__()

        self.activation = get_activation(hidden_act)
        self.initializer = tf.keras.initializers.get(initializer)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.type_vocab_size = type_vocab_size
        self.dropout_rate = hidden_dropout_prob
        self.attention_dropout_rate = attention_probs_dropout_prob

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'max_seq_len': self.max_position_embeddings,
            'intermediate_size': self.intermediate_size,
            'type_vocab_size': self.type_vocab_size,
            'activation': serialize_activation(self.activation),
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'initializer': tf.keras.initializers.serialize(self.initializer),
        }
        base_config = super(BertEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        embedding_width = self.hidden_size
        self._word_embedding_layer = OnDeviceEmbedding(
            vocab_size=self.vocab_size,
            embedding_width=embedding_width,
            initializer=self.initializer,
            name='word_embeddings')

        self._position_embedding_layer = PositionEmbedding(
            initializer=self.initializer,
            max_length=self.max_position_embeddings,
            name='position_embeddings')

        self._type_embedding_layer = OnDeviceEmbedding(
            vocab_size=self.type_vocab_size,
            embedding_width=embedding_width,
            initializer=self.initializer,
            use_one_hot=True,
            name='type_embeddings')

        self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
            name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

        self._embedding_dropout_layer = tf.keras.layers.Dropout(
            name='embeddings/dropout', rate=self.dropout_rate)

        self._transformer_layers = []
        for i in range(self.num_hidden_layers):
            layer = TransformerEncoder(
                num_attention_heads=self.num_attention_heads,
                intermediate_dim=self.intermediate_size,
                intermediate_activation=self.activation,
                kernel_initializer=self.initializer,
                output_dropout=self.dropout_rate,
                attention_dropout=self.attention_dropout_rate,
                name='transformer/layer_%d' % i)
            self._transformer_layers.append(layer)

        self._pooler_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation='tanh',
            kernel_initializer=self.initializer,
            name='pooler_transform')

    def call(self, inputs, training=None):
        word_ids, mask, type_ids = inputs
        word_embeddings = self._word_embedding_layer(word_ids)
        position_embeddings = self._position_embedding_layer(word_embeddings)
        type_embeddings = self._type_embedding_layer(type_ids)
        embeddings = tf.keras.layers.Add()(
            [word_embeddings, position_embeddings, type_embeddings])
        embeddings = self._embedding_norm_layer(embeddings)
        embeddings = self._embedding_dropout_layer(embeddings, training)

        attention_mask = SelfAttentionMask()(embeddings, mask)

        encoder_outputs = []
        data = embeddings
        for layer in self._transformer_layers:
            data = layer([data, attention_mask])
            encoder_outputs.append(data)

        first_token_tensor = (
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                encoder_outputs[-1]))

        pooler_output = self._pooler_layer(first_token_tensor)

        return [encoder_outputs, pooler_output]

    def get_embedding_table(self):
        return self._word_embedding_layer.embeddings

    def get_embedding_layer(self):
        return self._word_embedding_layer
