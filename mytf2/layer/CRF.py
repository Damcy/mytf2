import tensorflow as tf
import tensorflow_addons as tfa


@tf.keras.utils.register_keras_serializable(package="mytf2")
class CRF(tf.keras.layers.Layer):
    def __init__(self, label_size):
        super(CRF, self).__init__()
        self.trans_params = tf.Variable(
            tf.random.uniform(shape=(label_size, label_size)),
            name="transition"
        )
    
    @tf.function
    def call(self, inputs):
        logits, seq_lens, labels = inputs
        pred_ids, _ = tfa.text.crf_decode(logits, self.trans_params, seq_lens)
        log_likelihood, _ = tfa.text.crf_log_likelihood(
                                                logits, labels, seq_lens,
                                                transition_params=self.trans_params)
        return log_likelihood, pred_ids
