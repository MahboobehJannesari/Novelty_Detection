import tensorflow as tf
from tensorflow.python.ops.summary_ops_v2 import always_record_summaries
from tensorflow.python.eager import context
from tensorflow.keras.callbacks import Callback

class OCSVMLayer(tf.keras.layers.Layer):

    def __init__(self, n_svs, n_features):
        super(OCSVMLayer, self).__init__()
        self.num_outputs = 1
        self.output_dim = 1

        self.n_svs = n_svs
        self.n_features = n_features

        self.support_vectors = self.add_variable(
            name='support_vectors',
            shape=(self.n_svs, self.n_features),
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=0.0),
            trainable=False
        )
        self.coefficients = self.add_variable(
            name='coefficients',
            shape=(1, self.n_svs),
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=0.0),
            trainable=False
        )
        self.rho = self.add_variable(
            name='rho',
            shape=(1,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=0.0),
            trainable=False
        )
        self.gamma = self.add_variable(
            name='gamma',
            shape=(),
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=0.0),
            trainable=False
        )

    def build(self, input_shape):
        super(OCSVMLayer, self).build(input_shape)

    def call(self, inputs):

        # inputs must be of shape [n_samples, n_features]
        x = tf.expand_dims(inputs, axis=1) - tf.expand_dims(self.support_vectors, axis=0)       # (n_samples, n_svs, n_features)
        x = tf.reduce_sum(tf.square(x), axis=2)                                                 # (n_samples, n_svs)
        x = tf.math.exp(tf.multiply(x, -self.gamma))
        x = tf.reduce_sum(tf.multiply(x, self.coefficients), axis=1)                            # (n_samples)
        x = x - self.rho

        # # TODO think about that?
        # x = tf.greater_equal(x, 0)

        return x

    def get_config(self):
        return {'n_svs': self.n_svs, 'n_features': self.n_features}


def ocsvm_layer(sklearn_model):

    assert(sklearn_model.kernel == 'rbf')
    layer = OCSVMLayer(n_svs=sklearn_model.support_vectors_.shape[0], n_features=sklearn_model.support_vectors_.shape[1])

    layer.support_vectors.assign(sklearn_model.support_vectors_)
    layer.coefficients.assign(sklearn_model.dual_coef_)
    layer.rho.assign(sklearn_model.offset_)
    layer.gamma.assign(sklearn_model.gamma)

    return layer


class NormLayer(tf.keras.layers.Layer):

    def __init__(self, n_features):
        super(NormLayer, self).__init__()
        self.num_outputs = 1
        self.n_features = n_features

        self.mean = self.add_variable(
            name='mean',
            shape=(self.n_features,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=0.0),
            trainable=False
        )
        self.stddev = self.add_variable(
            name='stddev',
            shape=(self.n_features,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=0.0),
            trainable=False
        )

    def build(self, input_shape):
        super(NormLayer, self).build(input_shape)

    def call(self, inputs):
        return (inputs - self.mean)/self.stddev

    def get_config(self):
        return {'n_features': self.n_features}


def norm_layer(mean, stddev):
    layer = NormLayer(n_features=mean.shape[0])
    layer.mean.assign(mean)
    layer.stddev.assign(stddev)

    return layer

class OffsetLayer(tf.keras.layers.Layer):

    def __init__(self, threshold):
        super(OffsetLayer, self).__init__()
        self.num_outputs = 1
        self.threshold = threshold

    def build(self, input_shape):
        self.threshold_var = self.add_variable(
            name='threshold',
            shape=(1,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=self.threshold),
            trainable=False
        )
        super(OffsetLayer, self).build(input_shape)

    def call(self, inputs):
        return self.threshold_var - inputs

    def get_config(self):
        return {'threshold': self.threshold}



class TensorBoardImage(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, updata_freq=100):
        super(TensorBoardImage, self).__init__()
        self.log_dir =log_dir
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.update_freq = updata_freq

        self.count = 0

    def on_batch_end(self, batch, logs=None):
        if self.count % self.update_freq == 0:
            # with context.eager_mode(), self.summary_writer.as_default(), always_record_summaries():
            with self.summary_writer.as_default():
                # summary = tf.summary.scalar('original', self.model.output[2])
                tf.summary.image('original', self.model.input, step=batch)



class TestCallback(Callback):
    def __init__(self, test_data, steps, updata_freq=200):
        self.test_data = test_data
        self.steps = steps
        self.update_freq = updata_freq
        self.count = 0

    def on_batch_end(self, batch, logs={}):
        x= self.test_data
        if self.count % self.update_freq == 0:

            loss = self.model.evaluate(x, steps=self.steps, verbose=0)
            print('\nTesting loss: {}'.format(loss))
        self.count +=1

class Histories(Callback):

    def on_train_begin(self,logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))
