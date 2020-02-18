""" Autoencoder model for Novelty detection"""

from sklearn.svm import OneClassSVM
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Lambda, Activation
from tensorflow.keras.layers import BatchNormalization, ReLU, LeakyReLU, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2


def keras_autoencoder(original_image_shape, filters, kernels, strides, units, BN=False,
                      dropout_rate=0.1, activation=LeakyReLU(), reg = l2(0.000)):


    inputs = Input(shape=original_image_shape, name="encoder_input")
    input_shape = {}
    x = inputs

    input_shape["conv"] = []
    for f, k, s in zip(filters, kernels, strides):
        input_shape["conv"].append(x.get_shape().as_list()[1:3])
        x = Conv2D(filters=f,
                   kernel_size=k,
                   strides=s,
                   padding="same",
                   data_format="channels_last",
                   activation=None,
                   activity_regularizer=reg)(x)
        x = BatchNormalization()(x) if BN  else x
        x = activation(x)

    input_shape['flatten'] = x.get_shape().as_list()[1:]
    x = Flatten()(x)
    input_shape["first_dense"] = x.get_shape().as_list()[1]

    for i, u in enumerate(units):
        x = Dense(units=u, activation=None)(x)
        if i < len(units) - 1:
            x = Dropout(rate=dropout_rate)(x)
            x = BatchNormalization()(x) if BN  else x
            x = activation(x)

    # encoder = Model(inputs, x,  name="encoder")
    encoder = Lambda(lambda t: t, name='encoder')(x)


    ##build decoder model
    dec_filters = list(reversed([original_image_shape[-1]] + filters[:-1]))
    dec_kernels = list(reversed(kernels))
    dec_strides = list(reversed(strides))
    dec_cropping = list(reversed(input_shape["conv"]))
    dec_units = list(reversed([input_shape['first_dense']] + units[:-1]))


    for u in dec_units:
        x = Dense(units=u, activation=None)(x)
        x = Dropout(rate=dropout_rate)(x)
        x = BatchNormalization()(x) if BN else x
        x = activation(x)

    x = Reshape(target_shape=input_shape["flatten"])(x)

    for i, (f, k, s, (c0, c1))  in enumerate(zip(dec_filters, dec_kernels, dec_strides, dec_cropping)):
        x = Conv2DTranspose(filters=f,
                            kernel_size=k,
                            strides=s,
                            padding="same",
                            activation=None,
                            activity_regularizer=reg)(x)
        x = Lambda(
            function=lambda t: t[:, :c0, :c1, :],
            output_shape=(c0, c1, f)
        )(x)

        if i < len(units) - 1:

            x = BatchNormalization()(x) if BN  else x
            x = activation(x)

    reconstruction_error = Lambda(
        function=lambda pair: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=pair[0],
                                                                                    logits=pair[1]), axis=(1, 2, 3))
        , output_shape=(1,), name="error")([inputs, x])

    x = Activation("sigmoid")(x)
    decoder = Lambda(lambda t: t, name='decoder')(x)

    # reconstruction_error = Lambda(
    #     function=lambda pair: tf.sqrt(tf.reduce_sum((pair[0] - pair[1]) ** 2, axis=(1, 2, 3)) /
    #                                   tf.reduce_sum(pair[0] ** 2, axis=(1, 2, 3))),
    #     output_shape=(1,),
    #     name="error"
    #     )([inputs, decoder])
    # reconstruction_error = Lambda(
    #     function=lambda pair: tf.sqrt(tf.reduce_sum((pair[0] - pair[1]) ** 2, axis=(1, 2, 3)))
    #                                   ,output_shape=(1,), name="error")([inputs, decoder])



    # bce = tf.keras.losses.BinaryCrossentropy()
    #
    # reconstruction_error = Lambda(
    #     function=lambda pair: tf.reduce_sum(bce(pair[0], pair[1]), axis=(1,2))
    #     , output_shape=(1,), name="error")([inputs, decoder])

    # reconstruction_error = Lambda(
    #     function=lambda pair: tf.reduce_sum((-1/2)*tf.reduce_sum(pair[0]*tf.math.log(pair[1])+((1-pair[0])*tf.math.log(1-pair[1]))), axis=0),
    #     output_shape=(1,), name="error")([inputs, decoder])
    # print("............reconstruction.....", reconstruction_error.shape)

    model = Model(
        inputs=inputs,
        outputs=[encoder, decoder, reconstruction_error], name="autoencoder_model"
    )

    model.summary()

    return model





