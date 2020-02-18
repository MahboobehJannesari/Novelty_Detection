"""
Novelty detection
    Autoencoder + ocsvm
    tensorflow version: 2.0.0-alpha0
"""

import sys
sys.path.insert(0, '/usr/local/lib/python3.6/dist-packages')
import tensorflow as tf
import os
import imageio
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from tensorflow.keras.layers import Lambda
from matplotlib.patches import Rectangle
from sklearn.model_selection import ParameterGrid
from sklearn import svm
from ocsvm_model import *
from autoencoder_model import *
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import client

class NoveltyDetection(object):

    def __init__(self, config, train_number):
        self.train_number = train_number

        # image info
        self.img_channel = config.getint("Image", "channel")
        self.patch_height = config.getint("Image", "patch_height")
        self.patch_width = config.getint("Image", "patch_width")
        self.stride = config.getint("Image", "stride")

        # autoencoder parameters
        self.filters = [int(i) for i in config.get("Autoencoder", "filters").split(",")]
        self.kernel_size = [int(i) for i in config.get("Autoencoder", "kernel_size").split(",")]
        self.strides = [int(i) for i in config.get("Autoencoder", "strides").split(",")]
        self.units = [int(i) for i in config.get("Autoencoder", "units").split(",")]
        self.lr = config.getfloat("Autoencoder", "learning_rate")
        self.BN = config.getboolean("Autoencoder", "batch_normalization")
        self.dropout_rate = config.getfloat("Autoencoder", "dropout_rate")
        self.auto_threshold = config.getfloat("Autoencoder", "threshold")

        self.input_shape = (self.patch_height, self.patch_width, self.img_channel)

        # autoencoder part - trained in train_autoencoder
        self.autoencoder_model = None

        # ocsvm part - trained in train_ocsvm
        self.ocsvm_model = None

        # unified models of autoencoder and ocsvm built from self.autoencoder_model and self.keras_ocsvm
        self.novelty_detector_model = None

        # model including input layer, novelty_detector_model and output layer
        self.patch_model = None
        self.patch_random_model = None


    def _read_image(self, img_path):
        """
        read images as tensors and normalize them per image
        :param img_path: path of images
        :return:
        """
        img_file = tf.io.read_file(img_path)
        image = tf.image.decode_png(img_file, channels=self.img_channel, dtype=tf.uint16)
        image =(tf.subtract(image, tf.reduce_min(image))/
                      tf.subtract(tf.reduce_max(image),tf.reduce_min(image)))
        return image


    def logs_mkdir(self, logdir):
        """
        create new directory for logs
        :return: log_dir path
        """
        if not os.path.exists(logdir):
            log_dir = logdir + "/log0"
            os.makedirs(log_dir)
        else:
            log_dir = logdir + "/log" + str(max([int(s[3:]) for s in os.listdir(logdir)]) + 1)
            os.mkdir(log_dir)
        return log_dir


    def _create_patch(self, images):
        """
        create patches using tensorflow function
        :param images: image tensor of dimension 3
        :return:
        """
        patch_size = [1, self.patch_height, self.patch_width, self.img_channel]
        patches = tf.image.extract_image_patches(images,
                                           sizes=patch_size,
                                           strides=[1, self.stride, self.stride, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        x = tf.reshape(patches, shape=[-1, self.patch_height, self.patch_width, self.img_channel])
        x = tf.expand_dims(x, axis=0)

        return x


    def _create_patch_random(self, images):
        """
        create patches randomly using tensorflow function
        :param images: image tensor of dimension 3
        :return:
        """

        offset_height = tf.random.uniform(shape=(), minval=0, maxval=self.stride-1,
                                         dtype=tf.int32, seed=0)
        offset_width = tf.random.uniform(shape=(), minval=0, maxval= self.stride-1,
                                        dtype=tf.int32, seed=0)
        images = images[:, offset_height:, offset_width:, :]
        patch_size = [1, self.patch_height, self.patch_width, self.img_channel]
        patches = tf.image.extract_image_patches(images,
                                                 sizes=patch_size,
                                                 strides=[1, self.stride, self.stride, 1],
                                                 rates=[1, 1, 1, 1],
                                                 padding="VALID")
        x = tf.reshape(patches, shape=[-1, self.patch_height, self.patch_width,
                                       self.img_channel])
        tf.print("number of patches:    ", tf.shape(x))
        x = tf.expand_dims(x, axis=0)
        return x

    def flip(self, images):
        """
        :param images:
        :return: fliped images
        """
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)

        return images


    def double_fn(self, x):
        return x, x

    def _create_dataset(self, imgs_list, batch_size, create_patch=True, shuffle=True):

        dataset = tf.data.Dataset.from_tensor_slices(imgs_list)
        dataset = dataset.shuffle(buffer_size=len(imgs_list))
        dataset = dataset.map(self._read_image, num_parallel_calls=4)
        dataset = dataset.map(tf.image.rot90)


        if create_patch:
            dataset = dataset.batch(1)
            dataset = dataset.map(self._create_patch_random, num_parallel_calls=4)
            dataset = dataset.apply(tf.data.experimental.unbatch())
            dataset = dataset.apply(tf.data.experimental.unbatch())
            dataset = dataset.shuffle(buffer_size=500) if shuffle else dataset

        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        # dataset = dataset.map(self.flip)
        dataset = dataset.prefetch(10)
        dataset = dataset.map(self.double_fn)
        return dataset


    def train_autoencoder(self, data_path, logs_dir, num_epochs , batch_size,
                                   autoencoder_model_path=None, steps=None, create_patch=True):

        dataset_train = self._create_dataset(data_path[0], batch_size, create_patch)
        dataset_eval = self._create_dataset(data_path[1], batch_size, create_patch)

        if steps is None:
            train_steps = self.count_patch(data_path[0], train=True) // batch_size
            eval_steps = self.count_patch(data_path[1], train=True) // batch_size
            steps = (train_steps, eval_steps)

        self.autoencoder_model = keras_autoencoder(self.input_shape,
                                                   self.filters, self.kernel_size,
                                                   self.strides, self.units,
                                                   BN=self.BN,
                                                   dropout_rate=self.dropout_rate)

        self.autoencoder_model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr),
                                       loss={"decoder":tf.keras.losses.BinaryCrossentropy()},
                                       metrics=[tf.keras.losses.BinaryCrossentropy()])

        logsdir = self.logs_mkdir(logs_dir)
        tensorboard = TensorBoard(log_dir=logsdir, write_images=True, write_graph=True, update_freq="batch")
        histories = Histories()

        self.autoencoder_model.fit(dataset_train,
                             y=None,
                             steps_per_epoch=steps[0],
                             epochs=num_epochs,
                             validation_data=dataset_eval,
                             validation_steps=steps[1],
                             shuffle=False,
                             verbose=1,
                             # callbacks=[tensorboard])
                             callbacks=[tensorboard, TestCallback(dataset_eval, steps[1]), histories])
        tf.keras.experimental.export_saved_model(self.autoencoder_model, autoencoder_model_path)


    def train_ocsvm(self, train_data, nu, gamma):
        """

        :param nu:
        :param gamma:
        :return: self.ocvm_model
        """

        print("Training sklearn model ...")
        mean = train_data.mean(axis=0)
        stddev = train_data.std(axis=0)
        train_data = (train_data - mean) / stddev

        print("Training OCSVM ...")
        sklearn_model = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma).fit(train_data)

        inputs = tf.keras.layers.Input(train_data.shape[1:])
        normalized_encoder = norm_layer(mean, stddev)(inputs)
        ocsvm_pred = ocsvm_layer(sklearn_model)(normalized_encoder)
        self.ocsvm_model = tf.keras.Model(inputs=inputs, outputs=ocsvm_pred, name="ocsvm_model")


    def merge_models(self):
        inputs = tf.keras.layers.Input(self.input_shape)
        encoded, decoded, error = self.autoencoder_model(inputs)
        rec_score =  OffsetLayer(self.rec_error_thr)(error)
        score = self.ocsvm_model(encoded)
        self.novelty_detector_model = tf.keras.models.Model(inputs=inputs, outputs=[rec_score, score])


    def train_fn(self, workspace_dir, data_path, logs_dir, num_epochs, batch_size,
                          nu, gamma, train_autoencoder=True,
                          autoencoder_model_path=None, steps=None, create_patch=True, encoder_saved=True):
        """

        :param data_path: [train_data_path, test_data_path]
        :param logs_dir: log_dir path
        :param num_epochs: number of epochs for training autoencoder
        :param batch_size:
        :param nu: ocsvm parameter
        :param gamma: ocsvm parameter
        :param train_autoencoder: True or False
        :param autoencoder_model_path:
        :param steps: number of steps for trainig keras model
        :param create_patch: True or False
        :param encoder_saved: if True, encoded vector will be saved as .npy
        :return:
        """

        if steps is None:
            train_steps = self.count_patch(data_path[0]) // batch_size
            eval_steps = self.count_patch(data_path[1]) // batch_size
            steps = (train_steps, eval_steps)

        if train_autoencoder:
            print("Training Autoencoder ...")
            self.train_autoencoder(data_path, logs_dir, num_epochs , batch_size,
                                   autoencoder_model_path=autoencoder_model_path,
                                   steps=steps, create_patch=create_patch)
        else:
            print("Loading Autoencoder  ...")
            self.autoencoder_model = tf.keras.experimental.load_from_saved_model(autoencoder_model_path)

        self.make_patch_random_model()
        encoded_data_train = None
        error_train = None
        encoded_data_test = None
        error_test = None
        i = self.train_number

        if encoder_saved:
            print("Extracting encoder layer ...")
            for img in data_path[0]:
                input = imageio.imread(img)
                image = np.asarray(input)
                image = image.astype("float32")
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                image = np.expand_dims(image, axis=2)
                image = np.expand_dims(image, axis=0)
                patches = self.patch_model.predict(image)
                patches = np.squeeze(patches, axis=0)
                encoded_data_train1, _, error_train1 = self.autoencoder_model.predict(patches)
                print("............... patches............  ", patches.shape, encoded_data_train1.shape)
                encoded_data_train = np.concatenate((encoded_data_train, encoded_data_train1),
                                                    axis=0) if encoded_data_train is not None else encoded_data_train1
                error_train = np.concatenate((error_train, error_train1), axis=0) if error_train is not None else error_train1

            for img in data_path[1]:
                input = imageio.imread(img)
                image = np.asarray(input)
                image = image.astype("float32")
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                image = np.expand_dims(image, axis=2)
                image = np.expand_dims(image, axis=0)
                patches = self.patch_model.predict(image)
                patches = np.squeeze(patches, axis=0)
                encoded_data_test1, _, error_test1 = self.autoencoder_model.predict(patches)
                encoded_data_test = np.concatenate((encoded_data_test, encoded_data_test1),
                                                    axis=0) if encoded_data_test is not None else encoded_data_test1
                error_test = np.concatenate((error_test, error_test1),
                                             axis=0) if error_test is not None else error_test1

            print("saving_results....")
            encoder_output = os.path.join(workspace_dir, "encoder_output")
            if not os.path.exists(os.path.join(workspace_dir, "encoder_output")):
                os.makedirs(os.path.join(workspace_dir, "encoder_output"))
            np.save(encoder_output + "/encoded_data_train%d"%i, encoded_data_train)
            np.save(encoder_output + "/encoded_data_test%d"%i, encoded_data_test)
            np.save(encoder_output + "/error_train%d"%i, error_train)
            np.save(encoder_output + "/error_test%d"%i, error_test)

        print("loading_results....")
        encoded_data_train = np.load(encoder_output +"/encoded_data_train%d.npy"%i)
        error_train = np.load(encoder_output + "/error_train%d.npy"%i)
        encoded_data_test = np.load(encoder_output + "/encoded_data_test%d.npy"%i)
        print("encoded_train...........", encoded_data_train.shape)

        self.rec_error_thr = np.quantile(error_train, q=self.auto_threshold)
        # ocsvm_parms = self.train_ocsvm_gridsearch(encoded_data_train, encoded_data_test)
        # self.train_ocsvm(encoded_data_train, ocsvm_parms["nu"], ocsvm_parms["gamma"])
        self.train_ocsvm(encoded_data_train, nu, gamma)
        self.merge_models()

    def make_patch_random_model(self):
        """
        makes a model to create patches rendomly
        :return:
        """
        input = tf.keras.layers.Input(dtype=tf.float32, shape=[None, None, self.img_channel])
        patches = Lambda(lambda img: self._create_patch_random(img),
                         output_shape=[None, self.patch_height, self.patch_width, self.img_channel])(input)
        self.patch_model = tf.keras.models.Model(input, patches)

    def make_patch_model(self):
        """
        makes a model to create patches
        :return:
        """
        input = tf.keras.layers.Input(dtype=tf.float32, shape=[None, None, self.img_channel])
        patches = Lambda(lambda img: self._create_patch(img),
                         output_shape=[None, self.patch_height, self.patch_width, self.img_channel])(input)
        self.patch_model = tf.keras.models.Model(input, patches)



    def save_model(self, model_path):
        """
        :param model_path: path for saving model
        """
        print("Saving Models....")
        tf.keras.experimental.export_saved_model(self.novelty_detector_model, model_path,
                                                 custom_objects={"NormLayer": NormLayer,
                                                                 "OCSVMLayer": OCSVMLayer,
                                                                 "OffsetLayer": OffsetLayer
                                                                 })
        print("The model is saved in", model_path)



    def load_model(self, model_path):
        """
        load model from model_path
        """
        print("Loding Model from {}".format(model_path))
        self.novelty_detector_model = tf.keras.experimental.load_from_saved_model(model_path,
                                                    custom_objects={"NormLayer": NormLayer,
                                                                    "OCSVMLayer": OCSVMLayer,
                                                                    "OffsetLayer": OffsetLayer})

    def _patch_location(self, image):
        """
        :param images: a tensor of image_path
        :return: list of patches
        """

        img_height, img_width = image.shape[1:3]
        num_patches_col = math.floor((img_width - self.patch_width) / self.stride) + 1
        num_patches_row = math.floor((img_height - self.patch_height) / self.stride) + 1

        pos_y = np.tile(np.arange(num_patches_col), (num_patches_row, 1))
        pos_y = np.expand_dims(pos_y, axis=-1)

        pos_x = np.transpose(np.tile(np.arange(num_patches_row), (num_patches_col, 1)))
        pos_x = np.expand_dims(pos_x, axis=-1)

        pos = np.concatenate((pos_x, pos_y), axis=-1)
        pos = np.reshape(pos, (num_patches_row * num_patches_col, 2))

        return pos

    def predict(self, image):
        """
        predict for single image
        :param image: an array
        :return: list of nok patches predicted by autoencoder and ocsvm
        """

        patches = self.patch_model.predict(image)
        patches = np.squeeze(patches, axis=0)

        n_col = math.floor((image.shape[2] - self.patch_width) / self.stride) + 1
        n_row = math.floor((image.shape[1] - self.patch_height) / self.stride) + 1

        ## visualization reconstructed images
        visualization = 0
        if visualization:
            self.autoencoder_model = tf.keras.experimental.load_from_saved_model(
                "../saved_models/autoencoder_model%d" % self.train_number)

            encoded, decoded, error = self.autoencoder_model.predict(patches)


            f, a = plt.subplots(n_row, 2 * n_col, figsize=(50, 50), frameon=False)
            for i in range(n_row):
                for j in range(n_col):
                    a[i][j].imshow(np.reshape(decoded[(i * n_col) + j], (self.patch_height, self.patch_width)))
                    a[i][j].set_title(round(error[(i * n_col) + j], 2))
                    a[i][j + n_col].imshow(np.reshape(patches[(i * n_col) + j], (self.patch_height, self.patch_width)))
            plt.show()

        rec_score, ocsvm_score = self.novelty_detector_model.predict(patches)
        patches_pos = self._patch_location(image)
        nok_patches_ocsvm = patches_pos[ocsvm_score < 0]
        nok_patches_auto = patches_pos[rec_score < 0]

        return  nok_patches_auto, nok_patches_ocsvm


    def predict_client(self, image):
        """
        prediction with python client
        :param image: an array of image
        :param batch_size:
        :return: list of nok patches predicted by autoencoder and ocsvm
        """

        patches = self.patch_model.predict(image)
        patches = np.squeeze(patches, axis=0)
        # shape = (batch_size, self.patch_height, self.patch_width, self.img_channel)

        rec_score, ocsvm_score = client.predict(patches, patches.shape)

        patches_pos = self._patch_location(image)
        nok_patches_ocsvm = patches_pos[ocsvm_score < 0]
        nok_patches_auto = patches_pos[rec_score < 0]
        return nok_patches_auto, nok_patches_ocsvm



    def visual_res(self, image, boxes1=[], boxes2=[]):
        """
        draw rectangles for nok patches on the image
        :param image:
        :param boxes1: list of top and left coordinates of rectangles
        :param boxes2: list of top and left coordinates of rectangles
        """

        plt.imshow(image)
        for coord in boxes1:
            plt.gca().add_patch(
                Rectangle((self.stride * coord[1], self.stride * coord[0]),
                          self.patch_height, self.patch_width,
                          linewidth=1, edgecolor='r',
                          facecolor='none'))

        for coord in boxes2:
            plt.gca().add_patch(
                Rectangle((self.stride * coord[1], self.stride * coord[0]),
                          self.patch_height, self.patch_width,
                          linewidth=1, edgecolor='g',
                          facecolor='none'))
        plt.show()



    def count_patch(self, imgs_list, train=False):
        """
        count number of patches for all images in imgs_list
        :param imgs_path: path of images
        :return: number of all patches
        """
        total_patches = 0
        import cv2
        for img in imgs_list:
            # image = scp.misc.imread(img)
            image = cv2.imread(img)
            img_height, img_width,_ = image.shape

            num_patches_col = math.floor((img_width - self.patch_width) / self.stride) + 1
            num_patches_row = math.floor((img_height - self.patch_height) / self.stride) + 1
            if train:
                total_patches += (num_patches_col-1) * (num_patches_row-1)
            else:
                total_patches += num_patches_col * num_patches_row

        return total_patches


    def train_ocsvm_gridsearch(self, train_data, test_data):
        """
        find the best nu and gamma for ocsvm
        :param train_data: train dataset
        :param test_data: test dataset
        :return:
        """

        print("Training sklearn model ...")
        mean = train_data.mean(axis=0)
        stddev = train_data.std(axis=0)
        train_data = (train_data - mean) / stddev
        test_data = (test_data - mean) / stddev

        clf = svm.OneClassSVM()
        grid = {'gamma': [0.001,0.005, 0.0001, 0.0005, 0.00001],
                'nu': [ 0.001,0.005, 0.0001, 0.0005, 0.00001]}

        results = []
        param = []

        for z in ParameterGrid(grid):
            clf.set_params(**z)
            clf.fit(train_data)

            y_pred_train = clf.predict(train_data)
            y_pred_test = clf.predict(test_data)

            acc_train = sum([1 for x in y_pred_train if x>=0])/len(y_pred_train)
            acc_test = sum([1 for x in y_pred_test if x>=0])/len(y_pred_test)

            print("Accuracy train", z, sum([1 for x in y_pred_train if x>=0]), acc_train)
            print("Accuracy test", z,sum([1 for x in y_pred_test if x>=0]),len(y_pred_test),  acc_test)

            param.append(z)
            results.append(acc_test)

        print("The best parameters", param[results.index(max(results))])

        return  param[results.index(max(results))]





