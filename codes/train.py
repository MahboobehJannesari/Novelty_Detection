"""
Novelty detection model: Autoencoder + ocsvm
tensorflow version: 2.0.0-alpha0
prediction on python client with tenosflow seving

python train.py --model_version 1 --workspace_dir ../workspace  --batch_size 32 --epochs 1

"""

import glob
import os
import configparser
from absl import app
from absl import flags
import tensorflow as tf
from novelty_detection import NoveltyDetection
print("tensorflow version:", tf.__version__)

flags.DEFINE_integer("model_version", 1, "The number of model")
flags.DEFINE_string("workspace_dir","../workspace", "main directory for novelty detection task" )
flags.DEFINE_integer("batch_size", 32, "batch size for training autoencoder")
flags.DEFINE_integer("epochs", 1, "number of epochs for training autoencoder")
FLAGS = flags.FLAGS
config = configparser.ConfigParser()

def main(argv):
    print("Model number:", FLAGS.model_version, FLAGS.workspace_dir + "/config%d.ini" % FLAGS.model_version )
    config.read(FLAGS.workspace_dir + "/config%d.ini" % FLAGS.model_version)
    train_path = config.get("Path", "train_path")
    eval_path = config.get("Path", "eval_path")
    logs_dir = FLAGS.workspace_dir + config.get("Path", "logs_dir")
    novelty_detection_model_path = FLAGS.workspace_dir +\
                                   config.get("Path", "novelty_detection_model_path") + "/" + str(FLAGS.model_version)
    autoencoder_model_path = FLAGS.workspace_dir + \
                             config.get("Path", "autoencoder_model_path") + "/" + str(FLAGS.model_version)
    imgs_format = config.get("Image", "format")

    detector = NoveltyDetection(config, FLAGS.model_version)

    train_autoencoder = 1 #train autoencoder
    encoder_saved = 1 #extract encoded vector for training ocsvm
    ## ocsvm training parameters
    nu = 0.0001
    gamma = 0.001

    train_imgs_list = glob.glob(train_path + "/*." + imgs_format)
    eval_imgs_list = glob.glob(eval_path + "/*." + imgs_format)
    data_path = (train_imgs_list, eval_imgs_list)

    print("Number of train images and patches", len(data_path[0]), detector.count_patch(data_path[0]))
    print("Number of Images for validation and patches", len(data_path[1]), detector.count_patch(data_path[1]))
    if not os.path.exists(autoencoder_model_path):
        os.makedirs(autoencoder_model_path)

    detector.train_fn(FLAGS.workspace_dir, data_path, logs_dir, num_epochs=FLAGS.epochs , batch_size=FLAGS.batch_size,
                      nu=nu, gamma=gamma, train_autoencoder=train_autoencoder,
                      autoencoder_model_path=autoencoder_model_path, encoder_saved=encoder_saved)

    if not os.path.exists(novelty_detection_model_path):
        os.makedirs(novelty_detection_model_path)
    detector.save_model(novelty_detection_model_path)


if __name__ == '__main__':
    app.run(main)


