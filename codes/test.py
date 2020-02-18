"""
Novelty detection model: Autoencoder + ocsvm
tensorflow version: 2.0.0-alpha0
prediction on python client with tenosflow seving

python test.py --model_version 1 --workspace_dir ../workspace --test_path ../test_data

"""

import glob
import cv2
import imageio
import configparser
import numpy as np
import tensorflow as tf
from absl import app 
from absl import flags
print("tensorflow version:", tf.__version__)
from novelty_detection import NoveltyDetection

flags.DEFINE_integer("model_version", 1, "The number of model")
flags.DEFINE_string("workspace_dir","../workspace", "main directory for novelty detection task" )
flags.DEFINE_string("test_path", "/mount/projekte/QUAR/mesutronic/cups_data/nok_test_data" , "path of test images")
flags.DEFINE_boolean("client", False, "if predict using served model")
FLAGS = flags.FLAGS
config = configparser.ConfigParser()

def detect_spot(image_path):
    from imutils import contours
    from skimage import measure
    import imutils
    print("image_path", image_path)
    image = cv2.imread(image_path,0)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    thresh = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels >10:
            mask = cv2.add(mask, labelMask)

        # find the contours in the mask, then sort them from left to
        # right
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]

        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(cX), int(cY)), int(radius),
                       (0, 0, 255), 3)
            cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)


def main(argv):
    print("Model number:", FLAGS.model_version, FLAGS.workspace_dir + "/config%d.ini" % FLAGS.model_version)
    config.read(FLAGS.workspace_dir + "/config%d.ini" % FLAGS.model_version)
    novelty_detection_model_path = FLAGS.workspace_dir + \
                                   config.get("Path", "novelty_detection_model_path") + "/" + str(FLAGS.model_version)
    imgs_format = config.get("Image", "format")

    detector = NoveltyDetection(config, FLAGS.model_version)

    test_imgs_list = glob.glob(FLAGS.test_path + "/*." + imgs_format)
    print("Number of test images", len(test_imgs_list))
    detector.load_model(novelty_detection_model_path)
    patch_model = detector.make_patch_model()

    i=0
    for img in test_imgs_list:
        i +=1
        # detect_spot(img)
        input = imageio.imread(img)
        print("image:   ", img)
        image = np.asarray(input)
        image = image.astype("float32")
        image = (image - np.min(image)) / ((np.max(image) - np.min(image)))

        image = np.expand_dims(image, axis=2)
        image= np.expand_dims(image, axis=0)

        # #predict with python client
        if FLAGS.client:
            print("predict with python client ...")
            nok_patch_auto, nok_patch_ocsvm = detector.predict_client(image)
        else:
            print("predict without client ...")
            nok_patch_auto, nok_patch_ocsvm = detector.predict(image)

        detector.visual_res(input, nok_patch_auto, nok_patch_ocsvm)
        
if __name__ == '__main__':
    app.run(main)
            

















