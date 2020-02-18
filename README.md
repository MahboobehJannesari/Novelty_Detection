# Novelty Detection

This guide provides instructions for how to set up proposed model for novelty detection task. The main structure of model is keras based with tensorflow backend. By following the steps, you will be able to use a
combination of an autoencoder and One Class Support Vector Machine (OCSVM) models for different datasets, train it on python scripts and test it as python and c++ clients using tensorflow serving.

### Contents

* autoencoder_model.py- contains definitions of autoencoder architecture.
* ocsvm_model.py-  contains ocsvm and some custom layers definitions in keras.
* client.py- code for python client when the model is served.
* novelt_detection.py- contains main functions for training and prediction.
* train.py- code for training the model.
* test.py- code for prediction. 
* tf-serving-client-cpp- codes to run the c++ client.
* workspace:
* * config*.ini- a text file for setting parameters.
* * requirements.txt- list of python packages.

### Dependencies
python 3.6
* absl-py==0.7.1
* imageio==2.5.0 
* matplotlib==3.1.0  
* numpy==1.16.4
* opencv-python==4.1.0.25 
* openpyxl==2.6.2 
* pandas==0.24.2 
* Pillow==6.0.0
* protobuf==3.8.0 
* scikit-image==0.15.0
* scikit-learn==0.21.2 
* scipy==1.3.0  
* tensorflow==2.0.0a0
* tensorflow-serving-api==1.13.0  
* grpcio==1.21.1
* xlrd==1.2.0


### Installing
Install python (https://www.python.org/)

For install Dependencies: 
```
pip3 install -r requirements.txt
```



### Serving the model with Tensorflow serving and Docker
Serving machine learning models is the process of taking a trained model and making it available to serve prediction requests. One of the
easiest ways to serve machine learning models is by using Tensorflow serving with Docker. 
Docker is a tool that packages software into units called containers that include everything needed to run the software. We now provide 
Docker image for serving and development for both CPU and GPU models in bellow steps:

Installing docker CE:  This will provide us all the tools we need to run and manage Docker containers. (https://docs.docker.com/install/linux/docker-ce/ubuntu/)

Saving the model: TensorFlow Serving uses the SavedModel format for its ML models. After training, the model is exported in this format using
keras instruction and saved in novelty_detection_model_path.

Serving with docker: Now serving the model using Docker is done with pulling the latest released TensorFlow Serving environment image,
and pointing it to the model with the following commands:
```
$ docker pull tensorflow/serving:latest-devel
```
```
$ docker run -p 8500:8500--mount type=bind,source=path/saved_models/novelty_detection_models,\
target=/models/novelty_detection_models -e  MODEL_NAME=novelty_detection_models -t tensorflow/serving 
```
 
 
## Deployment
* Running the GRPC client: with python or c++ script, we can send the served model images and get back predictions.

* Python client: 
To run python client in the docker container, --client should be add to command line for test.

* c++ client: 
To use the tensorflow serving c++ client, it is necessary to  install next dependencies: 

* protocol buffers (https://github.com/google/protobuf/tree/master/src)
* gRPC 
(https://github.com/grpc/grpc/tree/master/src/cpp)
* OpenCV 
(http://opencv.org/)
  
* Before the compilation we make sure that we have provided some host with a running TensorFlow Serving model in the tf-serving-client-cpp/main.cpp 
file by definition HOST, PORT and MODEL_NAME , MODEL_SIGNATURE_NAME, MODEL_INPUT such as python client.
* The compilation use CMake system.  Now, we can get output using c++ client by running tf-serving-client-cpp_linux/main.cpp.


## Authors

* **Mahboubeh Jannesari**  - [MahboobehJannesari](https://github.com/MahboobehJannesari)



# Novelty_Detection
