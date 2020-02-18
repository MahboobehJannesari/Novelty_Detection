"""
model information:
saved_model_cli show --dir "novelty_detection_models/model_version" --all
"""

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util


def predict(patches, shape):
    hostport = "0.0.0.0:8500"
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "novelty_detection_models"
    request.model_spec.signature_name = "serving_default"
    # request.inputs["input_3"].CopyFrom(
    #     tf.make_tensor_proto(patches, shape=shape, dtype='float32')) #tensorflow 1.13
    request.inputs["input_3"].CopyFrom(
        tensor_util.make_tensor_proto(patches, shape=shape, dtype='float32')) # tensorflow 2.0.0 alpha

    result = stub.Predict(request, 30)
    prediction_auto = result.outputs['offset_layer']
    prediction_ocsvm = result.outputs['ocsvm_model']
    print("=======score_auto", tf.make_ndarray(prediction_auto))
    print("=======score_ocsvm", tf.make_ndarray(prediction_ocsvm))


    return  tf.make_ndarray(prediction_auto), tf.make_ndarray(prediction_ocsvm)


