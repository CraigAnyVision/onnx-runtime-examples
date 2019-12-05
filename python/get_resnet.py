#!/usr/bin/env python3

import urllib.request
import tarfile

## retrieve our model from the ONNX model zoo
onnx_model_tar = "resnet50v2.tar.gz"
onnx_model_url = "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/" + onnx_model_tar
labels_filename = "imagenet-simple-labels.json"
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/" + labels_filename
urllib.request.urlretrieve(onnx_model_url, filename=onnx_model_tar)
urllib.request.urlretrieve(imagenet_labels_url, filename=labels_filename)
tar = tarfile.open(onnx_model_tar)
tar.extractall()
tar.close()
