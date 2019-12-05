#!/usr/bin/env bash

set -eu

axel https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz
tar xvf squeezenet.tar.gz
cp squeezenet/model.onnx squeezenet.onnx
