#!/usr/bin/env python3

import onnxruntime
import numpy as np
from onnxruntime.datasets import get_example

example_model = get_example("sigmoid.onnx")
sess = onnxruntime.InferenceSession(example_model)

print(sess.get_providers())
sess.set_providers(['CUDAExecutionProvider'])
# session.set_providers(['CPUExecutionProvider'])

# identify our input name and shape
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)
input_type = sess.get_inputs()[0].type
print("Input type  :", input_type, "\n")

# identify our output name and shape
output_name = sess.get_outputs()[0].name
print("Output name  :", output_name)
output_shape = sess.get_outputs()[0].shape
print("Output shape :", output_shape)
output_type = sess.get_outputs()[0].type
print("Output type  :", output_type, "\n")

# pass in some input and compute our predictions
x = np.random.random(input_shape)
x = x.astype(np.float32)

print("x:\n", x, "\n")

result = sess.run([output_name], {input_name: x})

print("result:\n", result, "\n")
