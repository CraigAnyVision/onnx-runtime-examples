// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License

#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>

int main(int argc, char**)
{
	//*************************************************************************
	// initialize environment... one environment per process
	// enviroment maintains thread pools and other state info
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

	// initialize session options if needed
	Ort::SessionOptions session_options;

	session_options.SetIntraOpNumThreads(1);

	if (argc > 1)
    {
        // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
	    // session (we also need to include cuda_provider_factory.h above which defines it)
        printf("Using CUDA Execution Provider\n");
        if(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0))
        {
            printf("ERROR: Failed to set CUDA runtime!\n");
            exit(EXIT_FAILURE);
        }
    }

	// Sets graph optimization level
	// Available levels are
	// ORT_DISABLE_ALL -> To disable all optimizations
	// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
	// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node
	// fusions) ORT_ENABLE_ALL -> To Enable All possible optimizations
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	//*************************************************************************
	// create session and load model into memory
	// using squeezenet version 1.3
	// URL = https://github.com/onnx/models/tree/master/squeezenet
	const char* model_path = "../squeezenet.onnx";

	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_options);

	//*************************************************************************
	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;

	// print number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
	                                       // Otherwise need vector<vector<>>

	printf("Number of inputs = %zu\n", num_input_nodes);

	// iterate over all input nodes
	for (size_t i = 0; i < num_input_nodes; i++)
	{
		// print input node names
		char* input_name = session.GetInputName(i, allocator);
		printf("Input %lu : name=%s\n", i, input_name);
		input_node_names[i] = input_name;

		// print input node types
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %lu : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims = tensor_info.GetShape();
		printf("Input %lu : num_dims=%zu\n", i, input_node_dims.size());
		for (size_t j = 0; j < input_node_dims.size(); j++)
		{
			printf("Input %lu : dim %lu=%jd\n", i, j, input_node_dims[j]);
		}
	}

	// Results should be...
	// Number of inputs = 1
	// Input 0 : name = data_0
	// Input 0 : type = 1
	// Input 0 : num_dims = 4
	// Input 0 : dim 0 = 1
	// Input 0 : dim 1 = 3
	// Input 0 : dim 2 = 224
	// Input 0 : dim 3 = 224

	//*************************************************************************
	// Similar operations to get output node information.
	// Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
	// OrtSessionGetOutputTypeInfo() as shown above.

	//*************************************************************************
	// Score the model using sample data, and inspect values

	size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
	                                           // use OrtGetTensorShapeElementCount() to get official size!

	std::vector<float> input_tensor_values(input_tensor_size);
	std::vector<const char*> output_node_names = {"softmaxout_1"};

	auto t1 = std::chrono::high_resolution_clock::now();

	// initialize input data with values in [0.0, 1.0]
	for (unsigned int i = 0; i < input_tensor_size; i++)
    {
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);
    }

	// create input tensor object from data values
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
	                                                          input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

	// score model & input tensor, get back output tensor
	auto output_tensors =
	    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

	// Get pointer to output tensor float values
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	assert(abs(floatarr[0] - 0.000045) < 1e-6);

	// score the model, and print scores for first 5 classes
	for (int i = 0; i < 5; i++)
    {
        printf("Score for class [%d] =  %f\n", i, floatarr[i]);
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    std::cout << "Infer took " << fp_ms.count() << " ms\n";

	// Results should be as below...
	// Score for class[0] = 0.000045
	// Score for class[1] = 0.003846
	// Score for class[2] = 0.000125
	// Score for class[3] = 0.001180
	// Score for class[4] = 0.001317
}
