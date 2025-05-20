#include <iostream>
#include <fstream>
#include <algorithm> // for std::all_of
#include <chrono>

#include "engine.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;

void Logger::log(Severity severity, const char* msg) noexcept {

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

// Check if a engine file exists
bool Engine::doesFileExist(const std::string& filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

void Engine::clearGpuBuffers() {
    if (!m_buffers.empty()) {
        // Free GPU memory of outputs
        const auto numInputs = m_inputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
        }
        m_buffers.clear();
    }
}

Engine::Engine(const Options& options)
    : m_options(options) {
    if (!m_options.doesSupportDynamicBatchSize) {
        std::cout << "Model does not support dynamic batch size, using optBatchSize and maxBatchSize of 1" << std::endl;
        m_options.optBatchSize = 1;
        m_options.maxBatchSize = 1;
    }
}

Engine::~Engine() {

    // Free pinned host buffer
    if (m_pinnedInput) {
        cudaFreeHost(m_pinnedInput);
        m_pinnedInput = nullptr;
    }

    // Free the GPU memory
    for (auto& buffer : m_buffers) {
        checkCudaErrorCode(cudaFree(buffer));
    }

    m_buffers.clear();
}

bool Engine::loadNetwork(const std::string& engineFilePath) {
    // Read the serialized model from disk
    std::ifstream file(engineFilePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    std::unique_ptr<IRuntime> runtime{ createInferRuntime(m_logger) };
    if (!runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
            ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    clearGpuBuffers();  // Ensure previous buffers are cleared
    m_buffers.resize(m_engine->getNbIOTensors());

    m_outputLengthsFloat.clear();
    m_inputDims.clear();
    m_outputDims.clear();
    m_IOTensorNames.clear();

    // Create a cuda stream
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
		const auto tensorShape = m_engine->getTensorShape(tensorName); // (-1, 96, 1) for input tensor, (-1, 1, 1) for output tensor
        const auto tensorDataType = m_engine->getTensorDataType(tensorName);

        if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
            
            // Note: We assume inputs are of type float
            if (tensorDataType != nvinfer1::DataType::kFLOAT) {
                auto msg = "Error, only float inputs are supported!";
                throw std::runtime_error(msg);
            }

            // Allocate memory for the input
            // Allocate enough to fit the max batch size (we could end up using less later)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i],
				m_options.maxBatchSize * tensorShape.d[1] * tensorShape.d[2] * sizeof(float), stream));   // [500000 * 96 * 1 * 4 (float)] for input  

            // Store the input dims for later use
			m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2]);  //[96, 1] for input 
        }
        else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
            // Compute the total number of elements in output tensor (excluding batch dim) i.e. [1,1]
            uint32_t outputLenFloat = 1;
            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= tensorShape.d[j];
            }

			m_outputLengthsFloat.push_back(outputLenFloat); // [1] for output

            // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
            // Allocate output buffer
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_options.maxBatchSize * sizeof(float), stream)); //& [1 * 500000 * 4 (float)] for output

            // Store output dimensions
            m_outputDims.emplace_back(tensorShape.d[1], tensorShape.d[2]); //[1, 1] for output
        }
        else {
            auto msg = "Error, IO Tensor is neither an input nor an output!";
            throw std::runtime_error(msg);
        }


    }

    // Allocate one pinned host buffer for staging all inputs at max batch size
    if (!m_inputDims.empty()) {
        size_t sampleSize = size_t(m_inputDims[0].d[0]) * m_inputDims[0].d[1];  // e.g. 96*1
        m_pinnedInputBytes = size_t(m_options.maxBatchSize) * sampleSize * sizeof(float);
        checkCudaErrorCode(cudaMallocHost(reinterpret_cast<void**>(&m_pinnedInput), m_pinnedInputBytes));
        std::cout << "[Engine] Allocated pinned host buffer: " << (m_pinnedInputBytes / (1024.0 * 1024.0)) << " MB\n";
    }

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

void Engine::checkCudaErrorCode(cudaError_t code) {
    if (code != 0) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

bool Engine::runInference(const std::vector<std::vector<std::vector<float>>>& inputs, std::vector<float>& featureVectors) {
    // First we do some error checking
    if (inputs.empty() || inputs[0].empty()) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }
    const auto numInputs = m_inputDims.size(); //M_inputDims stores objects of 2D shape. Here we have only one 2D object of shape [96, 1],so this vectors has size one thus numInputs is set to 1.
    if (inputs.size() != numInputs) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Incorrect number of inputs provided!" << std::endl;
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
        std::cout << "===== Error =====";
        std::cout << "The batch size is larger than the model expects!";
        std::cout << "Model max batch size: {}", m_options.maxBatchSize;
        std::cout << "Batch size provided to call to runInference: {}", inputs[0].size();
        return false;
    }

    const auto batchSize = static_cast<int32_t>(inputs[0].size()); // Ensure same batchsize followed for all inputs
    auto starth = std::chrono::high_resolution_clock::now();
    // Create a CUDA stream for asynchronous operations.
    cudaStream_t inferenceCudaStream;
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto& batchInput = inputs[i];// batchInput: vector of samples (each sample is vector<float>)
        const auto& dims = m_inputDims[i];     // expected dims: e.g. [96, 1]

        // Calculate expected number of elements per sample.
        size_t expectedSize = static_cast<size_t>(dims.d[0]) * static_cast<size_t>(dims.d[1]); // e.g. 96 * 1 = 96

        // Check that every sample in the batch has the expected size.
        if (!std::all_of(batchInput.begin(), batchInput.end(), [expectedSize](const std::vector<float>& sample) {
            return sample.size() == expectedSize;
            }))
        {
            std::cerr << "Error: Not all input samples have " << expectedSize << " elements." << std::endl;
            cudaStreamDestroy(inferenceCudaStream);
            return false;
        }
        
        
        // Set the input binding dimensions.
        // The engine was built with input shape [-1, 96, 1] where the first dimension is dynamic.
        // At runtime, we supply the actual batch size.
        nvinfer1::Dims3 inputDims;
        inputDims.nbDims = 3;
        inputDims.d[0] = batchSize;  // Set the dynamic batch dimension to our optimal batch size.
        inputDims.d[1] = 96;         // spectral vector length
        inputDims.d[2] = 1;          // fixed

        // Set the binding dimensions for the input (binding index 0 i.e. i=0). 
		if (!m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims)) { // [260417, 96, 1]
            std::cerr << "Failed to set input binding dimensions." << std::endl;
            cudaStreamDestroy(inferenceCudaStream);
            return false;
        }

        // Pack into the long-lived pinned host buffer
        size_t sampleSize = size_t(dims.d[0]) * dims.d[1];  // e.g. 96*1
        float* dstPtr = m_pinnedInput;
        for (int s = 0; s < batchSize; ++s) {
            std::memcpy(dstPtr,
                batchInput[s].data(),
                sampleSize * sizeof(float));
            dstPtr += sampleSize;
        }
        size_t bytesToCopy = batchSize * sampleSize * sizeof(float);

        // Single async copy from pinned host buffer ? device
        checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i],m_pinnedInput,bytesToCopy,cudaMemcpyHostToDevice,inferenceCudaStream));
    }
    cudaStreamSynchronize(inferenceCudaStream);
    auto endh = std::chrono::high_resolution_clock::now();
    auto durationh = std::chrono::duration_cast<std::chrono::milliseconds>(endh- starth);
    std::cout << "CPU-GPU Processing time: " << durationh.count() << " milliseconds" << std::endl;

    // Ensure all dynamic bindings are specified.
    if (!m_context->allInputDimensionsSpecified()) {
        cudaStreamDestroy(inferenceCudaStream);
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    auto start1 = std::chrono::high_resolution_clock::now();
    // --- Run Inference ---
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        cudaStreamDestroy(inferenceCudaStream);
        return false;
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Inference Processing time: " << duration1.count() << " milliseconds" << std::endl;


    auto start2 = std::chrono::high_resolution_clock::now();
    featureVectors.clear();


    
    // Get output binding index (it comes after input tensors)
    int32_t outputBinding = numInputs;  // numInputs is 1 in your case

    // Calculate total size for all predictions
    auto outputLength = m_outputLengthsFloat[outputBinding - numInputs];
    size_t totalBytes = batchSize * outputLength * sizeof(float);

    // Allocate buffer for all predictions at once
    std::vector<float> allPredictions(batchSize * outputLength);

    // Copy all predictions in one operation
    checkCudaErrorCode(cudaMemcpyAsync(
        allPredictions.data(),
        m_buffers[outputBinding],
        totalBytes,
        cudaMemcpyDeviceToHost,
        inferenceCudaStream
    ));

    // Wait for copy to complete
    cudaStreamSynchronize(inferenceCudaStream);

    featureVectors = std::move(allPredictions);

    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "GPU-CPU Processing time: " << duration2.count() << " milliseconds" << std::endl;
    
    // Synchronize the stream to ensure all asynchronous operations are complete.
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;
}




void Engine::getDeviceNames(std::vector<std::string>& deviceNames) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device = 0; device < numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

