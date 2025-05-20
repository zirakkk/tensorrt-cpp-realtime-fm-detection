#pragma once
#include <vector> 
#include <memory>
#include <string> 
#include "NvInfer.h"

// Precision used for GPU inference
enum class Precision {
    FP32,
    FP16
};

// Default Options for the network
struct Options {
    bool doesSupportDynamicBatchSize = true;
    Precision precision = Precision::FP16;

    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;

    // Maximum allowable batch size
    int32_t maxBatchSize = 1;

	// Max allowable GPU memory to be used for model conversion, in bytes.
    // Applications should allow the engine builder as much workspace as they can afford;
    // at runtime, the SDK allocates no more than this and typically less.
    size_t maxWorkspaceSize = 3000000000;

    // GPU device index
    int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;

};

class Engine {

public:
    Engine(const Options& options);
    ~Engine();

    // Load and prepare the network for inference
    bool loadNetwork(const std::string& engineFilePath);

    // For testing: we run inference with dummy input data.
    // Input format [input_Tensor][batch][96 size vector], Output featureVector Format [batch] 
    //bool runInference(const std::vector<std::vector<std::vector<float>>> & inputs, std::vector<std::vector<std::vector<float>>>& featureVectors);
    bool runInference(const std::vector<std::vector<std::vector<float>>>& inputs,std::vector<float>& featureVectors);
private:
    void getDeviceNames(std::vector<std::string>& deviceNames);
    bool doesFileExist(const std::string& filepath);
    void clearGpuBuffers();

    // Pinned host buffer for input staging:
    float* m_pinnedInput{nullptr};
    size_t m_pinnedInputBytes{0};
    

    // Holds pointers to the input and output GPU buffers
    std::vector<void*> m_buffers;
    std::vector<uint32_t> m_outputLengthsFloat{};
	std::vector<nvinfer1::Dims2> m_inputDims; 
    std::vector<nvinfer1::Dims2> m_outputDims;
    std::vector<std::string> m_IOTensorNames;

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Options m_options;
    Logger m_logger;

    inline void checkCudaErrorCode(cudaError_t code);

};

