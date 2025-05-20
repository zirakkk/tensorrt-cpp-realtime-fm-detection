# TensorRT C++ Hyperspectral FM Detection Inference Pipeline

[![CUDA](https://img.shields.io/badge/CUDA-12.2-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.6-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8.svg)](https://opencv.org/)
[![C++](https://img.shields.io/badge/C++-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)

## 🚀 Overview

A high-performance, production-ready inference engine built on NVIDIA TensorRT for real-time hyperspectral image analysis. This system delivers microsecond-level latency for critical applications in spectral imaging, leveraging GPU acceleration and memory optimization techniques.

## ✨ Key Features

- **Ultra-Low Latency**: Optimized memory transfers and CUDA stream management
- **Dynamic Batch Processing**: Handles variable workloads from single samples to 500,000+ vectors
- **Mixed Precision Support**: FP16/FP32 configurable precision for optimal performance/accuracy balance
- **Memory-Efficient Design**: Minimizes allocations and maximizes throughput
- **Hyperspectral Format Support**: Native processing of industry-standard ENVI files
- **Visualization Pipeline**: Integrated display capabilities for real-time monitoring

## 🔧 Technical Architecture

### Core Components

#### `Engine` Class
The central inference orchestrator that manages:
- TensorRT execution contexts and CUDA streams
- GPU memory allocation and buffer management
- Tensor dimension handling and shape inference
- Optimized host-device and device-host data transfers

#### Memory Management System
- **Pinned Memory Allocator**: Uses CUDA page-locked memory for maximum bandwidth
- **Buffer Reuse Strategy**: Minimizes allocations during high-throughput inference
- **Zero-Copy Techniques**: Reduces unnecessary data movement

#### Inference Pipeline
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Input Data  │ →  │ Preprocess  │ →  │GPU Inference│  → │ Postprocess │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ↑                                                        ↓
┌─────────────┐                                          ┌─────────────┐
│ ENVI Files  │                                          │Visualization│
└─────────────┘                                          └─────────────┘
```

## 💻 Performance Optimizations

| Technique | Implementation | Benefit |
|-----------|----------------|---------|
| **Pinned Memory** | `cudaMallocHost` for host buffers | Eliminates extra staging copy → up to ~2×–3× higher H2D throughput and enables true async transfers 2-3x faster host-device transfers |
| **Asynchronous Execution** | CUDA streams with callbacks | Overlaps kernel execution and data movement, hiding transfer latency |
| **Batch Optimization** | Dynamic batch sizing | Maximum GPU utilization |

## 📊 Benchmarks

| Batch Size | Inference Time | Memory Transfer | Total Latency |
|------------|----------------|-----------------|---------------|
| 145222     | ~29 ms         | ~9 ms           | ~38 ms        |


*Measured on NVIDIA RTX 3090, TensorRT 10.6, CUDA 12.2*

## 🛠️ Requirements

- **CUDA**: 12.2 or higher
- **TensorRT**: 10.6.0.26 or higher
- **OpenCV**: 4.9.0+ (Debug) / 4.11.0+ (Release)
- **C++ Compiler**: C++17 compatible
- **GPU**: NVIDIA with compute capability 7.0+
- **OS**: Windows 10/11 with Visual Studio 2022

## 📝 Usage

### Model Preparation
```bash
# Convert pre-trained ONNX model to TensorRT engine
trtexec --onnx=./NirEncD_MuscleFixed.onnx \
        --minShapes=input:1x96x1 \
        --optShapes=input:145222x96x1 \
        --maxShapes=input:500000x96x1 \
        --fp16 \
        --saveEngine=./NirEncD_Ch1_Muscle_Dynamic_1_File_FP16.plan
```

### C++ Integration
```cpp
// Configure engine options
options.precision = Precision::FP16;
options.optBatchSize = 145222;
options.maxBatchSize = 500000;

// Initialize engine
Engine engine(options);

// Load TensorRT engine
if (!engine.loadNetwork("path/to/model.plan")) {
    std::cerr << "Failed to load model" << std::endl;
    return -1;
}

// Prepare input data - [Input Tensor][batchsize][96 bands] format
std::vector<std::vector<std::vector<float>>> inputs = createInputsFromENVIFile();

// Run inference
std::vector<float> results;
if (!engine.runInference(inputs, output)) {
    std::cerr << "Inference failed" << std::endl;
    return -1;
}

// Process output
// ...
```

## 🔗 Related Repositories

- [TensorRT Samples](https://github.com/NVIDIA/TensorRT)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
