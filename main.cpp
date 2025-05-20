#include "engine.h"
#include <chrono>
#include <stdexcept> // header for std::runtime_error
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>


using namespace std;

float muscle_channel_threshold = 0.8752055044318717;
float non_muscle_channel_threshold = 0.9076205231145336;

std::vector<std::vector<uint8_t>> convert_1d_to_2d(const std::vector<float>& data, int width, int height, int model_no) 
{
	if (data.size() != width * height) {
		throw std::invalid_argument("Data size does not match width and height.");
	}
	float threshold;
	if (model_no == 1)
		threshold = muscle_channel_threshold;
	else
		threshold = non_muscle_channel_threshold;

	std::vector<std::vector<uint8_t>> image_2d(height, std::vector<uint8_t>(width));
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (data[y * width + x] >= threshold)
				image_2d[y][x] = 255;
			else
				image_2d[y][x] = 0;

			//if (data[y * width + x] > 0 && data[y * width + x] < 0.9)
			//	printf("[y=%d,x=%d] %f\n", y, x, data[y * width + x]);
				//std::cout << std::format("({} x={}\n", y, x);

		}
	}
	return image_2d;
}

std::vector<std::vector<uint8_t>> convert_3d_to_2d(const std::vector<std::vector<std::vector<float>>>& data, int width, int height, int model_no, int band_no)
{
	if (data[0].size() != width * height) {
		throw std::invalid_argument("Data size does not match width and height.");
	}
	float threshold;
	if (model_no == 1)
		threshold = muscle_channel_threshold;
	else
		threshold = non_muscle_channel_threshold;

	std::vector<std::vector<uint8_t>> image_2d(height, std::vector<uint8_t>(width));
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			image_2d[y][x] = (data[0][y * width + x][band_no]) * 255;

			//if (data[y * width + x] > 0 && data[y * width + x] < 0.9)
			//	printf("[y=%d,x=%d] %f\n", y, x, data[y * width + x]);
				//std::cout << std::format("({} x={}\n", y, x);

		}
	}
	return image_2d;
}

void display_demo()
{

	// Load an image
	cv::Mat image = cv::imread("../../data/demo_img.jpg", cv::IMREAD_COLOR);

	// Check if the image was loaded successfully
	if (image.empty()) {
		cout << "Could not open or find the image!" << endl;
	}

	// Create a window to display the image
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

	// Show the image inside the window
	cv::imshow("Display window", image);

	// Wait for a key press until the window is closed
	cv::waitKey(0);

	// Destroy all opened windows
	cv::destroyAllWindows();
}

void display(std::vector<std::vector<uint8_t>> image_2d)
{
	int height = image_2d.size();
	int width = image_2d[0].size();
	cv::Mat image2D_gray(height, width, CV_8UC1);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			image2D_gray.at<uchar>(y, x) = image_2d[y][x];
		}
	}
	cv::imshow("Grayscale Image from 2D", image2D_gray);
	cv::waitKey(0);


}
std::vector<std::vector<std::vector<float>>> createInputsFromENVIFile(const std::vector<float>& longVector, const std::string filename, size_t width, size_t height, float dummyValue = 0.0f)
{
	// We need to read in chunks of 95 values.
	const size_t bandSize = 95;
	const size_t totalValues = longVector.size();
	
	// Ensure the long vector size is a multiple of 95.
	if (totalValues % bandSize != 0)
	{
		throw std::runtime_error("Input vector size is not a multiple of 95.");
	}

	// Number of samples (rows) we will create. we can also hardcode this value as 145222 
	const size_t numSamples = totalValues / bandSize;  // e.g., 145222 if totalValues is 13796090

	// Create the 2D input batch with shape [numSamples, 96]
	std::vector<std::vector<float>> inputBatch;
	inputBatch.reserve(numSamples);

	// 
	std::vector<std::vector<std::vector<float>>> inputs;

	// Read the ENVI image raw (FP32)
	std::ifstream file(filename, std::ios_base::binary);
	if (file.is_open()) {

		// Determine file size
		file.seekg(0, std::ios::end);
		int length = file.tellg();
		const size_t num_elements = file.tellg() / sizeof(float);

		// Create a buffer to hold the data
		//std::vector<float> buffer(num_elements);

		// Read the data
		file.seekg(0, std::ios::beg);
		for (size_t row = 0; row < height; ++row)
		{
			std::vector<float> sample;
			sample.resize(bandSize*width); 
			file.read(reinterpret_cast<char*>(sample.data()), bandSize*width *sizeof(float));
			
			std::vector<vector<float>> spec_2d(width, std::vector<float>(bandSize+1));
			for (size_t i = 0; i < (bandSize + 1); i++)
				for (size_t j = 0; j < width; j++)
					if (i < bandSize)
						spec_2d[j][i] = sample[i * width + j];
					else
						spec_2d[j][i] = 0.0f; // 95 values + 1 dummy value; use reserve when using push_back
			
			
			for (size_t t = 0; t < width; t++)
			// Now sample has 96 elements.
				inputBatch.push_back(std::move(spec_2d[t]));
		}


		inputs.push_back(std::move(inputBatch));

		if (file) {
			std::cout << "Successfully read " << numSamples << " spectral vectors from " << filename << std::endl;
			// Process the data in the buffer
		}
		else {
			std::cerr << "Error reading data from " << filename << std::endl;
		}

		file.close();
	}
	else {
		std::cerr << "Error opening file: " << filename << std::endl;
	}



	return inputs;
}


std::vector<std::vector<std::vector<float>>> createInputsFromLongVector(const std::vector<float>& longVector, float dummyValue = 0.0f)
{
	// We need to read in chunks of 95 values.
	const size_t chunkSize = 95;
	const size_t totalValues = longVector.size();

	// Ensure the long vector size is a multiple of 95.
	if (totalValues % chunkSize != 0)
	{
		throw std::runtime_error("Input vector size is not a multiple of 95.");
	}

	// Number of samples (rows) we will create. we can also hardcode this value as 145222 
	const size_t numSamples = totalValues / chunkSize;  // e.g., 145222 if totalValues is 13796090

	// Create the 2D input batch with shape [numSamples, 96]
	std::vector<std::vector<float>> inputBatch;
	inputBatch.reserve(numSamples);
	for (size_t row = 0; row < numSamples; ++row)
	{
		std::vector<float> sample;
		sample.reserve(chunkSize + 1); // 95 values + 1 dummy value
		size_t startIndex = row * chunkSize;

		// Copy 95 values for the sample.
		for (size_t i = 0; i < chunkSize; ++i)
		{
			sample.push_back(longVector[startIndex + i]);
		}

		// Append the dummy value.
		sample.push_back(dummyValue);

		// Now sample has 96 elements.
		inputBatch.push_back(std::move(sample));
	}

	// Move the 2D data into the 3D structure
	// Wrap the 2D batch in an outer vector to get shape [1, numSamples, 96]
	// The 2D data is transferred (moved) to the 3D data. Thus, this is not copying. 
	std::vector<std::vector<std::vector<float>>> inputs;
	inputs.push_back(std::move(inputBatch));

	return inputs;
}
bool fileExists(const std::string& filename) {
	std::ifstream file(filename);
	return file.is_open();
}

void printHelp(const std::string& programName) {
	std::cout << "------------------------------\n";
	std::cout << "\nUsage: Simulation.exe [options]\n\n"
		<< "Options:\n"
		<< "  -h, --help\t\tDisplay this help message\n"
		<< "  filename\t\tSpecify input ENVI file name (The header file (.hdr) is not needed.) \n"
		<< "  modelname\t\tSpecify TensorRT inference model\n\n"
		<< "Example: " << programName << "\n"
		<< "Example: " << programName << R"( ..\data\swir_bs_dirty_5x5_1.hyp ..\pretrained_models\NirEncD_Ch1_Muscle_Dynamic_1_File_FP16.plan)" << "\n";
}

int main(int argc, char* argv[])
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
	//std::string filename = "../../data/swir_bs_dirty_5x5_1.hyp";
	std::string filename = "swir_bs_dirty_5x5_1.hyp";

	//std::string enginePath = "C:/Users/Zirak.Khan/Projects/TensorRT_Cpp_Project/Pretrained_files/NirEncD_Ch0_Meat_Dynamic_100MB_FP16_New.plan";
	std::string enginePath = "../../pretrained_models/NirEncD_Ch1_Muscle_Dynamic_1_File_FP16.plan";
	//std::string enginePath = "NirEncD_Ch1_Muscle_Dynamic_1_File_FP16.plan";

	if (argc == 3) 
	{
		filename = argv[1];
		enginePath = argv[2];
	}
	else if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
	{
		printHelp(argv[0]);
	}

	if (!fileExists(filename) || !fileExists(enginePath)) {
		if (!fileExists(filename))
			std::cout << "Error: " << filename << " does not exist." << std::endl;
		if (!fileExists(enginePath))
			std::cout << "Error: " << enginePath << " does not exist." << std::endl;

		printHelp(argv[0]);
		return -1;
	}

	//display_demo();

	size_t width = 506, height = 287, bands = 95;
		
	// Specify our GPU inference configuration options
	Options options;

	options.doesSupportDynamicBatchSize = true;
	options.precision = Precision::FP16; 

	if (options.doesSupportDynamicBatchSize) {
		options.optBatchSize = 145222;
		options.maxBatchSize = 250000;
	}
	else {
		options.optBatchSize = 1;
		options.maxBatchSize = 1;
	}

	// Load the TensorRT engine from disk and prepare for inference
	Engine engine(options);


	bool succ = engine.loadNetwork(enginePath);
	if (!succ) {
		throw std::runtime_error("Unable to load TRT engine.");
	}


	std::vector<float> featureVectors;

	// For My Testing, You dont need this, you just have to procide the longVector.
	// For demonstration, we assume a long vector of the required size.
	const size_t totalValues = 13796090;  //  506 * 287 * 95 = 145222 * 95
	std::vector<float> longVector(totalValues, 1.0f); //Filling with dummy values.
	std::cout << "LongVector Shape is : [" << longVector.size() << "]" << std::endl;

	//auto inputs = createInputsFromLongVector(longVector, 0.0f); // Using 0.0f as the 96th dummy value.
	auto inputs = createInputsFromENVIFile(longVector, filename, width, height, 0.0f); // Using 0.0f as the 96th dummy value.


	// Now inputs has shape [1, 145222, 96].
	std::cout << "inputs Shape is : [" << inputs.size() << ", " << inputs[0].size() << ", "<< inputs[0][0].size()<< "]" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();

	// Now we run inference on dummy data.
	succ = engine.runInference(inputs, featureVectors);
	if (!succ) {
		throw std::runtime_error("Inference failed.");
	}

	std::cout << "featureVector Shape is: [" << featureVectors.size() << "]" << std::endl;

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "Total Processing time CPU-GPU + Inference + GPU-CPU: " << duration.count() << " milliseconds" << std::endl;


	std::vector<std::vector<uint8_t>> image_2d_one_band = convert_3d_to_2d(inputs, width, height, 1, 0);
	display(image_2d_one_band);

	//std::vector<std::vector<uint8_t>> image_2d = convert_1d_to_2d(flattenedOutput, width, height, 1);
	std::vector<std::vector<uint8_t>> image_2d = convert_1d_to_2d(featureVectors, width, height, 1);

	display(image_2d);

	return 0;
}
    