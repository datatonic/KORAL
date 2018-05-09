#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "KORAL.h"
#include "CUDAK2NN.h"
using namespace std::chrono;

#define cudaCalloc(A, B, STREAM) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B); \
        if (__cudaCalloc_err == cudaSuccess) cudaMemsetAsync(*A, 0, B, STREAM); \
} while (0)

struct Match {
	int q, t;
	Match() {}
	Match(const int _q, const int _t) : q(_q), t(_t) {}
};

int main() 
{
	constexpr int matchThreshold = 5;
	constexpr uint8_t KFAST_thresh = 60;
	constexpr float scale_factor = 1.2f;
	constexpr uint8_t scale_levels = 8;

	constexpr uint width = 1978;
	constexpr uint height = 1145;
	constexpr uint maxkpNum = 50000;
	
	cudaStream_t m_stream1, m_stream2;
	if (cudaStreamCreate(&m_stream1) == cudaErrorInvalidValue || cudaStreamCreate(&m_stream2) == cudaErrorInvalidValue)
		std::cerr << "Unable to create stream" << std::endl;

	uint64_t *d_desc1, *d_desc2;
	cudaCalloc((void**) &d_desc1, 64 * maxkpNum, m_stream1);
	cudaCalloc((void**) &d_desc2, 64 * maxkpNum, m_stream2);

	cv::Mat image1 = cv::imread("../1.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (!image1.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	std::vector<cv::KeyPoint> converted_kps1;
	KORAL koral(scale_factor, scale_levels, width, height, maxkpNum);
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (uint counter = 0; counter < 1000; ++counter) {
		koral.go(image1.data, image1.cols, image1.rows, KFAST_thresh);

	}
	high_resolution_clock::time_point end = high_resolution_clock::now();

	double sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(1);
	std::cout << "Processed 1000 images and retrieved " << koral.kps.size() << " keypoints and descriptors per image in " << sec << " s" << std::endl;

	cv::Mat image_with_kps1;
	for (const auto& kp : koral.kps) {
		// note that KORAL keypoint coordinates are on their native scale level,
		// so if you want to plot them accurately on scale level 0 (the original
		// image), you must multiply both the x- and y-coords by scale_factor^kp.scale,
		// as is done here.
		const float scale = static_cast<float>(std::pow(scale_factor, kp.scale));
		converted_kps1.emplace_back(scale*static_cast<float>(kp.x), scale*static_cast<float>(kp.y), 7.0f*scale, 180.0f / 3.1415926535f * kp.angle, static_cast<float>(kp.score));
	}
	int kpRef = koral.kps.size();
	cudaMemsetAsync(d_desc1, 0, 64 * (koral.kps.size() + 8), m_stream1);
	cudaMemcpyAsync(d_desc1, &koral.desc[0], 64 * (koral.kps.size() + 8), cudaMemcpyHostToDevice, m_stream1);
	cudaStreamSynchronize(m_stream1);

	cv::drawKeypoints(image1, converted_kps1, image_with_kps1, cv::Scalar::all(-1.0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("KORAL", CV_WINDOW_NORMAL);
	cv::imshow("KORAL", image_with_kps1);
	cv::waitKey(0);

	cv::Mat image2 = cv::imread("../2.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (!image2.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	std::vector<cv::KeyPoint> converted_kps2;
	start = high_resolution_clock::now();
	for (uint counter = 0; counter < 1000; ++counter) {
		koral.go(image2.data, image2.cols, image2.rows, KFAST_thresh);
	}
	end = high_resolution_clock::now();

	sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(1);
	std::cout << "Processed 1000 images and retrieved " << koral.kps.size() << " keypoints and descriptors per image in " << sec << " s" << std::endl;

	cv::Mat image_with_kps2;
	for (const auto& kp : koral.kps) {
		// note that KORAL keypoint coordinates are on their native scale level,
		// so if you want to plot them accurately on scale level 0 (the original
		// image), you must multiply both the x- and y-coords by scale_factor^kp.scale,
		// as is done here.
		const float scale = static_cast<float>(std::pow(scale_factor, kp.scale));
		converted_kps2.emplace_back(scale*static_cast<float>(kp.x), scale*static_cast<float>(kp.y), 7.0f*scale, 180.0f / 3.1415926535f * kp.angle, static_cast<float>(kp.score));
	}

	cudaMemsetAsync(d_desc2, 0, 64 * (koral.kps.size()), m_stream2);
	cudaMemcpyAsync(d_desc2, &koral.desc[0], 64 * (koral.kps.size()), cudaMemcpyHostToDevice, m_stream2);
	cudaStreamSynchronize(m_stream2);

	cv::drawKeypoints(image2, converted_kps2, image_with_kps2, cv::Scalar::all(-1.0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("KORAL", CV_WINDOW_NORMAL);
	cv::imshow("KORAL", image_with_kps2);
	cv::waitKey(0);
	//koral.freeGPUMemory();
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = d_desc2;
	resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
	resDesc.res.linear.desc.x = 32;
	resDesc.res.linear.desc.y = 32;
	resDesc.res.linear.sizeInBytes = 64 * koral.kps.size();

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t tex_q = 0;
	cudaCreateTextureObject(&tex_q, &resDesc, &texDesc, nullptr);

	cudaStreamSynchronize(m_stream1);
	cudaStreamSynchronize(m_stream2);

	// allocating space for match results
	int* d_matches;
	cudaMalloc(&d_matches, 4 * koral.kps.size());

	//std::cout << std::endl << "Warming up..." << std::endl;
	start = high_resolution_clock::now();	
	CUDAK2NN(d_desc1, static_cast<int>(kpRef), tex_q, static_cast<int>(koral.kps.size()), d_matches, matchThreshold);
	end = high_resolution_clock::now();

	// transferring matches back to host
	int* h_matches = reinterpret_cast<int*>(malloc(4 * koral.kps.size()));
	cudaMemcpy(h_matches, d_matches, 4 * koral.kps.size(), cudaMemcpyDeviceToHost);

	std::cout << "CUDA reports " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	std::vector<Match> matches;

    std::vector<cv::DMatch> dmatches;
	for (size_t i = 0; i < koral.kps.size(); ++i) {
		if (h_matches[i] != -1) {
			matches.emplace_back(i, h_matches[i]);
			dmatches.emplace_back(h_matches[i], i, 0.0f);
		}
	}
	sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(1);
	std::cout << "CUDAK2NN found " << matches.size() << " matches in " << sec * 1e3 << " ms" << std::endl;
	std::cout << "Throughput: " << static_cast<double>(koral.kps.size())*static_cast<double>(koral.kps.size()) / 1e6 << " billion comparisons/second." << std::endl << std::endl;

    cv::Mat image_with_matches;
    cv::drawMatches(image1, converted_kps1, image2, converted_kps2, dmatches, image_with_matches, cv::Scalar::all(-1.0), cv::Scalar::all(-1.0), std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow("Matches", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
    cv::imshow("Matches", image_with_matches);
    cv::waitKey(0);

	koral.freeGPUMemory();	

	cudaFree(d_desc1);
	cudaFree(d_desc2);
	cudaFree(d_matches);
	cudaDeviceReset();

	return 0;
}
