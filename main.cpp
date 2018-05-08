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
using namespace std::chrono;

struct Match {
	int q, t;
	Match() {}
	Match(const int _q, const int _t) : q(_q), t(_t) {}
};

int main() {
	constexpr uint8_t KFAST_thresh = 60;
	constexpr float scale_factor = 1.2f;
	constexpr uint8_t scale_levels = 8;

	constexpr uint width = 1920;
	constexpr uint height = 1080;
	constexpr uint maxkpNum = 50000;
	// --------------------------------

	cv::Mat image = cv::imread("../test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	std::vector<cv::KeyPoint> converted_kps;
	KORAL koral(scale_factor, scale_levels, width, height, maxkpNum);
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (uint counter = 0; counter < 1000; ++counter) {
		koral.go(image.data, image.cols, image.rows, KFAST_thresh);

	}
	high_resolution_clock::time_point end = high_resolution_clock::now();

	double sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(1);
	std::cout << "Processed 1000 images and retrieved " << koral.kps.size() << " keypoints and descriptors per image in " << sec << " s" << std::endl;

	cv::Mat image_with_kps;
	for (const auto& kp : koral.kps) {
		// note that KORAL keypoint coordinates are on their native scale level,
		// so if you want to plot them accurately on scale level 0 (the original
		// image), you must multiply both the x- and y-coords by scale_factor^kp.scale,
		// as is done here.
		const float scale = static_cast<float>(std::pow(scale_factor, kp.scale));
		converted_kps.emplace_back(scale*static_cast<float>(kp.x), scale*static_cast<float>(kp.y), 7.0f*scale, 180.0f / 3.1415926535f * kp.angle, static_cast<float>(kp.score));
	}

	cv::drawKeypoints(image, converted_kps, image_with_kps, cv::Scalar::all(-1.0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("KORAL", CV_WINDOW_NORMAL);
	cv::imshow("KORAL", image_with_kps);
	cv::waitKey(0);
	koral.freeGPUMemory();
	
	cudaDeviceReset();
}
