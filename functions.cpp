#include "functions.h"
#include <filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath)
{
	namespace fs = std::filesystem;
	std::vector<std::string> filePaths;
	for (auto& p : fs::directory_iterator(folderPath))
	{
		filePaths.push_back(p.path().u8string());
		//std::cout << p.path().u8string() << std::endl;
	}
	return filePaths;
}

std::vector<cv::Mat> getImages(std::string folderName, double scale)
{
	std::vector<std::string> files = getImagesPathsFromFolder(folderName);
	std::vector<cv::Mat> images;
	for (int i = 0; i < files.size(); i++) {
		std::cout << i << std::endl;
		images.push_back(cv::imread(files[i], cv::IMREAD_GRAYSCALE));
		cv::resize(images.back(), images.back(), cv::Size(), scale, scale);
	}
	return images;
}