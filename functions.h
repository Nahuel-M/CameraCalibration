#pragma once
#include <vector>
#include <string>
#include <opencv2/core.hpp>

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath);

std::vector<cv::Mat> getImages(std::string folderName, double scale = 1);