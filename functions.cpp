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

bool compareNat(const std::string& a, const std::string& b)
{
	if (a.empty())
		return true;
	if (b.empty())
		return false;
	if (std::isdigit(a[0]) && !std::isdigit(b[0]))
		return true;
	if (!std::isdigit(a[0]) && std::isdigit(b[0]))
		return false;
	if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
	{
		if (std::toupper(a[0]) == std::toupper(b[0]))
			return compareNat(a.substr(1), b.substr(1));
		return (std::toupper(a[0]) < std::toupper(b[0]));
	}

	// Both strings begin with digit --> parse both numbers
	std::istringstream issa(a);
	std::istringstream issb(b);
	int ia, ib;
	issa >> ia;
	issb >> ib;
	if (ia != ib)
		return ia < ib;

	// Numbers are the same --> remove numbers and recurse
	std::string anew, bnew;
	std::getline(issa, anew);
	std::getline(issb, bnew);
	return (compareNat(anew, bnew));
}

std::vector<std::string> getImagesPathsFromFolder(std::string folderPath)
{
	namespace fs = std::filesystem;
	std::vector<std::string> filePaths;
	for (auto& p : fs::directory_iterator(folderPath))
	{
		filePaths.push_back(p.path().u8string());
		//std::cout << p.path().u8string() << std::endl;
	}
	std::sort(filePaths.begin(), filePaths.end(), compareNat);
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