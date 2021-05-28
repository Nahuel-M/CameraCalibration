#pragma once 

const float  PI = 3.14159265358979f;

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "matplotlibcpp.h"

#include "functions.h"

namespace plt = matplotlibcpp;
using namespace cv;
using namespace std;
std::vector< std::vector< Point2f > > image_points;
std::vector< std::vector< Point2f > > undistorted_image_points;
std::vector< std::vector< Point3f > > object_points; 
vector<cv::Mat_<float>> doFs;
vector<cv::Mat_<float>> jointDoFs;
float avgAngle;
string imageFolderName;
string outputFolderName;
string calibrationFolderName = "calibration";

Mat_<uchar> prevGradientSign{ Size(6,1) , 255 };
Mat_<float> learningConsistency{ Size(6,1) };
Mat_<float> dampening{ Size(6,1), 1 };

Mat K, D;
float alpha;

enum class Method {
	Calculate, Import
};

Size imageSize;
int board_width, board_height;
float square_size;
double pixelSize = 1.55e-6;
double f = 6e-3;
float preMult = f / pixelSize;

/// Generate a chess grid using the degrees of freedom and the board dimensions
inline std::vector<cv::Point3f> generateChessGrid(cv::Mat_<float> doF)
{
	std::vector<cv::Point3f> points3D;
	for (float i = -(float)(board_height - 1) / 2; i <= board_height / 2; i += 1) {
		for (float j = -(float)(board_width - 1) / 2; j <= (float)(board_width) / 2; j+=1) {
			float   cosa = cos(doF(3)),
					cosb = cos(doF(4)),
					cosc = cos(doF(5)),
					sina = sin(doF(3)),
					sinb = sin(doF(4)),
					sinc = sin(doF(5));
			float   Px = doF(0) + j * square_size * cosa * cosb +   i * square_size * (cosa * sinb * sinc - sina * cosc),
					Py = doF(1) + j * square_size * sina * cosb +   i * square_size * (sina * sinb * sinc + cosa * cosc),
					Pz = doF(2) + j * square_size * -sinb +         i * square_size * cosb * sinc;
			points3D.push_back({ Px, Py, Pz });
		}
	}
	return points3D;
}
/// Project a chess grid (or any other set of points) to a camera located at O(0,0,0) with no rotation
inline std::vector<cv::Point2f> projectChessGrid(std::vector<cv::Point3f> points3D)
{
	std::vector<cv::Point2f> points2D;
	for (auto &p : points3D)
	{
		points2D.push_back(Point2f{ preMult * p.x / p.z + (float)imageSize.width/2, preMult * p.y / p.z + (float)imageSize.height/2});
	}
	return points2D;
}

void plot2Dpoints(std::vector<cv::Point2f>& points2D)
{
	Mat_<uchar> plot{ imageSize };
	for (auto &p : points2D) {
		if (p.x - 10 < 0 || p.x + 10 > imageSize.width || p.y - 10 < 0 || p.y + 10 > imageSize.height) continue;
		//std::cout << p << std::endl;
		plot(Rect{ Point2i(p), Size{10,10} }) = 255;
	}
	cv::resize(plot, plot, cv::Size(), 0.25, 0.25);
	cv::namedWindow("Plot", cv::WindowFlags::WINDOW_AUTOSIZE);
	cv::imshow("Plot", plot);
}

void plotPerformance(std::vector<cv::Point2f> points2D, std::vector<cv::Point2f> goal)
{
	Mat_<uchar> plot{ imageSize };
	for (int i = 0; i < points2D.size(); i++) {
		Point2f p = goal[i];
		bool f = 0;
		if (!(p.x - 10 < 0 || p.x + 10 > imageSize.width || p.y - 10 < 0 || p.y + 10 > imageSize.height)) {
			plot(Rect{ Point2i(p), Size{10,10} }) = 150;
		}
		else { f = 1; }

		p = points2D[i];
		if (!(p.x - 11 < 0 || p.x + 11 > imageSize.width || p.y - 11 < 0 || p.y + 11 > imageSize.height)) {
			plot(Rect{ Point2i(p), Size{10,10} }) = 255;
		}
		else { f = 1; }
		if (f == 0)
		{
			cv::line(plot, goal[i], points2D[i], 150, 5);
		}
	}
	cv::resize(plot, plot, cv::Size(), 0.25, 0.25);
	cv::namedWindow("Performance", cv::WindowFlags::WINDOW_AUTOSIZE);
	cv::imshow("Performance", plot);
}

inline float getError(Mat_<float>& doF, std::vector<Point2f>& points)
{
	float error = 0;
	std::vector<Point2f> minPoints2D = projectChessGrid(generateChessGrid(doF));
	for (int i = 0; i < board_height; i++) {
		for (int j = 0; j < board_width; j++) {
			int pSelect = i * board_width + j;
			error += norm(points[pSelect] - minPoints2D[pSelect]);
		}
	}
	return error;
}

inline float subGradientNumerical(Mat_<float>& doF, std::vector<Point2f>& points, int index, float stepSize)
{
	float minError = 0;
	float maxError = 0;
	Mat_<float> doFstep{ doF.size() };
	doFstep <<  0, 0, 0, 0, 0, 0, 0, 0, 0;
	doFstep(index) = stepSize;

	Mat_<float> minDof = doF - doFstep;
	Mat_<float> maxDof = doF + doFstep;

	return getError(maxDof, points) - getError(minDof, points);
}

inline Mat_<float> getGradient(Mat_<float>& doF, std::vector<Point2f>& points, bool verbose = false)
{
	Mat_<float> gradient{ doF.size() };
	gradient = 0;
	for (int d = 0; d < doF.cols; d++) {
		gradient(d) = subGradientNumerical(doF, points, d, 0.00001);
	}
	if (verbose) {
		std::vector<Point2f> points2D = projectChessGrid(generateChessGrid(doF));
		plotPerformance(points2D, points);
	}
	return gradient;
}

void analyzeGradient(Mat_<float>& doF, Mat_<float>& doFstep, std::vector<Point2f>& points, int steps, string title, int plotNum = -1)
{
	vector<float> error;
	vector<float> value;
	//vector<float> gradient;
	float min = 10, max = 20;
	float minError = 100000, maxError = 0;
	for (int k = 0; k < steps; k++) {
		Mat_<float> doF2 = doF + (k - steps/2) * doFstep;

		//gradient.push_back(getGradient(doF2, points)(plotNum));
		value.push_back(doF2(plotNum));
		error.push_back(getError(doF2, points) / board_height / board_width);

		if (error[k] < minError) {
			minError = error[k];
			min = value[k];
		} 
		if (error[k] > maxError) {
			maxError = error[k];
			max = value[k];
		}
		if (k == steps / 2) {
			plt::axvline(value.back());
		}
	}
	if (plotNum == -1) {
		plt::plot(error, "r");
		//plt::plot(gradient, "r--");
		plt::ylim(0, 2000);
		plt::title(title);
		//plt::save("Plot" + title);
		plt::show();
	}
	else
	{
		plt::plot(value, error, "r");
		//plt::plot(value, gradient, "r--");
		//plt::ylim(minError, minError *1.5f);
	}
   
}

void gradientDescentStep(Mat_<float>& doF, std::vector<Point2f>& points,float alpha, bool verbose = false)
{
   
	Mat_<float> gradient = getGradient(doF, points, verbose);
	
	learningConsistency = (gradient > 0)==prevGradientSign;
	prevGradientSign = gradient > 0;
	Mat dampen, undampen;
	multiply(dampening, learningConsistency, undampen, 1.f/255.f, dampening.type());
	multiply(dampening, 255-learningConsistency, dampen, 1.f / 255.f, dampening.type());
	dampening = 1.2f * undampen + 1.f / 1.2f * dampen;
	dampening = min(dampening, 1);

	float error = getError(doF, points);
	//learningConsistency = (gradient.mul(prevGradient)>0);
	//float dampening = error/(5000 + error);
	doF -= gradient.mul(dampening) * alpha;
	if (verbose) {
		std::cout << "Gradient: " << gradient << std::endl;
		std::cout << "Dampening: " << dampening << ", error: " << error / board_height / board_width << std::endl;
		std::cout << "doF: " << doF << std::endl;
	}
}

void jointGradientDescentStep(std::vector<Mat_<float>>& doFs, std::vector<std::vector<Point2f>>& pointSets, float alpha, bool verbose = false)
{
	// Get individual gradient for each camera
	std::vector<Mat_<float> > gradients(doFs.size());
	for (int i = 0; i < doFs.size(); i++) {
		gradients[i] = getGradient(doFs[i], pointSets[i], false);
	}
	// Combine to an average gradient
	Mat_<float> avgGradient{ gradients[0].size() }; 
	avgGradient = 0;
	for (int i = 0; i < doFs.size(); i++) {
		avgGradient += gradients[i];
	}
	avgGradient /= doFs.size();

	// Apply gradients to DoFs. The x and y position are taken from the respective gradients because they are different for each camera.
	float avgError = 0;
	for (int i = 0; i < doFs.size(); i++) {
		avgError += getError(doFs[i], pointSets[i]) / doFs.size();
	}

	learningConsistency = (avgGradient > 0) == prevGradientSign;
	prevGradientSign = avgGradient > 0;
	Mat dampen, undampen;
	multiply(dampening, learningConsistency, undampen, 1.f / 255.f, dampening.type());
	multiply(dampening, 255 - learningConsistency, dampen, 1.f / 255.f, dampening.type());
	dampening = 1.1f * undampen + 1.f / 1.2f * dampen;
	dampening = min(dampening, 1);
	dampening(0) = dampening(2);	// Set x and y dampening to z dampening, since x and y are different for each camera.
	dampening(1) = dampening(2);

	for (int i = 0; i <doFs.size(); i++) {
		doFs[i](0) -= gradients[i](0) * alpha * dampening(0);
		doFs[i](1) -= gradients[i](1) * alpha * dampening(1);
		doFs[i](2) -= gradients[i](2) * alpha * dampening(2);
		doFs[i](3) -= avgGradient(3) * alpha * dampening(3);
		doFs[i](4) -= avgGradient(4) * alpha * dampening(4);
		doFs[i](5) -= avgGradient(5) * alpha * dampening(5);
	}

	if (verbose)
	{
		Ptr<Formatter> Format = cv::Formatter::get();
		Format->set32fPrecision(2);
		std::cout.setf(std::ios::fixed, std::ios::floatfield);
		std::cout << std::left;
		//std::cout.precision(4);
		//std::cout << std::fixed;
		std::cout << "Joint descent step. 1st gradient: " << Format->format(avgGradient) << " Avg z, angles: "
			<< Format->format(doFs[0](Rect{ 2,0,4,1 })) << std::endl
			<< "avgError: " << avgError / board_height / board_width
			<< "\tdampFac: " << dampening(Rect{ 2,0,4,1 }) << std::endl;
	}
}

float minErrorOverDegree(Mat_<float>& doF, Mat_<float>& doFstep, std::vector<Point2f>& points, int steps, int degree)
{
	vector<float> error;
	vector<float> value;
	float min = 0;
	float minError = 10000000;
	for (int k = 0; k < steps; k++) {
		Mat_<float> doF2 = doF + (k - steps / 2) * doFstep;
		value.push_back(doF2(degree));
		error.push_back(getError(doF2, points) / board_height / board_width);
		if (error[k] < minError) {
			minError = error[k];
			min = value[k];
		}
	}
	return min;
}

Mat_<float> minimizeErrorWithDoF(Mat_<float>& doF, std::vector<Point2f>& points, float stepSize, int steps)
{
	Mat_<float> newdoF{ doF.size() };
	Mat_<float> doFstep{ doF.size() };
	for (int d = 0; d < doF.cols; d++) {
		doFstep << 0, 0, 0, 0, 0, 0, 0, 0, 0;
		doFstep(d) = stepSize;
		newdoF(d) = minErrorOverDegree(doF, doFstep, points, steps, d);
	}
	return newdoF;
}

void dofGradient(Mat_<float>& doF, std::vector<Point2f>& points) {
	plt::figure(1);
	plt::clf();
	plt::suptitle("Error vs doF value");
	for (int d = 0; d < doF.cols; d++) {
		plt::subplot(2, 3, d + 1);
		Mat_<float> doFstep{ doF.size() };
		doFstep << 0, 0, 0, 0, 0, 0, 0, 0, 0;
		if (d < 3) { doFstep(d) = 0.000005; }
		else { doFstep(d) = 0.0002; }
		//doFstep(d) = 0.002;// getGradient(doF, points)(d) / 30000;
		analyzeGradient(doF, doFstep, points, 2000, to_string(d), d);
	}
	plt::show(true);
	//plt::pause(1);
}

void setup_calibration(int board_width, int board_height, float square_size, vector<string>& imagePaths) {
	Size board_size = Size(board_width, board_height);
	std::cout << "Finding corners in patterns" << std::endl;
	std::vector< Point2f > corners;
	Mat gray;
	int board_n = board_width * board_height;
	int count = 0;
	for (auto &j : imagePaths) {
		gray = cv::imread(j, cv::IMREAD_GRAYSCALE);
		if (count == 0) {
			imageSize = gray.size();
		}
		std::cout << "Image nr " << count << std::endl;
		count++;
		bool found = false;
		//found = cv::findChessboardCorners(gray, board_size, corners,
		//	CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
		found = findChessboardCornersSB(gray, board_size, corners, CALIB_CB_ACCURACY | CALIB_CB_EXHAUSTIVE);
		//cv::SimpleBlobDetector::Params blobDetectorParams;
		//blobDetectorParams.filterByArea = true;
		//blobDetectorParams.maxArea = 250000;
		//blobDetectorParams.minArea = 10000;
		//blobDetectorParams.filterByCircularity = true;
		//blobDetectorParams.minCircularity = 0.8;
		//blobDetectorParams.maxCircularity = 1;
		//
		//blobDetectorParams.filterByConvexity = true;
		//blobDetectorParams.minConvexity = 0.7;
		//blobDetectorParams.maxConvexity = 1;
		//blobDetectorParams.filterByInertia = false;
		//blobDetectorParams.filterByColor = false;
		//Ptr< FeatureDetector > blobDetector = cv::SimpleBlobDetector::create(blobDetectorParams);
		//std::vector<KeyPoint> keypoints;
		//blobDetector->detect(gray, keypoints);
		//Mat im_with_keypoints;
		//drawKeypoints(gray, keypoints, im_with_keypoints, Scalar(0, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		//resize(im_with_keypoints, im_with_keypoints, Size(0, 0), 0.4, 0.4);
		//imshow("keypoints", im_with_keypoints);
		//waitKey(0);

		//cv::CirclesGridFinderParameters gridFinderParameter;
		//gridFinderParameter.gridType = cv::CirclesGridFinderParameters::GridType::SYMMETRIC_GRID;
		//gridFinderParameter.minGraphConfidence = 15;
		//gridFinderParameter.kmeansAttempts = 100;
		//gridFinderParameter.edgeGain = 20;
		//gridFinderParameter.edgePenalty = 0;
		//gridFinderParameter.convexHullFactor = 2;
		//gridFinderParameter.minDensity = 30;
		//gridFinderParameter.vertexGain = 20;



		//found = cv::findCirclesGrid(gray, board_size, corners, CALIB_CB_SYMMETRIC_GRID, blobDetector, gridFinderParameter);
		if (found)
		{
			//cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
			//	TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 120, 0.01));
			// If corners indexes are flipped: flip back
			if (corners[0].x > corners[1].x) {
				std::cout << "Reversing" << std::endl;
				reverse(corners.begin(), corners.end());
			}
			//drawChessboardCorners(gray, board_size, corners, found);
			//Rect desiredRoi = Rect{ (Point2i)corners[(corners.size())-8] - Point2i{100,100}, Size{200,200} };
			//Rect matRect = Rect{ 0,0,gray.cols, gray.rows };
			//resize(gray(desiredRoi & matRect), gray, Size{}, 3, 3, 0);
			//cv::imshow("image", gray);
			//waitKey(0);
		}
		
		vector< Point3f > obj;
		for (int i = 0; i < board_height; i++)
			for (int j = 0; j < board_width; j++)
				obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

		if (found) {
			std::cout << "Found " << corners.size() << " corners in the image" << std::endl;
			image_points.push_back(corners);
			object_points.push_back(obj);
		}
	}
}

void get2DPattern(Method method, string imageFolder)
{
	if (method == Method::Calculate) {
		/// Find 2D pattern points in images
		vector<string> imagePaths = getImagesPathsFromFolder(imageFolder);
		setup_calibration(board_width, board_height, square_size, imagePaths);
		FileStorage file(calibrationFolderName + "/Points", FileStorage::WRITE);
		file << "object_points" << object_points;
		file << "image_points" << image_points;
		file.release();
	}
	else if (method == Method::Import) {
		/// Read 2D pattern points from file
		FileStorage file(calibrationFolderName + "/Points", FileStorage::READ);
		file["image_points"] >> image_points;
		file["object_points"] >> object_points;
		file.release();
		// Get imagesize from probe image
		vector<string> imagePaths = getImagesPathsFromFolder(imageFolder);
		Mat sizeProbe = imread(imagePaths[0], cv::IMREAD_GRAYSCALE);
		imageSize = sizeProbe.size();
	}
}

void getIntrinsicCameraData(Method method) 
{
	if (method == Method::Calculate) {
		/// Calculate intrinsic camera calibration info
		std::vector<Mat> rvecs, tvecs;
		Mat stdInt, stdEx, perViewErrors;
		int flags = 0;
		double rmsReprojectionError = calibrateCamera(object_points, image_points, imageSize, K, D, 
			rvecs, tvecs, stdInt, stdEx, perViewErrors, flags,
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,100,DBL_EPSILON));
		std::cout << "Average reprojection error: " << rmsReprojectionError << std::endl;
		FileStorage fs(calibrationFolderName + "/CalibrationFile", FileStorage::WRITE);
		fs << "K" << K;
		fs << "D" << D;
		fs << "board_width" << board_width;
		fs << "board_height" << board_height;
		fs << "square_size" << square_size;
		fs << "reprojection_error" << rmsReprojectionError;
		fs.release();
		printf("Done Calibration\n");
	}
	else if (method == Method::Import) {
		/// Read intrinsic camera calibration info from file
		FileStorage fs(calibrationFolderName + "/CalibrationFile", FileStorage::READ);
		fs["K"] >> K;
		fs["D"] >> D;
		fs.release();
	}
}

void getUndistorted2DPattern(Method method)
{
	if (method == Method::Calculate) {
		/// Undistort detected 2D points
		for (auto &distortedPoints : image_points) {
			vector<Point2f> points;
			undistortPoints(distortedPoints, points, K, D, noArray(), K);
			undistorted_image_points.push_back(points);
		}
		FileStorage undist(calibrationFolderName + "/UndistortedPoints", FileStorage::WRITE);
		undist << "points" << undistorted_image_points;
		undist.release();
	}
	else if (method == Method::Import) {
		/// Read undistorted 2d pattern points from file
		FileStorage undist(calibrationFolderName + "/UndistortedPoints", FileStorage::READ);
		undist["points"] >> undistorted_image_points;
		undist.release();
	}
}

void estimatePatternDoFs(Method method)
{
	if (method == Method::Calculate) 
	{
		vector<float> errors;
		int counter = 0;

		for (std::vector< Point2f > points : undistorted_image_points) {
			std::cout << "Calibrating pattern nr: " << counter << std::endl;
			Point2f centerPoint{ 0,0 };
			for (auto& p : points)
			{
				centerPoint += p;
			}
			centerPoint /= (float)points.size();
			centerPoint = centerPoint - Point2f(imageSize / 2);
			float guessDepth = 0.77;
			Point3f initEstimate = { centerPoint.x * guessDepth / preMult, centerPoint.y * guessDepth / preMult, guessDepth };
			Mat_<float> doF{ Size(6,1) };
			doF << initEstimate.x, initEstimate.y, initEstimate.z,
				0, 0, 0;
			plotPerformance(projectChessGrid(generateChessGrid(doF)), points);
			waitKey(10);
			float preError = getError(doF, points) / board_height / board_width;
			// Set dampening factor back to 1 for each pattern
			dampening = 1;
			/// Perform initial gradient descent
			for (int i = 0; i < 20000; i++) {
				gradientDescentStep(doF, points, alpha, false);
			}
			float error = getError(doF, points) / board_height / board_width;
			if (error > 1.5) {
				for (int i = 0; i < 2000; i++) {
					gradientDescentStep(doF, points, alpha, true);
				}
			}
			error = getError(doF, points) / board_height / board_width;
			if (error > 1.5) {
				std::cout << "Error too large: " << error << std::endl;
				for (int i = 0; i < 10; i++) {
					gradientDescentStep(doF, points, alpha, true);
					plotPerformance(projectChessGrid(generateChessGrid(doF)), points);
					dofGradient(doF, points);
				}
			}

			doFs.push_back(doF);

			std::cout << "doF: " << doF << std::endl;
			std::cout << "Error from initial estimate: " << preError << ". Error after gradient descent: " << error << std::endl;
			errors.push_back(error);
			counter++;
		}

		FileStorage doFsFile(calibrationFolderName + "/doFs", FileStorage::WRITE);
		doFsFile << "doFs" << doFs;
		doFsFile << "rep_err" << errors;
		doFsFile.release();
		for (auto &doF : doFs)
		{
			std::cout << doF << std::endl;
		}
		std::cout << "Reprojection errors: " << std::endl;
		for (auto error : errors)
		{
			std::cout << error << std::endl;
		}
	}
	else if (method == Method::Import)
	{
		FileStorage doFsFile(calibrationFolderName + "/doFs", FileStorage::READ);
		doFsFile["doFs"] >> doFs;
		doFsFile.release();
	}
	
}

void estimateJointPatternDoFs(Method method)
{
	// take average z, a, b, c for all DoFs, since they should be equal.
	Mat_<float> meanDoF = Mat::zeros(doFs[0].size(), doFs[0].type());
	for (auto &doF : doFs)
	{
		meanDoF += doF;
	}
	meanDoF /= doFs.size();

	std::cout << "Starting joint gradient descent. Average DoF: " << meanDoF << std::endl;
	/// Setting angles and Z position of the pattern to the mean of the estimations
	for (int i = 0; i < doFs.size(); i++) {
		jointDoFs.push_back(doFs[i]);
		for (int d = 2; d < doFs[i].cols; d++) {
			jointDoFs[i](d) = meanDoF(d);
		}
	}
	/// Setting dampening matrix back to undampened
	dampening = 1;

	for (int i = 0; i < 20000; i++)
	{
		if (i % 100 == 0)
		{
			std::cout << i << " ";
			jointGradientDescentStep(jointDoFs, undistorted_image_points, alpha * 2e-1, true);
		}
		else {
			jointGradientDescentStep(jointDoFs, undistorted_image_points, alpha * 2e-1, false);
		}
	}
	std::cout << "Reprojection errors per camera: ";
	for (int e = 0; e < jointDoFs.size(); e++)
	{
		std::cout << getError(jointDoFs[e], undistorted_image_points[e]) / board_height / board_width << ", ";
	}
	std::cout << std::endl;
	/// Setting camera X and Y positions relative to center camera
	Mat centerDof = jointDoFs[(jointDoFs.size() - 1) / 2](Rect{ 0,0,2,1 }).clone();
	for (auto& DoF : jointDoFs)
	{
		DoF(Rect{ 0,0,2,1 }) -= centerDof;
		std::cout << DoF << std::endl;
	}

	FileStorage jointDoFsFile(calibrationFolderName + "/jointDoFs", FileStorage::WRITE);
	jointDoFsFile << "jointDoFs" << jointDoFs;
	jointDoFsFile.release();

	std::cout << "Finished mass gradient descent." << std::endl;
}

void estimateCameraAngles(float estimatedCamDistance)
{
	vector<cv::Mat_<float>> goalDoFs;
	for (auto &DoF : jointDoFs)
	{
		//std::cout << "b " << DoF << std::endl;
		Mat_<float > goalDoF = DoF.clone();
		goalDoF(0) = round(goalDoF(0) / estimatedCamDistance);
		goalDoF(1) = round(goalDoF(1) / estimatedCamDistance);
		goalDoFs.push_back(goalDoF);
		//std::cout << "a " << DoF << std::endl;
	}
	avgAngle = 0;
	for (int d = 0; d < jointDoFs.size(); d++)
	{
		float angle;
		if (abs(jointDoFs[d](1)) + abs(jointDoFs[d](0)) > estimatedCamDistance/2)
		{
			angle = atan2f(jointDoFs[d](1), jointDoFs[d](0));
		}
		else
		{
			angle = 0;
		}
		float goalAngle = atan2f(goalDoFs[d](1), goalDoFs[d](0));
		float diff;
		if (angle - goalAngle > PI/2) 
		{
			diff = angle - goalAngle - PI;
		}
		else if (abs(angle - goalAngle) < PI / 2)
		{
			diff = angle - goalAngle;
		}
		else 
		{
			diff = angle - goalAngle + PI;
		}
		avgAngle += diff;
		std::cout << "angles " << angle << ", " << goalAngle << ", " << diff << std::endl;
	}
	avgAngle /= jointDoFs.size() - 1;
	std::cout << "Average angle of cameras: "<< avgAngle << std::endl;
}

void exportImages(string imageFolderName, string outputFolderName)
{
	Mat map1, map2;
	vector<string> imagePaths = getImagesPathsFromFolder(imageFolderName);
	Mat sample = cv::imread(imagePaths[0], cv::IMREAD_GRAYSCALE);
	cv::initUndistortRectifyMap(K, D, Mat(), K, sample.size(), sample.type(), map1, map2);
	Mat gray;
	for (int i = 0; i < imagePaths.size(); i++) {
		// Undistort
		Mat color = cv::imread(imagePaths[i]);
		Mat gray;
		color.convertTo(color, CV_16UC3, 256);
		cvtColor(color, gray, COLOR_BGR2GRAY);
		Mat grayWarped;
		remap(gray, grayWarped, map1, map2, INTER_CUBIC);
		//cv::Mat mask;
		//std::vector< Point2i > patternCorners{ 
		//	undistorted_image_points[i][0], 
		//	undistorted_image_points[i][board_width-1],
		//	undistorted_image_points[i][(board_height-1)*board_width],
		//	undistorted_image_points[i][board_height*board_width-1]
		//};
		//fillConvexPoly(mask, patternCorners, 255, 8);
		//cv::Mat hist;
		//int histSize= 256;
		//float range[] = { 0, 256 };
		//const float* histRange = { range };
		//calcHist(&gray, 1, 0, mask, hist, 1, &histSize, &histRange, true, true);
		//float avgWhite = 0;
		//float totalCount = 0;
		//for (int h = 75; h < 180; h++)
		//{
		//	totalCount += hist.at<ushort>(0, h);
		//	avgWhite += hist.at<ushort>(0, h)*h;
		//}
		//avgWhite /= totalCount;
		//std::cout << "Average white on pattern: " << avgWhite << std::endl;
		//gray = gray * 110 / avgWhite;
		//// get rotation matrix for rotating the image around its center in pixel coordinates
		//cv::Point2f center((gray.cols - 1) / 2.0, (gray.rows - 1) / 2.0);
		//cv::Mat rotMat = cv::getRotationMatrix2D(center, avgAngle/PI*180, 1.0);
		//// determine bounding rectangle, center not relevant
		//cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), gray.size(), avgAngle).boundingRect2f();
		//// adjust transformation matrix
		//rotMat.at<double>(0, 2) += bbox.width / 2.0 - gray.cols / 2.0;
		//rotMat.at<double>(1, 2) += bbox.height / 2.0 - gray.rows / 2.0;

		//cv::Mat dst;
		//cv::warpAffine(gray, dst, rotMat, bbox.size());


		cv::imwrite(outputFolderName + "\\" + to_string(i) + ".png", grayWarped);
	}
}

void exportCamLocations(string outputFolderName)
{
	std::vector<Point3f> camPos;
	float cos_a = cos(-avgAngle);
	float sin_a = sin(-avgAngle);
	for (auto& doF : jointDoFs)
	{
		//camPos.push_back(Point3f{ -(cos_a * doF(0) - sin_a * doF(1)), (sin_a * doF(0) + cos_a * doF(1)), doF(2) });
		camPos.push_back(Point3f{ -doF(0), doF(1), doF(2) });
		doF(0) = -camPos.back().x;
		doF(1) = camPos.back().y;
		std::cout << camPos.back() << std::endl;
	}
	FileStorage camPosFile(outputFolderName + "/cameraPosition.xyz", FileStorage::WRITE);
	camPosFile << "camera_positions" << camPos;
	camPosFile << "camera_focal_length" << 5.f;
	camPosFile << "camera_pixel_size" << 1.55e-3f;
	camPosFile.release();
}

int main()
{
	imageFolderName = "sourceImages\\\\CalibrationBol5";		// Folder name for images
	outputFolderName = "processedImages\\\CalibrationBol5";
	calibrationFolderName = "calibration2";			// Folder name for files containing calibration data
	alpha = 1.5e-2;									// Multiplier for gradient in gradient descent
	board_width = 8, board_height = 6;				// Calibration pattern dimensions
	square_size = 2e-2;//28.7e-3;//43.2e-3			// Calibration pattern pitch (in meters)
	float camera_distance = 50e-3;					// Approximate distance between two cameras in the array

	/// Estimate and export OR import image and object points. Stored in image_points and object_points.
	get2DPattern(Method::Calculate, imageFolderName);
	/// Estimate and export OR import intrinsic camera data. Stored in K and D.
	getIntrinsicCameraData(Method::Import);
	/// Estimate and export OR import undistorted image points. Stored in undistorted_image_points.
	getUndistorted2DPattern(Method::Calculate);
	/// Estimate and export OR import the position and rotation of the calibration pattern with respect to each camera. Stored in DoFs.
	estimatePatternDoFs(Method::Calculate);
	/// Estimate the joint pattern DoFs, taking into account that the angles and depth of the pattern are equal for all cameras. Stored in jointDoFs.
	estimateJointPatternDoFs(Method::Calculate);
	/// Estimate camera angle around Z-axis (orthogonal to camera plane)
	estimateCameraAngles(camera_distance);
	/// Apply rotation to points and move from pattern doF to 
	exportCamLocations(outputFolderName);
	/// Apply rotation to images to compensate for camera angle.
	//exportImages(imageFolderName, outputFolderName);
	return 0;
}