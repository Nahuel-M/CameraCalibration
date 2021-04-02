#pragma once 

#include <stdio.h>
#include <iostream>
#include <sys/stat.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
string imageFolderName;
string calibrationFolderName = "calibration";

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
std::vector<cv::Point3f> generateChessGrid(cv::Mat_<float> doF)
{
    std::vector<cv::Point3f> points3D;
    for (int i = - (board_height - 1) / 2; i <= board_height / 2; i++) {
        for (int j = - (board_width - 1) / 2; j <= board_width / 2; j++) {
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
std::vector<cv::Point2f> projectChessGrid(std::vector<cv::Point3f> points3D)
{
    std::vector<cv::Point2f> points2D;
    float preMult = f / pixelSize;
    for (auto p : points3D)
    {
        points2D.push_back(Point2f{ preMult * p.x / p.z + imageSize.width/2, preMult * p.y / p.z +imageSize.height/2});
    }
    return points2D;
}

void plot2Dpoints(std::vector<cv::Point2f>& points2D)
{
    Mat_<uchar> plot{ imageSize };
    for (auto p : points2D) {
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

float getError(Mat_<float>& doF, std::vector<Point2f>& points)
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

float subGradientNumerical(Mat_<float>& doF, std::vector<Point2f>& points, int index, float stepSize)
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

Mat_<float> getGradient(Mat_<float>& doF, std::vector<Point2f>& points, bool verbose = false)
{
    Mat_<uchar> plot{ imageSize };
    Mat_<float> gradient{ doF.size() };
    gradient = 0;
    std::vector<Point2f> points2D = projectChessGrid(generateChessGrid(doF));
    for (int d = 0; d < doF.cols; d++) {
        gradient(d) = subGradientNumerical(doF, points, d, 0.00001)*3;
        if (d >= 2) { gradient(d) *= 10; }
        //if (d == 2){ gradient(d) /= 50; }
    }
    if (verbose) {
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
    
    Mat_<float> learningConsistency{ doF.size() };

    float error = getError(doF, points);
    //learningConsistency = (gradient.mul(prevGradient)>0);
    float dampening = error/(10000 + error);
    doF -= gradient * dampening * alpha;
    if (verbose) {
        std::cout << "Gradient: " << gradient << std::endl;
        std::cout << "Dampening: " << dampening << ", error: " << error / board_height / board_width << std::endl;
        std::cout << "doF: " << doF << std::endl;
    }
}

void massGradientDescentStep(std::vector<Mat_<float>>& doFs, std::vector<std::vector<Point2f>>& pointSets, float alpha, bool verbose = false)
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
    // Apply gradients to DoFs. The x and y position are taken from the respective gradients.
    float avgError = 0;
    for (int i = 0; i < doFs.size(); i++) {
        avgError += getError(doFs[i], pointSets[i]) / doFs.size();
    }

    float dampening = avgError / (10000 + avgError);
    for (int i = 0; i <doFs.size(); i++) {
        doFs[i](0) -= gradients[i](0) * alpha * dampening;
        doFs[i](1) -= gradients[i](1) * alpha * dampening;
        doFs[i](2) -= avgGradient(2) * alpha * dampening;
        doFs[i](3) -= avgGradient(3) * alpha * dampening;
        doFs[i](4) -= avgGradient(4) * alpha * dampening;
        doFs[i](5) -= avgGradient(5) * alpha * dampening;
    }

    if (verbose)
    {
        std::cout << "Mass descent step. Average gradient: " << avgGradient << "\t avgError: " << avgError / board_height / board_width 
            << "\t dampening factor: " << dampening << std::endl;
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
    std::vector< Point2f > corners;
    Mat gray;
    int board_n = board_width * board_height;
    int count = 0;
    for (auto j : imagePaths) {
        gray = cv::imread(j, cv::IMREAD_GRAYSCALE);
        std::cout << "Image nr " << count << std::endl;
        count++;
        bool found = false;
        found = cv::findChessboardCorners(gray, board_size, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
  
        if (found)
        {
            cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
                TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
            // If corners indexes are flipped: flip back
            if (corners[0].x > corners[1].x) {
                std::cout << "Reversing" << std::endl;
                reverse(corners.begin(), corners.end());
            }
            drawChessboardCorners(gray, board_size, corners, found);
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
    }
}

void getIntrinsicCameraData(Method method) 
{
    if (method == Method::Calculate) {
        /// Calculate intrinsic camera calibration info
        std::vector<Mat> rvecs, tvecs;
        Mat stdInt, stdEx, perViewErrors;
        std::cout << calibrateCamera(object_points, image_points, imageSize, K, D, rvecs, tvecs, stdInt, stdEx, perViewErrors) << std::endl;
        FileStorage fs(calibrationFolderName + "/CalibrationFile3", FileStorage::WRITE);
        fs << "K" << K;
        fs << "D" << D;
        fs << "board_width" << board_width;
        fs << "board_height" << board_height;
        fs << "square_size" << square_size;
        fs.release();
        printf("Done Calibration\n");
    }
    else if (method == Method::Import) {
        /// Read intrinsic camera calibration info from file
        FileStorage fs(calibrationFolderName + "/CalibrationFile3", FileStorage::READ);
        fs["K"] >> K;
        fs["D"] >> D;
        fs.release();
    }
}

void getUndistorted2DPattern(Method method)
{
    if (method == Method::Calculate) {
        /// Undistort detected 2D points
        for (auto distortedPoints : image_points) {
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
            Point2f centerPoint = points[(board_height * board_width - 1) / 2] - Point2f(imageSize / 2);
            float guessDepth = 0.94;
            Point3f initEstimate = { centerPoint.x * guessDepth / preMult, centerPoint.y * guessDepth / preMult, guessDepth };
            Mat_<float> doF{ Size(6,1) };
            doF << initEstimate.x, initEstimate.y, initEstimate.z,
                0, 0, 0;
            float preError = getError(doF, points) / board_height / board_width;

            /// Perform initial gradient descent
            for (int i = 0; i < 20000; i++) {
                gradientDescentStep(doF, points, alpha, false);
            }
            float error = getError(doF, points) / board_height / board_width;
            if (error > 1) {
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
        for (auto doF : doFs)
        {
            std::cout << doF << std::endl;
        }
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
    Mat_<float> meanDoF{ doFs[0].size() };
    for (auto doF : doFs)
    {
        meanDoF += doF;
    }
    meanDoF /= doFs.size();
    for (int i = 0; i < doFs.size(); i++) {
        jointDoFs.push_back(doFs[i]);
        for (int d = 2; d < doFs[i].cols; d++) {
            jointDoFs[i](d) = meanDoF(d);
        }
    }
    for (int i = 0; i < 3000; i++)
    {
        massGradientDescentStep(jointDoFs, undistorted_image_points, alpha * 1e-1, true);
    }
    FileStorage jointDoFsFile(calibrationFolderName + "/jointDoFs", FileStorage::WRITE);
    jointDoFsFile << "jointDoFs" << jointDoFs;
    jointDoFsFile.release();
    for (auto doF : jointDoFs)
    {
        std::cout << doF << std::endl;
    }
    std::cout << "Finished mass gradient descent." << std::endl;
}

int main()
{
    imageFolderName = "Series1";
    calibrationFolderName = "calibration";  // Foldername for files containing calibration data
    alpha = 5e-3;                           // Multiplier for gradient in gradient descent
    imageSize = Size{ 4056, 3040 };
    board_width = 7, board_height = 5;
    square_size = 28.7e-3;

    /// Calculate and export OR import image and object points. Stored in image_points and object_points.
    get2DPattern(Method::Import, imageFolderName);
    /// Calculate and export OR import intrinsic camera data. Stored in K and D.
    getIntrinsicCameraData(Method::Import);
    /// Calculate and export OR import undistorted image points. Stored in undistorted_image_points.
    getUndistorted2DPattern(Method::Import);
    /// Calculate and export OR import the position and rotation of the calibration pattern with respect to each camera. Stored in DoFs.
    estimatePatternDoFs(Method::Import);
    /// Calculates the joint pattern DoFs, taking into account that the angles and depth of the pattern are equal for all cameras.
    estimateJointPatternDoFs(Method::Calculate);

    return 0;
}