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

cv::Mat K;
cv::Mat D;

std::vector< std::vector< Point3f > > object_points;
std::vector< std::vector< Point2f > > image_points;
std::vector< Point2f > corners;
std::vector< std::vector< Point2f > > left_img_points;

Size imageSize = Size{4056, 3040};
int board_width = 7, board_height = 5;
float square_size = 28.7e-3;
double pixelSize = 1.55e-6;
double f = 6e-3;

Mat gray;
Size im_size;

std::vector<cv::Point3f> generateChessGrid(cv::Mat_<float> doF)
{
    std::vector<cv::Point3f> points3D;
    for (int i = - (board_height - 1) / 2; i <= board_height / 2; i++) {
        for (int j = - (board_width - 1) / 2; j <= board_width / 2; j++) {
            //float   Px = doF(2) / doF(0) + i * doF(3) + j * doF(6),
            //        Py = doF(2) / doF(1) + i * doF(4) + j * doF(7),
            //        Pz = doF(2) + i * doF(5) + j * doF(8);
            float cosa = cos(doF(3));
            float cosb = cos(doF(4));
            float cosc = cos(doF(5));
            float sina = sin(doF(3));
            float sinb = sin(doF(4));
            float sinc = sin(doF(5));
            float   Px = doF(0) + j * square_size * cosa * cosb +   i * square_size * (cosa * sinb * sinc - sina * cosc),
                    Py = doF(1) + j * square_size * sina * cosb +   i * square_size * (sina * sinb * sinc + cosa * cosc),
                    Pz = doF(2) + j * square_size * -sinb +         i * square_size * cosb * sinc;

            points3D.push_back({ Px, Py, Pz });
        }
    }
    return points3D;
}

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
    cv::namedWindow("Plot", cv::WindowFlags::WINDOW_NORMAL);
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

    cv::namedWindow("Performance", cv::WindowFlags::WINDOW_NORMAL);
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
        gradient(d) = subGradientNumerical(doF, points, d, 0.0001);
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
    float dampening = error/(13000 + error);
    doF -= gradient * dampening * alpha;
    if (verbose) {
        std::cout << "Gradient: " << gradient << std::endl;
        std::cout << "Dampening: " << dampening << ", error: " << error << std::endl;
        std::cout << "doF: " << doF << std::endl;
    }
    // Normalize i and j vectors to square_size
    //float iNorm = cv::pow(doF(3) * doF(3) + doF(4) * doF(4) + doF(5) * doF(5), 0.5);
    //doF(3) *= square_size / iNorm;
    //doF(4) *= square_size / iNorm;
    //doF(5) *= square_size / iNorm;
    //float jNorm = cv::pow(doF(6) * doF(6) + doF(7) * doF(7) + doF(8) * doF(8), 0.5);
    //doF(6) *= square_size / jNorm;
    //doF(7) *= square_size / jNorm;
    //doF(8) *= square_size / jNorm;
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
        if (d < 3) { doFstep(d) = 0.00005; }
        else { doFstep(d) = 0.001; }
        //doFstep(d) = 0.002;// getGradient(doF, points)(d) / 30000;
        analyzeGradient(doF, doFstep, points, 200, to_string(d), d);
    }
    plt::show(true);
    //plt::pause(1);
}

void setup_calibration(int board_width, int board_height, float square_size, vector<string>& images) {
    Size board_size = Size(board_width, board_height);
    int board_n = board_width * board_height;
    int count = 0;
    for (auto j : images) {
        gray = cv::imread(j, cv::IMREAD_GRAYSCALE);
        std::cout << count << std::endl;
        count++;
        char img_file[100];
        //cv::cvtColor(i, gray, COLOR_BGR2GRAY);

        bool found = false;

        found = cv::findChessboardCorners(gray, board_size, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
  
        
        if (found)
        {
            cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
                TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
            drawChessboardCorners(gray, board_size, corners, found);
        }
        
        vector< Point3f > obj;
        for (int i = 0; i < board_height; i++)
            for (int j = 0; j < board_width; j++)
                obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

        if (found) {
            //cout << k << ". Found corners!" << endl;
            std::cout << corners.size() << ", " << obj.size() << std::endl;
            image_points.push_back(corners);
            object_points.push_back(obj);
        }
    }
}

double computeReprojectionErrors(const vector< vector< Point3f > >& objectPoints,
    const std::vector< std::vector< Point2f > >& imagePoints,
    const std::vector< Mat >& rvecs, const vector< Mat >& tvecs,
    const Mat& cameraMatrix, const Mat& distCoeffs) {
    std::vector< Point2f > imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    std::vector< float > perViewErrors;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int)objectPoints.size(); ++i) {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
            distCoeffs, imagePoints2);
        err = cv::norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }
    return std::sqrt(totalErr / totalPoints);
}


int main()
{
    /// Find 2d pattern points in images
    //vector<string> images = getImagesPathsFromFolder("TestImgsCamPos");
    //setup_calibration(board_width, board_height, square_size, images);
    //FileStorage file("Points", FileStorage::WRITE);
    //file << "object_points" << object_points;
    //file << "image_points" << image_points;
    //file.release();
    //return 1;

    /// Read 2d pattern points from file
    //FileStorage file("Points", FileStorage::READ);
    //file["image_points"] >> image_points;
    //file["object_points"] >> object_points;
    std::vector< std::vector< Point2f > > undistorted_image_points;
    FileStorage undist("UndistortedPoints", FileStorage::READ);
    undist["points"] >> undistorted_image_points;

    /// Read intrinsic camera calibration info from file
    FileStorage fs("CalibrationFile2", FileStorage::READ);
    fs["K"] >> K;
    fs["D"] >> D;
    
    /// Undistort detected 2D points
    //for (auto pointsDist : image_points) {
    //    vector<Point2f> points;
    //    undistortPoints(pointsDist, points, K, D);
    //    for (int i = 0; i < points.size(); i++) {
    //        points[i] = Point2f{ points[i].x * imageSize.width + imageSize.width / 2, points[i].y * imageSize.height + imageSize.height / 2 };
    //    }
    //    undistorted_image_points.push_back(points);
    //}
    //FileStorage undist("UndistortedPoints", FileStorage::WRITE);
    //undist << "points" << undistorted_image_points;
    //undist.release();

    FileStorage campos("campos", FileStorage::WRITE);
    vector<Mat> doFs;
    vector<float> errors;
    int counter = 1;
    for (std::vector< Point2f > points : undistorted_image_points) {
        std::cout << "Point: " << counter << std::endl;
        counter++;

        Mat_<float> doF{ Size(6,1) };
        //doF << 0, 0, 0.75,
                //0, 0, 0;
        doF << 0.34649482, 0.19480404, 0.79643202, -0.092996247, 0.30407393, 0.37667418;

        //plotPerformance(projectChessGrid(generateChessGrid(doF)), points);
        //dofGradient(doF, points);

        float alpha = 1e-3;
        /// Perform initial gradient descent
        for (int i = 0; i < 2000; i++) {
            //doF = minimizeErrorWithDoF(doF, points, 0.02, 200);
            //std::cout << doF << std::endl;
            gradientDescentStep(doF, points, alpha, false);
            //waitKey(0);
            //plotPerformance(projectChessGrid(generateChessGrid(doF)), points);
            //dofGradient(doF, points);
        }
        std::cout << "doF: " << doF << std::endl;
        float error = getError(doF, points) / board_height / board_width;
        if (error > 30) {
            for (int i = 0; i < 10; i++) {
                gradientDescentStep(doF, points, alpha, true);
                plotPerformance(projectChessGrid(generateChessGrid(doF)), points);
                dofGradient(doF, points);
            }
        }

        doFs.push_back(doF);
        
        std::cout << "Error: " << error << std::endl;
        errors.push_back(error);
    }
    
    campos << "doFs" << doFs;
    campos << "rep_err" << errors;
    campos.release();
    for (auto doF : doFs)
    {
        std::cout << doF << std::endl;
    }
    for (auto error : errors)
    {
        std::cout << error << std::endl;
    }

    
    //calibrateCamera(object_points, image_points, gray.size(), K, D, rvecs, tvecs, stdInt, stdEx, perViewErrors, flag);

    //solvePnP(object_points, image_points, K, D, rvec, tvec, false);
    //cout << rvec << endl;

    //cout << "Calibration error: " << computeReprojectionErrors(object_points, image_points, rvecs, tvecs, K, D) << endl;

    //FileStorage fs("CalibrationFile2", FileStorage::WRITE);
    //fs << "K" << K;
    //fs << "D" << D;
    //fs << "board_width" << board_width;
    //fs << "board_height" << board_height;
    //fs << "square_size" << square_size;
    //printf("Done Calibration\n");

    return 0;
}