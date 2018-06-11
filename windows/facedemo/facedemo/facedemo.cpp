// facedemo.cpp : 定义控制台应用程序的入口点。
//
#include "facetracker.h"
#ifdef _DEBUG
#pragma comment(lib,"opencv_world341d.lib")  //include your opencv,  debug -- opencv_world341d ,release -- opencv_world341
#pragma comment(lib,"opencv_pvl341d.lib")
#else
#pragma comment(lib,"opencv_world341.lib")  //include your opencv,  debug -- opencv_world341d ,release -- opencv_world341
#pragma comment(lib,"opencv_pvl341.lib")
#endif

cv::Mat cameraMatrix;
void buildCameraMatrix(int cx, int cy, float focalLength) {
	if (!cameraMatrix.empty()) return;
	cameraMatrix = cv::Mat::zeros(3, 3, CV_32F);
	cameraMatrix.at<float>(0) = focalLength;
	cameraMatrix.at<float>(2) = static_cast<float>(cx);
	cameraMatrix.at<float>(4) = focalLength;
	cameraMatrix.at<float>(5) = static_cast<float>(cy);
	cameraMatrix.at<float>(8) = 1;
}

void drawAxes(cv::Mat& frame, cv::Point3f cpoint, FaceTracker::LandMarkInfo& headPose, float scale) {
	double yaw = headPose.x;
	double pitch = headPose.y;
	double roll = headPose.z;

	cv::Matx33f        Rx(1, 0, 0,
		0, cos(pitch), -sin(pitch),
		0, sin(pitch), cos(pitch));
	cv::Matx33f Ry(cos(yaw), 0, -sin(yaw),
		0, 1, 0,
		sin(yaw), 0, cos(yaw));
	cv::Matx33f Rz(cos(roll), -sin(roll), 0,
		sin(roll), cos(roll), 0,
		0, 0, 1);


	auto r = cv::Mat(Rz*Ry*Rx);
	buildCameraMatrix(frame.cols / 2, frame.rows / 2, 950.0);

	cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F), zAxis1(3, 1, CV_32F);

	xAxis.at<float>(0) = 1 * scale;
	xAxis.at<float>(1) = 0;
	xAxis.at<float>(2) = 0;

	yAxis.at<float>(0) = 0;
	yAxis.at<float>(1) = -1 * scale;
	yAxis.at<float>(2) = 0;

	zAxis.at<float>(0) = 0;
	zAxis.at<float>(1) = 0;
	zAxis.at<float>(2) = -1 * scale;

	zAxis1.at<float>(0) = 0;
	zAxis1.at<float>(1) = 0;
	zAxis1.at<float>(2) = 1 * scale;

	cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
	o.at<float>(2) = cameraMatrix.at<float>(0);

	xAxis = r * xAxis + o;
	yAxis = r * yAxis + o;
	zAxis = r * zAxis + o;
	zAxis1 = r * zAxis1 + o;

	cv::Point p1, p2;

	p2.x = static_cast<int>((xAxis.at<float>(0) / xAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
	p2.y = static_cast<int>((xAxis.at<float>(1) / xAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
	cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 0, 255), 2);

	p2.x = static_cast<int>((yAxis.at<float>(0) / yAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
	p2.y = static_cast<int>((yAxis.at<float>(1) / yAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
	cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 255, 0), 2);

	p1.x = static_cast<int>((zAxis1.at<float>(0) / zAxis1.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
	p1.y = static_cast<int>((zAxis1.at<float>(1) / zAxis1.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);

	p2.x = static_cast<int>((zAxis.at<float>(0) / zAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
	p2.y = static_cast<int>((zAxis.at<float>(1) / zAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
	cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 2);

	cv::circle(frame, p2, 3, cv::Scalar(255, 0, 0), 2);
}

int main()
{
	FaceTracker tracker;
	FaceTracker::FaceResult ret;

	cv::VideoCapture cap;
	cap.open(0);

	if (!cap.isOpened()) {
		std::cout << "can not open camera 0 \n";
		cap.release();
		getchar();
		return -1;
	}
	else {
		cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	}

	while (true) {
		cv::Mat frame;
		cap.read(frame);
		cv::Mat newframe = frame.clone(); 
		if (FaceTracker::INSERTIMG_GET == tracker.InsertImage(newframe, ret)) {  //newframe may change here
			for (int i = 0; i < ret.facedet[0].boxs.size(); i++) {   //to avoid memcpy, if only 1 batch use facedet[0]
				cv::rectangle(newframe, cvPoint(ret.facedet[0].boxs[i].left, ret.facedet[0].boxs[i].top),
					cvPoint(ret.facedet[0].boxs[i].right, ret.facedet[0].boxs[i].bottom), cv::Scalar(71, 99, 250), 2);
				for (int j = 0; j < LANDMARK_COUNT / 2; j++) {
					cv::circle(newframe, cv::Point(ret.landmark[i].fmark[2 * j], ret.landmark[i].fmark[2 * j + 1]), 1, cv::Scalar(0, 255, 0), 2);
				}
				cv::Point3f center((ret.facedet[0].boxs[i].right+ ret.facedet[0].boxs[i].left)/2, (ret.facedet[0].boxs[i].bottom+ret.facedet[0].boxs[i].top)/2, 0);
				drawAxes(newframe, center,ret.landmark[i], 50);
			}
		}
		cv::imshow("facelandmark", newframe);

		if (cv::waitKey(1) > 0)
			break;
	}

	return 0;
}

