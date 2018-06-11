#ifndef __POSEESTIMATE_H_
#define __POSEESTIMATE_H_

#include "facetracker.h"

class PoseEstimate
{
public:
	PoseEstimate();
	~PoseEstimate();
	void computeRadian(cv::Mat & frame, FaceTracker::LandMarkInfo & l);

private:
	bool isRotationMatrix(cv::Mat &R);
	void rotationMatrixToEulerAngles(cv::Mat &R, FaceTracker::LandMarkInfo & l);
	void getFeatures(const FaceTracker::LandMarkInfo& li, vector<cv::Point2d>& features);
};

#endif