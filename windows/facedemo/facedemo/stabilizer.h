#ifndef __STABILIZER__H_
#define __STABILIZER__H_

#include <opencv2\video\tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

class Stabilizer
{
public:
	Stabilizer();
	~Stabilizer();
	cv::Mat Correct(cv::Mat& measure);

private:
	virtual inline int step1() { return 8; }
	virtual inline int step2() { return 4; }
	void Init(int num);
	cv::KalmanFilter filter_;
	bool bInit;
	cv::Mat premeasure_;
};

class LandMarkStabilizer :public Stabilizer
{
public:
	LandMarkStabilizer() {}
private:
	virtual inline int step1() { return 2; }
	virtual inline int step2() { return 1; }
};

#endif