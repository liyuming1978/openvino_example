#include "stabilizer.h"

Stabilizer::Stabilizer()
{
	bInit = false;
}

Stabilizer::~Stabilizer()
{
}

void Stabilizer::Init(int num)
{
	if (bInit)
		return;
	int snum = num * 2;
	filter_.init(snum, num);
	filter_.transitionMatrix = cv::Mat::zeros(snum, snum, CV_32F);
	for (int i = 0; i < snum; i++) {
		for (int j = 0; j < snum; j++) {
			if (i == j )
				filter_.transitionMatrix.at<float>(i, j) = 1.0;
			else if((j - num) == i)
				filter_.transitionMatrix.at<float>(i, j) = 3.0;  //bigger , faster
			else
				filter_.transitionMatrix.at<float>(i, j) = 0.0;
		}
	}
	cv::setIdentity(filter_.measurementMatrix);
	cv::setIdentity(filter_.processNoiseCov, cv::Scalar::all(1e-5));
	cv::setIdentity(filter_.measurementNoiseCov, cv::Scalar::all(1e-1));   //default 1e-1
	cv::setIdentity(filter_.errorCovPost, cv::Scalar::all(1));
	bInit = true;
}
cv::Mat Stabilizer::Correct(cv::Mat& measure)
{
	if (!bInit) {
		Init(measure.rows);
		premeasure_ = measure.clone()+99999;
	}
	cv::Mat tmepm;
	cv::absdiff(measure, premeasure_, tmepm);
	int diff = cv::sum(tmepm)[0] / measure.rows;
	if(diff>step1())
		cv::setIdentity(filter_.measurementNoiseCov, cv::Scalar::all(1e-4));  //if moving fast, quciker follow
	else if(diff>step2())
		cv::setIdentity(filter_.measurementNoiseCov, cv::Scalar::all(1e-3)); //if not, keep more stable
	else
		cv::setIdentity(filter_.measurementNoiseCov, cv::Scalar::all(1e-1)); //if not, keep more stable
	filter_.predict();
	premeasure_ =  filter_.correct(measure).rowRange(0, filter_.transitionMatrix.cols/2);
	return premeasure_;
}