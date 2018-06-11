#ifndef __FACETRACKER_H_
#define __FACETRACKER_H_

#include "detector.h"
#include "landmark.h"
#include "stabilizer.h"

//#define LANDMARK_STABLE  

class FaceTracker
{
public:

	typedef enum {
		INSERTIMG_NULL = -1,
		INSERTIMG_INSERTED = 0,
		INSERTIMG_GET = 1,
	}InsertImgStatus;

	typedef struct _landmarkinfo {
		int fmark[LANDMARK_COUNT];
		double x;
		double y;
		double z;
	}LandMarkInfo;

	typedef struct __FaceResult {
		vector<Detector::DetctorResult> facedet;  
		vector<LandMarkInfo> landmark;  //the sequence  landmark is the same as vector<resultbox> boxs;  ?? vector+smartpoint(add LandMarkInfo in resultbox) good but too complicated
	}FaceResult;

	FaceTracker();
	~FaceTracker();
	InsertImgStatus InsertImage(cv::Mat& orgimg, FaceResult& ret);

private:

	typedef struct _facefilter {
		Stabilizer fstb;
#ifdef LANDMARK_STABLE
		LandMarkStabilizer mstb;//[LANDMARK_COUNT/2];
#endif
		int left;
		int right;
		int top;
		int bottom;
		int w;
		int h;
		bool bfounded;
		int id;
	}FaceFilter;

	Detector facedet_;
	LandMark lanmark_;
	vector<FaceFilter> facefilters_;
	cv::Mat facemeasure_;

	void CorrectFace(vector<Detector::resultbox>& boxs,int picw,int pich);
	void FillFaceMeause(const Detector::resultbox& b);
	void FillFaceBox(const cv::Mat& m, Detector::resultbox& b,int w, int h);
	void UpdateFaceFilter(FaceFilter& f, const Detector::resultbox& b);
};

#endif