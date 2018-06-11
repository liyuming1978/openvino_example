#include "facetracker.h"
#include "poseestimate.h"

FaceTracker::FaceTracker()
{
	facedet_.Load(std::string(".\\model\\facedet.xml"), std::string(".\\model\\facedet.bin"));
	if(facedet_.err_msg!="")
		throw std::logic_error(facedet_.err_msg);
	lanmark_.Load(std::string(".\\model\\landmark.xml"), std::string(".\\model\\landmark.bin"));
	if (lanmark_.err_msg != "")
		throw std::logic_error(lanmark_.err_msg);

	facemeasure_ = cv::Mat::zeros(4, 1, CV_32F);
}


FaceTracker::~FaceTracker()
{
}

FaceTracker::InsertImgStatus FaceTracker::InsertImage(cv::Mat & orgimg, FaceResult & ret)
{
	Detector::InsertImgStatus faceret;

	faceret = facedet_.InsertImage(orgimg, ret.facedet);
	if (Detector::INSERTIMG_GET == faceret) {  //aSync call, you must use the ret image
		orgimg = ret.facedet[0].orgimg;  //only 1 camera
		CorrectFace(ret.facedet[0].boxs, orgimg.cols, orgimg.rows); //?? CorrectFace here is better
		ret.landmark.clear();
		int id = 0;
		PoseEstimate pe;

		for (auto& it = ret.facedet[0].boxs.begin(); it != ret.facedet[0].boxs.end();) {
			int x = it->left;
			int y = it->top;
			int width = it->right - it->left + 1;
			int height = it->bottom - it->top + 1;
			if (lanmark_.insertImage(orgimg.clone()(cv::Rect(x, y, width, height)))) {
				float* plandmark = lanmark_.getLandmark();
				float flmin = 99999.0;
				float flmax = -99999.0;
				for (int j = 0; j < LANDMARK_COUNT; j++) {
					if (plandmark[j] < flmin)
						flmin = plandmark[j];
					if (plandmark[j] >flmax)
						flmax = plandmark[j];
				}
				int fw = (int)(flmax - flmin);  //the model is -128,128
				if (fw < 80 || fw>200) {  //some face wrong detection may cause landmark wrong too.
					it = ret.facedet[0].boxs.erase(it);
					id++;
					continue;
				}

				LandMarkInfo li;
#ifdef LANDMARK_STABLE  //if only stable two , the result is not good, if stable 136,  the cpu cost is too high , so just pass it
				for (auto& f : facefilters_) { //not sort, sort will cause more copy
					if (f.id == id) {
#if 0
						for (int j = 0; j < LANDMARK_COUNT/2; j++) {
							cv::Mat landmarkm = cv::Mat(2, 1, CV_32F, &plandmark[j*2]);
							cv::Mat landmarkmc = f.mstb[j].Correct(landmarkm);
							memcpy(&li.fmark[j * 2], landmarkmc.data, sizeof(li.fmark[0]) * 2);
						}
#else  //too much cpu cost
						cv::Mat landmarkm = cv::Mat(LANDMARK_COUNT, 1, CV_32F, plandmark);
						cv::Mat landmarkmc = f.mstb.Correct(landmarkm);
						plandmark = (float*)landmarkmc.data;
#endif
						break;
					}
				}
#endif
				for (int j = 0; j < LANDMARK_COUNT / 2; j++) {
					li.fmark[2 * j] = (int)(plandmark[2 * j] * (it->right - it->left + 1) / 128 + it->left);
					li.fmark[2 * j + 1] = (int)(plandmark[2 * j + 1] * (it->bottom - it->top + 1) / 128 + it->top);
				}
				pe.computeRadian(orgimg, li);

				ret.landmark.push_back(li); //more copy here, todo optimize
			}
			id++;
			it++;
		}

		return INSERTIMG_GET;
	}
	else {
		if (Detector::INSERTIMG_NULL == faceret)
			return INSERTIMG_NULL;
		else {
			if(Detector::INSERTIMG_PROCESSED== faceret)
				orgimg = ret.facedet[0].orgimg;  //only 1 camera
			return INSERTIMG_INSERTED;
		}
	}		
}

void FaceTracker::FillFaceMeause(const Detector::resultbox& b)
{
	facemeasure_.at<float>(0) = (float)b.left;
	facemeasure_.at<float>(1) = (float)b.right;
	facemeasure_.at<float>(2) = (float)b.top;
	facemeasure_.at<float>(3) = (float)b.bottom;
}

void FaceTracker::FillFaceBox(const cv::Mat& m, Detector::resultbox& b,int w,int h)
{
	b.left = (int)m.at<float>(0);
	b.right = (int)m.at<float>(1);
	b.top = (int)m.at<float>(2);
	b.bottom = (int)m.at<float>(3);
	if (b.left < 0) b.left = 0;
	if (b.top < 0) b.top = 0;
	if (b.right >= w) b.right = w - 1;
	if (b.bottom >= h) b.bottom = h - 1;
}

void FaceTracker::UpdateFaceFilter(FaceTracker::FaceFilter& f, const Detector::resultbox& b)
{
	f.left = b.left;
	f.right = b.right;
	f.top = b.top;
	f.bottom = b.bottom;
	f.bfounded = true;
}

void FaceTracker::CorrectFace(vector<Detector::resultbox>& boxs, int picw, int pich)
{
	int total = (int)facefilters_.size();
	int id = 0;
	for (auto& f : facefilters_) {
		f.bfounded = false;
		f.w = f.right - f.left + 1;
		f.h = f.bottom - f.top + 1;
	}
	for (auto& b : boxs) {
		int min_d = INT_MAX;
		int curw = b.right - b.left + 1;
		int curh = b.bottom - b.top + 1;
		int minindex = -1;
		for (int i = 0; i < total; i++){
			if (facefilters_[i].bfounded)
				continue;
			if ((curw > facefilters_[i].w && curw > facefilters_[i].w * 2) ||
				(facefilters_[i].w > curw && facefilters_[i].w > curw * 2))
				continue;
			if ((curh > facefilters_[i].h && curh > facefilters_[i].h * 2) ||
				(facefilters_[i].h > curh && facefilters_[i].h > curh * 2))
				continue;
			int d = abs(facefilters_[i].left + facefilters_[i].w / 2 - b.left - curw / 2) + abs(facefilters_[i].top + facefilters_[i].h / 2 - b.top - curh / 2);
			if (d < min_d) {
				min_d = d;
				minindex = i;
			}
		}
		if (minindex >= 0 && min_d< (curw/4+ curh/4) ){
			FillFaceMeause(b);
			FillFaceBox(facefilters_[minindex].fstb.Correct(facemeasure_), b,picw,pich);
			UpdateFaceFilter(facefilters_[minindex], b);
			facefilters_[minindex].id = id;
		}
		else {
			FaceFilter f;
			UpdateFaceFilter(f, b);
			FillFaceMeause(b);
			f.fstb.Correct(facemeasure_);
			f.id = id;
			facefilters_.push_back(f);//more copy here, todo optimize
		}
		id++;
	}
	for (auto& it = facefilters_.begin(); it != facefilters_.end();) {
		if (!it->bfounded) {
			it = facefilters_.erase(it);
		}
		else
			it++;
	}
}